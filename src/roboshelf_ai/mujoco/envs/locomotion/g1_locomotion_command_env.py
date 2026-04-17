"""
G1LocomotionCommandEnv — Fázis A locomotion tanítási env.

A Unitree G1 egy command-tracking locomotion policy-t tanul ebben az env-ben.
A policy bemenete: propriocepció + parancsvektor (v_forward, yaw_rate, ...).
A policy kimenete: 29 DoF aktuátorparancs.

Ez az env NEM tartalmaz retail bolt scene-t, NEM tartalmaz navigációs célt.
Egyetlen feladata: a robot tanuljon meg stabilan járni és parancsot követni.

Tanulság a legacy nav env-ből (v22):
  - Guggoló alappóz (hip_pitch=-0.1, knee=+0.3, ankle=-0.2) — unitree_rl_gym alapján
  - defaultctrl = keyframe-alapú, nem nulla — nulla ctrl ≠ egyensúly
  - 2 sub-step fizika — 1 sub-step instabilabb
  - Reset noise szükséges — nélküle determinisztikus reset → policy gyorsan overfit
  - Lábcsúszás és lábkeresztezés büntetés segít
"""

from __future__ import annotations

import os
import tempfile
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False
    logger.error("MuJoCo nem elérhető. Telepítsd: pip install mujoco>=3.6.0")


# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

G1_DOF = 29
G1_MASS_KG = 35.0

# G1 guggoló alappóz (unitree_rl_gym/unitree_rl_gym/envs/g1_config.py alapján)
# Ezek a qpos ízületi szögek, amelyek stabilis állást biztosítanak
_DEFAULT_JOINT_ANGLES = {
    # qpos index: érték (radián)
    7:  -0.1,   # bal hip_pitch
    10:  0.3,   # bal knee
    11: -0.2,   # bal ankle_pitch
    13: -0.1,   # jobb hip_pitch
    16:  0.3,   # jobb knee
    17: -0.2,   # jobb ankle_pitch
}

# defaultctrl indexek (aktuátor sorrend egyezik qpos-szal a G1-nél)
_DEFAULT_CTRL = {
    # ctrl index: érték
    0:  -0.1,   # bal hip_pitch
    3:   0.3,   # bal knee
    4:  -0.2,   # bal ankle_pitch
    6:  -0.1,   # jobb hip_pitch
    9:   0.3,   # jobb knee
    10: -0.2,   # jobb ankle_pitch
    # Karok (index 15-28)
    15:  0.2,
    16:  0.2,
    17:  0.0,
    18:  1.28,
    22:  0.2,
    23: -0.2,
    24:  0.0,
    25:  1.28,
}


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class G1LocomotionCommandEnv(gym.Env):
    """
    Unitree G1 command-tracking locomotion tanítási env (Fázis A).

    Observation space:
        - qpos[2:] (nq-2): ízületi szögek (freejoint xyz kihagyva, quat megtartva)
        - qvel (nv): ízületi sebességek + torzó lin/ang vel
        - IMU: roll, pitch (torzó dőlési szögei)
        - parancs vektor: (v_forward, v_lateral, yaw_rate, stance_w, step_h) — 5D
        Összesen: dinamikusan számított (model.nq - 2 + model.nv + 2 + 5)

    Action space:
        - 29 DoF aktuátorparancs, [-1, 1] normalizálva
        - 0.3 rad ACTION_SCALE — a defaultctrl körüli kis perturbációk

    Reward:
        - r_forward: parancs sebességkövetés (v_forward, yaw_rate)
        - r_upright: torzó függőleges tartása
        - r_alive:   talpon maradás per-lépés jutalma
        - r_smooth:  akció simasági büntetés
        - r_energy:  energiahatékonysági büntetés
        - r_feet_slip: lábcsúszás büntetés
        - r_feet_dist: lábak közötti min. távolság kényszer

    Termination:
        - torzó magassága < 0.4m vagy > 1.5m (elesett)
        - torzó dőlési szöge > 60° (felborult)
        - max_episode_steps (TimeLimitWrapper kezeli, vagy belső limit)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        # Parancs tartományok
        v_forward_range: Tuple[float, float] = (0.0, 1.5),
        yaw_rate_range:  Tuple[float, float] = (-1.0, 1.0),
        # Fizika
        sub_steps: int = 2,
        action_scale: float = 0.3,
        noise_scale: float = 0.01,
        # Reward súlyok
        w_forward: float = 2.0,
        w_yaw: float = 0.5,
        w_upright: float = 1.5,
        w_alive: float = 0.5,
        w_smooth: float = -0.1,
        w_energy: float = -0.05,
        w_feet_slip: float = -0.1,
        w_feet_dist: float = -1.0,
        feet_min_dist: float = 0.15,
    ):
        super().__init__()

        if not _MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo szükséges. Telepítsd: pip install mujoco>=3.6.0")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.sub_steps = sub_steps
        self.action_scale = action_scale
        self.noise_scale = noise_scale

        # Parancs tartományok
        self.v_forward_range = v_forward_range
        self.yaw_rate_range = yaw_rate_range

        # Reward súlyok
        self.w_forward = w_forward
        self.w_yaw = w_yaw
        self.w_upright = w_upright
        self.w_alive = w_alive
        self.w_smooth = w_smooth
        self.w_energy = w_energy
        self.w_feet_slip = w_feet_slip
        self.w_feet_dist = w_feet_dist
        self.feet_min_dist = feet_min_dist

        # Belső állapot
        self._episode_steps = 0
        self._last_action: Optional[np.ndarray] = None
        self._current_command: np.ndarray = np.zeros(5, dtype=np.float32)
        self._combined_xml_path: Optional[str] = None

        # MJCF betöltés
        self._load_model()

        # Observation és action space
        obs_size = self._obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # defaultctrl vektor (keyframe alapján)
        self._default_ctrl = self._build_default_ctrl()

        # Renderer
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, width=640, height=480)

        logger.info(
            f"G1LocomotionCommandEnv kész: "
            f"obs={obs_size}, act={self.model.nu}, "
            f"sub_steps={sub_steps}"
        )

    # ------------------------------------------------------------------
    # Betöltés
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """G1 MJCF betöltése. Csak a G1 modell, bolt nélkül."""
        g1_dir = self._find_g1_dir()

        # Minimális wrapper XML — csak a G1, semmi más
        combined_xml = '<mujoco model="g1_locomotion">\n  <include file="g1.xml"/>\n</mujoco>'

        # Process-safe temp fájl a G1 mappájában (így a meshdir helyes)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=g1_dir,
            prefix="_roboshelf_loco_", delete=False
        )
        tmp.write(combined_xml)
        tmp.close()
        self._combined_xml_path = tmp.name

        self.model = mujoco.MjModel.from_xml_path(self._combined_xml_path)
        self.data = mujoco.MjData(self.model)

        # Torzó (pelvis) body index
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self._torso_id == -1:
            self._torso_id = 1
            logger.warning("'pelvis' body nem található, fallback: body index 1")

        # Láb body indexek
        self._left_foot_id: Optional[int] = None
        self._right_foot_id: Optional[int] = None
        for left_name, right_name in [
            ("left_ankle_roll_link",  "right_ankle_roll_link"),
            ("left_ankle_link",       "right_ankle_link"),
            ("left_foot",             "right_foot"),
        ]:
            l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, left_name)
            r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, right_name)
            if l != -1 and r != -1:
                self._left_foot_id = l
                self._right_foot_id = r
                break

        print(
            f"  ✅ G1LocomotionCommandEnv betöltve: "
            f"{self.model.nbody} body, {self.model.nu} aktuátor, "
            f"nq={self.model.nq}, nv={self.model.nv}"
        )
        if self._left_foot_id is None:
            print("  ⚠️  Láb body-k nem találhatók — feet reward kikapcsolt")

    def _find_g1_dir(self) -> str:
        """Megkeresi a G1 MJCF könyvtárat."""
        candidates = [
            os.environ.get("G1_MODEL_PATH"),
            "/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/"
            "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
            "/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages/"
            "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
            os.path.expanduser("~/mujoco_menagerie/unitree_g1"),
        ]
        # site-packages keresés
        for p in sys.path:
            if "site-packages" in p:
                for sub in [
                    "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
                    "mujoco_menagerie/unitree_g1",
                ]:
                    full = os.path.join(p, sub)
                    candidates.append(full)

        for path in candidates:
            if path and os.path.exists(os.path.join(str(path), "g1.xml")):
                return str(path)

        raise FileNotFoundError(
            "Unitree G1 MJCF (g1.xml) nem található!\n"
            "Telepítsd: pip install mujoco-playground\n"
            "Vagy: git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie"
        )

    def _build_default_ctrl(self) -> np.ndarray:
        """Keyframe-alapú defaultctrl vektor."""
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for idx, val in _DEFAULT_CTRL.items():
            if idx < self.model.nu:
                ctrl[idx] = val
        return ctrl

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs_size(self) -> int:
        # qpos[2:] (freejoint xyz nélkül, quat megtartva) + qvel + IMU(2) + cmd(5)
        return (self.model.nq - 2) + self.model.nv + 2 + 5

    def _get_obs(self) -> np.ndarray:
        torso_xmat = self.data.body(self._torso_id).xmat.reshape(3, 3)

        # Roll és pitch kiszámítása a rotációs mátrixból
        roll  = np.arctan2(torso_xmat[2, 1], torso_xmat[2, 2])
        pitch = np.arctan2(-torso_xmat[2, 0],
                           np.sqrt(torso_xmat[2, 1]**2 + torso_xmat[2, 2]**2))

        obs = np.concatenate([
            self.data.qpos[2:].flat,           # freejoint xyz kihagyva, quat + ízületek
            self.data.qvel.flat,               # teljes qvel (torzó lin/ang + ízületek)
            [roll, pitch],                     # IMU: dőlési szögek
            self._current_command,             # parancs vektor (5D)
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Parancs mintavételezés
    # ------------------------------------------------------------------

    def _sample_command(self) -> np.ndarray:
        """Véletlen locomotion parancs mintavételezése (epizód elején)."""
        v_fwd = self.np_random.uniform(*self.v_forward_range)
        yaw   = self.np_random.uniform(*self.yaw_rate_range)
        return np.array([v_fwd, 0.0, yaw, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Állítsd be a guggoló alappózt (v22 tanulság)
        noise = self.np_random.uniform(
            -self.noise_scale, self.noise_scale, self.model.nq
        )

        if self.model.nq > 7:
            self.data.qpos[0] = 0.0 + noise[0]        # x
            self.data.qpos[1] = 0.0 + noise[1]        # y
            self.data.qpos[2] = 0.79                   # z — fix, stabil magasság
            self.data.qpos[3] = 1.0                    # qw
            self.data.qpos[4:7] = noise[4:7] * 0.001  # kis quaternion zaj

            # Guggoló alappóz ízületek
            for idx, val in _DEFAULT_JOINT_ANGLES.items():
                if idx < self.model.nq:
                    self.data.qpos[idx] = val + noise[idx]

            # Kar pozíciók (keyframe)
            if self.model.nq >= 36:
                self.data.qpos[22] =  0.2  + noise[22]
                self.data.qpos[23] =  0.2  + noise[23]
                self.data.qpos[25] =  1.28 + noise[25]
                self.data.qpos[29] =  0.2  + noise[29]
                self.data.qpos[30] = -0.2  + noise[30]
                self.data.qpos[32] =  1.28 + noise[32]

        # ctrl = defaultctrl (nulla ctrl ≠ egyensúly!)
        self.data.ctrl[:] = self._default_ctrl.copy()

        mujoco.mj_forward(self.model, self.data)

        # Epizód állapot
        self._episode_steps = 0
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)
        self._current_command = self._sample_command()

        obs = self._get_obs()
        info = {"command": self._current_command.copy()}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ctrl = defaultctrl + action * ACTION_SCALE
        raw_ctrl = self._default_ctrl + action * self.action_scale
        ctrl_range = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(raw_ctrl, ctrl_range[:, 0], ctrl_range[:, 1])

        # Fizikai szimuláció (2 sub-step — v22 tanulság)
        for _ in range(self.sub_steps):
            mujoco.mj_step(self.model, self.data)

        self._episode_steps += 1
        obs = self._get_obs()
        reward, info = self._compute_reward(action)

        # Terminálás: elesett
        terminated = not self._is_healthy()
        truncated = self._episode_steps >= self.max_episode_steps

        self._last_action = action.copy()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray):
        cmd_v_fwd = float(self._current_command[0])
        cmd_yaw   = float(self._current_command[2])

        torso_id = self._torso_id
        cvel = self.data.body(torso_id).cvel
        lin_vel_x = cvel[3]   # előre irányú sebesség (G1 X-tengely)
        ang_vel_z = cvel[2]   # Z-tengely körüli szögsebesség

        # 1. Sebességkövetés — exponenciális tracking (legged_gym stílusú)
        fwd_error = (lin_vel_x - cmd_v_fwd) ** 2
        yaw_error = (ang_vel_z - cmd_yaw) ** 2
        r_forward = self.w_forward * np.exp(-fwd_error / 0.25)
        r_yaw     = self.w_yaw     * np.exp(-yaw_error / 0.25)

        # 2. Upright: torzó függőleges tartása
        torso_xmat = self.data.body(torso_id).xmat.reshape(3, 3)
        upright = torso_xmat[2, 2]  # z-z elem: 1=függőleges, 0=oldalra dőlt
        r_upright = self.w_upright * upright

        # 3. Alive: talpon maradás per-lépés jutalma
        r_alive = self.w_alive if self._is_healthy() else 0.0

        # 4. Akció simaság büntetés
        if self._last_action is not None:
            r_smooth = self.w_smooth * float(
                np.sum(np.square(action - self._last_action))
            )
        else:
            r_smooth = 0.0

        # 5. Energia (aktuátor erőfeszítés) büntetés
        r_energy = self.w_energy * float(np.sum(np.square(action)))

        # 6. Lábcsúszás büntetés
        r_feet_slip = 0.0
        if self._left_foot_id is not None:
            for foot_id in (self._left_foot_id, self._right_foot_id):
                foot_contact = self.data.cfrc_ext[foot_id, 2] > 1.0
                if foot_contact:
                    foot_vel = self.data.body(foot_id).cvel[3:5]
                    r_feet_slip += self.w_feet_slip * float(np.sum(np.square(foot_vel)))

        # 7. Lábak közötti min. távolság kényszer
        r_feet_dist = 0.0
        if self._left_foot_id is not None:
            left_pos  = self.data.body(self._left_foot_id).xpos[:2]
            right_pos = self.data.body(self._right_foot_id).xpos[:2]
            foot_dist = float(np.linalg.norm(left_pos - right_pos))
            if foot_dist < self.feet_min_dist:
                violation = self.feet_min_dist - foot_dist
                r_feet_dist = self.w_feet_dist * (violation ** 2)

        total = r_forward + r_yaw + r_upright + r_alive + r_smooth + r_energy + r_feet_slip + r_feet_dist

        info = {
            "r_forward":    r_forward,
            "r_yaw":        r_yaw,
            "r_upright":    r_upright,
            "r_alive":      r_alive,
            "r_smooth":     r_smooth,
            "r_energy":     r_energy,
            "r_feet_slip":  r_feet_slip,
            "r_feet_dist":  r_feet_dist,
            "lin_vel_x":    lin_vel_x,
            "ang_vel_z":    ang_vel_z,
            "cmd_v_fwd":    cmd_v_fwd,
            "cmd_yaw":      cmd_yaw,
            "torso_z":      float(self.data.body(torso_id).xpos[2]),
            "episode_steps": self._episode_steps,
        }
        return float(total), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_healthy(self) -> bool:
        torso_z = float(self.data.body(self._torso_id).xpos[2])
        if not (0.4 < torso_z < 1.5):
            return False
        # Dőlési szög ellenőrzés: ha a torzó Z-tengelye túl elfordul
        torso_xmat = self.data.body(self._torso_id).xmat.reshape(3, 3)
        upright = torso_xmat[2, 2]  # cos(dőlési szög)
        if upright < 0.5:  # > ~60° dőlés
            return False
        return True

    # ------------------------------------------------------------------
    # Render / Close
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array" and self._renderer is not None:
            mujoco.mj_forward(self.model, self.data)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._combined_xml_path and os.path.exists(self._combined_xml_path):
            try:
                os.remove(self._combined_xml_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Standalone teszt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("G1LocomotionCommandEnv — standalone teszt")
    print("=" * 50)

    env = G1LocomotionCommandEnv()
    obs, info = env.reset()

    print(f"Obs shape:    {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Parancs:      v_fwd={info['command'][0]:.2f}, yaw={info['command'][2]:.2f}")
    print()

    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  Robot elesett a {step+1}. lépésnél (torso_z={info['torso_z']:.2f}m)")
            break
        if truncated:
            print(f"  Epizód max lépésszámnál vágva: {step+1}")
            break

    print(f"\n  Összesített reward: {total_reward:.2f}")
    print(f"  Utolsó lépés info:")
    for k in ("r_forward", "r_upright", "r_alive", "lin_vel_x", "torso_z"):
        print(f"    {k}: {info[k]:.4f}")
    print(f"\n✅ Env futott!")
    env.close()
