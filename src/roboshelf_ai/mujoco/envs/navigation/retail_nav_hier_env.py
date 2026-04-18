"""
Hierarchikus retail navigációs env — Fázis B.

Architektúra:
    High-level: NavPolicy (PPO, 5 Hz) → LocomotionCommand(v_forward, yaw_rate)
    Low-level:  UnitreeRLGymAdapter (motion.pt LSTM, 50 Hz) → torque[12]

Observation (9 dim):
    [0:2]  robot_xy               — torso pozíció
    [2]    goal_dist              — célpont távolság (m)
    [3]    goal_angle             — szöghiba a célponthoz (rad, robot frame)
    [4]    torso_heading          — torso orientáció (yaw, rad)
    [5:7]  torso_lin_vel          — előre/oldal sebesség (m/s)
    [7:9]  foot_contact           — bal/jobb talp kontakt (0/1)

Action (2 dim, normalizálva [-1, 1]):
    [0]  v_forward  → [0.0, 1.0] m/s
    [1]  yaw_rate   → [-0.8, 0.8] rad/s

Curriculum (4 szint, retail_nav_hier_v1.yaml alapján):
    1. Egyenes 0.5m
    2. Egyenes 1.5m
    3. Kanyar 2.0m
    4. Teljes 3.3m pálya akadályokkal

Elfogadási feltétel: 50%+ célpont elérés random startból.

Használat:
    from roboshelf_ai.mujoco.envs.navigation.retail_nav_hier_env import RetailNavHierEnv
    env = RetailNavHierEnv(cfg=cfg)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from roboshelf_ai.locomotion.policy_adapter import UnitreeRLGymAdapter, G1_DEFAULT_ANGLES
from roboshelf_ai.core.interfaces.locomotion_command import LocomotionCommand

# ---------------------------------------------------------------------------
# Útvonal konstansok
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]  # roboshelf-ai-redesign/

_SCENE_XML_CANDIDATES = [
    _REPO_ROOT / "unitree_rl_gym/resources/robots/g1_description/scene.xml",
    Path.home() / "unitree_rl_gym/resources/robots/g1_description/scene.xml",
]

_MOTION_PT_CANDIDATES = [
    _REPO_ROOT / "unitree_rl_gym/deploy/pre_train/g1/motion.pt",
    Path.home() / "unitree_rl_gym/deploy/pre_train/g1/motion.pt",
]

_STORE_XML = _REPO_ROOT / "src/envs/assets/roboshelf_retail_store.xml"

# ---------------------------------------------------------------------------
# Szimuláció paraméterei
# ---------------------------------------------------------------------------

SIM_DT           = 0.002   # 500 Hz
LOCO_HZ          = 50      # locomotion policy frekvencia
NAV_HZ           = 5       # nav policy frekvencia
LOCO_DECIMATION  = int(500 / LOCO_HZ)   # 10: minden 10. sim lépésnél fut loco
NAV_DECIMATION   = int(500 / NAV_HZ)    # 100: minden 100. sim lépésnél fut nav step

# G1 torso body neve (scene.xml-ben)
TORSO_BODY_NAME = "pelvis"

# Elfogadási thresholdok
LOCO_COLLAPSE_Z       = 0.5   # m — ez alatt: locomotion összeomlás
LOCO_COLLAPSE_UPRIGHT = 0.5   # cos — ez alatt: eldőlt

# Reward skálák (config-ból felülírhatók)
DEFAULT_REWARD = dict(
    w_goal_approach =  5.0,
    w_goal_reached  = 100.0,
    w_collision     = -20.0,
    w_loco_fail     = -10.0,
    w_orientation   =  -0.5,
    w_time          =  -0.01,
)


# ---------------------------------------------------------------------------
# RetailNavHierEnv
# ---------------------------------------------------------------------------

class RetailNavHierEnv(gym.Env):
    """Hierarchikus retail navigációs Gymnasium env.

    A nav policy (PPO) 5 Hz-en ad LocomotionCommand-ot,
    a locomotion adapter (motion.pt) 50 Hz-en futtatja a PD control-t,
    a MuJoCo szimuláció 500 Hz-en fut.

    Egy gym.step() = NAV_DECIMATION (100) szimulációs lépés.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": NAV_HZ}

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        curriculum_level: int = 1,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        cfg = cfg or {}

        env_cfg  = cfg.get("env", {})
        rew_cfg  = cfg.get("reward", DEFAULT_REWARD)

        # --- Reward súlyok ---
        self.w_goal_approach = rew_cfg.get("w_goal_approach",  5.0)
        self.w_goal_reached  = rew_cfg.get("w_goal_reached",  100.0)
        self.w_collision     = rew_cfg.get("w_collision",     -20.0)
        self.w_loco_fail     = rew_cfg.get("w_loco_fail",     -10.0)
        self.w_orientation   = rew_cfg.get("w_orientation",    -0.5)
        self.w_time          = rew_cfg.get("w_time",           -0.01)

        # --- Env paraméterek ---
        self.max_episode_steps = env_cfg.get("max_episode_steps", 500)
        self.goal_radius       = env_cfg.get("goal_radius", 0.5)
        self.v_forward_range   = tuple(env_cfg.get("v_forward_range", [0.0, 1.0]))
        self.yaw_rate_range    = tuple(env_cfg.get("yaw_rate_range", [-0.8, 0.8]))

        # --- Curriculum ---
        self.curriculum_level = curriculum_level
        self._curriculum_cfg  = cfg.get("curriculum", {}).get("levels", [])

        # --- MuJoCo modell betöltés ---
        self._load_model()

        # --- Locomotion adapter ---
        motion_pt = self._find_file(_MOTION_PT_CANDIDATES, "motion.pt")
        self._loco = UnitreeRLGymAdapter(motion_pt) if motion_pt else UnitreeRLGymAdapter("")
        if self._loco.is_dummy:
            import logging
            logging.getLogger(__name__).warning(
                "UnitreeRLGymAdapter dummy módban — motion.pt nem található!"
            )

        # --- Spaces ---
        obs_dim = env_cfg.get("obs_dim", 9)
        act_dim = env_cfg.get("action_dim", 2)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # --- Belső állapot ---
        self._step_count  = 0
        self._goal_xy     = np.zeros(2, dtype=np.float32)
        self._prev_dist   = 0.0
        self._loco_failed = False

        # --- Renderer ---
        self._renderer = None
        if render_mode == "human":
            self._init_renderer()

    # -----------------------------------------------------------------------
    # Betöltés
    # -----------------------------------------------------------------------

    def _find_file(self, candidates, name: str) -> Optional[Path]:
        for c in candidates:
            if Path(c).exists():
                return Path(c)
        import logging
        logging.getLogger(__name__).warning(f"{name} nem található: {candidates}")
        return None

    def _load_model(self) -> None:
        import mujoco

        scene_xml = self._find_file(_SCENE_XML_CANDIDATES, "scene.xml")
        if scene_xml is None:
            raise FileNotFoundError(
                "G1 scene.xml nem található. "
                "Ellenőrizd: unitree_rl_gym/resources/robots/g1_description/scene.xml"
            )

        self._model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self._model.opt.timestep = SIM_DT
        self._data  = mujoco.MjData(self._model)

        # Torso body id
        self._torso_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME
        )
        if self._torso_id < 0:
            # Fallback: első body a freejoint után
            self._torso_id = 1

    def _init_renderer(self) -> None:
        try:
            import mujoco
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Renderer nem elérhető: {e}")

    # -----------------------------------------------------------------------
    # Curriculum: cél pozíció meghatározás
    # -----------------------------------------------------------------------

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        """Curriculum szint alapján célpont mintavételezés."""
        level_cfg = self._get_level_cfg()

        if level_cfg and "goal_fixed" in level_cfg:
            return np.array(level_cfg["goal_fixed"], dtype=np.float32)

        if level_cfg and "goal_range" in level_cfg:
            r = level_cfg["goal_range"]
            x = rng.uniform(r["x"][0], r["x"][1])
            y = rng.uniform(r["y"][0], r["y"][1])
            return np.array([x, y], dtype=np.float32)

        # Default (szint 4): raktár pozíció
        return np.array([0.0, 3.8], dtype=np.float32)

    def _get_level_cfg(self) -> Optional[dict]:
        for lvl in self._curriculum_cfg:
            if lvl.get("id") == self.curriculum_level:
                return lvl
        return None

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        import mujoco
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # MuJoCo reset
        mujoco.mj_resetData(self._model, self._data)

        # G1 start pozíció: kis random offset a [0,0] körül
        # (hogy ne induljon mindig pont a cél előtt)
        start_x = float(rng.uniform(-0.3, 0.3))
        start_y = float(rng.uniform(-0.3, 0.3))
        self._data.qpos[0] = start_x
        self._data.qpos[1] = start_y
        self._data.qpos[2] = 0.8          # torso magasság

        # Random yaw orientáció (a robot ne nézzen mindig egyenesen a cél felé)
        yaw = float(rng.uniform(-np.pi, np.pi))
        self._data.qpos[3] = np.cos(yaw / 2)  # qw
        self._data.qpos[4] = 0.0               # qx
        self._data.qpos[5] = 0.0               # qy
        self._data.qpos[6] = np.sin(yaw / 2)  # qz

        self._data.qpos[7:19] = G1_DEFAULT_ANGLES
        self._data.qvel[:]    = 0.0
        self._data.ctrl[:]    = 0.0

        # Kis reset zaj a lábakra
        self._data.qpos[7:19] += rng.uniform(-0.02, 0.02, 12).astype(np.float32)

        mujoco.mj_forward(self._model, self._data)

        # Locomotion adapter reset
        self._loco.reset()

        # Cél
        self._goal_xy   = self._sample_goal(rng)
        self._prev_dist = self._get_dist_to_goal()
        self._step_count = 0
        self._loco_failed = False

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import mujoco

        # Action → LocomotionCommand (denormalizálás)
        v_fwd = float(np.clip(
            (action[0] + 1.0) / 2.0 * (self.v_forward_range[1] - self.v_forward_range[0])
            + self.v_forward_range[0],
            *self.v_forward_range
        ))
        yaw = float(np.clip(
            action[1] * self.v_forward_range[1] if len(action) < 2
            else action[1] * abs(self.yaw_rate_range[1]),
            *self.yaw_rate_range
        ))
        cmd = LocomotionCommand(v_forward=v_fwd, yaw_rate=yaw)

        # NAV_DECIMATION szimulációs lépés futtatása
        self._loco_failed = False
        for sim_step in range(NAV_DECIMATION):
            tau = self._loco.step_mujoco(self._data, cmd)
            self._data.ctrl[:] = tau
            mujoco.mj_step(self._model, self._data)

            # Locomotion összeomlás ellenőrzés
            if self._check_loco_collapse():
                self._loco_failed = True
                break

        self._step_count += 1

        # Reward
        reward, reward_info = self._compute_reward(cmd)

        # Terminálás
        goal_reached = self._get_dist_to_goal() < self.goal_radius
        terminated   = self._loco_failed or goal_reached
        truncated    = self._step_count >= self.max_episode_steps

        obs  = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info["goal_reached"]   = goal_reached
        info["loco_collapsed"] = self._loco_failed

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Megfigyelés
    # -----------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """9 dim obs: [robot_xy(2), goal_dist, goal_angle, heading, lin_vel(2), foot_contact(2)]"""
        torso_xy  = self._data.qpos[:2].astype(np.float32)
        goal_dist = float(self._get_dist_to_goal())
        goal_angle = float(self._get_angle_to_goal())
        heading    = float(self._get_torso_yaw())
        lin_vel    = self._data.qvel[:2].astype(np.float32)  # global x, y vel

        # Talp kontakt — scene.xml-ben nincs dedikált kontakt geom,
        # közelítés: torso_z alapján
        torso_z = float(self._data.qpos[2])
        foot_contact = np.array([
            1.0 if torso_z > 0.6 else 0.0,  # bal talp (proxy)
            1.0 if torso_z > 0.6 else 0.0,  # jobb talp (proxy)
        ], dtype=np.float32)

        return np.concatenate([
            torso_xy,
            [goal_dist, goal_angle, heading],
            lin_vel,
            foot_contact,
        ]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(self, cmd: LocomotionCommand) -> Tuple[float, dict]:
        dist     = self._get_dist_to_goal()
        approach = self._prev_dist - dist   # pozitív ha közeledik
        self._prev_dist = dist

        r_approach = self.w_goal_approach * approach
        r_reached  = self.w_goal_reached  if dist < self.goal_radius else 0.0
        r_loco     = self.w_loco_fail     if self._loco_failed else 0.0
        r_orient   = self.w_orientation   * abs(self._get_angle_to_goal())
        r_time     = self.w_time

        total = r_approach + r_reached + r_loco + r_orient + r_time

        info = {
            "r_approach": r_approach,
            "r_reached":  r_reached,
            "r_loco":     r_loco,
            "r_orient":   r_orient,
            "r_time":     r_time,
        }
        return float(total), info

    # -----------------------------------------------------------------------
    # Segéd metódusok
    # -----------------------------------------------------------------------

    def _get_dist_to_goal(self) -> float:
        torso_xy = self._data.qpos[:2]
        return float(np.linalg.norm(torso_xy - self._goal_xy))

    def _get_angle_to_goal(self) -> float:
        """Szöghiba a célponthoz a robot test frame-jében (rad)."""
        torso_xy = self._data.qpos[:2]
        diff     = self._goal_xy - torso_xy
        goal_yaw = math.atan2(diff[1], diff[0])
        heading  = self._get_torso_yaw()
        angle    = goal_yaw - heading
        # -π … π normalizálás
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def _get_torso_yaw(self) -> float:
        """Torso yaw szög (rad) a freejoint quaternion-ból."""
        qw, qx, qy, qz = self._data.qpos[3:7]
        return float(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))

    def _check_loco_collapse(self) -> bool:
        """True ha a locomotion összeomlott (elesett)."""
        torso_z = float(self._data.qpos[2])
        qw, qx, qy, _ = self._data.qpos[3:7]
        upright = 1.0 - 2.0 * (qx**2 + qy**2)
        return torso_z < LOCO_COLLAPSE_Z or upright < LOCO_COLLAPSE_UPRIGHT

    def _get_info(self) -> dict:
        return {
            "torso_xy":        self._data.qpos[:2].copy(),
            "torso_z":         float(self._data.qpos[2]),
            "goal_xy":         self._goal_xy.copy(),
            "dist_to_goal":    self._get_dist_to_goal(),
            "curriculum_level": self.curriculum_level,
            "step":            self._step_count,
        }

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
