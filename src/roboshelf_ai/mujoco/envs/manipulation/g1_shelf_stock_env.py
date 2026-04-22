"""
G1 Shelf Stocking Manipulációs Env — Fázis C.

Rögzített testű sandbox: a robot törzse nem mozog, csak a jobb kar (14 DoF)
veszi fel az asztalon lévő terméket és teszi a polcra.

Architektúra:
    ManipPolicy (PPO, ~20 Hz) → position_target[14] → MuJoCo position actuator

Aktuátorok (nu=14, position control):
    0-3:  right_shoulder (pitch, roll, yaw) + right_elbow
    4-6:  right_wrist (roll, pitch, yaw)
    7-9:  right_hand_thumb (0, 1, 2)
    10-11: right_hand_index (0, 1)
    12-13: right_hand_middle (0, 1)

Observation (38 dim):
    [0:3]   right_hand_xyz  — kéz pozíciója world frame-ben
    [3:6]   stock_xyz       — termék pozíciója
    [6:9]   target_xyz      — célpozíció a polcon
    [9:23]  joint_pos[14]   — jobb kar ízületi szögek (normalizált)
    [23:37] joint_vel[14]   — jobb kar ízületi sebességek
    [37]    grasp_flag      — 1 ha a termék emelkedett (0.02m felett start)

Action (14 dim, [-1, 1]):
    → position target, denormalizálva az ízületi határokra

4 tanítási fázis (automatikusan detektált):
    REACH  — kéz közelít a termékhez
    GRASP  — kéz elérte a terméket, ujjak zárnak
    LIFT   — termék emelkedik
    PLACE  — termék mozog a polc felé

Elfogadási feltétel: 70%+ sikeres elhelyezés random startból.

Használat:
    from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv
    env = G1ShelfStockEnv(cfg=cfg)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Útvonal konstansok
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]   # roboshelf-ai-redesign/

# v2: az eredeti g1_29dof_with_hand.xml kinematikájából generált scene
_SCENE_XML = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"

# ---------------------------------------------------------------------------
# Szimuláció paraméterei
# ---------------------------------------------------------------------------

SIM_DT      = 0.002    # 500 Hz (scene XML-ből)
MANIP_HZ    = 20       # policy frekvencia
DECIMATION  = int(500 / MANIP_HZ)   # 25 szimulációs lépés / policy lépés

# v2 scene: teljes G1 kinematika (43 hinge joint) + stock_1 freejoint
# Aktív jointok: csak jobb váll(3) + könyök(1) = 4 DOF (kp=20, stabil)
# Csukló + ujjak: equality constraint rögzíti (passzív)
# Jobb kar qpos: [29:43], de csak [29:33] aktív (4 joint)
# ctrl: nu=4, ctrl[0:4] = right_shoulder_pitch/roll/yaw + right_elbow
N_TOTAL_DOF      = 43   # összes robot joint
N_ARM_DOF        = 4    # aktív: shoulder(3) + elbow(1)
ARM_QPOS_START   = 29   # qpos[29:33] = aktív jobb kar jointok
ARM_CTRL_START   = 0    # ctrl[0:4]
STOCK_QPOS_START = 43   # freejoint: 43..49

# Ízületi határok (scene XML alapján, ugyanaz mint g1_29dof_with_hand.xml)
# Csak az aktív 4 joint határai (váll×3 + könyök)
_JOINT_RANGES = np.array([
    [-3.0892,  2.6704],  # right_shoulder_pitch
    [-2.2515,  1.5882],  # right_shoulder_roll
    [-2.6180,  2.6180],  # right_shoulder_yaw
    [-1.0472,  2.0944],  # right_elbow
], dtype=np.float32)

# Alapértelmezett karállás: csak 4 aktív joint (váll×3 + könyök)
# Robot +x irányba néz, asztal x=1.2-nél
# shoulder_pitch pozitív → kar előre+felfelé emelkedik
_DEFAULT_ARM_POS = np.array([
     1.0,   # right_shoulder_pitch — előre emelve
    -0.2,   # right_shoulder_roll  — kicsit oldalra
     0.0,   # right_shoulder_yaw
     1.0,   # right_elbow          — hajlítva
], dtype=np.float32)

# Ujjak nyitott / zárt állás — nem használt (ujjak rögzítve)
_FINGERS_OPEN   = np.zeros(0, dtype=np.float32)
_FINGERS_CLOSED = np.zeros(0, dtype=np.float32)

# Ujjak nyitott / zárt állás (grasp)
_FINGERS_OPEN   = np.zeros(7, dtype=np.float32)   # ujjak indexei: 7–13
_FINGERS_CLOSED = np.array([0.5, 0.5, -0.8, 1.2, 1.5, 1.2, 1.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fázis enum
# ---------------------------------------------------------------------------

class ManipPhase(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT  = 2
    PLACE = 3


# ---------------------------------------------------------------------------
# G1ShelfStockEnv
# ---------------------------------------------------------------------------

class G1ShelfStockEnv(gym.Env):
    """Shelf stocking manipulációs Gymnasium env (Fázis C sandbox).

    A jobb kar 14 DoF-ja position actuator-okkal vezérelhető.
    A test rögzített, locomotion nem fut.
    Egy gym.step() = DECIMATION (25) szimulációs lépés @ 500 Hz.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": MANIP_HZ}

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        cfg = cfg or {}

        env_cfg = cfg.get("env", {})
        rew_cfg = cfg.get("reward", {})

        # --- Reward súlyok ---
        self.w_reach        = rew_cfg.get("w_reach",        2.0)
        self.w_grasp        = rew_cfg.get("w_grasp",       20.0)
        self.w_lift         = rew_cfg.get("w_lift",         5.0)
        self.w_place        = rew_cfg.get("w_place",       10.0)
        self.w_placed       = rew_cfg.get("w_placed",     100.0)
        self.w_joint_limit  = rew_cfg.get("w_joint_limit",  -1.0)
        self.w_smooth       = rew_cfg.get("w_smooth",       -0.01)
        self.w_time         = rew_cfg.get("w_time",         -0.005)

        # --- Env paraméterek ---
        self.max_episode_steps    = env_cfg.get("max_episode_steps", 500)
        self.goal_radius          = env_cfg.get("goal_radius", 0.08)
        self.grasp_dist_threshold = env_cfg.get("grasp_dist_threshold", 0.06)
        self.lift_height          = env_cfg.get("lift_height", 0.15)

        # --- Scene XML betöltés ---
        scene_path = cfg.get("scene", {}).get("xml_path", None)
        if scene_path:
            xml = _REPO_ROOT / scene_path
        else:
            xml = _SCENE_XML

        self._load_model(xml)

        # --- Spaces ---
        obs_dim = env_cfg.get("obs_dim", 38)
        act_dim = env_cfg.get("action_dim", 14)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # --- Belső állapot ---
        self._step_count   = 0
        self._phase        = ManipPhase.REACH
        self._grasp_flag   = 0.0
        self._stock_z0     = 0.0    # termék kezdeti z (emelés méréshez)
        self._prev_hand_dist = 0.0
        self._prev_stock_dist = 0.0
        self._prev_action  = np.zeros(N_ARM_DOF, dtype=np.float32)

        # --- Renderer ---
        self._renderer = None
        if render_mode == "human":
            self._init_renderer()

    # -----------------------------------------------------------------------
    # Modell betöltés
    # -----------------------------------------------------------------------

    def _load_model(self, xml_path: Path) -> None:
        import mujoco
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Manip sandbox scene XML nem található: {xml_path}\n"
                "Ellenőrizd: src/envs/assets/scene_manip_sandbox.xml"
            )
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._model.opt.timestep = SIM_DT
        self._data  = mujoco.MjData(self._model)

        # Body / site ID-k
        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")
        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")

    def _init_renderer(self) -> None:
        try:
            import mujoco
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Renderer nem elérhető: {e}")

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

        mujoco.mj_resetData(self._model, self._data)

        # Passzív jointok (lábak, derék, bal kar): 0-ban maradnak (mj_resetData nulláz)
        # Jobb kar alapállásba + kis reset zaj
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = (
            _DEFAULT_ARM_POS + rng.uniform(-0.05, 0.05, N_ARM_DOF).astype(np.float32)
        )
        self._data.qvel[:] = 0.0
        # ctrl[0:14] = csak jobb kar (nu=14, passzív jointoknak nincs actuator)
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = _DEFAULT_ARM_POS.copy()

        # Stock termék: asztalfelszínen, kis y-offset (véletlenszerű)
        # v2 scene: asztal x=1.2, stock world pos ~ (1.2, ±0.15, 0.870)
        # stock_1 freejoint: qpos[STOCK_QPOS_START..+7] = [x, y, z, qw, qx, qy, qz]
        stock_y = float(rng.uniform(-0.15, 0.15))
        self._data.qpos[STOCK_QPOS_START + 0] = 1.2         # x: asztal közepe
        self._data.qpos[STOCK_QPOS_START + 1] = stock_y     # y: kis véletlen eltérés
        self._data.qpos[STOCK_QPOS_START + 2] = 0.870       # z: asztalfelszín + doboz félmagasság
        self._data.qpos[STOCK_QPOS_START + 3] = 1.0         # qw
        self._data.qpos[STOCK_QPOS_START + 4:STOCK_QPOS_START + 7] = 0.0

        mujoco.mj_forward(self._model, self._data)

        # Belső állapot reset
        self._step_count     = 0
        self._phase          = ManipPhase.REACH
        self._grasp_flag     = 0.0
        self._stock_z0       = float(self._data.xpos[self._stock_body_id][2])
        self._prev_hand_dist  = self._get_hand_stock_dist()
        self._prev_stock_dist = self._get_stock_target_dist()
        self._prev_action    = _DEFAULT_ARM_POS.copy()

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import mujoco

        # Action denormalizálás: [-1,1] → ízületi határok
        target_pos = self._denorm_action(action)

        # DECIMATION szimulációs lépés
        # ctrl[0:14] = jobb kar position target (nu=14)
        for _ in range(DECIMATION):
            self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_pos
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        # Fázis frissítés
        self._update_phase()

        # Reward
        reward, reward_info = self._compute_reward(action)

        # Terminálás
        placed   = self._get_stock_target_dist() < self.goal_radius
        terminated = placed
        truncated  = self._step_count >= self.max_episode_steps

        self._prev_action = target_pos.copy()

        obs  = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info["placed"]  = placed
        info["phase"]   = int(self._phase)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Fázis logika
    # -----------------------------------------------------------------------

    def _update_phase(self) -> None:
        hand_dist  = self._get_hand_stock_dist()
        stock_z    = float(self._data.xpos[self._stock_body_id][2])
        stock_rise = stock_z - self._stock_z0

        if self._phase == ManipPhase.REACH:
            if hand_dist < self.grasp_dist_threshold:
                self._phase = ManipPhase.GRASP

        elif self._phase == ManipPhase.GRASP:
            if stock_rise > 0.02:   # termék megmozdult
                self._grasp_flag = 1.0
                self._phase = ManipPhase.LIFT

        elif self._phase == ManipPhase.LIFT:
            if stock_rise > self.lift_height:
                self._phase = ManipPhase.PLACE

    # -----------------------------------------------------------------------
    # Megfigyelés
    # -----------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        hand_xyz   = self._data.site_xpos[self._hand_site_id].astype(np.float32)
        stock_xyz  = self._data.xpos[self._stock_body_id].astype(np.float32)
        target_xyz = self._data.site_xpos[self._target_site_id].astype(np.float32)

        joint_pos = self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].astype(np.float32)
        joint_vel = self._data.qvel[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].astype(np.float32)

        # Normalizált ízületi pozíciók [-1, 1]
        mid   = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
        half  = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
        joint_pos_norm = (joint_pos - mid) / (half + 1e-6)

        return np.concatenate([
            hand_xyz,
            stock_xyz,
            target_xyz,
            joint_pos_norm,
            np.clip(joint_vel * 0.1, -5, 5),
            [self._grasp_flag],
        ]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, dict]:
        hand_dist  = self._get_hand_stock_dist()
        stock_dist = self._get_stock_target_dist()
        stock_z    = float(self._data.xpos[self._stock_body_id][2])
        stock_rise = stock_z - self._stock_z0

        # Reach reward: dense distance alapú (nem delta!)
        # -w_reach * dist → mindig negatív, de kisebb ha közelebb; kényszeríti a mozgást
        r_reach = -self.w_reach * hand_dist

        # Grasp reward: termék emelkedik
        r_grasp = self.w_grasp * max(0.0, stock_rise - 0.01) if self._phase >= ManipPhase.GRASP else 0.0

        # Lift reward: magasság növekedés
        r_lift = self.w_lift * max(0.0, stock_rise) if self._phase >= ManipPhase.LIFT else 0.0

        # Place reward: dense distance alapú
        r_place = -self.w_place * stock_dist if self._phase == ManipPhase.PLACE else 0.0

        # Placed bónusz
        r_placed = self.w_placed if stock_dist < self.goal_radius else 0.0

        # Joint limit büntetés
        joint_pos = self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF]
        limit_violation = np.sum(np.maximum(
            joint_pos - _JOINT_RANGES[:, 1] * 0.95,
            _JOINT_RANGES[:, 0] * 0.95 - joint_pos,
        ).clip(0))
        r_limit = self.w_joint_limit * limit_violation

        # Simaság büntetés
        target_pos = self._denorm_action(action)
        r_smooth = self.w_smooth * float(np.sum((target_pos - self._prev_action) ** 2))

        r_time = self.w_time

        total = r_reach + r_grasp + r_lift + r_place + r_placed + r_limit + r_smooth + r_time

        self._prev_hand_dist  = hand_dist
        self._prev_stock_dist = stock_dist

        info = dict(
            r_reach=r_reach, r_grasp=r_grasp, r_lift=r_lift,
            r_place=r_place, r_placed=r_placed, r_limit=r_limit,
            r_smooth=r_smooth,
        )
        return float(total), info

    # -----------------------------------------------------------------------
    # Segédmetódusok
    # -----------------------------------------------------------------------

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        """[-1,1] → ízületi pozíció target."""
        mid  = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
        half = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
        return (np.clip(action, -1, 1) * half + mid).astype(np.float32)

    def _get_hand_stock_dist(self) -> float:
        hand  = self._data.site_xpos[self._hand_site_id]
        stock = self._data.xpos[self._stock_body_id]
        return float(np.linalg.norm(hand - stock))

    def _get_stock_target_dist(self) -> float:
        stock  = self._data.xpos[self._stock_body_id]
        target = self._data.site_xpos[self._target_site_id]
        return float(np.linalg.norm(stock - target))

    def _get_info(self) -> dict:
        return {
            "hand_xyz":        self._data.site_xpos[self._hand_site_id].copy(),
            "stock_xyz":       self._data.xpos[self._stock_body_id].copy(),
            "target_xyz":      self._data.site_xpos[self._target_site_id].copy(),
            "hand_stock_dist": self._get_hand_stock_dist(),
            "stock_target_dist": self._get_stock_target_dist(),
            "stock_rise":      float(self._data.xpos[self._stock_body_id][2]) - self._stock_z0,
            "phase":           int(self._phase),
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
