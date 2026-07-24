"""
T1 Shelf Stocking Push Env — v1 (Vendor-independence track).

Booster T1 humanoid — 4-DOF jobb kar (Shoulder Pitch/Roll + Elbow Pitch/Yaw).
Cél: lateral push task — stock dobozt target pozícióba tolni a polcon.
Nincs gripper (T1 standard nem tartalmaz kézfej aktuátorokat).

Vendor-independence motiváció:
    Ugyanaz a push feladat mint G1 Phase 030, de más gyártó robotján (Booster T1).
    Metrika: azonos SR target (≥80%) és reward skála, összehasonlítható checkpointok.
    Ha mindkét robot eléri → validált "vendor-independent" approach.

Architektúra:
    PushPolicy (PPO, 20 Hz) → position_target[4] → MuJoCo position actuator

Aktuátorok (nu=4):
    0: right_shoulder_pitch   (qpos[6])
    1: right_shoulder_roll    (qpos[7])
    2: right_elbow_pitch      (qpos[8])
    3: right_elbow_yaw        (qpos[9])

Observation (24 dim — kompatibilis a G1 obs struktúrával):
    [0:3]   hand_xyz          — kéz pozíciója world frame-ben
    [3:6]   stock_xyz         — termék pozíciója
    [6:9]   target_xyz        — célpozíció
    [9:12]  hand→stock vec    — relatív vektor
    [12:15] stock→target vec  — relatív vektor
    [15:19] joint_pos[4]      — normalizált ízületi szögek
    [19:23] joint_vel[4]      — ízületi sebességek (clippelve)
    [23]    contact_flag      — 1 ha kéz érinti a terméket

Reward (tanh alapú, bounded):
    reach:   1 - tanh(5 * hand→stock dist)
    push:    contact_force alapú + stock→target közelség
    placed:  success bónusz

2 fázis:
    REACH → PUSH

Fő különbségek a G1 env-hez képest:
    - Nincs gripper (action_dim=4, nem 5)
    - ARM_QPOS_INDICES = [6, 7, 8, 9]  (G1: 29..32)
    - STOCK_QPOS_START = 23            (G1: 43)
    - DEFAULT_ARM_POS = [-1.0, 1.5, 1.5, 0.0]  (diag grid search eredménye)
    - Push task: stock y-offset → target (y=0), nem pick-and-place
    - Nincsenek GRASP/LIFT fázisok

Elfogadási feltétel: ≥80% sikeres push 100 epizódon (G1-gyel azonos target).

Referenciák:
    G1 env:   g1_shelf_stock_env.py (v9, Phase 030)
    Scene:    scene_manip_sandbox_t1_v1.xml
    Diag:     results/diag/t1_reach_*.csv  (DEFAULT_ARM_POS forrása)
    Obsidian: [[multi_robot_strategy_2026-07]]
"""

from __future__ import annotations

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
_REPO_ROOT = _HERE.parents[5]
_SCENE_XML = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_t1_v1.xml"

# ---------------------------------------------------------------------------
# Szimuláció paraméterei
# ---------------------------------------------------------------------------

SIM_DT     = 0.001          # 1000 Hz
MANIP_HZ   = 20             # policy frekvencia
DECIMATION = int(1000 / MANIP_HZ)  # 50 sim lépés / policy lépés

N_ARM_DOF = 4   # Right Shoulder Pitch/Roll + Right Elbow Pitch/Yaw

# T1 arm joint qpos indexek (depth-first XML traversal sorrendben):
#   qpos[0]:  AAHead_yaw
#   qpos[1]:  Head_pitch
#   qpos[2]:  Left_Shoulder_Pitch
#   qpos[3]:  Left_Shoulder_Roll
#   qpos[4]:  Left_Elbow_Pitch
#   qpos[5]:  Left_Elbow_Yaw
#   qpos[6]:  Right_Shoulder_Pitch  ← ARM_QPOS_INDICES[0]
#   qpos[7]:  Right_Shoulder_Roll
#   qpos[8]:  Right_Elbow_Pitch
#   qpos[9]:  Right_Elbow_Yaw
#   qpos[10..22]: Waist + 2×6 leg DOF
#   qpos[23..29]: stock_1 freejoint (3pos + 4quat)
ARM_QPOS_INDICES  = [6, 7, 8, 9]  # T1 jobb kar qpos indexei
ARM_CTRL_INDICES  = [0, 1, 2, 3]  # T1 actuator ctrl indexei
STOCK_QPOS_START  = 23            # stock_1 freejoint kezdő indexe

# T1 jobb kar ízületi határok (t1.xml alapján):
#   Right_Shoulder_Pitch:  range="-3.31  1.22"
#   Right_Shoulder_Roll:   range="-1.57  1.74"
#   Right_Elbow_Pitch:     range="-2.27  2.27"
#   Right_Elbow_Yaw:       range=" 0     2.44"
_JOINT_RANGES = np.array([
    [-3.31,  1.22],  # Right_Shoulder_Pitch
    [-1.57,  1.74],  # Right_Shoulder_Roll
    [-2.27,  2.27],  # Right_Elbow_Pitch
    [ 0.00,  2.44],  # Right_Elbow_Yaw
], dtype=np.float32)

# Diag grid search eredménye (t1_reach_20260724_*.csv):
# pitch=-1.0, roll=1.5, elbow_p=1.5, elbow_yaw=0.0
# → hand_x=0.399m, hand_z=0.778m, stable=True
_DEFAULT_ARM_POS = np.array([
    -1.0,  # Right_Shoulder_Pitch
     1.5,  # Right_Shoulder_Roll
     1.5,  # Right_Elbow_Pitch
     0.0,  # Right_Elbow_Yaw
], dtype=np.float32)

# Push task célpozíció (scene_manip_sandbox_t1_v1.xml):
#   target_shelf site: pos="0.30 0.0 0.77"
#
# Stock spawn tartomány:
#   T1 jobb kar természetes y pozíciója ≈ -0.18m (negatív oldal).
#   A robot CSAK a negatív y oldalból tud hatékonyan tolni (jobb karból jobbra).
#   Stock mindig y ∈ [−STOCK_Y_MAX, −STOCK_Y_MIN] → negatív y, a kar eléri.
#   STOCK_Y_MIN > goal_radius (0.08m), hogy a start pozíció ne legyen már "sikerben".
STOCK_X_FIXED   = 0.30    # tároló asztal x pozíciója
STOCK_Z_ON_SURF = 0.77    # stock center z (asztal felszínén)
STOCK_Y_MIN     = 0.10    # min y-offset (> goal_radius=0.08m)
STOCK_Y_MAX     = 0.15    # max y-offset

# Reward tanh skálák
TANH_SCALE_REACH = 5.0
TANH_SCALE_PUSH  = 5.0

# Contact force küszöb
CONTACT_FORCE_THRESHOLD = 0.05  # N (push esetén kisebb érték megbízhatóbb)

# OBS dim: 3+3+3+3+3+4+4+1 = 24 (G1-gyel azonos!)
OBS_DIM = 24


# ---------------------------------------------------------------------------
# Fázis enum
# ---------------------------------------------------------------------------

class PushPhase(IntEnum):
    REACH = 0
    PUSH  = 1


# ---------------------------------------------------------------------------
# Segédfüggvény: tanh alapú smooth distance reward
# ---------------------------------------------------------------------------

def smooth_dist_reward(dist: float, scale: float = 5.0) -> float:
    """
    1 - tanh(scale * dist): közel=1, 0→1.0, 0.1m→~0.46, 0.3m→~0.03
    """
    return float(np.clip(1.0 - np.tanh(scale * dist), 0.0, 1.0))


# ---------------------------------------------------------------------------
# T1ShelfStockEnv
# ---------------------------------------------------------------------------

class T1ShelfStockEnv(gym.Env):
    """
    Shelf stocking push env — Booster T1 humanoid.
    Laterális push task: stock dobozt y-offset → target (y=0).
    Nincs gripper, nincs lift — csak REACH + PUSH fázis.

    Kompatibilis a G1ShelfStockEnv obs/action struktúrájával
    (azonos OBS_DIM=24, hasonló reward skála) a vendor-independence
    összehasonlíthatóság érdekében.
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
        self.w_reach       = rew_cfg.get("w_reach",       1.0)
        self.w_push        = rew_cfg.get("w_push",        2.0)   # contact + közelség
        self.w_place       = rew_cfg.get("w_place",       2.0)   # stock→target shaping
        self.w_placed      = rew_cfg.get("w_placed",     10.0)   # success bónusz
        self.w_joint_limit = rew_cfg.get("w_joint_limit", -0.5)
        self.w_smooth      = rew_cfg.get("w_smooth",      -0.001)

        self.tanh_scale_reach = rew_cfg.get("tanh_scale_reach", TANH_SCALE_REACH)
        self.tanh_scale_push  = rew_cfg.get("tanh_scale_push",  TANH_SCALE_PUSH)

        # --- Env paraméterek ---
        self.max_episode_steps    = env_cfg.get("max_episode_steps", 500)
        self.goal_radius          = env_cfg.get("goal_radius", 0.08)
        self.reach_dist_threshold = env_cfg.get("reach_dist_threshold", 0.12)

        # --- Scene XML ---
        scene_path = cfg.get("scene", {}).get("xml_path", None)
        xml = (_REPO_ROOT / scene_path) if scene_path else _SCENE_XML
        self._load_model(xml)

        # --- Spaces ---
        obs_dim = env_cfg.get("obs_dim", OBS_DIM)
        act_dim = env_cfg.get("action_dim", N_ARM_DOF)  # 4, nincs gripper

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # --- Belső állapot ---
        self._step_count   = 0
        self._phase        = PushPhase.REACH
        self._contact_flag = 0.0
        self._prev_action  = np.zeros(N_ARM_DOF, dtype=np.float32)

        self._renderer = None
        if render_mode == "human":
            self._init_renderer()

    # -----------------------------------------------------------------------
    # Modell betöltés
    # -----------------------------------------------------------------------

    def _load_model(self, xml_path: Path) -> None:
        import mujoco
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML nem található: {xml_path}")
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._model.opt.timestep = SIM_DT
        self._data = mujoco.MjData(self._model)

        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")
        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")

        # Ellenőrzés
        assert self._stock_body_id >= 0,  "stock_1 body nem található az XML-ben"
        assert self._hand_site_id  >= 0,  "right_hand_site nem található az XML-ben"
        assert self._target_site_id >= 0, "target_shelf site nem található az XML-ben"

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

        # Kar alapállásba + kis reset zaj
        arm_init = _DEFAULT_ARM_POS + rng.uniform(-0.05, 0.05, N_ARM_DOF).astype(np.float32)
        for i, qi in enumerate(ARM_QPOS_INDICES):
            self._data.qpos[qi] = arm_init[i]
        for i, ci in enumerate(ARM_CTRL_INDICES):
            self._data.ctrl[ci] = arm_init[i]

        self._data.qvel[:] = 0.0

        # Stock pozíció: x=0.30 (asztalon), y ∈ [−STOCK_Y_MAX, −STOCK_Y_MIN], z=0.77
        # Mindig negatív y → T1 jobb kar eléri (természetes oldal)
        y_offset = float(rng.uniform(STOCK_Y_MIN, STOCK_Y_MAX))
        stock_y  = -y_offset  # mindig negatív

        self._data.qpos[STOCK_QPOS_START + 0] = STOCK_X_FIXED
        self._data.qpos[STOCK_QPOS_START + 1] = stock_y
        self._data.qpos[STOCK_QPOS_START + 2] = STOCK_Z_ON_SURF
        self._data.qpos[STOCK_QPOS_START + 3] = 1.0  # quaternion w
        self._data.qpos[STOCK_QPOS_START + 4] = 0.0  # x
        self._data.qpos[STOCK_QPOS_START + 5] = 0.0  # y
        self._data.qpos[STOCK_QPOS_START + 6] = 0.0  # z

        mujoco.mj_forward(self._model, self._data)

        # Belső állapot reset
        self._step_count   = 0
        self._phase        = PushPhase.REACH
        self._contact_flag = 0.0
        self._prev_action  = _DEFAULT_ARM_POS.copy()

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import mujoco

        target_pos = self._denorm_action(action[:N_ARM_DOF])

        for _ in range(DECIMATION):
            for i, ci in enumerate(ARM_CTRL_INDICES):
                self._data.ctrl[ci] = target_pos[i]
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        # Contact flag és fázis frissítés
        self._contact_flag = self._get_contact_flag()
        self._update_phase()

        reward, reward_info = self._compute_reward(action)

        placed     = self._get_stock_target_dist() < self.goal_radius
        terminated = placed
        truncated  = self._step_count >= self.max_episode_steps

        self._prev_action = target_pos.copy()

        obs  = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info["placed"] = placed
        info["phase"]  = int(self._phase)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Fázis logika (REACH → PUSH)
    # -----------------------------------------------------------------------

    def _update_phase(self) -> None:
        if self._phase == PushPhase.REACH:
            hand_dist = self._get_hand_stock_dist()
            if hand_dist < self.reach_dist_threshold:
                self._phase = PushPhase.PUSH
        # PUSH fázisból nincs továbblépés (terminated = placed)

    # -----------------------------------------------------------------------
    # Contact flag — MuJoCo contact force alapú
    # -----------------------------------------------------------------------

    def _get_contact_flag(self) -> float:
        """
        1.0 ha a right_hand_link / right_hand_site körüli geom-ok érintik a stock_1-et.
        T1 nincs ujj → csak right_hand_link geom-jait vizsgáljuk.
        """
        import mujoco

        stock_body_id = self._stock_body_id

        # Stock geom-ok
        stock_geoms = {
            i for i in range(self._model.ngeom)
            if self._model.geom_bodyid[i] == stock_body_id
        }

        # T1 jobb kéz body (right_hand_link)
        hand_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link")
        hand_geoms = {
            i for i in range(self._model.ngeom)
            if self._model.geom_bodyid[i] == hand_body_id
        } if hand_body_id >= 0 else set()

        for c in range(self._data.ncon):
            contact = self._data.contact[c]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 in hand_geoms and g2 in stock_geoms) or \
               (g2 in hand_geoms and g1 in stock_geoms):
                force = np.zeros(6)
                mujoco.mj_contactForce(self._model, self._data, c, force)
                if np.linalg.norm(force[:3]) > CONTACT_FORCE_THRESHOLD:
                    return 1.0

        return 0.0

    # -----------------------------------------------------------------------
    # Observation (OBS_DIM=24 — G1-gyel azonos struktúra)
    # -----------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        hand_xyz   = self._data.site_xpos[self._hand_site_id].astype(np.float32)
        stock_xyz  = self._data.xpos[self._stock_body_id].astype(np.float32)
        target_xyz = self._data.site_xpos[self._target_site_id].astype(np.float32)

        hand_to_stock   = (stock_xyz  - hand_xyz).astype(np.float32)
        stock_to_target = (target_xyz - stock_xyz).astype(np.float32)

        # Joint state normalizálva
        joint_pos = np.array(
            [self._data.qpos[qi] for qi in ARM_QPOS_INDICES], dtype=np.float32
        )
        joint_vel = np.array(
            [self._data.qvel[qi] for qi in ARM_QPOS_INDICES], dtype=np.float32
        )

        mid  = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
        half = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
        joint_pos_norm = (joint_pos - mid) / (half + 1e-6)

        return np.concatenate([
            hand_xyz,          # [0:3]
            stock_xyz,         # [3:6]
            target_xyz,        # [6:9]
            hand_to_stock,     # [9:12]
            stock_to_target,   # [12:15]
            joint_pos_norm,    # [15:19]
            np.clip(joint_vel * 0.1, -5, 5),  # [19:23]
            [self._contact_flag],              # [23]
        ]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, dict]:
        hand_dist  = self._get_hand_stock_dist()
        stock_dist = self._get_stock_target_dist()

        # REACH: kéz közelítése a stockhoz
        r_reach = self.w_reach * smooth_dist_reward(hand_dist, self.tanh_scale_reach)

        # PUSH: contact bónusz + stock→target közelség, PUSH fázisban aktiválva
        near_for_push = smooth_dist_reward(hand_dist, scale=10.0)
        contact_bonus = self._contact_flag
        if self._phase >= PushPhase.PUSH:
            r_push = self.w_push * (0.5 * contact_bonus + 0.5 * near_for_push)
        else:
            r_push = self.w_push * 0.1 * near_for_push  # kis előjelzés REACH-ben

        # PLACE: stock→target shaping (csak ha már érintkezés volt)
        place_raw = smooth_dist_reward(stock_dist, self.tanh_scale_push)
        if self._phase >= PushPhase.PUSH or self._contact_flag > 0.5:
            r_place = self.w_place * place_raw
        else:
            r_place = self.w_place * 0.05 * place_raw

        # SUCCESS
        r_placed = self.w_placed if stock_dist < self.goal_radius else 0.0

        # Regularizáció: joint limit közelség
        joint_pos = np.array(
            [self._data.qpos[qi] for qi in ARM_QPOS_INDICES], dtype=np.float32
        )
        limit_viol = np.sum(np.maximum(
            joint_pos - _JOINT_RANGES[:, 1] * 0.95,
            _JOINT_RANGES[:, 0] * 0.95 - joint_pos,
        ).clip(0))
        r_limit = self.w_joint_limit * float(limit_viol)

        # Mozgási simítás
        target_pos = self._denorm_action(action[:N_ARM_DOF])
        r_smooth = self.w_smooth * float(np.sum((target_pos - self._prev_action) ** 2))

        total = r_reach + r_push + r_place + r_placed + r_limit + r_smooth

        return float(total), dict(
            r_reach=r_reach, r_push=r_push, r_place=r_place,
            r_placed=r_placed, r_limit=r_limit, r_smooth=r_smooth,
            hand_dist=hand_dist, stock_dist=stock_dist,
            contact_flag=self._contact_flag,
        )

    # -----------------------------------------------------------------------
    # Segédmetódusok
    # -----------------------------------------------------------------------

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        """Normalizált [-1,1] akció → joint target (ízületi határon belül)."""
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
            "hand_xyz":          self._data.site_xpos[self._hand_site_id].copy(),
            "stock_xyz":         self._data.xpos[self._stock_body_id].copy(),
            "target_xyz":        self._data.site_xpos[self._target_site_id].copy(),
            "hand_stock_dist":   self._get_hand_stock_dist(),
            "stock_target_dist": self._get_stock_target_dist(),
            "phase":             int(self._phase),
            "contact_flag":      self._contact_flag,
            "step":              self._step_count,
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
