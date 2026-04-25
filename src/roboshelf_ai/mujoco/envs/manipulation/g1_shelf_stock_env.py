"""
G1 Shelf Stocking Manipulációs Env — v8 (Phase 030 F3).

Változások v7-hez képest (2026-04-25, policy collapse fix):
  - lift_trigger_threshold: 0.03m — lift reward csak valódi emelkedésnél számít
  - w_lift: 1.0 → 3.0 config-ból (erősebb lift signal)
  - ent_coef: 0.01 → 0.05 config-ból (policy collapse ellen)

Változások v6-hoz képest (2026-04-24, Panda-szerű reward & obs):
  - Reward: -w*dist → 1 - tanh(5*dist)  (DeepMind PandaPickCube minta)
  - Obs: hand→stock és stock→target relatív vektorok hozzáadva (obs_dim: 18 → 24)
  - Grasp: contact force alapú flag (nem csak magassági heurisztika)
  - DEFAULT_ARM_POS: debug-igazolt optimum [-1.0, 0.2, -0.2, 1.2]
  - Target: robot előtt (x=0.45, y=0, z=0.97) — kar eléri (debug: 0.025m)

Architektúra:
    ManipPolicy (PPO, 20 Hz) → position_target[4] → MuJoCo position actuator

Aktuátorok (nu=4, equality constraint eltávolítva — Phase 030):
    0: right_shoulder_pitch
    1: right_shoulder_roll
    2: right_shoulder_yaw
    3: right_elbow

Observation (24 dim):
    [0:3]   hand_xyz          — kéz pozíciója world frame-ben
    [3:6]   stock_xyz         — termék pozíciója
    [6:9]   target_xyz        — célpozíció
    [9:12]  hand→stock vec    — relatív vektor (ÚJ — PandaPickCube minta)
    [12:15] stock→target vec  — relatív vektor (ÚJ)
    [15:19] joint_pos[4]      — normalizált ízületi szögek
    [19:23] joint_vel[4]      — ízületi sebességek (clippelve)
    [23]    contact_flag      — 1 ha kéz érinti a terméket (ÚJ — contact force alapú)

Reward (tanh alapú, bounded [0,1]):
    reach:  1 - tanh(5 * hand→stock dist)
    grasp:  contact_force alapú bónusz
    lift:   emelési magasság alapú
    place:  1 - tanh(5 * stock→target dist)
    placed: success bónusz

4 fázis:
    REACH → GRASP → LIFT → PLACE

Referenciák:
    DeepMind PandaPickCube: 1-tanh(k*dist) shaping
    Obsidian: [[Panda-szerű reward-függvényt és obs-designt]]
    Obsidian: [[mujoco pick place reward shaping PPO kutatás Github]]

Elfogadási feltétel: 70%+ sikeres elhelyezés 20 epizódon.
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
_SCENE_XML = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"

# ---------------------------------------------------------------------------
# Szimuláció paraméterei
# ---------------------------------------------------------------------------

SIM_DT     = 0.001          # 1000 Hz
MANIP_HZ   = 20             # policy frekvencia
DECIMATION = int(1000 / MANIP_HZ)  # 50 sim lépés / policy lépés

N_ARM_DOF        = 4    # shoulder(3) + elbow(1)
ARM_QPOS_START   = 29   # qpos[29:33]
ARM_CTRL_START   = 0    # ctrl[0:4]
STOCK_QPOS_START = 43   # freejoint: [43:50]

# Ízületi határok
_JOINT_RANGES = np.array([
    [-3.0892,  2.6704],  # right_shoulder_pitch
    [-2.2515,  1.5882],  # right_shoulder_roll
    [-2.6180,  2.6180],  # right_shoulder_yaw
    [-1.0472,  2.0944],  # right_elbow
], dtype=np.float32)

# Debug-igazolt optimális alapállás (2026-04-24):
# grid search 12^4=20736 kombináción → best hand→stock dist = 0.025m
# shoulder_pitch=-0.995, roll=0.192, yaw=-0.238, elbow=1.238
_DEFAULT_ARM_POS = np.array([
    -1.0,   # right_shoulder_pitch
     0.2,   # right_shoulder_roll
    -0.2,   # right_shoulder_yaw
     1.2,   # right_elbow
], dtype=np.float32)

# Reward tanh skála (DeepMind PandaPickCube minta)
TANH_SCALE_REACH = 5.0   # 1 - tanh(5 * dist): közel=1, 0.3m-nél ≈ 0.03
TANH_SCALE_PLACE = 5.0

# Contact force threshold a grasp detektáláshoz
CONTACT_FORCE_THRESHOLD = 0.1  # N

# OBS dim: 3+3+3+3+3+4+4+1 = 24
OBS_DIM = 24


# ---------------------------------------------------------------------------
# Fázis enum
# ---------------------------------------------------------------------------

class ManipPhase(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT  = 2
    PLACE = 3


# ---------------------------------------------------------------------------
# Segédfüggvény: tanh alapú smooth distance reward
# ---------------------------------------------------------------------------

def smooth_dist_reward(dist: float, scale: float = 5.0) -> float:
    """
    DeepMind PandaPickCube minta: 1 - tanh(scale * dist)
    - dist=0.0  → reward=1.0 (maximum)
    - dist=0.1m → reward≈0.46
    - dist=0.3m → reward≈0.03
    - Bounded [0, 1], erős gradiens közel, lapos távolabb
    """
    return float(np.clip(1.0 - np.tanh(scale * dist), 0.0, 1.0))


# ---------------------------------------------------------------------------
# G1ShelfStockEnv
# ---------------------------------------------------------------------------

class G1ShelfStockEnv(gym.Env):
    """
    Shelf stocking manipulációs env — v7.
    PandaPickCube-szerű reward és obs design.
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

        # --- Reward súlyok (tanh alapú, bounded) ---
        self.w_reach       = rew_cfg.get("w_reach",       1.0)   # reach shaping súlya
        self.w_grasp       = rew_cfg.get("w_grasp",       2.0)   # grasp bónusz
        self.w_lift        = rew_cfg.get("w_lift",        1.0)   # lift shaping
        self.w_place       = rew_cfg.get("w_place",       2.0)   # place shaping súlya
        self.w_placed      = rew_cfg.get("w_placed",     10.0)   # success bónusz
        self.w_joint_limit = rew_cfg.get("w_joint_limit", -0.5)
        self.w_smooth      = rew_cfg.get("w_smooth",      -0.001)

        self.tanh_scale_reach = rew_cfg.get("tanh_scale_reach", TANH_SCALE_REACH)
        self.tanh_scale_place = rew_cfg.get("tanh_scale_place", TANH_SCALE_PLACE)

        # --- Env paraméterek ---
        self.max_episode_steps      = env_cfg.get("max_episode_steps", 500)
        self.goal_radius            = env_cfg.get("goal_radius", 0.08)
        self.grasp_dist_threshold   = env_cfg.get("grasp_dist_threshold", 0.10)  # kéz→stock
        self.lift_height            = env_cfg.get("lift_height", 0.10)
        self.lift_trigger_threshold = env_cfg.get("lift_trigger_threshold", 0.03)  # v8: küszöb

        # --- Scene XML ---
        scene_path = cfg.get("scene", {}).get("xml_path", None)
        xml = (_REPO_ROOT / scene_path) if scene_path else _SCENE_XML
        self._load_model(xml)

        # --- Spaces ---
        act_dim = env_cfg.get("action_dim", self._model.nu)
        obs_dim = env_cfg.get("obs_dim", OBS_DIM)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # --- Belső állapot ---
        self._step_count      = 0
        self._phase           = ManipPhase.REACH
        self._contact_flag    = 0.0   # contact force alapú (v7)
        self._reached_box     = False  # egyszer már elérte a kart a terméket
        self._stock_z0        = 0.0
        self._prev_action     = np.zeros(N_ARM_DOF, dtype=np.float32)

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

        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,  "stock_1")
        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE,  "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE,  "target_shelf")

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
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = (
            _DEFAULT_ARM_POS + rng.uniform(-0.05, 0.05, N_ARM_DOF).astype(np.float32)
        )
        self._data.qvel[:] = 0.0
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = _DEFAULT_ARM_POS.copy()

        # Stock pozíció: asztalfelszínen, kis y-offset
        stock_y = float(rng.uniform(-0.10, 0.10))
        self._data.qpos[STOCK_QPOS_START + 0] = 0.45
        self._data.qpos[STOCK_QPOS_START + 1] = stock_y
        self._data.qpos[STOCK_QPOS_START + 2] = 0.870
        self._data.qpos[STOCK_QPOS_START + 3] = 1.0
        self._data.qpos[STOCK_QPOS_START + 4:STOCK_QPOS_START + 7] = 0.0

        mujoco.mj_forward(self._model, self._data)

        # Belső állapot reset
        self._step_count   = 0
        self._phase        = ManipPhase.REACH
        self._contact_flag = 0.0
        self._reached_box  = False
        self._stock_z0     = float(self._data.xpos[self._stock_body_id][2])
        self._prev_action  = _DEFAULT_ARM_POS.copy()

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import mujoco

        target_pos = self._denorm_action(action)

        for _ in range(DECIMATION):
            self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_pos
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        # Contact force frissítés
        self._contact_flag = self._get_contact_flag()

        # Fázis frissítés
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
    # Fázis logika
    # -----------------------------------------------------------------------

    def _update_phase(self) -> None:
        hand_dist  = self._get_hand_stock_dist()
        stock_z    = float(self._data.xpos[self._stock_body_id][2])
        stock_rise = stock_z - self._stock_z0

        if self._phase == ManipPhase.REACH:
            if hand_dist < self.grasp_dist_threshold:
                self._reached_box = True
                self._phase = ManipPhase.GRASP

        elif self._phase == ManipPhase.GRASP:
            # Contact force VAGY magassági emelkedés → LIFT
            if self._contact_flag > 0.5 and stock_rise > 0.01:
                self._phase = ManipPhase.LIFT

        elif self._phase == ManipPhase.LIFT:
            if stock_rise > self.lift_height:
                self._phase = ManipPhase.PLACE

    # -----------------------------------------------------------------------
    # Contact flag — MuJoCo contact force alapú
    # -----------------------------------------------------------------------

    def _get_contact_flag(self) -> float:
        """
        1.0 ha a kéz (right_hand_site körüli geom-ok) érintkezik a stock_1-gyel.
        MuJoCo contact listából olvassa — nem magassági heurisztika.
        """
        import mujoco

        stock_body_id = self._stock_body_id
        # Stock body geom-jainak összegyűjtése
        stock_geoms = set()
        for i in range(self._model.ngeom):
            if self._model.geom_bodyid[i] == stock_body_id:
                stock_geoms.add(i)

        # Kéz body-k: right_hand_palm és ujjak
        hand_body_names = [
            "right_hand_palm_link",
            "right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
            "right_hand_index_0_link", "right_hand_index_1_link",
            "right_hand_middle_0_link", "right_hand_middle_1_link",
        ]
        hand_bodies = set()
        for name in hand_body_names:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                hand_bodies.add(bid)

        hand_geoms = set()
        for i in range(self._model.ngeom):
            if self._model.geom_bodyid[i] in hand_bodies:
                hand_geoms.add(i)

        # Contact keresés
        for c in range(self._data.ncon):
            contact = self._data.contact[c]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 in hand_geoms and g2 in stock_geoms) or \
               (g2 in hand_geoms and g1 in stock_geoms):
                # Contact force nagyság
                force = np.zeros(6)
                mujoco.mj_contactForce(self._model, self._data, c, force)
                if np.linalg.norm(force[:3]) > CONTACT_FORCE_THRESHOLD:
                    return 1.0

        return 0.0

    # -----------------------------------------------------------------------
    # Observation — v7: relatív vektorok + contact flag
    # -----------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        hand_xyz   = self._data.site_xpos[self._hand_site_id].astype(np.float32)
        stock_xyz  = self._data.xpos[self._stock_body_id].astype(np.float32)
        target_xyz = self._data.site_xpos[self._target_site_id].astype(np.float32)

        # Relatív vektorok — PandaPickCube minta
        hand_to_stock  = (stock_xyz  - hand_xyz).astype(np.float32)   # (3,)
        stock_to_target = (target_xyz - stock_xyz).astype(np.float32)  # (3,)

        # Joint state (normalizált)
        joint_pos = self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].astype(np.float32)
        joint_vel = self._data.qvel[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].astype(np.float32)

        mid  = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
        half = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
        joint_pos_norm = (joint_pos - mid) / (half + 1e-6)

        return np.concatenate([
            hand_xyz,          # [0:3]
            stock_xyz,         # [3:6]
            target_xyz,        # [6:9]
            hand_to_stock,     # [9:12]   ÚJ
            stock_to_target,   # [12:15]  ÚJ
            joint_pos_norm,    # [15:19]
            np.clip(joint_vel * 0.1, -5, 5),  # [19:23]
            [self._contact_flag],              # [23]    ÚJ
        ]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reward — v7: tanh alapú, PandaPickCube minta
    # -----------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, dict]:
        hand_dist  = self._get_hand_stock_dist()
        stock_dist = self._get_stock_target_dist()
        stock_z    = float(self._data.xpos[self._stock_body_id][2])
        stock_rise = stock_z - self._stock_z0

        # --- REACH: 1 - tanh(scale * dist) ---
        # Bounded [0,1]: dist=0 → 1.0, dist=0.1m → 0.46, dist=0.3m → 0.03
        r_reach = self.w_reach * smooth_dist_reward(hand_dist, self.tanh_scale_reach)

        # --- GRASP: contact force bónusz + közelségi shaping ---
        near_for_grasp = smooth_dist_reward(hand_dist, scale=10.0)  # élesebb
        contact_bonus  = self._contact_flag
        r_grasp = self.w_grasp * (0.5 * contact_bonus + 0.5 * near_for_grasp) \
                  if self._phase >= ManipPhase.GRASP else \
                  self.w_grasp * 0.1 * near_for_grasp  # kis előjelzés REACH-ben is

        # --- LIFT: emelési magasság — v8: trigger küszöb alatt 0 (nem csak rezgés) ---
        if self._phase >= ManipPhase.LIFT and stock_rise >= self.lift_trigger_threshold:
            r_lift = self.w_lift * np.clip(stock_rise / self.lift_height, 0.0, 1.0)
        else:
            r_lift = 0.0

        # --- PLACE: tanh alapú, gating: csak ha már elérte a dobozt ---
        place_raw = smooth_dist_reward(stock_dist, self.tanh_scale_place)
        if self._reached_box or self._phase >= ManipPhase.GRASP:
            r_place = self.w_place * place_raw
        else:
            r_place = self.w_place * 0.05 * place_raw  # minimális signal előtte

        # --- SUCCESS bónusz ---
        r_placed = self.w_placed if stock_dist < self.goal_radius else 0.0

        # --- Regularizáció ---
        joint_pos = self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF]
        limit_viol = np.sum(np.maximum(
            joint_pos - _JOINT_RANGES[:, 1] * 0.95,
            _JOINT_RANGES[:, 0] * 0.95 - joint_pos,
        ).clip(0))
        r_limit = self.w_joint_limit * limit_viol

        target_pos = self._denorm_action(action)
        r_smooth = self.w_smooth * float(np.sum((target_pos - self._prev_action) ** 2))

        total = r_reach + r_grasp + r_lift + r_place + r_placed + r_limit + r_smooth

        self._prev_hand_dist  = hand_dist
        self._prev_stock_dist = stock_dist

        return float(total), dict(
            r_reach=r_reach, r_grasp=r_grasp, r_lift=r_lift,
            r_place=r_place, r_placed=r_placed,
            r_limit=r_limit, r_smooth=r_smooth,
            hand_dist=hand_dist, stock_dist=stock_dist,
            contact_flag=self._contact_flag,
        )

    # -----------------------------------------------------------------------
    # Segédmetódusok
    # -----------------------------------------------------------------------

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
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
            "stock_rise":        float(self._data.xpos[self._stock_body_id][2]) - self._stock_z0,
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
