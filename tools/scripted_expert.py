"""
Scripted Expert Policy — F3c PUSH TASK (Phase 030).

Stratégia (v3 — push task, F3c pivot):
    F3b pick-and-place blokkolt: a G1 kar felülről közelít és lenyomja a stock-ot,
    nem képes emelni (fizikai korlát, known issue #20/#23).

    F3c feladat: PUSH — stock laterális tolása target (0.45, 0.0, 0.77) pozícióba.
    Az emelés nem szükséges: target z = stock settled z = 0.77m.

    2-fázisú push stratégia:
        APPROACH → kar a stock MÖGÉ megy (push irányával ellentétes oldalra),
                   z=0.90m magasságon (stock teteje 0.81m fölött → nem ütközik).
        PUSH     → kar leesik push magasságra (z=0.79m = stock center),
                   majd söpör target irányba → stock-ot oldalra tolja → SUCCESS.

    Geometria (kimérve):
        asztal felszín z=0.730m, stock félmagasság=0.040m → settled z=0.770m
        stock teteje z=0.810m → APPROACH_HEIGHT=0.90m biztonságos
        PUSH_HEIGHT=0.79m → stock center magasságán → laterális erő, nem lefelé

Reset tartomány (F3c — szélesített):
    x∈[0.25, 0.65], y∈[-0.15, 0.15] — max Δxy=0.22m → push szükséges
    (az env edzési reset megmarad szélesnek: [0.35,0.55]×[-0.15,0.15])

Kimenet:
    data/demos/scripted_v1/raw_demos.pkl  — EpisodeBuffer lista
    (kompatibilis a tools/lerobot_export.py-val)

Futtatás (repo gyökeréből):
    python3 tools/scripted_expert.py --n-demos 50 --out-dir data/demos/scripted_v1
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Útvonal konstansok
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_SCENE_XML = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"

# ---------------------------------------------------------------------------
# Env konstansok (g1_shelf_stock_env.py-ból szinkronizálva)
# ---------------------------------------------------------------------------

SIM_DT      = 0.001
MANIP_HZ    = 20
DECIMATION  = int(1000 / MANIP_HZ)   # 50 sim lépés / policy lépés

N_ARM_DOF          = 4
ARM_QPOS_START     = 29
ARM_CTRL_START     = 0
GRIPPER_CTRL_START = 4
STOCK_QPOS_START   = 43

_JOINT_RANGES = np.array([
    [-3.0892,  2.6704],
    [-2.2515,  1.5882],
    [-2.6180,  2.6180],
    [-1.0472,  2.0944],
], dtype=np.float32)

_DEFAULT_ARM_POS = np.array([-1.0, 0.2, -0.2, 1.2], dtype=np.float32)
_GRIPPER_CLOSED  = np.array([-0.8,  0.5, -1.2,  1.3,  1.5,  1.3,  1.5], dtype=np.float32)
_GRIPPER_OPEN    = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)

# Contact detekció: hand body nevek (g1_shelf_stock_env.py _get_contact_flag() alapján)
_HAND_BODY_NAMES = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
    "right_hand_index_0_link", "right_hand_index_1_link",
    "right_hand_middle_0_link", "right_hand_middle_1_link",
]

CONTACT_FORCE_THRESHOLD = 0.5   # N — érintkezési erő küszöb
GOAL_RADIUS             = 0.08  # m — sikeres elhelyezési távolság
STOCK_RESET_Z           = 0.870 # m — kinematikai reset z (env-vel azonos)

# F3c push task konstansok
APPROACH_BEHIND_DIST = 0.15  # m — ennyivel megy a stock mögé (push iránnyal ellentétesen)
APPROACH_HEIGHT      = 0.90  # m — APPROACH fázis z (stock tető 0.81m fölé biztonsággal)
PUSH_HEIGHT          = 0.79  # m — PUSH fázis z (stock center 0.77m ≈ laterális kontakt)
PUSH_THROUGH         = 0.05  # m — dinamikus push: stock_pos + push_dir * 0.05 (nem fixált target+offset)
APPROACH_XY_THRESH   = 0.10  # m — APPROACH→PUSH átmenet laterális távolság küszöb
APPROACH_TIMEOUT     = 120   # lépés — fallback PUSH-ra ha APPROACH nem konvergál (volt: 60)

# F3c v2 reset tartomány — FIX PUSH IRÁNY redesign (2026-05-01):
#
# Probléma (v1, 7.1% SR): széles range → stock néha a target "túloldalán" →
#   az arm APPROACH behind-pozíciója x>0.60 → workspace-en kívül → fail.
#
# Megoldás: stock MINDIG a target robot-oldali részén (x < target_x=0.45):
#   - Push irány: mindig közel +x (robot→target irány)
#   - APPROACH behind = stock_x − 0.15 → mindig x<0.22 → workspace garantált
#   - MIN távolság: stock_x_max=0.36 → place_dist_min=0.09m > GOAL_RADIUS → nem triviális
#
# Range: x∈[0.20, 0.36], y∈[-0.08, 0.08]
#   Max push: sqrt(0.25²+0.08²)=0.26m | Min push: sqrt(0.09²+0.08²)=0.12m
STOCK_RESET_X_RANGE = (0.20, 0.36)
STOCK_RESET_Y_RANGE = (-0.08, 0.08)

# Minimális lépésszám a sikerhez:
# - 25 (eredeti): stock z-esés guard (z=0.870→0.770, steps 0-25)
# - 50 (v2): stock + arm-deflection guard. A stock a reset utáni esés közben
#   nekiütközhet az arm alapállású kezének (z≈0.75m) → kis lateral push →
#   beesik targetbe, anélkül hogy az expert valóban tolna (triviális siker).
#   50 lépés = 2.5s → az arm-deflection sikerek kiszűrve, valódi tolások megmaradnak.
MIN_SUCCESS_STEP = 50


# ---------------------------------------------------------------------------
# Fázis enum — F3c push task: APPROACH → PUSH → DONE
# ---------------------------------------------------------------------------

class ExpertPhase(IntEnum):
    APPROACH = 0  # stock mögé megy z=0.90m magasságon (akadálymentesen)
    PUSH     = 1  # leesik z=0.79m-re, söpör target irányba → stock-ot tolja
    DONE     = 2


# ---------------------------------------------------------------------------
# Adatstruktúrák
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepData:
    obs:    np.ndarray   # (24,) float32 — raw obs
    action: np.ndarray   # (5,) float32  — [4 arm norm + 1 gripper]
    reward: float
    done:   bool
    info:   dict


@dataclasses.dataclass
class EpisodeBuffer:
    steps:   List[StepData]
    success: bool
    length:  int


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _norm_action(target_qpos: np.ndarray) -> np.ndarray:
    lo, hi = _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
    return 2.0 * (target_qpos - lo) / (hi - lo) - 1.0


def _denorm_action(action_norm: np.ndarray) -> np.ndarray:
    lo, hi = _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
    return lo + (action_norm + 1.0) * 0.5 * (hi - lo)


def _get_contact_flag(model: mujoco.MjModel, data: mujoco.MjData,
                      stock_body_id: int, hand_body_ids: set) -> float:
    """1.0 ha a kéz geomok érintkeznek a stock geomokkal (g1_shelf_stock_env.py alapján)."""
    stock_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == stock_body_id}
    hand_geoms  = {i for i in range(model.ngeom) if model.geom_bodyid[i] in hand_body_ids}

    for c in range(data.ncon):
        contact = data.contact[c]
        g1, g2  = contact.geom1, contact.geom2
        if (g1 in hand_geoms and g2 in stock_geoms) or \
           (g2 in hand_geoms and g1 in stock_geoms):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, force)
            if np.linalg.norm(force[:3]) > CONTACT_FORCE_THRESHOLD:
                return 1.0
    return 0.0


def _get_obs(model: mujoco.MjModel, data: mujoco.MjData,
             hand_site_id: int, target_site_id: int, stock_body_id: int,
             hand_body_ids: set) -> np.ndarray:
    """24-dimenziós obs (g1_shelf_stock_env._get_obs()-sal szinkronban)."""
    hand_xyz    = data.site_xpos[hand_site_id].copy()
    stock_xyz   = data.xpos[stock_body_id].copy()
    target_xyz  = data.site_xpos[target_site_id].copy()

    hand_to_stock   = stock_xyz  - hand_xyz
    stock_to_target = target_xyz - stock_xyz

    joint_pos = data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].copy().astype(np.float32)
    joint_vel = np.clip(
        data.qvel[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF], -10.0, 10.0
    ).astype(np.float32)

    contact_flag = _get_contact_flag(model, data, stock_body_id, hand_body_ids)

    return np.concatenate([
        hand_xyz,           # [0:3]
        stock_xyz,          # [3:6]
        target_xyz,         # [6:9]
        hand_to_stock,      # [9:12]
        stock_to_target,    # [12:15]
        joint_pos,          # [15:19]
        joint_vel,          # [19:23]
        [contact_flag],     # [23]
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Jacobian IK segédfüggvény
# ---------------------------------------------------------------------------

def _ik_step(model: mujoco.MjModel, data: mujoco.MjData,
             site_id: int, arm_dof_ids: list,
             goal_xyz: np.ndarray,
             current_qpos: np.ndarray,
             n_iter: int = 5,
             lam: float = 0.01,
             max_dq: float = 0.06) -> np.ndarray:
    """
    Jacobian damped LS IK, n_iter micro-lépéssel.
    Visszaad: new_qpos (NINCS beírva az adatba — csak visszaadja az értéket).
    """
    new_qpos = current_qpos.copy()
    saved_qpos = data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].copy()

    for _ in range(n_iter):
        delta = goal_xyz - data.site_xpos[site_id]
        if np.linalg.norm(delta) < 0.003:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J    = jacp[:, arm_dof_ids]
        dq   = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), delta)
        dq   = np.clip(dq, -max_dq, max_dq)
        new_qpos = np.clip(
            new_qpos + dq, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
        )
        data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = new_qpos
        mujoco.mj_kinematics(model, data)

    # Visszaállítás (step() végzi a tényleges szimulációt)
    data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = saved_qpos

    return new_qpos


# ---------------------------------------------------------------------------
# Scripted Expert
# ---------------------------------------------------------------------------

class ScriptedExpert:
    """
    F3c push task scripted expert: APPROACH → PUSH → DONE.

    Fizikai valóság (kimérve):
        - asztal felszín z=0.730m, stock félmagasság=0.040m → settled z=0.770m
        - stock tető z=0.810m → APPROACH_HEIGHT=0.90m akadálymentesen átmegy fölötte
        - PUSH_HEIGHT=0.79m → stock center magasságán → laterális erő dominál
        - target z=0.77m = settled z → nincs emelés szükséges

    Fázisok:
        APPROACH: stock mögé megy z=0.90m-en (push iránnyal ellentétesen 0.15m)
        PUSH:     leesik z=0.79m-re, söpör target irányba → stock oldalát tolja
        DONE:     dist(stock, target) < GOAL_RADIUS=0.08m
    """

    def __init__(self, xml_path: Path = _SCENE_XML, seed: int = 0):
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data  = mujoco.MjData(self._model)
        self._rng   = np.random.default_rng(seed)
        self._phase      = ExpertPhase.APPROACH
        self._step_count = 0

        # Site / body ID-k
        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")
        self._arm_dof_ids    = list(range(ARM_QPOS_START, ARM_QPOS_START + N_ARM_DOF))

        # Hand body ID-k (contact detection)
        self._hand_body_ids: set = set()
        for name in _HAND_BODY_NAMES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._hand_body_ids.add(bid)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Kinematikai reset — F3c push task.
        Stock x,y: szélesített tartomány, push szükséges a sikerhez.
        Fázis: APPROACH indul azonnal.
        """
        mujoco.mj_resetData(self._model, self._data)

        # Kar alapállás
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = _DEFAULT_ARM_POS
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF]  = _DEFAULT_ARM_POS
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7]  = _GRIPPER_OPEN

        # Stock reset — szélesített F3c tartomány
        for _ in range(50):
            stock_x = float(self._rng.uniform(*STOCK_RESET_X_RANGE))
            stock_y = float(self._rng.uniform(*STOCK_RESET_Y_RANGE))
            self._data.qpos[STOCK_QPOS_START + 0] = stock_x
            self._data.qpos[STOCK_QPOS_START + 1] = stock_y
            self._data.qpos[STOCK_QPOS_START + 2] = STOCK_RESET_Z
            self._data.qpos[STOCK_QPOS_START + 3:STOCK_QPOS_START + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            h = self._data.site_xpos[self._hand_site_id]
            s = self._data.xpos[self._stock_body_id]
            if np.linalg.norm(h - s) >= 0.12:
                break

        self._phase      = ExpertPhase.APPROACH
        self._step_count = 0

        mujoco.mj_forward(self._model, self._data)
        return _get_obs(self._model, self._data,
                        self._hand_site_id, self._target_site_id,
                        self._stock_body_id, self._hand_body_ids)

    # ------------------------------------------------------------------
    # Lépés
    # ------------------------------------------------------------------

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        arm_action     = action_norm[:4]
        gripper_signal = float(np.clip(action_norm[4], -1.0, 1.0))

        # Kar: position control
        target_qpos = _denorm_action(arm_action)
        target_qpos = np.clip(target_qpos, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1])
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_qpos

        # Gripper: lineáris interpoláció OPEN↔CLOSED
        t = (gripper_signal + 1.0) / 2.0
        gripper_targets = (1.0 - t) * _GRIPPER_OPEN + t * _GRIPPER_CLOSED
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = gripper_targets

        for _ in range(DECIMATION):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = _get_obs(self._model, self._data,
                       self._hand_site_id, self._target_site_id,
                       self._stock_body_id, self._hand_body_ids)

        hand_pos   = self._data.site_xpos[self._hand_site_id].copy()
        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        place_dist = float(np.linalg.norm(stock_pos - target_pos))
        contact_f  = _get_contact_flag(self._model, self._data,
                                       self._stock_body_id, self._hand_body_ids)

        # F3c push task: target z=0.77m = stock settled z → nincs emelés.
        # MIN_SUCCESS_STEP guard: stock kinematikai z=0.870-ről esik steps 0-25-ben,
        # áthalad target z=0.77-n → trivális siker kizárva (stock nem stabil még).
        success = (place_dist < GOAL_RADIUS) and (self._step_count >= MIN_SUCCESS_STEP)
        timeout = self._step_count >= 300   # rövidebb timeout (push task gyorsabb)
        done    = success or timeout

        # Push task reward: közelség a target-hoz + contact bónusz
        reward = float(
            3.0 * (1.0 - np.tanh(5.0 * place_dist))         # stock→target közelség
            + 1.0 * float(contact_f)                          # érintkezés bónusz
            + (10.0 if success else 0.0)                      # siker
        )

        info = {
            "phase":           self._phase.name,
            "hand_stock_dist": float(np.linalg.norm(hand_pos - stock_pos)),
            "place_dist":      place_dist,
            "contact_f":       float(contact_f),
            "success":         success,
            "placed":          success,
            "timeout":         timeout,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Expert action
    # ------------------------------------------------------------------

    def expert_action(self) -> np.ndarray:
        """
        F3c push task: APPROACH → PUSH → DONE.

        APPROACH:
            Push iránya: stock → target (laterálisan).
            Cél: stock MÖGÖTT z=APPROACH_HEIGHT=0.90m (stock tető 0.81m fölé).
            "Mögött" = stock pozíciójától APPROACH_BEHIND_DIST=0.15m-rel
                       a push iránnyal ellentétesen.
            Átmenet: hand_xy közel behind_xy-hoz (< APPROACH_XY_THRESH) VAGY timeout.
            Gripper: nyitva (szabad mozgás).

        PUSH:
            Cél: target mögött PUSH_THROUGH=0.06m-rel, z=PUSH_HEIGHT=0.79m.
            A kar söpör a stock oldalán (z≈stock center), tolja target felé.
            Gripper: nyitva.
            Átmenet: place_dist < GOAL_RADIUS → DONE.
        """
        hand_pos   = self._data.site_xpos[self._hand_site_id].copy()
        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        place_dist = float(np.linalg.norm(stock_pos - target_pos))

        # Push irány: stock → target (csak xy)
        push_vec = target_pos[:2] - stock_pos[:2]
        push_len = float(np.linalg.norm(push_vec))
        push_dir = push_vec / (push_len + 1e-8)

        # Stock mögötti pont (APPROACH célja)
        behind_xy = stock_pos[:2] - push_dir * APPROACH_BEHIND_DIST

        # --- Fázis-átmenet ---
        if self._phase == ExpertPhase.APPROACH:
            hand_behind_dist = float(np.linalg.norm(hand_pos[:2] - behind_xy))
            if hand_behind_dist < APPROACH_XY_THRESH or self._step_count >= APPROACH_TIMEOUT:
                self._phase = ExpertPhase.PUSH

        elif self._phase == ExpertPhase.PUSH:
            if place_dist < GOAL_RADIUS:
                self._phase = ExpertPhase.DONE

        # --- Cél pozíció fázis szerint ---
        if self._phase == ExpertPhase.APPROACH:
            # Stock mögé, magas z-n (akadálymentesen átmegy a stock fölött)
            goal_xyz = np.array([behind_xy[0], behind_xy[1], APPROACH_HEIGHT])
            gripper  = -1.0   # nyitva

        elif self._phase == ExpertPhase.PUSH:
            # Fix push-through: target mögött PUSH_THROUGH-val, stock center magasságán.
            # A stock mindig robot-oldalon van (x < target_x) → push-through pont mindig
            # elérhető workspace-ben (target_x + 0.10 ≈ 0.55, ami az arm számára OK).
            push_through_xy = target_pos[:2] + push_dir * PUSH_THROUGH
            goal_xyz = np.array([push_through_xy[0], push_through_xy[1], PUSH_HEIGHT])
            gripper  = -1.0   # nyitva

        else:  # DONE
            goal_xyz = hand_pos.copy()
            gripper  = -1.0

        # --- Jacobian IK (5 micro-lépés) ---
        current_qpos = self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].copy()
        new_qpos     = _ik_step(self._model, self._data,
                                self._hand_site_id, self._arm_dof_ids,
                                goal_xyz, current_qpos,
                                n_iter=5, lam=0.01, max_dq=0.06)

        arm_action_norm = _norm_action(new_qpos.astype(np.float32))
        return np.append(arm_action_norm, gripper).astype(np.float32)

    # ------------------------------------------------------------------
    # Epizód futtatás
    # ------------------------------------------------------------------

    def run_episode(self, max_steps: int = 300) -> EpisodeBuffer:
        """Egyetlen epizód: scripted expert vezérli, minden lépést ment."""
        obs    = self.reset()
        steps: List[StepData] = []

        for _ in range(max_steps):
            action = self.expert_action()
            next_obs, reward, done, info = self.step(action)

            steps.append(StepData(
                obs    = obs.copy(),
                action = action.copy(),
                reward = reward,
                done   = done,
                info   = info,
            ))
            obs = next_obs

            if done:
                break

        last_info = steps[-1].info if steps else {}
        success   = last_info.get("success", False) or last_info.get("placed", False)
        return EpisodeBuffer(steps=steps, success=success, length=len(steps))


# ---------------------------------------------------------------------------
# Demonstráció gyűjtés
# ---------------------------------------------------------------------------

def collect_demonstrations(
    n_demos:     int  = 50,
    max_retries: int  = 2000,
    seed:        int  = 42,
    verbose:     bool = True,
) -> List[EpisodeBuffer]:
    """Gyűjt n_demos sikeres epizódot."""
    expert = ScriptedExpert(seed=seed)
    demos: List[EpisodeBuffer] = []
    tried = 0

    while len(demos) < n_demos and tried < max_retries:
        buf = expert.run_episode()
        tried += 1

        if buf.success:
            demos.append(buf)
            if verbose:
                print(f"[{len(demos):3d}/{n_demos}] ✅ siker  ({buf.length} lépés) | #{tried}")
        elif verbose and tried % 20 == 0:
            sr = 100 * len(demos) / tried
            print(f"[{len(demos):3d}/{n_demos}] ... #{tried}  {sr:.1f}% sikerességi arány")

    if verbose:
        sr = 100 * len(demos) / tried if tried > 0 else 0
        print(f"\nEredmény: {len(demos)}/{n_demos} demo ({tried} próbából, {sr:.1f}% SR)")

    return demos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scripted Expert demo gyűjtő")
    parser.add_argument("--n-demos",     type=int, default=50)
    parser.add_argument("--out-dir",     type=str, default="data/demos/scripted_v1")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2000)
    parser.add_argument("--no-save",     action="store_true",
                        help="Ne mentse a raw_demos.pkl-t (alapértelmezett: ment)")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Demonstráció gyűjtés: {args.n_demos} sikeres epizód")
    print("─" * 60)

    t0    = time.time()
    demos = collect_demonstrations(
        n_demos     = args.n_demos,
        max_retries = args.max_retries,
        seed        = args.seed,
    )
    elapsed = time.time() - t0
    print(f"\n⏱  {elapsed:.1f}s  ({elapsed/max(1,len(demos)):.1f}s/demo)")

    if demos and not args.no_save:
        import pickle
        raw_path = out_dir / "raw_demos.pkl"
        with open(raw_path, "wb") as f:
            pickle.dump(demos, f)
        print(f"Mentve: {raw_path} ({len(demos)} demo)")

    print("\n✅ Következő lépés:")
    print(f"   python3 tools/lerobot_export.py --in-dir {out_dir} --out-dir data/lerobot/scripted_v1")


if __name__ == "__main__":
    main()
