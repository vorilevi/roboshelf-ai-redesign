"""
Scripted Expert Policy — F3b demonstráció gyűjtés (Phase 030).

Célja:
    IK-alapú, determinisztikus vezérlő, ami végrehajtja a
    reach → grasp → lift → place pipeline-t és trajectóriákat
    ment LeRobot-kompatibilis formátumba (lerobot_export.py-n keresztül).

Architektúra:
    ScriptedExpert.run_episode()
        └── fázisok: REACH → GRASP → LIFT → PLACE → DONE
        └── minden lépésnél: obs, action, reward, done mentés
        └── sikeres epizód → EpisodeBuffer-be kerül
    collect_demonstrations(n_episodes)
        └── n sikeres epizódig futtatja a run_episode-ot
        └── visszaad: list[EpisodeBuffer]

Kimenet:
    tools/lerobot_export.py felhasználja a buffereket LeRobotDataset v3.0 formátumba.

Futtatás (repo gyökeréből):
    python3 tools/scripted_expert.py --n-demos 50 --out-dir data/demos/scripted_v1

Ismert korlátok:
    - 4-DOF kar: nincs wrist DOF, ezért a grasp szög fixált
    - IK: analitikus, nem iteratív → csak a reachable workspace-ben működik
    - Contact: a grasp sikerességét contact force-szal mérjük (0.1N threshold)

Referenciák:
    - g1_shelf_stock_env.py: env implementáció (ugyanazt a MuJoCo modellt használja)
    - known_issues.md #15: finger joint class fix (finger_joint, nem passive_joint)
    - known_issues.md #18: min 15cm start dist a reset-ben
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from enum import IntEnum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Útvonal konstansok (repo gyökeréhez relatív)
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_SCENE_XML = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"

# ---------------------------------------------------------------------------
# Env konstansok (g1_shelf_stock_env.py-ból másolva — szinkronban kell tartani)
# ---------------------------------------------------------------------------

SIM_DT     = 0.001
MANIP_HZ   = 20
DECIMATION = int(1000 / MANIP_HZ)

N_ARM_DOF         = 4
ARM_QPOS_START    = 29
ARM_CTRL_START    = 0
GRIPPER_CTRL_START = 4
STOCK_QPOS_START  = 43

_JOINT_RANGES = np.array([
    [-3.0892,  2.6704],
    [-2.2515,  1.5882],
    [-2.6180,  2.6180],
    [-1.0472,  2.0944],
], dtype=np.float32)

_DEFAULT_ARM_POS = np.array([-1.0, 0.2, -0.2, 1.2], dtype=np.float32)

_GRIPPER_CLOSED = np.array([-0.8,  0.5, -1.2,  1.3,  1.5,  1.3,  1.5], dtype=np.float32)
_GRIPPER_OPEN   = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)

CONTACT_FORCE_THRESHOLD = 0.1   # N
MIN_START_DIST = 0.15           # m (known_issues #18)
GOAL_RADIUS    = 0.08           # m

# ---------------------------------------------------------------------------
# Fázis enum
# ---------------------------------------------------------------------------

class ExpertPhase(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT  = 2
    PLACE = 3
    DONE  = 4


# ---------------------------------------------------------------------------
# Adatstruktúrák
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepData:
    obs:    np.ndarray   # (24,) float32
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
    """Abszolút joint szögöket → [-1, 1] normalizált action."""
    lo = _JOINT_RANGES[:, 0]
    hi = _JOINT_RANGES[:, 1]
    return 2.0 * (target_qpos - lo) / (hi - lo) - 1.0


def _denorm_action(action_norm: np.ndarray) -> np.ndarray:
    """[-1, 1] normalizált action → abszolút joint szögök."""
    lo = _JOINT_RANGES[:, 0]
    hi = _JOINT_RANGES[:, 1]
    return lo + (action_norm + 1.0) * 0.5 * (hi - lo)


def _get_contact_force(model: mujoco.MjModel, data: mujoco.MjData,
                       geom_name: str = "right_hand_geom") -> float:
    """Contact force a jobb kéz és a stock között."""
    try:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    except Exception:
        return 0.0

    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == geom_id or c.geom2 == geom_id:
            # contact force magnitude
            cf = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, cf)
            total += float(np.linalg.norm(cf[:3]))
    return total


def _get_obs(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """24-dimenziós obs vektor (g1_shelf_stock_env.py-val szinkronban)."""
    # Pozíciók
    hand_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
    stock_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

    hand_xyz   = data.site_xpos[hand_site_id].copy()
    target_xyz = data.site_xpos[target_site_id].copy()
    stock_xyz  = data.xpos[stock_body_id].copy()

    # Relatív vektorok
    hand_to_stock  = stock_xyz - hand_xyz
    stock_to_target = target_xyz - stock_xyz

    # Joint állapot
    joint_pos = data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].copy().astype(np.float32)
    joint_vel = np.clip(
        data.qvel[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF],
        -10.0, 10.0
    ).astype(np.float32)

    # Contact flag
    contact_force = _get_contact_force(model, data)
    contact_flag  = float(contact_force > CONTACT_FORCE_THRESHOLD)

    obs = np.concatenate([
        hand_xyz,          # [0:3]
        stock_xyz,         # [3:6]
        target_xyz,        # [6:9]
        hand_to_stock,     # [9:12]
        stock_to_target,   # [12:15]
        joint_pos,         # [15:19]
        joint_vel,         # [19:23]
        [contact_flag],    # [23]
    ]).astype(np.float32)
    return obs


# ---------------------------------------------------------------------------
# Scripted Expert
# ---------------------------------------------------------------------------

class ScriptedExpert:
    """
    IK-mentes, egyszerű P-szabályozó alapú expert policy.

    Stratégia:
        REACH: karral közelíts a stock fölé, gripper nyitva
        GRASP: ereszkedj a stock-ra, gripper zár
        LIFT:  emeld 10cm-rel
        PLACE: vidd a target_site-ra, gripper nyit (elenged)
    """

    def __init__(self, xml_path: Path = _SCENE_XML, seed: int = 0):
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data  = mujoco.MjData(self._model)
        self._rng   = np.random.default_rng(seed)
        self._phase = ExpertPhase.REACH
        self._initial_stock_z: float = 0.0
        self._step_count: int = 0

        # Site / body ID-k (egyszer lekérdezve)
        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset env, véletlenszerű stock pozíció (min 15cm kéztől)."""
        mujoco.mj_resetData(self._model, self._data)

        # Robot alapállás
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = _DEFAULT_ARM_POS
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = _GRIPPER_OPEN

        # Stock véletlenszerű pozíció (known_issues #18: min 15cm)
        for _ in range(50):
            stock_x = float(self._rng.uniform(0.35, 0.55))
            stock_y = float(self._rng.uniform(-0.15, 0.15))
            self._data.qpos[STOCK_QPOS_START + 0] = stock_x
            self._data.qpos[STOCK_QPOS_START + 1] = stock_y
            self._data.qpos[STOCK_QPOS_START + 2] = 0.415   # z: asztal magassága
            # quaternion: identity
            self._data.qpos[STOCK_QPOS_START + 3:STOCK_QPOS_START + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            hand_pos  = self._data.site_xpos[self._hand_site_id].copy()
            stock_pos = self._data.xpos[self._stock_body_id].copy()
            if np.linalg.norm(hand_pos - stock_pos) >= MIN_START_DIST:
                break

        self._initial_stock_z = float(self._data.xpos[self._stock_body_id][2])
        self._phase     = ExpertPhase.REACH
        self._step_count = 0

        mujoco.mj_forward(self._model, self._data)
        return _get_obs(self._model, self._data)

    # ------------------------------------------------------------------
    # Lépés
    # ------------------------------------------------------------------

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """action_norm: [-1,1]^4 (arm) + [-1,1]^1 (gripper)."""
        arm_action = action_norm[:4]
        gripper_signal = float(np.clip(action_norm[4], -1.0, 1.0))

        # Kar: position control
        target_qpos = _denorm_action(arm_action)
        target_qpos = np.clip(target_qpos, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1])
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_qpos

        # Gripper: +1 = zárva, -1 = nyitva
        gripper_targets = (
            _GRIPPER_CLOSED if gripper_signal > 0 else _GRIPPER_OPEN
        )
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = gripper_targets

        # Szimuláció
        for _ in range(DECIMATION):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = _get_obs(self._model, self._data)

        # Állapot lekérdezés
        hand_pos   = self._data.site_xpos[self._hand_site_id].copy()
        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        stock_rise = float(stock_pos[2] - self._initial_stock_z)
        place_dist = float(np.linalg.norm(stock_pos - target_pos))
        contact_f  = _get_contact_force(self._model, self._data)

        # Siker / fail
        success = place_dist < GOAL_RADIUS
        timeout = self._step_count >= 500
        done    = success or timeout

        # Egyszerű reward (csak loggoláshoz — ACT BC nem használja)
        reward = float(
            1.0 - np.tanh(5.0 * np.linalg.norm(hand_pos - stock_pos))
            + 2.0 * float(contact_f > CONTACT_FORCE_THRESHOLD)
            + 5.0 * np.tanh(10.0 * max(0.0, stock_rise))
            + 2.0 * (1.0 - np.tanh(5.0 * place_dist))
            + (10.0 if success else 0.0)
        )

        info = {
            "phase":      self._phase.name,
            "hand_stock_dist": float(np.linalg.norm(hand_pos - stock_pos)),
            "stock_rise": stock_rise,
            "place_dist": place_dist,
            "contact_f":  contact_f,
            "success":    success,
            "timeout":    timeout,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Expert action (scripted)
    # ------------------------------------------------------------------

    def expert_action(self) -> np.ndarray:
        """
        Fázis-alapú P-vezérlő:
            REACH  → közelíts a stock fölé (y=stock_y, x=stock_x, z=stock_z+0.15)
            GRASP  → ereszkedj (z=stock_z+0.03), gripper zár
            LIFT   → emeld (z=stock_z+0.15 → target_z)
            PLACE  → vidd a target_xyz-re, gripper nyit
        """
        hand_pos   = self._data.site_xpos[self._hand_site_id].copy()
        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        stock_rise = float(stock_pos[2] - self._initial_stock_z)
        contact_f  = _get_contact_force(self._model, self._data)
        reach_dist = float(np.linalg.norm(hand_pos - stock_pos))
        place_dist = float(np.linalg.norm(stock_pos - target_pos))

        # Fázis-átmenet
        if self._phase == ExpertPhase.REACH and reach_dist < 0.05:
            self._phase = ExpertPhase.GRASP
        elif self._phase == ExpertPhase.GRASP and contact_f > CONTACT_FORCE_THRESHOLD and stock_rise > 0.005:
            self._phase = ExpertPhase.LIFT
        elif self._phase == ExpertPhase.LIFT and stock_rise > 0.08:
            self._phase = ExpertPhase.PLACE
        elif self._phase == ExpertPhase.PLACE and place_dist < GOAL_RADIUS:
            self._phase = ExpertPhase.DONE

        # Cél kéz pozíció fázis szerint
        if self._phase == ExpertPhase.REACH:
            goal_xyz = stock_pos + np.array([0.0, 0.0, 0.15])
            gripper  = -1.0   # nyitva
        elif self._phase == ExpertPhase.GRASP:
            goal_xyz = stock_pos + np.array([0.0, 0.0, 0.03])
            gripper  =  1.0   # zárva
        elif self._phase == ExpertPhase.LIFT:
            goal_xyz = stock_pos + np.array([0.0, 0.0, 0.15])
            gripper  =  1.0
        elif self._phase == ExpertPhase.PLACE:
            goal_xyz = target_pos.copy()
            gripper  = -1.0   # elenged
        else:
            goal_xyz = hand_pos.copy()
            gripper  = -1.0

        # P-vezérlő: Δq ≈ J^T * Δx (nagyon egyszerűsített — 4-DOF linearizált Jacobian)
        # TODO: cseréld le analitikus IK-ra vagy MuJoCo mj_jacSite-ra ha rossz a coverage
        delta_xyz = goal_xyz - hand_pos
        current_qpos = self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF].copy()

        # Jacobian alapú update (mj_jacSite)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._hand_site_id)
        # Csak az ARM DOF-ok Jacobian-ja
        arm_dof_ids = list(range(ARM_QPOS_START, ARM_QPOS_START + N_ARM_DOF))
        J_arm = jacp[:, arm_dof_ids]  # (3, 4)

        # Pseudo-inverse IK lépés
        lam = 0.01   # damping (numerical stability)
        JJT = J_arm @ J_arm.T + lam * np.eye(3)
        delta_q = J_arm.T @ np.linalg.solve(JJT, delta_xyz)
        delta_q = np.clip(delta_q, -0.15, 0.15)   # max lépés per policy step
        new_qpos = current_qpos + delta_q
        new_qpos = np.clip(new_qpos, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1])

        arm_action_norm = _norm_action(new_qpos.astype(np.float32))
        return np.append(arm_action_norm, gripper).astype(np.float32)

    # ------------------------------------------------------------------
    # Epizód futtatás
    # ------------------------------------------------------------------

    def run_episode(self, max_steps: int = 500) -> EpisodeBuffer:
        """Egyetlen epizód: scripted expert vezérli, minden lépést ment."""
        obs = self.reset()
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

        success = steps[-1].info.get("success", False) if steps else False
        return EpisodeBuffer(steps=steps, success=success, length=len(steps))


# ---------------------------------------------------------------------------
# Demonstráció gyűjtés
# ---------------------------------------------------------------------------

def collect_demonstrations(
    n_demos:     int  = 50,
    max_retries: int  = 500,
    seed:        int  = 42,
    verbose:     bool = True,
) -> List[EpisodeBuffer]:
    """
    Gyűjt n_demos sikeres epizódot.

    Args:
        n_demos:     Kívánt sikeres demonstrációk száma.
        max_retries: Max próbálkozás (sikertelen epizódok beleértve).
        seed:        Véletlenszám seed.
        verbose:     Progress kiírás.

    Returns:
        list[EpisodeBuffer]: csak sikeres epizódok.
    """
    expert  = ScriptedExpert(seed=seed)
    demos:  List[EpisodeBuffer] = []
    tried   = 0

    while len(demos) < n_demos and tried < max_retries:
        buf = expert.run_episode()
        tried += 1

        if buf.success:
            demos.append(buf)
            if verbose:
                print(
                    f"[{len(demos):3d}/{n_demos}] ✅ siker "
                    f"({buf.length} lépés) | próba #{tried}"
                )
        elif verbose and tried % 10 == 0:
            print(f"[{len(demos):3d}/{n_demos}] ❌ sikertelen | próba #{tried}")

    if verbose:
        sr = 100 * len(demos) / tried if tried > 0 else 0
        print(f"\nEredmény: {len(demos)}/{n_demos} demo gyűjtve "
              f"({tried} próbából, {sr:.1f}% sikerességi arány)")

    return demos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scripted Expert demonstráció gyűjtő")
    parser.add_argument("--n-demos",  type=int,  default=50,
                        help="Gyűjtendő sikeres demonstrációk száma (default: 50)")
    parser.add_argument("--out-dir",  type=str,  default="data/demos/scripted_v1",
                        help="Kimeneti könyvtár (lerobot_export.py bemenetének)")
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--max-retries", type=int, default=500)
    parser.add_argument("--save-raw", action="store_true",
                        help="Nyers numpy adatokat is ment (debug célra)")
    args = parser.parse_args()

    out_dir = Path(_REPO_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Demonstráció gyűjtés: {args.n_demos} sikeres epizód")
    print(f"Kimeneti könyvtár: {out_dir}")
    print("─" * 60)

    t0 = time.time()
    demos = collect_demonstrations(
        n_demos     = args.n_demos,
        max_retries = args.max_retries,
        seed        = args.seed,
    )
    elapsed = time.time() - t0
    print(f"\n⏱  Gyűjtési idő: {elapsed:.1f}s ({elapsed/max(1,len(demos)):.1f}s/demo)")

    if args.save_raw:
        import pickle
        raw_path = out_dir / "raw_demos.pkl"
        with open(raw_path, "wb") as f:
            pickle.dump(demos, f)
        print(f"Nyers adatok mentve: {raw_path}")

    print("\n✅ Kész! Következő lépés:")
    print(f"   python3 tools/lerobot_export.py --in-dir {out_dir} --out-dir data/lerobot/scripted_v1")


if __name__ == "__main__":
    main()
