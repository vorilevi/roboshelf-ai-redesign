"""
GR1T1 push task demo gyűjtő — scripted expert v2 (Track 3).

2 fázisú determinisztikus vezérlő:
  SETTLE → PUSH

A kulcs: PUSH_ARM_POS ahol a kézfej BOX geom teljesen a stock "mögött" van
(geom_y_hi=-0.203 < stock_back_face_y_min=-0.14) ÉS a z range-ben van
(geom_z_lo=0.802 < stock_z_hi=0.810). Ez teszi lehetővé a tiszta laterális
push-t J-transpose vezérlővel: 100% SR 200 epizódon.

Miért nem DEFAULT_ARM_POS?
  DEFAULT_ARM_POS = [-1.0, 0.0, 0.5, 0.083]:
    geom_z_lo = 0.865m  (stock top = 0.810m → 5.5cm fölötte)
    geom_y_hi = -0.103m  (stock back face ≈ -0.14m → geom nem mögötte)
  Eredmény: csak chaotikus brief contact, 67% SR.

  PUSH_ARM_POS = [-0.654, 0.111, 1.875, 0.500]:
    geom_z_lo = 0.802m  (stock top = 0.810m → geom ÁT FED a stockon 8mm-t)
    geom_y_hi = -0.203m  (stock back face ∈ [-0.19, -0.14] → geom MÖGÖTTE van)
  Eredmény: tiszta sustained contact, 100% SR 200 epizódon.

Output:
  results/demos/gr1_push_demos_YYYYMMDD_HHMM.npz
  - obs:            (N,) object array, elemek: (T, 24) float32
  - actions:        (N,) object array, elemek: (T, 4)  float32 (normalizált -1..1)
  - rewards:        (N,) object array, elemek: (T,)    float32
  - successes:      (N,) bool
  - episode_lengths:(N,) int

Futtatás (repo gyökeréből):
  python3 tools/scripted_expert_gr1.py
  python3 tools/scripted_expert_gr1.py --n-episodes 100 --verbose
  python3 tools/scripted_expert_gr1.py --n-episodes 1000 --out results/demos/gr1_demos.npz

Referenciák:
  Env:      src/roboshelf_ai/mujoco/envs/manipulation/gr1_shelf_stock_env.py
  Scene:    src/envs/assets/scene_manip_sandbox_gr1_v1.xml
  kp:       results/diag/gr1_kp_sweep_final.csv  (kp=150 validált)
  T1 ref:   tools/scripted_expert_t1.py
  Geom diag: PUSH_ARM_POS keresés → geom AABB < stock z_top, geom y_hi < stock back face
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mujoco
from roboshelf_ai.mujoco.envs.manipulation.gr1_shelf_stock_env import (
    GR1ShelfStockEnv,
    ARM_QPOS_INDICES,
    ARM_CTRL_INDICES,
    _DEFAULT_ARM_POS,
    _JOINT_RANGES,
    N_ARM_DOF,
    OBS_DIM,
)

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

DEFAULT_N_EPISODES = 1000

# PUSH_ARM_POS: keresve AABB grid-del, 2026-07-26
# Kritérium: geom_y_hi < min(stock_back_face) = -0.14 (mögötte minden stocknak)
#            geom_z_lo < stock_z_hi = 0.810 (z kontakt lehetséges)
#            site_x > 0.36 (közel a stock x=0.45-höz)
# Eredmény:  geom_y=[−0.307, −0.203], geom_z=[0.802, 0.933]
#            site=(0.395, −0.264, 0.840) @ forward kinematics
PUSH_ARM_POS = np.array([-0.654, 0.111, 1.875, 0.500], dtype=np.float32)

SETTLE_STEPS = 5      # PUSH_ARM_POS-ra settler — kp=150-nél ~0.25s alatt stabilizálódik
                      # (volt: 80 → training data 94% konstans akció → 20% SR)
                      # Fix: 5 settle + ~3 push → push arány 37% vs korábbi 6%
PUSH_GAIN    = 5.0    # Jacobian transpose gain
MAX_DELTA_Q  = 0.10   # max joint delta per policy step (rad)
PUSH_OVERSHOOT = 0.03 # m — target y-on túllövés

# Normalizált PUSH_ARM_POS (gyorsítótár)
_mid         = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
_half        = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
_PAP_NORM    = np.clip((PUSH_ARM_POS - _mid) / (_half + 1e-6), -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Jacobian transpose segédfüggvények
# ---------------------------------------------------------------------------

def get_position_jacobian(
    model:        mujoco.MjModel,
    data:         mujoco.MjData,
    site_id:      int,
    qvel_indices: List[int],
) -> np.ndarray:
    """3×4 pozíció Jacobian a hand site-hoz, az ARM DOF-okra vetítve."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, qvel_indices]


def joint_to_norm(joint_target: np.ndarray) -> np.ndarray:
    """Joint szögek (rad) → normalizált akció [-1, 1]."""
    return np.clip(
        (joint_target - _mid) / (_half + 1e-6), -1.0, 1.0
    ).astype(np.float32)


def norm_to_joint(norm_action: np.ndarray) -> np.ndarray:
    """Normalizált akció [-1, 1] → joint szögek (rad)."""
    return (np.clip(norm_action, -1, 1) * _half + _mid).astype(np.float32)


def jt_step(
    model:       mujoco.MjModel,
    data:        mujoco.MjData,
    site_id:     int,
    delta_xyz:   np.ndarray,
    gain:        float,
    max_delta_q: float = MAX_DELTA_Q,
) -> np.ndarray:
    """Jacobian transpose lépés: delta_xyz → új joint szögek (rad)."""
    J       = get_position_jacobian(model, data, site_id, ARM_QPOS_INDICES)
    delta_q = np.clip(gain * J.T @ delta_xyz, -max_delta_q, max_delta_q)
    current_q = np.array([data.qpos[qi] for qi in ARM_QPOS_INDICES], dtype=np.float64)
    return np.clip(current_q + delta_q, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]).astype(np.float32)


# ---------------------------------------------------------------------------
# Scripted push vezérlő — 2 fázis
# ---------------------------------------------------------------------------

class ScriptedPushController:
    """
    GR1T1 push task scripted vezérlő — SETTLE → PUSH.

    SETTLE (első SETTLE_STEPS lépés):
        PUSH_ARM_POS-ra hozza a kart. Ez az a konfiguráció ahol:
          - a kézfej BOX geom y range: [-0.307, -0.203]
            → a geom teljesen a stock back face mögött van (back face ≥ -0.19)
          - a kézfej BOX geom z range: [0.802, 0.933]
            → a geom z-ben átfed a stockkal (stock top = 0.810)
        Nincs APPROACH fázis — elkerüli a chaotikus brief contact-ot.

    PUSH (SETTLE_STEPS után):
        Jacobian transpose: kéz a target irányába (y = target_y + PUSH_OVERSHOOT).
        A geom kontaktban van a stock back face-szel → sustained push.
        avg_len ≈ 83 lépés (80 settle + 3 push).
    """

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def compute_action(
        self,
        obs:     np.ndarray,
        model:   mujoco.MjModel,
        data:    mujoco.MjData,
        site_id: int,
    ) -> np.ndarray:
        """Returns: (4,) float32 normalizált akció [-1, 1]."""
        self._step += 1

        # SETTLE: PUSH_ARM_POS
        if self._step <= SETTLE_STEPS:
            return _PAP_NORM

        # PUSH: Jacobian transpose toward target
        hand_xyz   = obs[0:3].astype(np.float64)
        stock_xyz  = obs[3:6].astype(np.float64)
        target_xyz = obs[6:9].astype(np.float64)

        desired_y = target_xyz[1] + PUSH_OVERSHOOT
        desired_z = stock_xyz[2]

        delta_xyz = np.array([0.0, desired_y - hand_xyz[1], desired_z - hand_xyz[2]])
        new_q     = jt_step(model, data, site_id, delta_xyz, PUSH_GAIN)
        return joint_to_norm(new_q)


# ---------------------------------------------------------------------------
# Egy epizód gyűjtése
# ---------------------------------------------------------------------------

def collect_episode(
    env:        GR1ShelfStockEnv,
    controller: ScriptedPushController,
    seed:       Optional[int] = None,
) -> Dict:
    obs, info = env.reset(seed=seed)
    controller.reset()

    obs_list    = [obs]
    action_list = []
    reward_list = []

    done = False
    while not done:
        action = controller.compute_action(
            obs,
            env._model,
            env._data,
            env._hand_site_id,
        )
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)

        done = terminated or truncated

    success = info.get("placed", False)

    return {
        "obs":     np.array(obs_list[:-1], dtype=np.float32),
        "actions": np.array(action_list,   dtype=np.float32),
        "rewards": np.array(reward_list,   dtype=np.float32),
        "success": success,
        "length":  len(action_list),
    }


# ---------------------------------------------------------------------------
# Fő gyűjtő loop
# ---------------------------------------------------------------------------

def collect_demos(
    n_episodes: int  = DEFAULT_N_EPISODES,
    verbose:    bool = True,
    seed_base:  int  = 0,
) -> Dict:
    env        = GR1ShelfStockEnv()
    controller = ScriptedPushController()

    obs_all     = np.empty(n_episodes, dtype=object)
    actions_all = np.empty(n_episodes, dtype=object)
    rewards_all = np.empty(n_episodes, dtype=object)
    successes   = np.zeros(n_episodes, dtype=bool)
    lengths     = np.zeros(n_episodes, dtype=np.int32)

    t0        = time.time()
    n_success = 0

    for ep in range(n_episodes):
        seed    = seed_base + ep
        ep_data = collect_episode(env, controller, seed=seed)

        obs_all[ep]     = ep_data["obs"]
        actions_all[ep] = ep_data["actions"]
        rewards_all[ep] = ep_data["rewards"]
        successes[ep]   = ep_data["success"]
        lengths[ep]     = ep_data["length"]

        if ep_data["success"]:
            n_success += 1

        if verbose and (ep + 1) % 100 == 0:
            sr       = n_success / (ep + 1) * 100
            elapsed  = time.time() - t0
            eps_per_s = (ep + 1) / elapsed
            eta      = (n_episodes - ep - 1) / eps_per_s
            print(
                f"  ep {ep+1:>5}/{n_episodes}  "
                f"SR: {sr:5.1f}%  "
                f"avg_len: {lengths[:ep+1].mean():.0f}  "
                f"elapsed: {elapsed:.0f}s  "
                f"ETA: {eta:.0f}s"
            )

    env.close()

    final_sr = successes.sum() / n_episodes * 100
    elapsed  = time.time() - t0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Kész — {n_episodes} epizód, {elapsed:.1f}s")
        print(f"Success rate: {final_sr:.1f}%  ({successes.sum()}/{n_episodes})")
        print(f"Átlag epizód hossz: {lengths.mean():.1f} lépés")
        print(f"Min/Max hossz:      {lengths.min()} / {lengths.max()}")
        target_ok = "✅" if final_sr >= 70.0 else "❌"
        print(f"Target (≥70%):      {target_ok}  [vendor-independence threshold]")
        scripted_ok = "✅" if final_sr >= 95.0 else "⚠"
        print(f"Scripted SR:        {scripted_ok}  (≥95% ajánlott demo minőség)")

    return {
        "obs":             obs_all,
        "actions":         actions_all,
        "rewards":         rewards_all,
        "successes":       successes,
        "episode_lengths": lengths,
        "success_rate":    final_sr,
        "n_episodes":      n_episodes,
    }


# ---------------------------------------------------------------------------
# Mentés
# ---------------------------------------------------------------------------

def save_demos(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        obs             = data["obs"],
        actions         = data["actions"],
        rewards         = data["rewards"],
        successes       = data["successes"],
        episode_lengths = data["episode_lengths"],
        success_rate    = np.array([data["success_rate"]]),
        n_episodes      = np.array([data["n_episodes"]]),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Demo mentve: {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GR1T1 push task scripted expert demo gyűjtő v2 (Track 3)")
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES,
                        help=f"Gyűjtendő epizódok száma (default: {DEFAULT_N_EPISODES})")
    parser.add_argument("--out", type=str, default=None,
                        help="Output NPZ útvonala")
    parser.add_argument("--seed", type=int, default=0,
                        help="Véletlen seed alap (default: 0)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if verbose:
        print(f"\nGR1T1 Push Expert Demo Gyűjtő v2 (Track 3)")
        print(f"  Epizódok:    {args.n_episodes}")
        print(f"  Seed alap:   {args.seed}")
        print(f"  Vezérlő:     SETTLE(PUSH_ARM_POS) → J-transpose PUSH")
        print(f"  PUSH_ARM_POS: {PUSH_ARM_POS.tolist()}")
        print(f"  Target SR:   ≥70% vendor-independence, ≥95% demo minőség")
        print()

    data = collect_demos(
        n_episodes = args.n_episodes,
        verbose    = verbose,
        seed_base  = args.seed,
    )

    out_path = Path(args.out) if args.out else \
        _REPO_ROOT / f"results/demos/gr1_push_demos_{time.strftime('%Y%m%d_%H%M')}.npz"
    save_demos(data, out_path)

    if args.quiet:
        print(f"{data['success_rate']:.1f}")


if __name__ == "__main__":
    main()
