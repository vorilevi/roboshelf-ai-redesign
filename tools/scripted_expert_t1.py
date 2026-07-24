"""
T1 push task demo gyűjtő — scripted expert.

Determinisztikus operational-space vezérlővel (Jacobian transpose)
gyűjt N epizódot a T1ShelfStockEnv-ből.

Vezérlő logika:
  APPROACH: kéz közelít a stock "mögé" (a target ellentétes oldalára)
  PUSH:     kéz áttolja a stockot a target felé

Output:
  results/demos/t1_push_demos_YYYYMMDD_HHMM.npz
  - obs:            (N,) object array, elemek: (T, 24) float32
  - actions:        (N,) object array, elemek: (T, 4)  float32 (normalizált -1..1)
  - rewards:        (N,) object array, elemek: (T,)    float32
  - successes:      (N,) bool
  - episode_lengths:(N,) int

Futtatás (repo gyökeréből):
  python3 tools/scripted_expert_t1.py
  python3 tools/scripted_expert_t1.py --n-episodes 100 --verbose
  python3 tools/scripted_expert_t1.py --n-episodes 1000 --out results/demos/t1_demos.npz

Referenciák:
  Env:    src/roboshelf_ai/mujoco/envs/manipulation/t1_shelf_stock_env.py
  Diag:   results/diag/t1_reach_*.csv  (DEFAULT_ARM_POS)
  kp:     results/diag/t1_kp_sweep_*.csv  (kp=150 validált)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mujoco
from roboshelf_ai.mujoco.envs.manipulation.t1_shelf_stock_env import (
    T1ShelfStockEnv,
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

# Vezérlő paraméterek
# T1 jobb kar természetes egyensúlyi pozíciója DEFAULT_ARM_POS-on:
#   hand_x ≈ 0.39m, hand_y ≈ -0.18m, hand_z ≈ 0.78m  (diag eredmény)
# Stock spawn: x=0.30, y ∈ [-0.15, -0.10], z=0.77
# → a kéz (y≈-0.18) már a stock (y≈-0.12) mögött van — közvetlen push lehetséges!
# → csak z-t kell emelni (0.69→0.78) a SETTLE fázisban, aztán pusholni
SETTLE_STEPS       = 40     # lépések száma mielőtt push kezdődik (arm settlements)
PUSH_GAIN          = 4.0    # Jacobian transpose gain (PUSH fázis)
MAX_DELTA_Q        = 0.12   # max joint delta per policy step (rad)


# ---------------------------------------------------------------------------
# Jacobian transpose vezérlő segédfüggvények
# ---------------------------------------------------------------------------

def get_position_jacobian(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    site_id: int,
    qvel_indices: List[int],
) -> np.ndarray:
    """
    Visszaadja a hand site 3×4 pozíció Jacobianját az arm DOF-okra.
    T1 karban minden joint 1-DOF revolute → qpos_idx == qvel_idx.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, qvel_indices]  # 3×4


def joint_to_norm(joint_target: np.ndarray) -> np.ndarray:
    """Joint szögek (rad) → normalizált akció [-1, 1]."""
    mid  = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
    half = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
    return np.clip((joint_target - mid) / (half + 1e-6), -1.0, 1.0).astype(np.float32)


def norm_to_joint(norm_action: np.ndarray) -> np.ndarray:
    """Normalizált akció [-1, 1] → joint szögek (rad)."""
    mid  = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
    half = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
    return (np.clip(norm_action, -1, 1) * half + mid).astype(np.float32)


# ---------------------------------------------------------------------------
# Scripted push vezérlő
# ---------------------------------------------------------------------------

class ScriptedPushController:
    """
    T1 push task scripted vezérlő — SETTLE + PUSH stratégia.

    A T1 jobb kar természetes egyensúlyi helyzete:
        hand_y ≈ -0.18m, hand_z ≈ 0.78m  (DEFAULT_ARM_POS)
    A stock spawnol: y ∈ [-0.15, -0.10], z = 0.77
    → A kéz már a stock MÖGÖTT van (hand_y < stock_y), csak a z magasságot
      kell beállítani (arm settle), majd y-ban előre kell tolni.

    Fázisok:
      SETTLE (első SETTLE_STEPS lépés):
        Tartja a DEFAULT_ARM_POS-t. Az aktuátor a kart equilibriumba hozza
        (z: 0.69 → 0.78m). Contact nélkül.
      PUSH:
        Jacobian transpose: kéz a target y=0 felé mozog (push irány: +y).
        Csak y és z komponenst vezérli (x-et szabadon hagyja → természetes).
        A kéz sweepeli a stockot a cél felé.
    """

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def compute_action(
        self,
        obs:      np.ndarray,
        model:    mujoco.MjModel,
        data:     mujoco.MjData,
        site_id:  int,
    ) -> np.ndarray:
        """
        Returns:
            action: (4,) float32, normalizált [-1, 1]
        """
        self._step += 1

        # ── SETTLE fázis: tartjuk a DEFAULT pozíciót ─────────────────────
        if self._step <= SETTLE_STEPS:
            return joint_to_norm(_DEFAULT_ARM_POS)

        # ── PUSH fázis: Jacobian transpose, csak y és z ──────────────────
        hand_xyz   = obs[0:3].astype(np.float64)
        stock_xyz  = obs[3:6].astype(np.float64)
        target_xyz = obs[6:9].astype(np.float64)

        # Cél: kéz toljon a target y pozíciója felé, stock magasságán
        # "Túllövés" +0.03m a target felé (contact erő pusholja a stockot)
        desired_y = target_xyz[1] + 0.03   # y=0 + kis túllövés
        desired_z = stock_xyz[2]           # stock magasságán marad

        # Csak y és z komponens (x szabadon)
        delta_yz = np.array([0.0, desired_y - hand_xyz[1], desired_z - hand_xyz[2]])

        J = get_position_jacobian(model, data, site_id, ARM_QPOS_INDICES)  # 3×4
        delta_q = PUSH_GAIN * J.T @ delta_yz                               # (4,)
        delta_q = np.clip(delta_q, -MAX_DELTA_Q, MAX_DELTA_Q)

        current_q = np.array([data.qpos[qi] for qi in ARM_QPOS_INDICES], dtype=np.float64)
        new_q     = np.clip(current_q + delta_q, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1])

        return joint_to_norm(new_q.astype(np.float32))


# ---------------------------------------------------------------------------
# Egy epizód gyűjtése
# ---------------------------------------------------------------------------

def collect_episode(
    env:        T1ShelfStockEnv,
    controller: ScriptedPushController,
    seed:       Optional[int] = None,
) -> Dict:
    """
    Lefuttat egy epizódot a scripted policy-val.

    Returns:
        dict: obs, actions, rewards, success, length
    """
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
        )  # type: ignore[arg-type]
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)

        done = terminated or truncated

    success = info.get("placed", False)

    return {
        "obs":     np.array(obs_list[:-1], dtype=np.float32),   # (T, 24)
        "actions": np.array(action_list,   dtype=np.float32),   # (T, 4)
        "rewards": np.array(reward_list,   dtype=np.float32),   # (T,)
        "success": success,
        "length":  len(action_list),
    }


# ---------------------------------------------------------------------------
# Fő gyűjtő loop
# ---------------------------------------------------------------------------

def collect_demos(
    n_episodes: int = DEFAULT_N_EPISODES,
    verbose:    bool = True,
    seed_base:  int  = 0,
) -> Dict:
    """
    Gyűjt n_episodes demo epizódot és visszaadja az összesített adatot.
    """
    env        = T1ShelfStockEnv()
    controller = ScriptedPushController()

    obs_all     = np.empty(n_episodes, dtype=object)
    actions_all = np.empty(n_episodes, dtype=object)
    rewards_all = np.empty(n_episodes, dtype=object)
    successes   = np.zeros(n_episodes, dtype=bool)
    lengths     = np.zeros(n_episodes, dtype=np.int32)

    t0          = time.time()
    n_success   = 0

    for ep in range(n_episodes):
        seed = seed_base + ep
        ep_data = collect_episode(env, controller, seed=seed)

        obs_all[ep]     = ep_data["obs"]
        actions_all[ep] = ep_data["actions"]
        rewards_all[ep] = ep_data["rewards"]
        successes[ep]   = ep_data["success"]
        lengths[ep]     = ep_data["length"]

        if ep_data["success"]:
            n_success += 1

        if verbose and (ep + 1) % 100 == 0:
            sr    = n_success / (ep + 1) * 100
            elapsed = time.time() - t0
            eps_per_s = (ep + 1) / elapsed
            eta = (n_episodes - ep - 1) / eps_per_s
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
        target_ok = "✅" if final_sr >= 80.0 else "❌"
        print(f"Target (≥80%):      {target_ok}")

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
    parser = argparse.ArgumentParser(description="T1 push task scripted expert demo gyűjtő")
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
        print(f"\nT1 Push Expert Demo Gyűjtő")
        print(f"  Epizódok:  {args.n_episodes}")
        print(f"  Seed alap: {args.seed}")
        print(f"  Vezérlő:   Jacobian transpose (APPROACH → PUSH)")
        print(f"  Env:       T1ShelfStockEnv, push task")
        print()

    data = collect_demos(
        n_episodes = args.n_episodes,
        verbose    = verbose,
        seed_base  = args.seed,
    )

    out_path = Path(args.out) if args.out else \
        _REPO_ROOT / f"results/demos/t1_push_demos_{time.strftime('%Y%m%d_%H%M')}.npz"
    save_demos(data, out_path)

    if args.quiet:
        print(f"{data['success_rate']:.1f}")


if __name__ == "__main__":
    main()
