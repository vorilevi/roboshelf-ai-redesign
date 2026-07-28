"""
Policy Rollout Demo Gyűjtő — F3b alternatív megközelítés (Phase 030).

Miért ez a scripted_expert.py helyett:
    A 4-DOF kar workspace-e fizikailag nem éri el a stock pozícióját
    (right_hand_site legjobb közelítés: 0.328m a stock-tól). A scripted IK
    ezért sosem tud sikeres epizódot generálni.

    Ez a script ehelyett a v9 PPO policy (best_model.zip, 10% sikerességi ráta)
    determinisztikus rollout-jaiból gyűjt sikeres epizódokat.
    ~3000-5000 epizód szükséges 500 sikeres demo-hoz.

Kimenet:
    data/demos/policy_v1/raw_demos.pkl  — EpisodeBuffer lista
    (kompatibilis a tools/lerobot_export.py-val)

Futtatás (repo gyökeréből):
    # Validáció: 200 epizód, leáll ha ≥ 20 sikeres
    python3 tools/policy_demo_collector.py --max-episodes 200 --target-demos 20

    # Teljes gyűjtés: 500 sikeres demo
    python3 tools/policy_demo_collector.py --target-demos 500 --out-dir data/demos/policy_v1

    # Legjobb checkpoint tesztelés
    python3 tools/policy_demo_collector.py --model results/manip_checkpoints_v9/best_model.zip \\
        --vec-normalize results/manip_checkpoints_v9/vec_normalize.pkl --target-demos 50
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv

import yaml

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

DEFAULT_MODEL     = "results/manip_checkpoints_v9/best_model.zip"
DEFAULT_VEC_NORM  = "results/manip_checkpoints_v9/vec_normalize.pkl"
DEFAULT_CONFIG    = "configs/manipulation/shelf_stock_v9.yaml"
DEFAULT_OUT_DIR   = "data/demos/policy_v1"

# known issue #18/#21 guard: az első MIN_SUCCESS_STEP lépésben bekövetkező
# siker lucky reset artifact (kinematikai z=0.870 > target z=0.85 → 1. lépésen még a
# target felett van). Ez az igazi siker mérésének kulcsa.
# eval_with_metrics.py-val konzisztens értéket kell tartani!
MIN_SUCCESS_STEP = 25

# ---------------------------------------------------------------------------
# Adatstruktúrák (azonos a scripted_expert.py-val → lerobot_export.py kompatibilis)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepData:
    obs:    np.ndarray   # (24,) float32 — RAW obs (VecNorm előtti)
    action: np.ndarray   # (5,) float32
    reward: float
    done:   bool
    info:   dict


@dataclasses.dataclass
class EpisodeBuffer:
    steps:   List[StepData]
    success: bool
    length:  int


# ---------------------------------------------------------------------------
# Demo gyűjtő
# ---------------------------------------------------------------------------

def collect_demos(
    model_path:     Path,
    vec_norm_path:  Path,
    config_path:    Path,
    target_demos:   int  = 500,
    max_episodes:   int  = 10000,
    seed:           int  = 42,
    deterministic:  bool = False,
    verbose:        bool = True,
) -> List[EpisodeBuffer]:

    # --- Config betöltés ---
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # --- Env létrehozás (single, DummyVecEnv) ---
    def make_env():
        env = G1ShelfStockEnv(cfg=cfg)
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
    vec_env.training   = False   # ne frissítse a statisztikákat
    vec_env.norm_reward = False  # reward-ot ne normalizálja

    # --- Model betöltés ---
    model = PPO.load(str(model_path), env=vec_env)

    if verbose:
        print(f"Model:      {model_path}")
        print(f"VecNorm:    {vec_norm_path}")
        print(f"Config:     {config_path}")
        print(f"Cél:        {target_demos} sikeres demo")
        print("─" * 60)

    demos:  List[EpisodeBuffer] = []
    ep_idx  = 0
    t0      = time.time()

    norm_obs = vec_env.reset()   # (1, 24) normalizált

    # Aktuális epizód buffer
    current_steps: List[StepData] = []

    while len(demos) < target_demos and ep_idx < max_episodes:
        # Policy döntés
        action, _ = model.predict(norm_obs, deterministic=deterministic)

        # Raw obs mentés (VecNorm visszafordítása)
        raw_obs = vec_env.unnormalize_obs(norm_obs)[0].astype(np.float32)

        # Lépés
        norm_obs, rewards, dones, infos = vec_env.step(action)

        current_steps.append(StepData(
            obs    = raw_obs,
            action = action[0].astype(np.float32),
            reward = float(rewards[0]),
            done   = bool(dones[0]),
            info   = infos[0],
        ))

        # Epizód vége
        if dones[0]:
            ep_idx += 1
            success = infos[0].get("placed", False) or infos[0].get("success", False)

            # SB3 DummyVecEnv: auto-reset után az obs már az új epizód első obs-a
            # → az aktuális norm_obs már az új epizód első obs-a, nem kell újra reset

            # SB3 DummyVecEnv: auto-reset után a terminal info a dones step-ben van
            # Ha placed nem jött át, ellenőrzük a lépések utolsó info-ját is
            if not success and current_steps:
                success = current_steps[-1].info.get("placed", False)

            # known issue #18/#21: MIN_SUCCESS_STEP guard
            # Ha az epizód az első MIN_SUCCESS_STEP lépésen belül terminál, az
            # nagy valószínűséggel lucky reset artifact (nem tanult viselkedés).
            # Az ilyen epizódot sikertelennek tekintjük a demo gyűjtéshez.
            ep_len = len(current_steps)
            if success and ep_len < MIN_SUCCESS_STEP:
                if verbose:
                    print(
                        f"  ⚠ SUSPICIOUS ep #{ep_idx}: len={ep_len} < {MIN_SUCCESS_STEP} "
                        f"— lucky reset artifact, kihagyva"
                    )
                success = False

            buf = EpisodeBuffer(
                steps   = current_steps,
                success = success,
                length  = ep_len,
            )

            if success:
                demos.append(buf)
                if verbose:
                    elapsed = time.time() - t0
                    rate = len(demos) / elapsed
                    eta  = (target_demos - len(demos)) / rate if rate > 0 else 0
                    print(
                        f"[{len(demos):4d}/{target_demos}] ✅ siker | "
                        f"ep #{ep_idx:5d} | {buf.length} lépés | "
                        f"ETA: {eta/60:.1f} perc"
                    )
            elif verbose and ep_idx % 100 == 0:
                sr = 100 * len(demos) / ep_idx if ep_idx > 0 else 0
                print(
                    f"[{len(demos):4d}/{target_demos}] ... "
                    f"ep #{ep_idx:5d} | sikerességi arány: {sr:.1f}%"
                )

            current_steps = []

    vec_env.close()

    elapsed = time.time() - t0
    sr = 100 * len(demos) / ep_idx if ep_idx > 0 else 0
    if verbose:
        print()
        print(f"Eredmény: {len(demos)}/{target_demos} demo | "
              f"{ep_idx} epizód | {sr:.1f}% sikerességi arány | "
              f"{elapsed/60:.1f} perc")

    return demos


# ---------------------------------------------------------------------------
# Checkpoint összehasonlítás
# ---------------------------------------------------------------------------

def compare_checkpoints(config_path: Path, verbose: bool = True) -> None:
    """Összehasonlítja a v9 összes checkpoint-ját, megkeresi a legjobbat."""
    checkpoints = sorted(
        (_REPO_ROOT / "results/manip_checkpoints_v9").glob("manip_shelf_stock_*_steps.zip")
    )
    vec_norms = sorted(
        (_REPO_ROOT / "results/manip_checkpoints_v9").glob("manip_shelf_stock_vecnormalize_*_steps.pkl")
    )

    print("Checkpoint összehasonlítás (50 epizód/checkpoint):")
    print(f"{'Checkpoint':<45} {'Siker':>6} {'Arány':>7}")
    print("─" * 60)

    best_ckpt = None; best_sr = 0

    for ckpt, vn in zip(checkpoints, vec_norms):
        demos = collect_demos(
            model_path    = ckpt,
            vec_norm_path = vn,
            config_path   = config_path,
            target_demos  = 50,
            max_episodes  = 300,
            verbose       = False,
        )
        # count episodes run
        sr = len(demos)
        print(f"{ckpt.name:<45} {sr:>6}/50  {sr/50*100:>6.1f}%")
        if sr > best_sr:
            best_sr   = sr
            best_ckpt = (ckpt, vn)

    if best_ckpt:
        print(f"\nLegjobb: {best_ckpt[0].name}  ({best_sr}/50)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO policy rollout demo gyűjtő")
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL,
                        help=f"PPO model zip (default: {DEFAULT_MODEL})")
    parser.add_argument("--vec-normalize", type=str, default=DEFAULT_VEC_NORM,
                        help=f"VecNormalize pkl (default: {DEFAULT_VEC_NORM})")
    parser.add_argument("--config",      type=str, default=DEFAULT_CONFIG,
                        help=f"Env config yaml (default: {DEFAULT_CONFIG})")
    parser.add_argument("--target-demos", type=int, default=500,
                        help="Gyűjtendő sikeres demo-k száma (default: 500)")
    parser.add_argument("--max-episodes", type=int, default=10000,
                        help="Max futtatott epizód (default: 10000)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Stochasztikus policy (exploráció bekapcsolva) — jobb sikerességi arány")
    parser.add_argument("--out-dir",     type=str, default=DEFAULT_OUT_DIR,
                        help=f"Kimeneti könyvtár (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--compare-checkpoints", action="store_true",
                        help="Összehasonlítja a v9 összes checkpoint-ját")
    args = parser.parse_args()

    model_path    = _REPO_ROOT / args.model
    vec_norm_path = _REPO_ROOT / args.vec_normalize
    config_path   = _REPO_ROOT / args.config
    out_dir       = _REPO_ROOT / args.out_dir

    if not model_path.exists():
        print(f"❌ Model nem található: {model_path}")
        return
    if not vec_norm_path.exists():
        print(f"❌ VecNormalize nem található: {vec_norm_path}")
        return

    if args.compare_checkpoints:
        compare_checkpoints(config_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    demos = collect_demos(
        model_path    = model_path,
        vec_norm_path = vec_norm_path,
        config_path   = config_path,
        target_demos  = args.target_demos,
        max_episodes  = args.max_episodes,
        seed          = args.seed,
        deterministic = not args.stochastic,
    )

    if not demos:
        print("❌ Nulla sikeres demo. Próbáld más checkpoint-tal (--model, --vec-normalize).")
        return

    raw_path = out_dir / "raw_demos.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"\n✅ Mentve: {raw_path}  ({len(demos)} demo)")
    print(f"\nKövetkező lépés:")
    print(f"   python3 tools/lerobot_export.py --in-dir {out_dir} --out-dir data/lerobot/policy_v1")


if __name__ == "__main__":
    main()
