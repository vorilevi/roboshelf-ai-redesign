"""
Megbízható policy kiértékelő — Phase 030 F3b (known issue #21 fix).

Problémák, amelyeket megold:
  1. Lucky reset artifact: ha az epizód lépés-1-én terminál, a siker
     valójában lucky seed (stock a reset után már a success szférán belül).
     → MIN_SUCCESS_STEP guard kiszűri.
  2. Aggregált ep_lengths.mean() elrejti az outliereket.
     → Per-epizód logging, suspicious_pct metrika.
  3. 10 epizódos eval nem elég statisztikailag.
     → Default: 50 epizód.

Output JSON:
  {
    "sr":                float,   # nyers siker arány
    "sr_valid":          float,   # siker arány MIN_SUCCESS_STEP után (az igazi)
    "mean_ep_length":    float,
    "mean_dist":         float,   # stock→target átlag távolság epizód végén
    "suspicious_pct":    float,   # % ahol ep_length < MIN_SUCCESS_STEP
    "n_episodes":        int,
    "min_success_step":  int,
    "model_path":        str,
  }

Futtatás:
  python3 tools/eval_with_metrics.py \\
    --config configs/manipulation/shelf_stock_v9.yaml \\
    --model  results/manip_checkpoints_v9/best_model.zip \\
    --vec-normalize results/manip_checkpoints_v9/vec_normalize.pkl \\
    --episodes 50

  # Gyors visszamérés (v9 igaz SR = 0% ellenőrzés):
  python3 tools/eval_with_metrics.py --episodes 50 --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

DEFAULT_MODEL     = "results/manip_checkpoints_v9/best_model.zip"
DEFAULT_VEC_NORM  = "results/manip_checkpoints_v9/vec_normalize.pkl"
DEFAULT_CONFIG    = "configs/manipulation/shelf_stock_v9.yaml"
DEFAULT_EPISODES  = 50

# Ez az igazi siker mérésének kulcsa (known issue #21):
# Az első MIN_SUCCESS_STEP lépésben bekövetkező siker lucky reset artifact.
# (25 lépés × 50 MuJoCo step = 1250 sim step → stock teljesen lezuhant)
MIN_SUCCESS_STEP  = 25


# ---------------------------------------------------------------------------
# Kiértékelés
# ---------------------------------------------------------------------------

def evaluate(
    model_path:    Path,
    vec_norm_path: Path,
    config_path:   Path,
    n_episodes:    int  = DEFAULT_EPISODES,
    seed:          int  = 42,
    verbose:       bool = True,
) -> Dict:

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def make_env():
        env = G1ShelfStockEnv(cfg=cfg)
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env)

    if verbose:
        print(f"\nEval: {model_path.name}  ({n_episodes} epizód, MIN_SUCCESS_STEP={MIN_SUCCESS_STEP})")
        print("─" * 60)

    ep_lengths:    List[int]   = []
    ep_dists:      List[float] = []
    successes:     List[bool]  = []
    valid_successes: List[bool] = []
    suspicious:    List[bool]  = []

    obs     = vec_env.reset()
    ep_len  = 0
    ep_dist = 0.0

    while len(ep_lengths) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        ep_len  += 1

        info   = infos[0]
        dist   = float(info.get("place_dist", info.get("stock_target_dist", 0.0)))
        ep_dist = dist  # utolsó lépés távolsága

        if dones[0]:
            success = info.get("placed", False) or info.get("success", False)
            is_suspicious = ep_len < MIN_SUCCESS_STEP
            valid_success = success and not is_suspicious

            ep_lengths.append(ep_len)
            ep_dists.append(ep_dist)
            successes.append(success)
            valid_successes.append(valid_success)
            suspicious.append(is_suspicious and success)

            if verbose:
                flag = "✅" if valid_success else ("⚠" if is_suspicious and success else "❌")
                note = " (SUSPICIOUS — lucky reset?)" if is_suspicious and success else ""
                print(f"  Ep {len(ep_lengths):3d}: {flag} len={ep_len:3d}  dist={ep_dist:.4f}{note}")

            ep_len = 0

    vec_env.close()

    sr        = float(np.mean(successes))
    sr_valid  = float(np.mean(valid_successes))
    susp_pct  = 100.0 * sum(suspicious) / n_episodes

    result = {
        "sr":               round(sr, 4),
        "sr_valid":         round(sr_valid, 4),
        "mean_ep_length":   round(float(np.mean(ep_lengths)), 1),
        "mean_dist":        round(float(np.mean(ep_dists)), 4),
        "suspicious_pct":   round(susp_pct, 1),
        "n_episodes":       n_episodes,
        "min_success_step": MIN_SUCCESS_STEP,
        "model_path":       str(model_path),
    }

    if verbose:
        print("─" * 60)
        print(f"  SR (nyers):       {sr*100:.1f}%")
        print(f"  SR (valid ✅):    {sr_valid*100:.1f}%  ← ez az igazi metrika")
        print(f"  Átlag ep hossz:   {result['mean_ep_length']:.1f} lépés")
        print(f"  Átlag dist:       {result['mean_dist']:.4f}m")
        print(f"  Suspicious ep:    {susp_pct:.1f}%  (len < {MIN_SUCCESS_STEP})")
        if susp_pct > 5:
            print(f"  ⚠ FIGYELEM: {susp_pct:.0f}% gyanús rövid sikerű epizód → lucky reset artifact!")
        print()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Policy kiértékelő — sr_valid (igaz SR)")
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--vec-normalize",type=str, default=DEFAULT_VEC_NORM)
    parser.add_argument("--config",       type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes",     type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--out",          type=str, default=None,
                        help="JSON output útvonal (opcionális)")
    parser.add_argument("--quiet",        action="store_true",
                        help="Csak a JSON-t írja ki stdout-ra")
    args = parser.parse_args()

    model_path    = _REPO_ROOT / args.model
    vec_norm_path = _REPO_ROOT / args.vec_normalize
    config_path   = _REPO_ROOT / args.config

    for p, name in [(model_path, "model"), (vec_norm_path, "vec_normalize"), (config_path, "config")]:
        if not p.exists():
            print(f"❌ {name} nem található: {p}")
            sys.exit(1)

    result = evaluate(
        model_path    = model_path,
        vec_norm_path = vec_norm_path,
        config_path   = config_path,
        n_episodes    = args.episodes,
        seed          = args.seed,
        verbose       = not args.quiet,
    )

    json_out = json.dumps(result, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_out)
        if not args.quiet:
            print(f"JSON mentve: {args.out}")
    else:
        print(json_out)


if __name__ == "__main__":
    main()
