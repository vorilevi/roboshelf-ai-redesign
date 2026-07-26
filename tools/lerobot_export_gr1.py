"""
GR1T1 push demo → LeRobot v3.0 dataset konverter.

Input:
    results/demos/gr1_push_demos_v1.npz
        obs:            (1000,) object array, elemek: (T, 24) float32
        actions:        (1000,) object array, elemek: (T, 4)  float32
        rewards:        (1000,) object array, elemek: (T,)    float32
        successes:      (1000,) bool
        episode_lengths:(1000,) int

Output:
    data/lerobot/gr1_push_v1/
        meta/
            info.json       — dataset metaadatok (action_dim=4, obs_dim=24)
            stats.json      — obs/action norm statisztikák
            episodes.jsonl  — egy sor per epizód
        data/
            chunk-000/
                episode_000000.parquet
                episode_000001.parquet
                ...

Csak sikeres epizódokat exportál (successes == True).

Futtatás (repo gyökeréből):
    python3 tools/lerobot_export_gr1.py
    python3 tools/lerobot_export_gr1.py --in results/demos/gr1_push_demos_v1.npz
    python3 tools/lerobot_export_gr1.py --out data/lerobot/gr1_push_v1 --all-episodes

Referenciák:
    T1 exporter:  tools/lerobot_export_t1.py
    GR1 env:      src/roboshelf_ai/mujoco/envs/manipulation/gr1_shelf_stock_env.py
    Kaggle train: notebooks/kaggle_unifolm_vla_finetune_gr1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_ARROW = True
except ImportError:
    _HAS_ARROW = False
    print("⚠  pyarrow nem elérhető.")
    print("   pip install pyarrow --break-system-packages")

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent

DEFAULT_IN  = _REPO_ROOT / "results/demos/gr1_push_demos_v1.npz"
DEFAULT_OUT = _REPO_ROOT / "data/lerobot/gr1_push_v1"

MANIP_HZ   = 20
OBS_DIM    = 24
ACTION_DIM = 4   # GR1T1: 4-DOF right arm (shoulder_pitch/roll/yaw + elbow_pitch)

TASK_LANG  = "Push the stock to the target position on the shelf."


# ---------------------------------------------------------------------------
# Fő konverzió
# ---------------------------------------------------------------------------

def export_to_lerobot(
    npz_path:     Path,
    out_dir:      Path,
    success_only: bool = True,
    verbose:      bool = True,
) -> None:
    """
    GR1T1 NPZ demos → LeRobot v3.0 Parquet dataset.
    """
    if not _HAS_ARROW:
        raise ImportError("pyarrow szükséges. pip install pyarrow --break-system-packages")

    if verbose:
        print(f"Betöltés: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    obs_all      = data["obs"]
    actions_all  = data["actions"]
    rewards_all  = data["rewards"]
    successes    = data["successes"]
    n_total      = len(successes)
    n_success    = int(successes.sum())

    if verbose:
        print(f"  Összes epizód: {n_total}, sikeres: {n_success} ({n_success/n_total*100:.1f}%)")

    if success_only:
        indices = np.where(successes)[0]
        if verbose:
            print(f"  Exportálás: {len(indices)} sikeres epizód")
    else:
        indices = np.arange(n_total)
        if verbose:
            print(f"  Exportálás: mind {n_total} epizód (--all-episodes)")

    meta_dir = out_dir / "meta"
    data_dir = out_dir / "data" / "chunk-000"
    for d in [meta_dir, data_dir, out_dir / "videos"]:
        d.mkdir(parents=True, exist_ok=True)

    all_obs:     List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episodes_meta = []

    for out_idx, src_idx in enumerate(indices):
        obs_ep    = obs_all[src_idx].astype(np.float32)
        action_ep = actions_all[src_idx].astype(np.float32)
        reward_ep = rewards_all[src_idx].astype(np.float64)
        T         = len(reward_ep)

        done_ep = np.zeros(T, dtype=bool)
        done_ep[-1] = True
        success_ep = np.full(T, bool(successes[src_idx]), dtype=bool)

        rows: dict = {
            "episode_index": np.full(T, out_idx, dtype=np.int64),
            "frame_index":   np.arange(T, dtype=np.int64),
            "timestamp":     np.arange(T, dtype=np.float64) / MANIP_HZ,
            "reward":        reward_ep,
            "done":          done_ep,
            "success":       success_ep,
        }

        for i in range(OBS_DIM):
            rows[f"obs_{i}"] = obs_ep[:, i]
        for i in range(ACTION_DIM):
            rows[f"action_{i}"] = action_ep[:, i]

        table = pa.table(rows)
        ep_path = data_dir / f"episode_{out_idx:06d}.parquet"
        pq.write_table(table, ep_path)

        all_obs.append(obs_ep)
        all_actions.append(action_ep)

        episodes_meta.append({
            "episode_index": out_idx,
            "length":        T,
            "success":       bool(successes[src_idx]),
        })

        if verbose and (out_idx + 1) % 100 == 0:
            print(f"  Parquet: {out_idx+1}/{len(indices)} epizód")

    all_obs_np    = np.concatenate(all_obs,    axis=0)
    all_action_np = np.concatenate(all_actions, axis=0)

    stats = {
        "obs": {
            "mean": all_obs_np.mean(axis=0).tolist(),
            "std":  (all_obs_np.std(axis=0) + 1e-8).tolist(),
            "min":  all_obs_np.min(axis=0).tolist(),
            "max":  all_obs_np.max(axis=0).tolist(),
        },
        "action": {
            "mean": all_action_np.mean(axis=0).tolist(),
            "std":  (all_action_np.std(axis=0) + 1e-8).tolist(),
            "min":  all_action_np.min(axis=0).tolist(),
            "max":  all_action_np.max(axis=0).tolist(),
        },
    }

    info = {
        "dataset_name":    "roboshelf_gr1_push_v1",
        "robot":           "fourier_gr1t1",
        "task":            TASK_LANG,
        "lerobot_version": "v3.0",
        "obs_dim":         OBS_DIM,
        "action_dim":      ACTION_DIM,
        "fps":             MANIP_HZ,
        "total_episodes":  len(indices),
        "total_frames":    int(all_obs_np.shape[0]),
        "success_rate":    float(successes[indices].mean()),
        "cameras":         [],
        "obs_keys":        [f"obs_{i}" for i in range(OBS_DIM)],
        "action_keys":     [f"action_{i}" for i in range(ACTION_DIM)],
        "source_npz":      str(npz_path),
        "success_only":    success_only,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_meta in episodes_meta:
            f.write(json.dumps(ep_meta) + "\n")

    if verbose:
        size_mb = sum(
            p.stat().st_size for p in data_dir.glob("*.parquet")
        ) / 1024 / 1024
        print(f"\n✅ LeRobot dataset mentve: {out_dir}")
        print(f"   Epizódok:     {len(indices)}")
        print(f"   Total frame:  {all_obs_np.shape[0]}")
        print(f"   Parquet size: {size_mb:.1f} MB")
        print(f"   obs_dim:      {OBS_DIM}")
        print(f"   action_dim:   {ACTION_DIM}")
        print(f"\nKövetkező lépés:")
        print(f"   HF push: huggingface-cli upload vorilevi/roboshelf-gr1-push-v1 {out_dir} --repo-type dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GR1T1 push demo NPZ → LeRobot v3.0 dataset"
    )
    parser.add_argument(
        "--in", dest="npz_in", type=str, default=str(DEFAULT_IN),
        help=f"Bemeneti NPZ (default: {DEFAULT_IN})"
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT),
        help=f"Kimeneti könyvtár (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--all-episodes", action="store_true",
        help="Sikertelen epizódokat is exportál"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    export_to_lerobot(
        npz_path     = Path(args.npz_in),
        out_dir      = Path(args.out),
        success_only = not args.all_episodes,
        verbose      = not args.quiet,
    )


if __name__ == "__main__":
    main()
