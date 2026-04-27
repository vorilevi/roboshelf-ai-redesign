"""
LeRobot Dataset Exporter — F3b (Phase 030).

Feladata:
    scripted_expert.py által gyűjtött EpisodeBuffer listát
    LeRobotDataset v3.0 formátumba konvertálni (Parquet + metadata).

LeRobot v3.0 könyvtárstruktúra:
    out_dir/
        meta/
            info.json          — dataset metaadatok
            stats.json         — obs/action normalizációs statisztikák
            episodes.jsonl     — egy sor per epizód (length, success, etc.)
        data/
            chunk-000/
                episode_000000.parquet
                episode_000001.parquet
                ...
        videos/                — (opcionális, most üres)

Parquet séma per sor (egy policy lépés):
    episode_index   int64
    frame_index     int64
    timestamp       float64   — lépés sorszáma * (1/MANIP_HZ)
    obs_*           float32   — 24 obs dim (obs_0 .. obs_23)
    action_*        float32   — 5 action dim (action_0 .. action_4)
    reward          float64
    done            bool
    success         bool

Futtatás (repo gyökeréből):
    python3 tools/lerobot_export.py \\
        --in-dir  data/demos/scripted_v1 \\
        --out-dir data/lerobot/scripted_v1

    (--in-dir-ben raw_demos.pkl kell, lásd scripted_expert.py --save-raw)

Referenciák:
    LeRobot v3.0 dataset spec: https://github.com/huggingface/lerobot
    ACT training: configs/bc/act_shelf_stock_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_ARROW = True
except ImportError:
    _HAS_ARROW = False
    print("⚠️  pyarrow nem elérhető — Parquet mentés nem lehetséges.")
    print("   Telepítés: pip install pyarrow --break-system-packages")

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

MANIP_HZ = 20
OBS_DIM  = 24
ACT_DIM  = 5

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent


# ---------------------------------------------------------------------------
# Fő konverzió
# ---------------------------------------------------------------------------

def export_to_lerobot(
    demos:   list,
    out_dir: Path,
    verbose: bool = True,
) -> None:
    """
    EpisodeBuffer lista → LeRobotDataset v3.0 struktúra.

    Args:
        demos:   collect_demonstrations() kimenete (list[EpisodeBuffer])
        out_dir: Kimeneti könyvtár (létrehozza, ha nem létezik)
        verbose: Progress kiírás
    """
    if not _HAS_ARROW:
        raise ImportError("pyarrow szükséges. pip install pyarrow --break-system-packages")

    out_dir = Path(out_dir)
    meta_dir  = out_dir / "meta"
    data_dir  = out_dir / "data" / "chunk-000"
    video_dir = out_dir / "videos"

    for d in [meta_dir, data_dir, video_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Obs / action statisztikák ---
    all_obs    = []
    all_actions = []
    episodes_meta = []

    for ep_idx, buf in enumerate(demos):
        obs_arr    = np.stack([s.obs    for s in buf.steps])   # (T, 24)
        action_arr = np.stack([s.action for s in buf.steps])   # (T, 5)
        all_obs.append(obs_arr)
        all_actions.append(action_arr)

        # --- Parquet írás ---
        T = len(buf.steps)
        rows = {
            "episode_index": np.full(T, ep_idx, dtype=np.int64),
            "frame_index":   np.arange(T, dtype=np.int64),
            "timestamp":     np.arange(T, dtype=np.float64) / MANIP_HZ,
            "reward":        np.array([s.reward for s in buf.steps], dtype=np.float64),
            "done":          np.array([s.done   for s in buf.steps], dtype=bool),
            "success":       np.array([s.info.get("success", False) for s in buf.steps], dtype=bool),
        }

        # Obs és action oszlopok
        for i in range(OBS_DIM):
            rows[f"obs_{i}"] = obs_arr[:, i].astype(np.float32)
        for i in range(ACT_DIM):
            rows[f"action_{i}"] = action_arr[:, i].astype(np.float32)

        table = pa.table(rows)
        ep_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, ep_path)

        episodes_meta.append({
            "episode_index": ep_idx,
            "length":        T,
            "success":       buf.success,
        })

        if verbose and ep_idx % 10 == 0:
            print(f"  Parquet: {ep_idx+1}/{len(demos)} epizód")

    # --- Globális statisztikák ---
    all_obs_np    = np.concatenate(all_obs,    axis=0)   # (N_total, 24)
    all_action_np = np.concatenate(all_actions, axis=0)  # (N_total, 5)

    stats = {
        "obs": {
            "mean": all_obs_np.mean(axis=0).tolist(),
            "std":  all_obs_np.std(axis=0).tolist(),
            "min":  all_obs_np.min(axis=0).tolist(),
            "max":  all_obs_np.max(axis=0).tolist(),
        },
        "action": {
            "mean": all_action_np.mean(axis=0).tolist(),
            "std":  all_action_np.std(axis=0).tolist(),
            "min":  all_action_np.min(axis=0).tolist(),
            "max":  all_action_np.max(axis=0).tolist(),
        },
    }

    # --- info.json ---
    info = {
        "dataset_name":    "roboshelf_scripted_v1",
        "robot":           "unitree_g1",
        "task":            "shelf_stock",
        "lerobot_version": "v3.0",
        "obs_dim":         OBS_DIM,
        "action_dim":      ACT_DIM,
        "fps":             MANIP_HZ,
        "total_episodes":  len(demos),
        "total_frames":    int(all_obs_np.shape[0]),
        "success_rate":    float(sum(b.success for b in demos) / len(demos)),
        "obs_keys":        [f"obs_{i}" for i in range(OBS_DIM)],
        "action_keys":     [f"action_{i}" for i in range(ACT_DIM)],
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_meta in episodes_meta:
            f.write(json.dumps(ep_meta) + "\n")

    if verbose:
        print(f"\n✅ LeRobot dataset mentve: {out_dir}")
        print(f"   Epizódok:    {len(demos)}")
        print(f"   Total frame: {all_obs_np.shape[0]}")
        print(f"   Success rate: {info['success_rate']*100:.1f}%")
        print(f"\nKövetkező lépés:")
        print(f"   python3 tools/train_act.py --dataset {out_dir} --config configs/bc/act_shelf_stock_v1.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LeRobot v3.0 dataset exporter")
    parser.add_argument("--in-dir",  type=str, required=True,
                        help="scripted_expert.py --save-raw kimeneti könyvtár")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="LeRobot dataset kimeneti könyvtár")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    in_dir  = Path(_REPO_ROOT / args.in_dir)
    out_dir = Path(_REPO_ROOT / args.out_dir)

    raw_path = in_dir / "raw_demos.pkl"
    if not raw_path.exists():
        print(f"❌ Nem található: {raw_path}")
        print("   Futtasd előbb: python3 tools/scripted_expert.py --save-raw --out-dir ...")
        return

    print(f"Betöltés: {raw_path}")
    with open(raw_path, "rb") as f:
        demos = pickle.load(f)

    print(f"Betöltött demonstrációk: {len(demos)}")
    export_to_lerobot(demos, out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
