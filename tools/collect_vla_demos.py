"""
VLA Demo Gyűjtő — F3e fázis.

Scriptelt expert demókat gyűjt KAMERA FRAMEKKEL együtt.
Kimenet: LeRobot v2.1 kompatibilis dataset (képekkel), UnifoLM-VLA fine-tune-hoz.

Kétmenetes gyűjtés (gyors):
    1. menet: ScriptedExpert renderelés NÉLKÜL — csak sikeres epizódok stock pozícióját
              menti. ~0.5s/epizód → 2500 kísérlet ≈ 20 perc.
    2. menet: Sikeres epizódok újrafuttatása kamera renderelésssel (fix kezdőpozícióból).
              ~50 lépés × render → 200 ep ≈ 15 perc.
    Összesen: ~35 perc (vs. ~7.5 óra egymenetben)

Videók azonnal kiírva disk-re (nem RAM-ban tárolva) — alacsony memóriaigény.

Futtatás (repo gyökeréből):
    python3 tools/collect_vla_demos.py \\
        --n-demos 200 \\
        --out-dir data/lerobot/vla_v1 \\
        --cameras front_cam

Kimenet:
    data/lerobot/vla_v1/
        meta/info.json, stats.json, episodes.jsonl
        data/chunk-000/episode_XXXXXX.parquet
        videos/chunk-000/observation.images.front_cam/episode_XXXXXX.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mujoco
import numpy as np

# ── Repo path setup ──────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_TOOLS_DIR = _HERE.parent

if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import scripted_expert as _exp
from scripted_expert import ScriptedExpert, MIN_SUCCESS_STEP

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_ARROW = True
except ImportError:
    _HAS_ARROW = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── Konstansok ───────────────────────────────────────────────────────────────

MANIP_HZ   = 20
OBS_DIM    = 24
ACT_DIM    = 5
IMG_H      = 224
IMG_W      = 224
LANGUAGE_INSTRUCTION = "Push the box to the target position on the shelf."

_XML_PATH = (_REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml").resolve()


# ── Adatstruktúra ─────────────────────────────────────────────────────────────

class FastEpisode:
    """1. menet: csak obs+action, nincs kép."""
    __slots__ = ("ep_idx", "obs_list", "action_list", "reward_list",
                 "done_list", "success_list", "success",
                 "initial_stock_x", "initial_stock_y")

    def __init__(self, ep_idx: int):
        self.ep_idx        = ep_idx
        self.obs_list:    List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.reward_list: List[float] = []
        self.done_list:   List[bool]  = []
        self.success_list: List[bool] = []
        self.success       = False
        self.initial_stock_x = 0.0
        self.initial_stock_y = 0.0


# ── CameraRenderer ────────────────────────────────────────────────────────────

class CameraRenderer:
    def __init__(self, model: mujoco.MjModel, cameras: List[str],
                 height: int = IMG_H, width: int = IMG_W):
        self._model    = model
        self._cameras  = cameras
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        frames = {}
        for cam in self._cameras:
            self._renderer.update_scene(data, camera=cam)
            frames[cam] = self._renderer.render().copy()
        return frames

    def close(self):
        self._renderer.close()


# ── VideoWriter wrapper ───────────────────────────────────────────────────────

class EpisodeVideoWriter:
    """Per-kamera VideoWriter, azonnal disk-re ír."""

    def __init__(self, vid_dirs: dict, ep_idx: int, fps: int = MANIP_HZ):
        self._writers = {}
        for cam, vd in vid_dirs.items():
            path = vd / f"episode_{ep_idx:06d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writers[cam] = cv2.VideoWriter(
                str(path), fourcc, fps, (IMG_W, IMG_H)
            )

    def write(self, frames: dict):
        for cam, rgb in frames.items():
            bgr = rgb[:, :, ::-1]
            self._writers[cam].write(bgr)

    def close(self):
        for w in self._writers.values():
            w.release()


# ── 1. MENET: gyors gyűjtés renderelés nélkül ────────────────────────────────

def collect_fast(n_demos: int,
                 max_retries: int,
                 seed: int,
                 fixed_pos_x: Optional[float],
                 fixed_pos_y: Optional[float]) -> List[FastEpisode]:
    """
    Fut renderelés nélkül. Visszaadja a sikeres epizódok listáját
    (obs/action + kezdeti stock pozíció benne).
    """
    if fixed_pos_x is not None:
        eps = 1e-6
        _exp.STOCK_RESET_X_RANGE = (fixed_pos_x - eps, fixed_pos_x + eps)
    if fixed_pos_y is not None:
        eps = 1e-6
        _exp.STOCK_RESET_Y_RANGE = (fixed_pos_y - eps, fixed_pos_y + eps)

    env       = ScriptedExpert(seed=seed)
    collected: List[FastEpisode] = []
    attempts  = 0
    t0 = time.time()

    print(f"\n1. MENET: gyors gyűjtés (renderelés nélkül)")
    print(f"Cél: {n_demos} sikeres ep | MAX_RETRIES={max_retries}")
    print("─" * 60)

    while len(collected) < n_demos and attempts < max_retries:
        obs_np = env.reset()
        ep = FastEpisode(ep_idx=len(collected))
        # stock pozíció az obs-ból (index 3:6)
        ep.initial_stock_x = float(obs_np[3])
        ep.initial_stock_y = float(obs_np[4])
        attempts += 1

        for _ in range(300):
            action_np = env.expert_action()
            next_obs_np, reward, done, info = env.step(action_np)
            success = bool(info.get("success", False))
            ep.obs_list.append(obs_np.astype(np.float32))
            ep.action_list.append(action_np.astype(np.float32))
            ep.reward_list.append(float(reward))
            ep.done_list.append(bool(done))
            ep.success_list.append(success)
            obs_np = next_obs_np
            if done:
                ep.success = success
                break

        if ep.success:
            collected.append(ep)
            if len(collected) % 10 == 0 or len(collected) <= 5:
                elapsed = time.time() - t0
                sr = 100 * len(collected) / attempts
                print(f"  [{len(collected):3d}/{n_demos}] "
                      f"attempts={attempts}  SR={sr:.1f}%  {elapsed:.0f}s")
        elif attempts % 200 == 0:
            sr = 100 * len(collected) / attempts
            print(f"  ... {attempts} kísérlet, {len(collected)} siker (SR={sr:.1f}%)")

    elapsed = time.time() - t0
    sr = 100 * len(collected) / max(attempts, 1)
    print(f"\n1. menet kész: {len(collected)} ep / {attempts} kísérlet "
          f"(SR={sr:.1f}%)  {elapsed:.0f}s")
    return collected


# ── 2. MENET: renderelés a sikeres epizódokhoz ───────────────────────────────

def render_successful(episodes: List[FastEpisode],
                      cameras: List[str],
                      vid_dirs: dict,
                      seed: int) -> None:
    """
    Sikeres epizódok újrafuttatása renderelésssel.
    Fix kezdőpozíció (elmentett stock x,y) → determinisztikus replay.
    """
    env = ScriptedExpert(seed=seed)
    renderer = CameraRenderer(env._model, cameras)
    t0 = time.time()

    print(f"\n2. MENET: renderelés ({len(episodes)} ep, kamerák: {cameras})")
    print("─" * 60)

    failed_replay = 0
    for i, ep in enumerate(episodes):
        # Fix pozíció = az 1. menetben mért kezdeti stock pozíció
        eps = 1e-6
        _exp.STOCK_RESET_X_RANGE = (ep.initial_stock_x - eps, ep.initial_stock_x + eps)
        _exp.STOCK_RESET_Y_RANGE = (ep.initial_stock_y - eps, ep.initial_stock_y + eps)

        obs_np = env.reset()
        video_writer = EpisodeVideoWriter(vid_dirs, ep.ep_idx)

        for step_i in range(len(ep.obs_list)):
            frames = renderer.render(env._data)
            video_writer.write(frames)

            action_np = env.expert_action()
            next_obs_np, _, done, info = env.step(action_np)
            obs_np = next_obs_np
            if done:
                # Extra frame az utolsó obs-hoz
                frames = renderer.render(env._data)
                video_writer.write(frames)
                if not info.get("success", False):
                    failed_replay += 1
                break

        video_writer.close()

        if (i + 1) % 20 == 0 or i < 3:
            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{len(episodes)}] ep_idx={ep.ep_idx}  {elapsed:.0f}s")

    renderer.close()
    elapsed = time.time() - t0
    print(f"\n2. menet kész: {len(episodes)} ep renderelve  {elapsed:.0f}s"
          + (f"  ⚠️ {failed_replay} replay nem sikeres (pozíció drift?)" if failed_replay else ""))


# ── LeRobot v2.1 export (Parquet + meta, videók már kiírva) ──────────────────

def export_lerobot_meta(episodes: List[FastEpisode],
                        out_dir: Path,
                        cameras: List[str]) -> None:
    """
    Parquet + meta fájlok kiírása. A videók már megvannak a vid_dirs-ben.
    """
    meta_dir = out_dir / "meta"
    data_dir = out_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    episodes_meta = []

    print(f"\nParquet + meta export → {out_dir}")

    for ep in episodes:
        T          = len(ep.obs_list)
        obs_arr    = np.stack(ep.obs_list)    # (T, 24)
        action_arr = np.stack(ep.action_list)  # (T, 5)
        all_obs.append(obs_arr)
        all_act.append(action_arr)

        rows = {
            "episode_index":        np.full(T, ep.ep_idx, dtype=np.int64),
            "frame_index":          np.arange(T, dtype=np.int64),
            "timestamp":            np.arange(T, dtype=np.float64) / MANIP_HZ,
            "reward":               np.array(ep.reward_list, dtype=np.float64),
            "done":                 np.array(ep.done_list, dtype=bool),
            "success":              np.array(ep.success_list, dtype=bool),
            "language_instruction": [LANGUAGE_INSTRUCTION] * T,
        }
        for i in range(OBS_DIM):
            rows[f"obs_{i}"] = obs_arr[:, i].astype(np.float32)
        for i in range(ACT_DIM):
            rows[f"action_{i}"] = action_arr[:, i].astype(np.float32)

        pq.write_table(pa.table(rows),
                       data_dir / f"episode_{ep.ep_idx:06d}.parquet")
        episodes_meta.append({"episode_index": ep.ep_idx, "length": T,
                               "success": ep.success,
                               "task": LANGUAGE_INSTRUCTION})

    all_obs_np = np.concatenate(all_obs, axis=0)
    all_act_np = np.concatenate(all_act, axis=0)

    stats = {
        "obs":    {"mean": all_obs_np.mean(0).tolist(), "std": all_obs_np.std(0).tolist(),
                   "min":  all_obs_np.min(0).tolist(),  "max": all_obs_np.max(0).tolist()},
        "action": {"mean": all_act_np.mean(0).tolist(), "std": all_act_np.std(0).tolist(),
                   "min":  all_act_np.min(0).tolist(),  "max": all_act_np.max(0).tolist()},
    }
    info = {
        "dataset_name": "roboshelf_vla_v1", "robot": "unitree_g1",
        "task": LANGUAGE_INSTRUCTION, "lerobot_version": "v2.1",
        "obs_dim": OBS_DIM, "action_dim": ACT_DIM, "fps": MANIP_HZ,
        "image_height": IMG_H, "image_width": IMG_W, "cameras": cameras,
        "total_episodes": len(episodes),
        "total_frames": int(all_obs_np.shape[0]),
        "success_rate": 1.0,
    }

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")

    print(f"✅ Export kész: {len(episodes)} ep, {int(all_obs_np.shape[0])} frame")


# ── Fő pipeline ───────────────────────────────────────────────────────────────

def run(args):
    # Ellenőrzések
    if not _HAS_ARROW:
        print("❌ pyarrow: pip install pyarrow --break-system-packages"); return
    if not _HAS_CV2:
        print("❌ opencv:  pip install opencv-python --break-system-packages"); return
    xml_str = _XML_PATH.read_text()
    for cam in args.cameras:
        if f'name="{cam}"' not in xml_str:
            print(f"❌ Kamera '{cam}' nem találhtó az XML-ben."); return

    out_dir = _REPO_ROOT / args.out_dir
    vid_dirs = {}
    for cam in args.cameras:
        vd = out_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
        vd.mkdir(parents=True, exist_ok=True)
        vid_dirs[cam] = vd

    print(f"Kimenet: {out_dir}")
    print(f"Kamerák: {args.cameras}")

    # ── 1. menet: gyors gyűjtés ──
    episodes = collect_fast(
        n_demos     = args.n_demos,
        max_retries = args.max_retries,
        seed        = args.seed,
        fixed_pos_x = args.fixed_pos_x,
        fixed_pos_y = args.fixed_pos_y,
    )
    if not episodes:
        print("❌ Nincs sikeres epizód."); return

    # ── 2. menet: renderelés ──
    render_successful(episodes, args.cameras, vid_dirs, args.seed)

    # ── Export meta + parquet ──
    export_lerobot_meta(episodes, out_dir, args.cameras)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VLA demo gyűjtés — F3e (kétmenetes)")
    p.add_argument("--n-demos",     type=int,   default=200)
    p.add_argument("--out-dir",     default="data/lerobot/vla_v1")
    p.add_argument("--cameras",     nargs="+",  default=["front_cam"])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max-retries", type=int,   default=5000)
    p.add_argument("--fixed-pos-x", type=float, default=None)
    p.add_argument("--fixed-pos-y", type=float, default=None)
    run(p.parse_args())

if __name__ == "__main__":
    main()
