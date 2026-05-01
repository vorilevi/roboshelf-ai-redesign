"""
ACT Policy Vizualizáció — F3d Phase 030.

Betölti a tanított ACT modellt és interaktív MuJoCo viewer-ben mutatja a viselkedést.

⚠️  macOS: mjpython kell (nem python3)! — known_issues.md #1
    Hiba python3-mal: RuntimeError: `launch_passive` requires mjpython on macOS

Futtatás (repo gyökeréből):
    mjpython tools/viz_act_policy.py \\
        --ckpt    results/bc_checkpoints_act_v2 \\
        --stats   data/lerobot/scripted_v1/meta/stats.json \\
        --n-ep    5 \\
        --exec-horizon 5

Kezelés:
    - Viewer ablakban: egérrel forgatható a kamera
    - q / Esc / ablak bezárás → kilépés
    - Epizódonként: 1.5 mp szünet reset után (látható az újraindulás)

Kimenet:
    Konzolban: epizód eredménye (✅/❌), place_dist, lépésszám, SR összesítő
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_TOOLS_DIR = _HERE.parent

if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import scripted_expert as _exp
from scripted_expert import (
    _get_obs,
    _JOINT_RANGES,
    _GRIPPER_CLOSED,
    _GRIPPER_OPEN,
    _DEFAULT_ARM_POS,
    ARM_QPOS_START,
    ARM_CTRL_START,
    GRIPPER_CTRL_START,
    N_ARM_DOF,
    DECIMATION,
    GOAL_RADIUS,
    STOCK_QPOS_START,
    STOCK_RESET_Z,
    MIN_SUCCESS_STEP,
)
from train_act import load_policy, ACTModel
from eval_act import StatsNormalizer


# ─── Env ─────────────────────────────────────────────────────────────────────

def _reset_env(model, data, rng):
    mujoco.mj_resetData(model, data)
    data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = _DEFAULT_ARM_POS
    data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF]  = _DEFAULT_ARM_POS
    data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7]  = _GRIPPER_OPEN

    hand_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    stock_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

    lo_x, hi_x = _exp.STOCK_RESET_X_RANGE
    lo_y, hi_y = _exp.STOCK_RESET_Y_RANGE

    for _ in range(50):
        sx = float(rng.uniform(lo_x, hi_x))
        sy = float(rng.uniform(lo_y, hi_y))
        data.qpos[STOCK_QPOS_START + 0] = sx
        data.qpos[STOCK_QPOS_START + 1] = sy
        data.qpos[STOCK_QPOS_START + 2] = STOCK_RESET_Z
        data.qpos[STOCK_QPOS_START + 3:STOCK_QPOS_START + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        h = data.site_xpos[hand_site_id]
        s = data.xpos[stock_body_id]
        if np.linalg.norm(h - s) >= 0.12:
            break

    mujoco.mj_forward(model, data)


def _get_full_obs(model, data):
    hand_site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
    stock_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

    hand_body_ids: set = set()
    for name in _exp._HAND_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            hand_body_ids.add(bid)

    return (
        _get_obs(model, data, hand_site_id, target_site_id, stock_body_id, hand_body_ids),
        hand_site_id, target_site_id, stock_body_id,
    )


def _apply_action(model, data, action_norm: np.ndarray):
    """action_norm: (5,) float32, eredeti [-1,1] térben (scripted_expert _norm_action)."""
    arm_action     = np.array(action_norm[:4], dtype=np.float32)
    gripper_signal = float(np.clip(action_norm[4], -1.0, 1.0))

    lo, hi = _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
    target_qpos = lo + (arm_action + 1.0) * 0.5 * (hi - lo)
    target_qpos = np.clip(target_qpos, lo, hi)
    data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_qpos

    t = (gripper_signal + 1.0) / 2.0
    data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = (
        (1.0 - t) * _GRIPPER_OPEN + t * _GRIPPER_CLOSED
    )

    for _ in range(DECIMATION):
        mujoco.mj_step(model, data)


# ─── Viz fő loop ─────────────────────────────────────────────────────────────

def run_viz(
    ckpt_dir:     str,
    stats_path:   str | None,
    n_episodes:   int,
    exec_horizon: int,
    max_steps:    int,
    seed:         int,
    realtime:     bool,
):
    # --- Model ---
    print(f"Checkpoint betöltés: {ckpt_dir}")
    model_ac, cfg = load_policy(str(ckpt_dir))
    device = next(model_ac.parameters()).device
    chunk_size = cfg["model"]["chunk_size"]
    if exec_horizon < 0:
        exec_horizon = chunk_size
    print(f"Device: {device} | chunk_size: {chunk_size} | exec_horizon: {exec_horizon}")

    # --- Stats ---
    if stats_path is None:
        ds_path    = Path(_REPO_ROOT) / cfg["dataset"]["path"]
        stats_path = str(ds_path / "meta" / "stats.json")
    stats_path = Path(stats_path) if Path(stats_path).is_absolute() \
                 else _REPO_ROOT / stats_path
    with open(stats_path) as f:
        normalizer = StatsNormalizer(json.load(f))
    print(f"Stats: {stats_path.name} ✅")

    # --- MuJoCo env ---
    xml_path = _REPO_ROOT / cfg["eval"]["env"]["xml_path"]
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data  = mujoco.MjData(mj_model)
    rng      = np.random.default_rng(seed)

    hand_site_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    target_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
    stock_body_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")
    hand_body_ids: set = set()
    for name in _exp._HAND_BODY_NAMES:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            hand_body_ids.add(bid)

    print(f"\nViewer indul — {n_episodes} epizód | q/Esc: kilépés\n{'─'*50}")

    ep_results = []

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for ep in range(n_episodes):
            if not viewer.is_running():
                break

            # Reset
            _reset_env(mj_model, mj_data, rng)
            viewer.sync()
            time.sleep(1.0)   # reset látható szünet

            chunk_buf = []
            buf_idx   = 0
            step_count = 0
            success    = False
            place_dist = float("inf")

            for _ in range(max_steps):
                if not viewer.is_running():
                    break

                # Re-query
                if buf_idx >= len(chunk_buf):
                    obs = _get_obs(mj_model, mj_data,
                                   hand_site_id, target_site_id,
                                   stock_body_id, hand_body_ids)
                    obs_norm = normalizer.normalize_obs(obs)
                    obs_t    = torch.from_numpy(obs_norm).unsqueeze(0).to(device)
                    with torch.no_grad():
                        actions_pred, _ = model_ac(obs_t, actions=None)
                    actions_np = actions_pred[0].cpu().numpy()   # (chunk, 5)
                    chunk_buf  = actions_np[:exec_horizon]
                    buf_idx    = 0

                # Action denorm + végrehajtás
                action_dn = normalizer.denormalize_action(chunk_buf[buf_idx])
                buf_idx += 1
                _apply_action(mj_model, mj_data, action_dn)
                step_count += 1

                # Állapot kiolvasás
                stock_pos  = mj_data.xpos[stock_body_id].copy()
                target_pos = mj_data.site_xpos[target_site_id].copy()
                place_dist = float(np.linalg.norm(stock_pos - target_pos))
                success    = (place_dist < GOAL_RADIUS) and (step_count >= MIN_SUCCESS_STEP)

                viewer.sync()

                if realtime:
                    time.sleep(mj_model.opt.timestep * DECIMATION)

                if success or step_count >= max_steps:
                    break

            status = "✅ SIKER" if success else "❌ FAIL "
            print(f"Ep {ep+1:2d}/{n_episodes} | {status} | "
                  f"dist={place_dist:.3f}m | lépés={step_count}")
            ep_results.append(success)

            if viewer.is_running():
                time.sleep(1.5)   # eredmény látható

    # --- Összesítő ---
    n_ok = sum(ep_results)
    sr   = 100.0 * n_ok / max(1, len(ep_results))
    print(f"\n{'═'*50}")
    print(f"VIZ ÖSSZESÍTŐ: {n_ok}/{len(ep_results)} siker → SR = {sr:.1f}%")
    print(f"{'═'*50}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACT policy vizualizáció — ⚠️  mjpython kell macOS-en!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Futtatás (repo gyökeréből):
  mjpython tools/viz_act_policy.py \\
      --ckpt    results/bc_checkpoints_act_v2 \\
      --stats   data/lerobot/scripted_v1/meta/stats.json \\
      --n-ep    5 \\
      --exec-horizon 5
        """,
    )
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--stats",        default=None)
    parser.add_argument("--n-ep",         type=int, default=5,
                        help="Epizódok száma (alapért.: 5)")
    parser.add_argument("--exec-horizon", type=int, default=5,
                        help="Lépések re-query előtt (alapért.: 5; -1 → chunk_size)")
    parser.add_argument("--max-steps",    type=int, default=300)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--realtime",     action="store_true",
                        help="Realtime sebesség (timestep szünet minden sim lépés után)")
    args = parser.parse_args()

    run_viz(
        ckpt_dir     = args.ckpt,
        stats_path   = args.stats,
        n_episodes   = args.n_ep,
        exec_horizon = args.exec_horizon,
        max_steps    = args.max_steps,
        seed         = args.seed,
        realtime     = args.realtime,
    )


if __name__ == "__main__":
    main()
