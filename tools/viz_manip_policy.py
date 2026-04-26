#!/usr/bin/env mjpython
"""
Manipulációs policy vizualizáció — interaktív MuJoCo viewer.

Futtatás (repo gyökeréből):
    mjpython tools/viz_manip_policy.py
    mjpython tools/viz_manip_policy.py --config configs/manipulation/shelf_stock_v9.yaml \
        --model results/manip_checkpoints_v9/best_model.zip \
        --vec-normalize results/manip_checkpoints_v9/vec_normalize.pkl \
        --episodes 5

Alapértelmezett: v8 best_model, 5 epizód egymás után.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import mujoco
import mujoco.viewer
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv
from roboshelf_ai.core.callbacks import make_vec_normalize

PHASE_NAMES = ["REACH", "GRASP", "LIFT", "PLACE"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/manipulation/shelf_stock_v8.yaml")
    p.add_argument("--model",        default="results/manip_checkpoints_v8/best_model.zip")
    p.add_argument("--vec-normalize",default="results/manip_checkpoints_v8/vec_normalize.pkl")
    p.add_argument("--episodes",     type=int, default=5)
    p.add_argument("--slowdown",     type=float, default=1.0,
                   help="Lassítás szorzó (1.0=realtime, 2.0=kétszer lassabb)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    norm_cfg = cfg.get("vec_normalize", {})

    # --- Env ---
    env = G1ShelfStockEnv(cfg=cfg)

    # --- VecNormalize (obs normalizáláshoz) ---
    vec_env = DummyVecEnv([lambda: G1ShelfStockEnv(cfg=cfg)])
    vn_path = _ROOT / args.vec_normalize
    if vn_path.exists():
        vec_env = make_vec_normalize(
            vec_env,
            load_path=str(vn_path),
            norm_obs=norm_cfg.get("norm_obs", True),
            norm_reward=False,
            clip_obs=norm_cfg.get("clip_obs", 10.0),
            gamma=cfg.get("ppo", {}).get("gamma", 0.99),
        )
        print(f"✅ VecNormalize betöltve: {vn_path.name}")
    else:
        print(f"⚠️  VecNormalize nem található, obs normalizálás nélkül fut")

    # --- Policy ---
    model_path = _ROOT / args.model
    model = PPO.load(str(model_path), device="cpu")
    print(f"✅ Policy betöltve: {model_path.name}")
    print(f"   Epizódok: {args.episodes} | Lassítás: {args.slowdown}×\n")

    dt = env._model.opt.timestep * 50  # 20 Hz policy → 50 sim lépés/policy lépés
    sleep_per_step = dt * args.slowdown

    # --- Viewer ---
    with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
        # Kamera: közelről, kissé felülről, oldalról
        viewer.cam.lookat[:] = [0.35, 0.0, 0.92]
        viewer.cam.distance  = 1.8
        viewer.cam.azimuth   = 150.0
        viewer.cam.elevation = -18.0

        for ep in range(args.episodes):
            if not viewer.is_running():
                break

            obs, _ = env.reset()
            obs_n  = vec_env.normalize_obs(obs.reshape(1, -1)) \
                     if hasattr(vec_env, "normalize_obs") else obs.reshape(1, -1)

            ep_reward = 0.0
            print(f"Ep {ep+1}/{args.episodes} indul...", end="", flush=True)

            for step in range(cfg.get("env", {}).get("max_episode_steps", 500)):
                if not viewer.is_running():
                    break

                action, _ = model.predict(obs_n, deterministic=True)
                obs, reward, term, trunc, info = env.step(action[0])
                obs_n = vec_env.normalize_obs(obs.reshape(1, -1)) \
                        if hasattr(vec_env, "normalize_obs") else obs.reshape(1, -1)

                ep_reward += reward
                viewer.sync()

                elapsed_start = time.time()
                remaining = sleep_per_step - (time.time() - elapsed_start)
                if remaining > 0:
                    time.sleep(remaining)

                if term or trunc:
                    break

            phase = info.get("phase", 0)
            dist  = info.get("stock_target_dist", -1.0)
            rise  = info.get("stock_rise", 0.0)
            contact = info.get("contact_flag", 0.0)
            print(f" → {PHASE_NAMES[min(phase,3)]:5s} | "
                  f"dist={dist:.3f}m | emelés={rise:.3f}m | "
                  f"contact={contact:.0f} | reward={ep_reward:.1f}")

            # Szünet epizódok között
            pause_end = time.time() + 1.5
            while viewer.is_running() and time.time() < pause_end:
                viewer.sync()
                time.sleep(0.02)

        print("\n👋 Kész.")
    env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
