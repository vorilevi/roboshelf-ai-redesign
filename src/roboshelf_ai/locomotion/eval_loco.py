#!/usr/bin/env python3
"""
Locomotion policy evaluátor — Fázis A.

Betölt egy betanított locomotion policy-t és futtatja az env-ben.
Kiírja az epizód-hosszakat, reward összegeket és command-tracking hibát.
Opcionálisan MuJoCo viewer-rel megjeleníti a mozgást.

Használat:
    # Repo gyökeréből:
    python src/roboshelf_ai/locomotion/eval_loco.py \
        --model roboshelf-results/loco/v1/best_model.zip \
        --config configs/locomotion/g1_command_v1.yaml \
        --episodes 5

    # Vizuális megjelenítés:
    python src/roboshelf_ai/locomotion/eval_loco.py \
        --model roboshelf-results/loco/v1/best_model.zip \
        --config configs/locomotion/g1_command_v1.yaml \
        --episodes 3 --render
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import yaml

# Fájl helye: src/roboshelf_ai/locomotion/eval_loco.py
# parents[0] = locomotion/, parents[1] = roboshelf_ai/, parents[2] = src/
_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

from roboshelf_ai.mujoco.envs.locomotion.g1_locomotion_command_env import (
    G1LocomotionCommandEnv
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, render: bool = False) -> G1LocomotionCommandEnv:
    env_cfg = cfg.get("env", {})
    rew_cfg = cfg.get("reward", {})
    return G1LocomotionCommandEnv(
        render_mode="human" if render else None,
        max_episode_steps=env_cfg.get("max_episode_steps", 1000),
        sub_steps=env_cfg.get("sub_steps", 2),
        action_scale=env_cfg.get("action_scale", 0.3),
        noise_scale=env_cfg.get("noise_scale", 0.01),
        v_forward_range=tuple(env_cfg.get("v_forward_range", [0.0, 1.5])),
        yaw_rate_range=tuple(env_cfg.get("yaw_rate_range", [-1.0, 1.0])),
        w_forward=rew_cfg.get("w_forward", 2.0),
        w_yaw=rew_cfg.get("w_yaw", 0.5),
        w_upright=rew_cfg.get("w_upright", 1.5),
        w_alive=rew_cfg.get("w_alive", 0.5),
        w_smooth=rew_cfg.get("w_smooth", -0.1),
        w_energy=rew_cfg.get("w_energy", -0.05),
        w_feet_slip=rew_cfg.get("w_feet_slip", -0.1),
        w_feet_dist=rew_cfg.get("w_feet_dist", -1.0),
        feet_min_dist=rew_cfg.get("feet_min_dist", 0.15),
    )


def run_episode(env, model, deterministic: bool = True, render: bool = False):
    """Lefuttat egy epizódot, visszaadja a statisztikákat."""
    obs, info = env.reset()
    command = info["command"]

    total_reward = 0.0
    steps = 0
    fwd_errors = []
    yaw_errors = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Command-tracking hiba
        fwd_errors.append(abs(info["lin_vel_x"] - info["cmd_v_fwd"]))
        yaw_errors.append(abs(info["ang_vel_z"] - info["cmd_yaw"]))

        if render:
            env.render()

        if terminated or truncated:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "mean_fwd_error": float(np.mean(fwd_errors)),
        "mean_yaw_error": float(np.mean(yaw_errors)),
        "final_torso_z": info["torso_z"],
        "command_v_fwd": float(command[0]),
        "command_yaw": float(command[2]),
    }


def evaluate(model_path: str, cfg: dict, n_episodes: int, render: bool,
             deterministic: bool) -> None:
    from stable_baselines3 import PPO

    print("\n" + "=" * 60)
    print("  Roboshelf AI — Locomotion Policy Evaluáció")
    print("=" * 60)
    print(f"  Modell:   {model_path}")
    print(f"  Epizódok: {n_episodes}")
    print(f"  Render:   {render}")
    print("=" * 60 + "\n")

    env = make_env(cfg, render=render)

    print(f"  Modell betöltése: {model_path}")
    model = PPO.load(model_path, device="cpu")
    print("  ✅ Betöltve\n")

    results = []
    for ep in range(n_episodes):
        stats = run_episode(env, model, deterministic=deterministic, render=render)
        results.append(stats)

        status = "ELESETT" if stats["terminated"] else "OK"
        print(
            f"  Ep {ep+1:2d}: lépés={stats['steps']:4d}  "
            f"reward={stats['total_reward']:7.1f}  "
            f"fwd_err={stats['mean_fwd_error']:.3f} m/s  "
            f"yaw_err={stats['mean_yaw_error']:.3f} rad/s  "
            f"[{status}]  "
            f"cmd=(v={stats['command_v_fwd']:.2f}, yaw={stats['command_yaw']:.2f})"
        )

    env.close()

    # Összesítő
    steps_arr  = [r["steps"] for r in results]
    reward_arr = [r["total_reward"] for r in results]
    fwd_err    = [r["mean_fwd_error"] for r in results]
    yaw_err    = [r["mean_yaw_error"] for r in results]
    survived   = sum(1 for r in results if not r["terminated"])

    print("\n" + "-" * 60)
    print("  ÖSSZESÍTŐ")
    print("-" * 60)
    print(f"  Túlélési arány:          {survived}/{n_episodes}")
    print(f"  Átlag epizód hossz:      {np.mean(steps_arr):.0f} ± {np.std(steps_arr):.0f} lépés")
    print(f"  Átlag reward:            {np.mean(reward_arr):.1f} ± {np.std(reward_arr):.1f}")
    print(f"  Átlag fwd tracking hiba: {np.mean(fwd_err):.4f} m/s")
    print(f"  Átlag yaw tracking hiba: {np.mean(yaw_err):.4f} rad/s")

    # Elfogadási feltétel ellenőrzés
    accept_cfg = cfg.get("acceptance", {})
    min_ep_len = accept_cfg.get("min_episode_length", 300)
    print()
    if np.mean(steps_arr) >= min_ep_len and survived == n_episodes:
        print(f"  ✅ ELFOGADVA: átlag epizód hossz {np.mean(steps_arr):.0f} >= {min_ep_len}")
    else:
        print(f"  ❌ NEM ELFOGADVA: átlag epizód hossz {np.mean(steps_arr):.0f} (cél: {min_ep_len}+)")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Locomotion policy evaluátor")
    parser.add_argument("--model", required=True,
                        help="Betanított modell .zip útvonala")
    parser.add_argument("--config", required=True,
                        help="Config YAML útvonala")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Futtatandó epizódok száma (default: 5)")
    parser.add_argument("--render", action="store_true",
                        help="MuJoCo viewer megjelenítés")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sztochasztikus akció mintavételezés (default: determinisztikus)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(
        model_path=args.model,
        cfg=cfg,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )
