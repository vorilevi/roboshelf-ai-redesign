#!/usr/bin/env python3
"""
Fázis A locomotion tanítás — G1LocomotionCommandEnv + PPO.

Használat:
    # Repo gyökeréből:
    python src/roboshelf_ai/training/train_loco_v1.py --config configs/locomotion/g1_command_v1.yaml

    # Sanity run (10k lépés, nincs mentés):
    python src/roboshelf_ai/training/train_loco_v1.py --config configs/locomotion/g1_command_v1.yaml --total-timesteps 10000 --no-save
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# PYTHONPATH: src mappa hozzáadása (ha nem telepített csomagként fut)
# Fájl helye: src/roboshelf_ai/training/train_loco_v1.py
# parents[0] = training/, parents[1] = roboshelf_ai/, parents[2] = src/
_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from roboshelf_ai.mujoco.envs.locomotion.g1_locomotion_command_env import (
    G1LocomotionCommandEnv
)


# ---------------------------------------------------------------------------
# Config betöltés
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI argumentumok felülírják a config értékeit."""
    if args.total_timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg["ppo"]["n_envs"] = args.n_envs
    if args.no_save:
        cfg["_no_save"] = True
    return cfg


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(cfg: dict, rank: int = 0, seed: int = 0):
    """Env factory függvény SubprocVecEnv-hez."""
    def _init():
        env_cfg = cfg.get("env", {})
        rew_cfg = cfg.get("reward", {})
        env = G1LocomotionCommandEnv(
            max_episode_steps=env_cfg.get("max_episode_steps", 1000),
            sub_steps=env_cfg.get("sub_steps", 2),
            action_scale=env_cfg.get("action_scale", 0.3),
            noise_scale=env_cfg.get("noise_scale", 0.01),
            v_forward_range=tuple(env_cfg.get("v_forward_range", [0.0, 1.5])),
            yaw_rate_range=tuple(env_cfg.get("yaw_rate_range", [-1.0, 1.0])),
            healthy_z_min=env_cfg.get("healthy_z_min", 0.4),
            healthy_z_max=env_cfg.get("healthy_z_max", 1.5),
            upright_threshold=env_cfg.get("upright_threshold", 0.5),
            buoyancy_force_start=env_cfg.get("buoyancy_force_start", 0.0),
            buoyancy_force_end=env_cfg.get("buoyancy_force_end", 0.0),
            buoyancy_anneal_steps=env_cfg.get("buoyancy_anneal_steps", 3_000_000),
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
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Fő tanítás
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    ppo_cfg  = cfg.get("ppo", {})
    eval_cfg = cfg.get("eval", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg  = cfg.get("logging", {})
    no_save  = cfg.get("_no_save", False)

    n_envs     = ppo_cfg.get("n_envs", 8)
    total_steps = ppo_cfg.get("total_timesteps", 5_000_000)
    save_path  = ckpt_cfg.get("save_path", "roboshelf-results/loco/v1/")
    tb_log     = log_cfg.get("tensorboard_log", "data/logs/loco_v1/")

    print("\n" + "=" * 60)
    print("  Roboshelf AI — Fázis A Locomotion Tanítás")
    print("=" * 60)
    print(f"  Envek száma:   {n_envs} (SubprocVecEnv)")
    print(f"  Total lépések: {total_steps:,}")
    print(f"  Mentési hely:  {save_path}")
    print(f"  TensorBoard:   {tb_log}")
    print(f"  No-save mód:   {no_save}")
    print("=" * 60 + "\n")

    # Tanítási env-ek (SubprocVecEnv — M2 CPU magok kihasználása)
    train_env = SubprocVecEnv([make_env(cfg, rank=i, seed=42) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    # Eval env (egyetlen env, determinisztikus)
    eval_env = SubprocVecEnv([make_env(cfg, rank=0, seed=99)])
    eval_env = VecMonitor(eval_env)

    # Callback-ek
    callbacks = []

    if not no_save:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(tb_log, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=max(eval_cfg.get("eval_freq", 50_000) // n_envs, 1),
            n_eval_episodes=eval_cfg.get("n_eval_episodes", 5),
            deterministic=eval_cfg.get("deterministic", True),
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(ckpt_cfg.get("save_freq", 100_000) // n_envs, 1),
            save_path=save_path,
            name_prefix="loco_v1",
            verbose=1,
        )

        callbacks = [eval_callback, checkpoint_callback]

    # PPO modell
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=tb_log if not no_save else None,
        verbose=log_cfg.get("verbose", 1),
    )

    print(f"  Policy paraméterei: {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    # Tanítás
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList(callbacks) if callbacks else None,
        progress_bar=True,
    )

    # Mentés
    if not no_save:
        final_path = os.path.join(save_path, "final_model")
        model.save(final_path)
        print(f"\n✅ Tanítás kész! Végső modell mentve: {final_path}.zip")
        print(f"   Legjobb modell: {os.path.join(save_path, 'best_model.zip')}")
    else:
        print("\n✅ Sanity run kész! (--no-save, nincs mentés)")

    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fázis A locomotion tanítás")
    parser.add_argument(
        "--config", required=True,
        help="Config YAML fájl útvonala (pl. configs/locomotion/g1_command_v1.yaml)"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Felülírja a config total_timesteps értékét"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Felülírja a config n_envs értékét"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Sanity run mód: nincs mentés, nincs TensorBoard"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)
    train(cfg)
