#!/usr/bin/env python3
"""
Fázis B hierarchikus navigáció tanítás — RetailNavHierEnv + PPO.

Használat (repo gyökeréből, miniforge env):
    # Sanity run (1000 lépés, nincs mentés):
    python src/roboshelf_ai/tasks/navigation/train_nav_hierarchical.py \
        --config configs/navigation/retail_nav_hier_v1.yaml \
        --total-timesteps 1000 --no-save

    # Teljes tanítás:
    python src/roboshelf_ai/tasks/navigation/train_nav_hierarchical.py \
        --config configs/navigation/retail_nav_hier_v1.yaml

    # Curriculum szint 2-től:
    python src/roboshelf_ai/tasks/navigation/train_nav_hierarchical.py \
        --config configs/navigation/retail_nav_hier_v1.yaml \
        --curriculum-level 2

VecNormalize:
    Mentve: <save_path>/vec_normalize.pkl
    Fine-tune-hoz: add hozzá a config load_path értékét.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

_src = str(Path(__file__).resolve().parents[3])
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)

from roboshelf_ai.mujoco.envs.navigation.retail_nav_hier_env import RetailNavHierEnv
from roboshelf_ai.core.callbacks import (
    EpisodeStatsCallback,
    make_vec_normalize,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    if args.total_timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg["ppo"]["n_envs"] = args.n_envs
    if args.curriculum_level is not None:
        cfg["_curriculum_level"] = args.curriculum_level
    if args.no_save:
        cfg["_no_save"] = True
    return cfg


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(cfg: dict, rank: int = 0, seed: int = 0, eval_env: bool = False):
    def _init():
        curriculum_level = cfg.get("_curriculum_level", 1)
        env = RetailNavHierEnv(cfg=cfg, curriculum_level=curriculum_level)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_lr_schedule(lr_init: float, schedule: str):
    """Lineáris vagy konstans LR schedule."""
    if schedule == "linear":
        return lambda progress: lr_init * progress  # progress: 1→0
    return lr_init


# ---------------------------------------------------------------------------
# Fő tanítás
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    ppo_cfg  = cfg.get("ppo", {})
    eval_cfg = cfg.get("eval", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg  = cfg.get("logging", {})
    norm_cfg = cfg.get("vec_normalize", {})
    no_save  = cfg.get("_no_save", False)

    n_envs      = ppo_cfg.get("n_envs", 4)
    total_steps = ppo_cfg.get("total_timesteps", 3_000_000)
    save_path   = ckpt_cfg.get("save_path", "roboshelf-results/nav/hier_v1/")
    tb_log      = log_cfg.get("tensorboard_log", "data/logs/nav_hier_v1/")
    curriculum_level = cfg.get("_curriculum_level", 1)
    use_vec_norm = norm_cfg.get("enabled", True)
    vec_norm_path = os.path.join(save_path, "vec_normalize.pkl")

    print("\n" + "=" * 62)
    print("  Roboshelf AI — Fázis B Hierarchikus Nav Tanítás")
    print("=" * 62)
    print(f"  Envek:         {n_envs} (SubprocVecEnv)")
    print(f"  Timesteps:     {total_steps:,}")
    print(f"  Curriculum:    Szint {curriculum_level}")
    print(f"  Mentési hely:  {save_path}")
    print(f"  TensorBoard:   {tb_log}")
    print(f"  VecNormalize:  {use_vec_norm}")
    print(f"  No-save mód:   {no_save}")
    print("=" * 62 + "\n")

    # --- Train env ---
    train_env = SubprocVecEnv([make_env(cfg, rank=i, seed=42) for i in range(n_envs)])
    train_env = VecMonitor(train_env)
    if use_vec_norm:
        load_norm = norm_cfg.get("load_path", None)
        train_env = make_vec_normalize(
            train_env,
            load_path=load_norm,
            norm_obs=norm_cfg.get("norm_obs", True),
            norm_reward=norm_cfg.get("norm_reward", True),
            clip_obs=norm_cfg.get("clip_obs", 10.0),
            gamma=ppo_cfg.get("gamma", 0.999),
        )

    # --- Eval env ---
    eval_env = SubprocVecEnv([make_env(cfg, rank=0, seed=99, eval_env=True)])
    eval_env = VecMonitor(eval_env)
    if use_vec_norm:
        eval_env = make_vec_normalize(
            eval_env,
            load_path=vec_norm_path if os.path.exists(vec_norm_path) else None,
            norm_obs=norm_cfg.get("norm_obs", True),
            norm_reward=False,
            clip_obs=norm_cfg.get("clip_obs", 10.0),
            gamma=ppo_cfg.get("gamma", 0.999),
        )

    # --- Callbacks ---
    callbacks = [EpisodeStatsCallback(verbose=0)]

    if not no_save:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(tb_log, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=max(eval_cfg.get("eval_freq", 25_000) // n_envs, 1),
            n_eval_episodes=eval_cfg.get("n_eval_episodes", 10),
            deterministic=eval_cfg.get("deterministic", True),
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(ckpt_cfg.get("save_freq", 50_000) // n_envs, 1),
            save_path=save_path,
            name_prefix=f"nav_hier_lvl{curriculum_level}",
            verbose=1,
        )
        callbacks += [eval_callback, checkpoint_callback]

    # --- LR schedule ---
    lr_schedule = ppo_cfg.get("lr_schedule", "constant")
    lr = make_lr_schedule(ppo_cfg.get("learning_rate", 3e-4), lr_schedule)

    # --- PPO modell ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        learning_rate=lr,
        gamma=ppo_cfg.get("gamma", 0.999),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=tb_log if not no_save else None,
        verbose=log_cfg.get("verbose", 1),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    print(f"  Policy paraméterei: {sum(p.numel() for p in model.policy.parameters()):,}\n")

    # --- Tanítás ---
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList(callbacks) if callbacks else None,
        progress_bar=True,
    )

    # --- Mentés ---
    if not no_save:
        final_path = os.path.join(save_path, f"final_model_lvl{curriculum_level}")
        model.save(final_path)
        print(f"\n✅ Tanítás kész! Végső modell: {final_path}.zip")
        print(f"   Legjobb modell: {os.path.join(save_path, 'best_model.zip')}")
        if use_vec_norm and hasattr(train_env, "save"):
            train_env.save(vec_norm_path)
            print(f"   VecNormalize:   {vec_norm_path}")
    else:
        print("\n✅ Sanity run kész! (--no-save)")

    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fázis B hierarchikus nav tanítás")
    p.add_argument("--config", required=True, help="Config YAML útvonala")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--curriculum-level", type=int, default=None,
                   help="Curriculum szint (1-4, default: config alapján)")
    p.add_argument("--no-save", action="store_true",
                   help="Sanity run — nincs mentés, nincs TensorBoard")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = merge_args(cfg, args)
    train(cfg)
