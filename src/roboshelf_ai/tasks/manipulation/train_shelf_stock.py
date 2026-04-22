#!/usr/bin/env python3
"""
Fázis C manipulációs tanítás — G1ShelfStockEnv + PPO.

Használat (repo gyökeréből, base conda env):
    # Sanity run (1000 lépés, nincs mentés):
    python src/roboshelf_ai/tasks/manipulation/train_shelf_stock.py \\
        --config configs/manipulation/shelf_stock_v1.yaml \\
        --total-timesteps 1000 --no-save

    # Teljes tanítás (5M lépés):
    python src/roboshelf_ai/tasks/manipulation/train_shelf_stock.py \\
        --config configs/manipulation/shelf_stock_v1.yaml

VecNormalize mentve: <save_path>/vec_normalize.pkl
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

from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv
from roboshelf_ai.core.callbacks import EpisodeStatsCallback, make_vec_normalize


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    if args.total_timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg["ppo"]["n_envs"] = args.n_envs
    if args.no_save:
        cfg["_no_save"] = True
    return cfg


def make_env(cfg: dict, rank: int = 0, seed: int = 0):
    def _init():
        env = G1ShelfStockEnv(cfg=cfg)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_lr_schedule(lr_init: float, schedule: str):
    if schedule == "linear":
        return lambda progress: lr_init * progress
    return lr_init


def train(cfg: dict) -> None:
    ppo_cfg  = cfg.get("ppo", {})
    eval_cfg = cfg.get("eval", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg  = cfg.get("logging", {})
    norm_cfg = cfg.get("vec_normalize", {})
    no_save  = cfg.get("_no_save", False)

    n_envs      = ppo_cfg.get("n_envs", 8)
    total_steps = ppo_cfg.get("total_timesteps", 5_000_000)
    save_path   = ckpt_cfg.get("save_path", "roboshelf-results/manip/shelf_stock_v1/")
    tb_log      = log_cfg.get("tensorboard_log", "data/logs/manip_shelf_stock_v1/")
    use_vec_norm = norm_cfg.get("enabled", True)
    vec_norm_path = os.path.join(save_path, "vec_normalize.pkl")

    net_arch_cfg = ppo_cfg.get("net_arch", {})
    net_arch = dict(
        pi=net_arch_cfg.get("pi", [256, 256, 128]),
        vf=net_arch_cfg.get("vf", [256, 256, 128]),
    )

    print("\n" + "=" * 62)
    print("  Roboshelf AI — Fázis C Shelf Stocking Tanítás")
    print("=" * 62)
    print(f"  Envek:         {n_envs} (SubprocVecEnv)")
    print(f"  Timesteps:     {total_steps:,}")
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
            gamma=ppo_cfg.get("gamma", 0.99),
        )

    # --- Eval env ---
    eval_env = SubprocVecEnv([make_env(cfg, rank=0, seed=99)])
    eval_env = VecMonitor(eval_env)
    if use_vec_norm:
        eval_env = make_vec_normalize(
            eval_env,
            load_path=vec_norm_path if (os.path.exists(vec_norm_path) and os.path.getsize(vec_norm_path) > 0) else None,
            norm_obs=norm_cfg.get("norm_obs", True),
            norm_reward=False,
            clip_obs=norm_cfg.get("clip_obs", 10.0),
            gamma=ppo_cfg.get("gamma", 0.99),
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
            eval_freq=max(eval_cfg.get("eval_freq", 50_000) // n_envs, 1),
            n_eval_episodes=eval_cfg.get("n_eval_episodes", 10),
            deterministic=eval_cfg.get("deterministic", True),
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(ckpt_cfg.get("save_freq", 100_000) // n_envs, 1),
            save_path=save_path,
            name_prefix="manip_shelf_stock",
            verbose=1,
        )
        callbacks += [eval_callback, checkpoint_callback]

    # --- LR schedule ---
    lr = make_lr_schedule(ppo_cfg.get("learning_rate", 3e-4), ppo_cfg.get("lr_schedule", "constant"))

    # --- PPO modell ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 512),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        learning_rate=lr,
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=tb_log if not no_save else None,
        verbose=log_cfg.get("verbose", 1),
        policy_kwargs=dict(net_arch=net_arch),
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
        final_path = os.path.join(save_path, "final_model")
        model.save(final_path)
        print(f"\n✅ Tanítás kész! Végső modell: {final_path}.zip")
        print(f"   Legjobb modell: {os.path.join(save_path, 'best_model.zip')}")
        if use_vec_norm and hasattr(train_env, "save"):
            train_env.save(vec_norm_path)
            print(f"   VecNormalize:   {vec_norm_path}")
        _run_final_eval(cfg, save_path, vec_norm_path, norm_cfg)
    else:
        print("\n✅ Sanity run kész! (--no-save)")

    train_env.close()
    eval_env.close()


def _run_final_eval(cfg, save_path, vec_norm_path, norm_cfg):
    """Tanítás utáni automatikus kiértékelés."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    eval_cfg   = cfg.get("eval", {})
    accept_cfg = cfg.get("acceptance", {})
    n_episodes = eval_cfg.get("n_eval_episodes", 10)
    best_path  = os.path.join(save_path, "best_model.zip")

    print("\n" + "=" * 62)
    print("  Automatikus kiértékelés — tanítás utáni eval")
    print("=" * 62)

    def _init():
        return G1ShelfStockEnv(cfg=cfg)

    env = DummyVecEnv([_init])
    env = VecMonitor(env)
    if norm_cfg.get("enabled", True) and os.path.exists(vec_norm_path) and os.path.getsize(vec_norm_path) > 0:
        env = make_vec_normalize(
            env,
            load_path=vec_norm_path,
            norm_obs=norm_cfg.get("norm_obs", True),
            norm_reward=False,
            clip_obs=norm_cfg.get("clip_obs", 10.0),
            gamma=cfg.get("ppo", {}).get("gamma", 0.99),
        )

    model = PPO.load(best_path, device="cpu")
    results = []
    obs = env.reset()

    for ep in range(n_episodes):
        ep_reward, ep_steps, done, info_last = 0.0, 0, False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1
            info_last  = info_arr[0]
            done = bool(done_arr[0])

        placed     = info_last.get("placed", False)
        phase      = info_last.get("phase", 0)
        dist_final = info_last.get("stock_target_dist", -1.0)
        results.append(dict(placed=placed, phase=phase, steps=ep_steps,
                            reward=ep_reward, dist_final=dist_final))

        phase_name = ["REACH", "GRASP", "LIFT", "PLACE"][min(phase, 3)]
        status = "✅ ELHELYEZVE" if placed else f"⏱ {phase_name}"
        print(f"  Ep {ep+1:3d}: {status:14s} | lépés={ep_steps:4d} | "
              f"reward={ep_reward:8.1f} | dist={dist_final:.3f}m")
        obs = env.reset()

    env.close()

    n_placed = sum(1 for r in results if r["placed"])
    sr = n_placed / n_episodes
    print(f"\n{'-'*62}")
    print(f"  ÖSSZESÍTŐ")
    print(f"{'-'*62}")
    print(f"  Elhelyezve:   {n_placed}/{n_episodes} ({100*sr:.0f}%)")
    print(f"  Átlag lépés:  {np.mean([r['steps'] for r in results]):.0f}")
    print(f"  Átlag reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Átlag dist:   {np.mean([r['dist_final'] for r in results]):.3f}m")

    min_sr = accept_cfg.get("min_success_rate", 0.7)
    print()
    if sr >= min_sr:
        print(f"  ✅ ELFOGADVA: {100*sr:.0f}% >= {100*min_sr:.0f}%")
    else:
        print(f"  ❌ NEM ELFOGADVA: {100*sr:.0f}% < {100*min_sr:.0f}%")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Fázis C shelf stocking tanítás")
    p.add_argument("--config", required=True)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = merge_args(cfg, args)
    train(cfg)
