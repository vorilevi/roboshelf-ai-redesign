#!/usr/bin/env python3
"""
Fázis B navigációs policy evaluátor.

Betölt egy betanított nav policy-t és futtatja a RetailNavHierEnv-ben.
Méri a célpont-elérési arányt, epizód hosszt, locomotion összeomlást.

Használat (repo gyökeréből):
    python src/roboshelf_ai/tasks/navigation/eval_nav.py \
        --model roboshelf-results/nav/hier_v1/best_model.zip \
        --vec-normalize roboshelf-results/nav/hier_v1/vec_normalize.pkl \
        --config configs/navigation/retail_nav_hier_v1.yaml \
        --episodes 20

    # Vizuális megjelenítés:
    ... --episodes 5 --render
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_src = str(Path(__file__).resolve().parents[3])
if _src not in sys.path:
    sys.path.insert(0, _src)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from roboshelf_ai.mujoco.envs.navigation.retail_nav_hier_env import RetailNavHierEnv
from roboshelf_ai.core.callbacks import make_vec_normalize


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_eval_env(cfg: dict, curriculum_level: int, render: bool):
    def _init():
        env = RetailNavHierEnv(
            cfg=cfg,
            curriculum_level=curriculum_level,
            render_mode="human" if render else None,
        )
        return env
    return _init


def evaluate(
    model_path: str,
    vec_norm_path: str,
    cfg: dict,
    n_episodes: int,
    curriculum_level: int,
    render: bool,
    deterministic: bool,
    seed: int,
) -> None:
    norm_cfg = cfg.get("vec_normalize", {})

    print("\n" + "=" * 62)
    print("  Roboshelf AI — Fázis B Nav Policy Evaluáció")
    print("=" * 62)
    print(f"  Modell:      {model_path}")
    print(f"  Epizódok:    {n_episodes}")
    print(f"  Curriculum:  Szint {curriculum_level}")
    print(f"  Render:      {render}")
    print("=" * 62 + "\n")

    # Env
    env = DummyVecEnv([make_eval_env(cfg, curriculum_level, render)])
    env = VecMonitor(env)
    if norm_cfg.get("enabled", True) and vec_norm_path:
        from pathlib import Path as P
        if P(vec_norm_path).exists():
            env = make_vec_normalize(
                env,
                load_path=vec_norm_path,
                norm_obs=norm_cfg.get("norm_obs", True),
                norm_reward=False,
                clip_obs=norm_cfg.get("clip_obs", 10.0),
                gamma=cfg.get("ppo", {}).get("gamma", 0.999),
            )
            print(f"  VecNormalize betöltve: {vec_norm_path}\n")
        else:
            print(f"  ⚠️  VecNormalize nem található: {vec_norm_path}\n")

    # Modell
    model = PPO.load(model_path, device="cpu")
    print(f"  ✅ Modell betöltve\n")

    results = []
    obs = env.reset()

    for ep in range(n_episodes):
        ep_reward = 0.0
        ep_steps  = 0
        done = False
        info_last = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, info_arr = env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1
            info_last  = info_arr[0]
            done = bool(done_arr[0])

        goal_reached   = info_last.get("goal_reached", False)
        loco_collapsed = info_last.get("loco_collapsed", False)
        dist_final     = info_last.get("dist_to_goal", -1.0)

        results.append({
            "ep":            ep + 1,
            "steps":         ep_steps,
            "reward":        ep_reward,
            "goal_reached":  goal_reached,
            "loco_collapsed": loco_collapsed,
            "dist_final":    dist_final,
        })

        status = "✅ CÉLBA" if goal_reached else ("❌ ELESETT" if loco_collapsed else "⏱ TIMEOUT")
        print(
            f"  Ep {ep+1:3d}: {status:12s} | "
            f"lépés={ep_steps:4d} | "
            f"reward={ep_reward:8.1f} | "
            f"dist={dist_final:.2f}m"
        )

        obs = env.reset()

    env.close()

    # Összesítő
    n_goal    = sum(1 for r in results if r["goal_reached"])
    n_loco    = sum(1 for r in results if r["loco_collapsed"])
    n_timeout = sum(1 for r in results if not r["goal_reached"] and not r["loco_collapsed"])
    success_rate = n_goal / n_episodes
    mean_steps   = np.mean([r["steps"] for r in results])
    mean_reward  = np.mean([r["reward"] for r in results])
    mean_dist    = np.mean([r["dist_final"] for r in results])

    print("\n" + "-" * 62)
    print("  ÖSSZESÍTŐ")
    print("-" * 62)
    print(f"  Célba ért:        {n_goal}/{n_episodes} ({100*success_rate:.0f}%)")
    print(f"  Loco összeomlás:  {n_loco}/{n_episodes} ({100*n_loco/n_episodes:.0f}%)")
    print(f"  Timeout:          {n_timeout}/{n_episodes}")
    print(f"  Átlag lépés:      {mean_steps:.0f}")
    print(f"  Átlag reward:     {mean_reward:.1f}")
    print(f"  Átlag záró dist:  {mean_dist:.2f}m")
    print()

    # Elfogadási feltétel
    accept_cfg = cfg.get("acceptance", {})
    min_success = accept_cfg.get("min_success_rate", 0.5)
    max_loco    = accept_cfg.get("loco_collapse_max", 0.1)

    ok_success = success_rate >= min_success
    ok_loco    = (n_loco / n_episodes) <= max_loco

    if ok_success and ok_loco:
        print(f"  ✅ ELFOGADVA: sikerességi arány {100*success_rate:.0f}% >= {100*min_success:.0f}%")
        print(f"              loco összeomlás {100*n_loco/n_episodes:.0f}% <= {100*max_loco:.0f}%")
    else:
        reasons = []
        if not ok_success:
            reasons.append(f"siker {100*success_rate:.0f}% < {100*min_success:.0f}%")
        if not ok_loco:
            reasons.append(f"loco összeomlás {100*n_loco/n_episodes:.0f}% > {100*max_loco:.0f}%")
        print(f"  ❌ NEM ELFOGADVA: {', '.join(reasons)}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Nav policy evaluátor")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--vec-normalize", default=None,
                   help="VecNormalize .pkl útvonala")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--curriculum-level", type=int, default=1)
    p.add_argument("--render", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    evaluate(
        model_path=args.model,
        vec_norm_path=args.vec_normalize,
        cfg=cfg,
        n_episodes=args.episodes,
        curriculum_level=args.curriculum_level,
        render=args.render,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
