#!/usr/bin/env python3
"""
Fázis C shelf stocking policy kiértékelő.

Betölt egy betanított manip policy-t és futtatja a G1ShelfStockEnv-ben.
Méri a sikerességi arányt és a fázis-szintű teljesítményt.

Használat (repo gyökeréből):
    python src/roboshelf_ai/tasks/manipulation/eval_shelf_stock.py \\
        --model roboshelf-results/manip/shelf_stock_v1/best_model.zip \\
        --vec-normalize roboshelf-results/manip/shelf_stock_v1/vec_normalize.pkl \\
        --config configs/manipulation/shelf_stock_v1.yaml \\
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
from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv
from roboshelf_ai.core.callbacks import make_vec_normalize

PHASE_NAMES = ["REACH", "GRASP", "LIFT", "PLACE"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(
    model_path: str,
    vec_norm_path: str,
    cfg: dict,
    n_episodes: int,
    render: bool,
    deterministic: bool,
) -> None:
    norm_cfg   = cfg.get("vec_normalize", {})
    accept_cfg = cfg.get("acceptance", {})

    print("\n" + "=" * 62)
    print("  Roboshelf AI — Fázis C Shelf Stocking Eval")
    print("=" * 62)
    print(f"  Modell:    {model_path}")
    print(f"  Epizódok:  {n_episodes}")
    print(f"  Render:    {render}")
    print("=" * 62 + "\n")

    def _init():
        return G1ShelfStockEnv(
            cfg=cfg,
            render_mode="human" if render else None,
        )

    env = DummyVecEnv([_init])
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
                gamma=cfg.get("ppo", {}).get("gamma", 0.99),
            )
            print(f"  VecNormalize betöltve: {vec_norm_path}\n")
        else:
            print(f"  ⚠️  VecNormalize nem található: {vec_norm_path}\n")

    model = PPO.load(model_path, device="cpu")
    print(f"  ✅ Modell betöltve\n")

    results = []
    obs = env.reset()

    for ep in range(n_episodes):
        ep_reward, ep_steps, done, info_last = 0.0, 0, False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, info_arr = env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1
            info_last  = info_arr[0]
            done = bool(done_arr[0])

        placed     = info_last.get("placed", False)
        phase      = info_last.get("phase", 0)
        dist_final = info_last.get("stock_target_dist", -1.0)
        stock_rise = info_last.get("stock_rise", 0.0)

        results.append(dict(
            placed=placed, phase=phase, steps=ep_steps,
            reward=ep_reward, dist_final=dist_final, stock_rise=stock_rise,
        ))

        phase_str = PHASE_NAMES[min(phase, 3)]
        status = "✅ ELHELYEZVE" if placed else f"⏱ {phase_str:5s}"
        print(
            f"  Ep {ep+1:3d}: {status:16s} | "
            f"lépés={ep_steps:4d} | "
            f"reward={ep_reward:8.1f} | "
            f"dist={dist_final:.3f}m | "
            f"emelés={stock_rise:.3f}m"
        )
        obs = env.reset()

    env.close()

    # Összesítő
    n_placed  = sum(1 for r in results if r["placed"])
    n_reached = sum(1 for r in results if r["phase"] >= 1)
    n_grasped = sum(1 for r in results if r["phase"] >= 2)
    n_lifted  = sum(1 for r in results if r["phase"] >= 3)
    sr = n_placed / n_episodes

    print(f"\n{'-'*62}")
    print(f"  ÖSSZESÍTŐ")
    print(f"{'-'*62}")
    print(f"  Elhelyezve (PLACE ✅): {n_placed}/{n_episodes}  ({100*sr:.0f}%)")
    print(f"  Reach fázis elért:    {n_reached}/{n_episodes} ({100*n_reached/n_episodes:.0f}%)")
    print(f"  Grasp fázis elért:    {n_grasped}/{n_episodes} ({100*n_grasped/n_episodes:.0f}%)")
    print(f"  Lift  fázis elért:    {n_lifted}/{n_episodes}  ({100*n_lifted/n_episodes:.0f}%)")
    print(f"  Átlag lépés:          {np.mean([r['steps'] for r in results]):.0f}")
    print(f"  Átlag reward:         {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Átlag záró dist:      {np.mean([r['dist_final'] for r in results]):.3f}m")
    print(f"  Átlag emelés:         {np.mean([r['stock_rise'] for r in results]):.3f}m")

    min_sr = accept_cfg.get("min_success_rate", 0.7)
    print()
    if sr >= min_sr:
        print(f"  ✅ ELFOGADVA: sikerességi arány {100*sr:.0f}% >= {100*min_sr:.0f}%")
    else:
        print(f"  ❌ NEM ELFOGADVA: {100*sr:.0f}% < {100*min_sr:.0f}%")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Fázis C shelf stocking evaluátor")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--vec-normalize", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    evaluate(
        model_path=args.model,
        vec_norm_path=args.vec_normalize,
        cfg=cfg,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )
