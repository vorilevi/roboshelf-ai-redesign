#!/usr/bin/env python3
"""
Roboshelf AI — Policy replay vizualizáció

A betanított v20 (vagy bármely) policy futtatása MuJoCo interaktív viewerben.

Használat:
  # Legutóbbi modell automatikus betöltése:
  python replay_policy.py

  # Konkrét modell:
  python replay_policy.py --model roboshelf-results/phase2/models/g1_retail_nav_m2_20m_v20_1776336729_final.zip

  # Lassabb lejátszás (alapértelmezett: realtime):
  python replay_policy.py --slowdown 3.0

  # Csak N epizód:
  python replay_policy.py --episodes 3
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# --- Import útvonal fix ---
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "src" / "envs"))
sys.path.insert(0, str(_THIS_DIR / "src" / "training"))

def find_latest_model(results_dir: Path):
    """Megkeresi a legfrissebb _final.zip modellt."""
    models = sorted(results_dir.rglob("*_final.zip"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise FileNotFoundError(f"Nincs _final.zip modell: {results_dir}")
    return models[-1]


def main():
    parser = argparse.ArgumentParser(description="Roboshelf policy replay")
    parser.add_argument("--model",    type=str, default=None,
                        help="Modell .zip útvonala (alapértelmezett: legfrissebb)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Lejátszandó epizódok száma (0 = végtelen)")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="Lassítás szorzó (1.0=realtime, 2.0=kétszer lassabb)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Determinisztikus policy (alapértelmezett: igen)")
    args = parser.parse_args()

    # --- Modell megkeresése ---
    results_dir = _THIS_DIR / "roboshelf-results" / "phase2" / "models"
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model(results_dir)

    vecnorm_path = Path(str(model_path).replace("_final.zip", "_final_vecnormalize.pkl"))
    if not vecnorm_path.exists():
        # Fallback: best_model VecNormalize
        vecnorm_path = results_dir.parent.parent / "best_model" / "best_model_vecnormalize.pkl"

    print(f"\n🤖 Roboshelf Policy Replay")
    print(f"=" * 50)
    print(f"  Modell:    {model_path.name}")
    print(f"  VecNorm:   {vecnorm_path.name}")
    print(f"  Epizódok:  {'végtelen' if args.episodes == 0 else args.episodes}")
    print(f"  Lassítás:  {args.slowdown}×")
    print(f"=" * 50)

    # --- SB3 + env betöltése ---
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv
    import mujoco
    import mujoco.viewer

    # Policy betöltése
    model = PPO.load(str(model_path), device="cpu")
    print(f"  ✅ Policy betöltve ({sum(p.numel() for p in model.policy.parameters()):,} paraméter)")

    # Env létrehozása (NEM VecEnv, közvetlen MuJoCo hozzáféréshez)
    env = RoboshelfRetailNavEnv()
    env.penalty_scale = 1.0  # eval: teljes büntetés

    # VecNormalize statisztikák betöltése (obs normalizáláshoz)
    vec_env = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  ✅ VecNormalize statisztikák betöltve")
    else:
        print(f"  ⚠️  VecNormalize nem található — obs normalizálás nélkül fut")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Lépési idő (50 Hz, 2 substep → 0.02s/lépés)
    dt = env.model.opt.timestep * 2  # ~0.02s
    sleep_per_step = dt * args.slowdown

    # --- MuJoCo viewer indítása ---
    print(f"\n  🎬 Viewer indítása... (kilépés: Esc vagy ablak bezárása)\n")

    ep_num = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Kamera beállítása: oldalnézet a bolt hosszában
        viewer.cam.lookat[:] = [0.0, 2.0, 1.0]
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -15.0

        # Végtelen ciklus — Esc vagy ablakbezárás állítja meg
        # Ha --episodes N van megadva: N epizód után 3mp szünet, majd újraindul
        while viewer.is_running():
            ep_num += 1
            batch_ep = ((ep_num - 1) % max(args.episodes, 1)) + 1  # ep száma a batch-en belül

            # Reset
            obs_raw, info = env.reset()
            obs_norm = vec_env.normalize_obs(obs_raw.reshape(1, -1))

            total_reward = 0.0
            steps = 0
            ep_start = time.time()

            print(f"  Ep {ep_num} (batch {batch_ep}/{max(args.episodes,1)}): "
                  f"indul... (táv: {info['dist_to_target']:.2f}m)", end="", flush=True)

            while viewer.is_running():
                step_start = time.time()

                # Policy döntés
                action, _ = model.predict(obs_norm, deterministic=args.deterministic)

                # Lépés
                obs_raw, reward, terminated, truncated, info = env.step(action[0])
                obs_norm = vec_env.normalize_obs(obs_raw.reshape(1, -1))

                total_reward += reward
                steps += 1

                # Viewer frissítése
                viewer.sync()

                # Realtime timing
                elapsed = time.time() - step_start
                remaining = sleep_per_step - elapsed
                if remaining > 0:
                    time.sleep(remaining)

                if terminated or truncated:
                    break

            ep_time = time.time() - ep_start
            dist = info.get("dist_to_target", 0)
            improvement = 3.3 - dist
            print(f" → reward={total_reward:.1f}, lépés={steps}, "
                  f"táv={dist:.2f}m ({'+' if improvement>0 else ''}{improvement:.2f}m), "
                  f"idő={ep_time:.1f}s")

            # Ha batch tele: 3mp szünet a viewer frissítésével, majd újraindul
            if args.episodes > 0 and batch_ep >= args.episodes:
                print(f"  --- {args.episodes} epizód kész, 3mp múlva újraindul... ---")
                pause_end = time.time() + 3.0
                while viewer.is_running() and time.time() < pause_end:
                    viewer.sync()
                    time.sleep(0.02)
            else:
                # Rövid szünet epizódok között
                pause_end = time.time() + 1.0
                while viewer.is_running() and time.time() < pause_end:
                    viewer.sync()
                    time.sleep(0.02)

    env.close()
    vec_env.close()
    print(f"\n  👋 Viewer bezárva.")


if __name__ == "__main__":
    main()
