#!/usr/bin/env python3
"""
Scripted expert demonstráció gyűjtő — BC csatornához.

A scripted expert egy egyszerű rule-based vezérlő:
  - Navigáció: P-szabályozó a célpont irányában
    (yaw_rate ∝ szög-hiba, v_forward csökken kanyarban)
  - Locomotion: UnitreeRLGymAdapter (motion.pt) kezeli a járásmintát

Kimenet: data/demonstrations/scripted_<env>_v<n>.npz

Használat (repo gyökeréből):
    python src/roboshelf_ai/scripts/collect_scripted_expert.py \\
        --env nav \\
        --n-episodes 100 \\
        --config configs/navigation/retail_nav_hier_v1.yaml \\
        --output data/demonstrations/scripted_nav_v1.npz

    # Gyors próba (5 epizód, vizualizáció nélkül):
    python src/roboshelf_ai/scripts/collect_scripted_expert.py \\
        --env nav --n-episodes 5 --no-render

Megjegyzés:
    A scripted expert közepes minőségű demókat ad.
    Az első BC iteráció után a policy rollout-ot is érdemes gyűjteni
    (--env nav --source policy_rollout --model-path <zip>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# PYTHONPATH setup
_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

from roboshelf_ai.core.interfaces.demonstration import DemoCollector


# ---------------------------------------------------------------------------
# Scripted nav expert
# ---------------------------------------------------------------------------

class ScriptedNavExpert:
    """Rule-based nav policy — P-szabályozó célpont felé.

    Obs (9 dim, retail_nav_hier_v1.yaml alapján):
        [0:2]  robot_xy
        [2:4]  goal_rel_polar: [dist, angle_to_goal]   ← fontos
        [4]    torso_heading
        [5:7]  torso_lin_vel
        [7:9]  foot_contact (2)

    Action (2 dim): [v_forward_norm, yaw_rate_norm] ∈ [-1, 1]
    """

    def __init__(
        self,
        kp_yaw: float = 1.2,          # yaw P-gain
        v_forward_max: float = 0.8,   # normalizált max előre sebesség
        slow_angle_threshold: float = 0.5,  # rad — ennél nagyobb szögnél lassít
    ) -> None:
        self.kp_yaw = kp_yaw
        self.v_forward_max = v_forward_max
        self.slow_angle_threshold = slow_angle_threshold

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Obs → normalizált [v_forward, yaw_rate] akció."""
        dist        = float(obs[2])   # célpont távolság
        angle_err   = float(obs[3])   # szöghiba a célponthoz (rad)

        # Yaw szabályozó
        yaw_rate = np.clip(self.kp_yaw * angle_err, -1.0, 1.0)

        # Sebesség: nagy szögnél lassul (fordulj meg mielőtt rohansz)
        angle_factor = max(0.0, 1.0 - abs(angle_err) / (self.slow_angle_threshold * 2))
        # Közel a célhoz is lassít
        dist_factor = min(1.0, dist / 0.5)
        v_forward = self.v_forward_max * angle_factor * dist_factor

        return np.array([v_forward, yaw_rate], dtype=np.float32)


# ---------------------------------------------------------------------------
# Gyűjtés nav env-ben
# ---------------------------------------------------------------------------

def collect_nav_demos(
    cfg_path: str,
    n_episodes: int,
    output_path: str,
    render: bool = False,
    seed: int = 42,
) -> None:
    """Scripted expert demókat gyűjt a nav hierarchikus env-ben."""
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("env", {})
    obs_dim = env_cfg.get("obs_dim", 9)
    act_dim = env_cfg.get("action_dim", 2)

    # Nav env importálása (csak ha fut a MuJoCo)
    try:
        from roboshelf_ai.mujoco.envs.navigation.retail_nav_hier_env import (
            RetailNavHierEnv
        )
        env = RetailNavHierEnv(cfg=cfg, render_mode="human" if render else None)
    except ImportError as e:
        print(f"⚠️  Nav env nem importálható: {e}")
        print("   Futtasd ezt a scriptet a saját gépen (Mac miniforge).")
        print("   A sandbox Linux-on hiányoznak a MuJoCo / SB3 csomagok.")
        return

    expert = ScriptedNavExpert()
    collector = DemoCollector(obs_dim=obs_dim, act_dim=act_dim)

    print(f"\nScripted nav expert gyűjtés: {n_episodes} epizód...")
    print(f"Config: {cfg_path}")
    print(f"Output: {output_path}\n")

    success_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        collector.start_episode()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = expert.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            collector.record(obs, action, reward, done)
            obs = next_obs
            ep_reward += reward
            steps += 1

        collector.end_episode()
        if info.get("goal_reached", False):
            success_count += 1

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(
                f"  Epizód {ep+1:>3}/{n_episodes} | "
                f"lépések={steps:>4} | "
                f"reward={ep_reward:>7.1f} | "
                f"siker: {success_count}/{ep+1} "
                f"({100*success_count/(ep+1):.0f}%)"
            )

    env.close()

    print(f"\nSzkripted expert sikeresség: {100*success_count/n_episodes:.1f}%")
    collector.save(output_path, source="scripted", config_name=Path(cfg_path).stem)


# ---------------------------------------------------------------------------
# Scripted loco expert (egyszerű: csak egyenesen előre megy)
# ---------------------------------------------------------------------------

def collect_loco_demos(
    cfg_path: str,
    n_episodes: int,
    output_path: str,
    render: bool = False,
    seed: int = 42,
) -> None:
    """Scripted expert demókat gyűjt a locomotion env-ben.

    A scripted loco expert fix parancsot ad: v_forward=0.5, yaw=0.
    Főleg BC-vel inicializált loco policy-hoz hasznos baseline.
    """
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("env", {})

    try:
        from roboshelf_ai.mujoco.envs.locomotion.g1_locomotion_command_env import (
            G1LocomotionCommandEnv
        )
        env = G1LocomotionCommandEnv(
            max_episode_steps=env_cfg.get("max_episode_steps", 1000),
        )
    except ImportError as e:
        print(f"⚠️  Loco env nem importálható: {e}")
        return

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    collector = DemoCollector(obs_dim=obs_dim, act_dim=act_dim)

    print(f"\nScripted loco expert gyűjtés: {n_episodes} epizód...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        collector.start_episode()
        done = False
        steps = 0

        while not done:
            # Konstans "egyenesen előre" parancs → defaultctrl körüli kis perturbáció
            action = env.action_space.sample() * 0.05  # közel nullához
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            collector.record(obs, action, reward, done)
            obs = next_obs
            steps += 1

        collector.end_episode()

        if (ep + 1) % 20 == 0 or ep == n_episodes - 1:
            print(f"  Epizód {ep+1:>3}/{n_episodes} | lépések={steps}")

    env.close()
    collector.save(output_path, source="scripted", config_name=Path(cfg_path).stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Scripted expert demonstráció gyűjtő")
    p.add_argument(
        "--env", choices=["nav", "loco"], default="nav",
        help="Melyik env-ben gyűjtsük a demókat"
    )
    p.add_argument(
        "--n-episodes", type=int, default=100,
        help="Gyűjtendő epizódok száma"
    )
    p.add_argument(
        "--config", default="configs/navigation/retail_nav_hier_v1.yaml",
        help="Config YAML fájl"
    )
    p.add_argument(
        "--output", default=None,
        help="Kimeneti .npz fájl (default: data/demonstrations/scripted_<env>_v1.npz)"
    )
    p.add_argument(
        "--render", action="store_true",
        help="MuJoCo viewer megjelenítése"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Véletlenszám seed"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output = args.output or f"data/demonstrations/scripted_{args.env}_v1.npz"

    if args.env == "nav":
        collect_nav_demos(
            cfg_path=args.config,
            n_episodes=args.n_episodes,
            output_path=output,
            render=args.render,
            seed=args.seed,
        )
    else:
        collect_loco_demos(
            cfg_path=args.config,
            n_episodes=args.n_episodes,
            output_path=output,
            render=args.render,
            seed=args.seed,
        )
