"""MuJoCo retail navigation environment for Roboshelf AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class RetailNavConfig:
    name: str = "phase2_retail_nav"
    robot_name: str = "unitree_g1"
    episode_steps: int = 1000
    control_substeps: int = 10
    target_xy: tuple[float, float] = (0.0, 3.8)
    start_xy: tuple[float, float] = (0.0, 0.5)
    healthy_z_min: float = 0.45
    healthy_z_max: float = 1.6
    forward_reward_weight: float = 2.0
    healthy_reward: float = 0.02
    ctrl_cost_weight: float = 0.001
    goal_bonus: float = 50.0
    target_tolerance: float = 0.35
    render_mode: str | None = None
    xml_path: str | None = None


class RetailNavEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal real MuJoCo wrapper for retail navigation."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: RetailNavConfig | None = None):
        super().__init__()
        self.config = config or RetailNavConfig()

        self.xml_path = self._resolve_xml_path(self.config.xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._step_count = 0
        self._prev_xy = np.zeros(2, dtype=np.float64)
        self._target_xy = np.array(self.config.target_xy, dtype=np.float64)
        self._start_xy = np.array(self.config.start_xy, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv + 2,),
            dtype=np.float32,
        )

        action_low = np.full((self.model.nu,), -1.0, dtype=np.float32)
        action_high = np.full((self.model.nu,), 1.0, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self._renderer = None
        if self.config.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)

    @staticmethod
    def _resolve_xml_path(xml_path: str | None) -> Path:
        if xml_path is not None:
            candidate = Path(xml_path).expanduser().resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Retail MJCF not found: {candidate}")
            return candidate

        repo_root = Path(__file__).resolve().parents[4]
        candidates = [
            repo_root / "src" / "roboshelf_ai" / "mujoco" / "assets" / "roboshelf_retail_store.xml",
            repo_root / "src" / "envs" / "assets" / "roboshelf_retail_store.xml",
            repo_root / "roboshelf_retail_store.xml",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError("Could not locate roboshelf_retail_store.xml in expected locations.")

    def _get_torso_xy(self) -> np.ndarray:
        return np.array(self.data.qpos[:2], dtype=np.float64)

    def _get_torso_z(self) -> float:
        if self.model.nq >= 3:
            return float(self.data.qpos[2])
        return 0.0

    def _get_obs(self) -> np.ndarray:
        torso_xy = self._get_torso_xy()
        target_rel = self._target_xy - torso_xy
        obs = np.concatenate(
            [
                np.asarray(self.data.qpos, dtype=np.float32),
                np.asarray(self.data.qvel, dtype=np.float32),
                target_rel.astype(np.float32),
            ]
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        torso_xy = self._get_torso_xy()
        dist_to_target = float(np.linalg.norm(self._target_xy - torso_xy))
        return {
            "xml_path": str(self.xml_path),
            "step_count": self._step_count,
            "torso_x": float(torso_xy[0]),
            "torso_y": float(torso_xy[1]),
            "torso_z": self._get_torso_z(),
            "dist_to_target": dist_to_target,
        }

    def _is_healthy(self) -> bool:
        torso_z = self._get_torso_z()
        return self.config.healthy_z_min <= torso_z <= self.config.healthy_z_max

    def _compute_reward(self, action: np.ndarray, torso_xy: np.ndarray) -> tuple[float, bool]:
        prev_dist = float(np.linalg.norm(self._target_xy - self._prev_xy))
        curr_dist = float(np.linalg.norm(self._target_xy - torso_xy))
        progress_reward = self.config.forward_reward_weight * (prev_dist - curr_dist)
        healthy_reward = self.config.healthy_reward if self._is_healthy() else 0.0
        ctrl_cost = self.config.ctrl_cost_weight * float(np.sum(np.square(action)))
        reached_goal = curr_dist <= self.config.target_tolerance
        goal_bonus = self.config.goal_bonus if reached_goal else 0.0
        reward = progress_reward + healthy_reward + goal_bonus - ctrl_cost
        return reward, reached_goal

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        if self.model.nq >= 3:
            self.data.qpos[0] = self._start_xy[0]
            self.data.qpos[1] = self._start_xy[1]
            self.data.qpos[2] = max(self.data.qpos[2], 0.78)

        if self.model.nv > 0:
            self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_xy = self._get_torso_xy().copy()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.model.nu > 0:
            self.data.ctrl[:] = action

        for _ in range(self.config.control_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        torso_xy = self._get_torso_xy()

        reward, reached_goal = self._compute_reward(action, torso_xy)
        terminated = reached_goal or (not self._is_healthy())
        truncated = self._step_count >= self.config.episode_steps

        obs = self._get_obs()
        info = self._get_info()

        self._prev_xy = torso_xy.copy()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.config.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = RetailNavEnv()
    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("action shape:", env.action_space.shape)
    print("info:", info)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={info['step_count']} reward={reward:.3f} "
            f"dist={info['dist_to_target']:.3f} z={info['torso_z']:.3f}"
        )
        if terminated or truncated:
            break

    env.close()
