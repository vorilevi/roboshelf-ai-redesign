# Roboshelf AI

MuJoCo-based reinforcement learning for Unitree G1 retail navigation and shelf-restocking tasks.

## Overview

Roboshelf AI is a reinforcement learning project focused on training a humanoid robot to operate in a retail store environment. The current goal is robust in-store navigation, followed by object manipulation and hierarchical task execution.

## Current Status

- Phase 1: locomotion baseline completed.
- Phase 2: G1 retail navigation is actively being trained.
- Phase 3: manipulation environment drafted.
- Phase 4–5: hierarchical policy and demo pipeline planned.

## Current Focus

The active experiment is a fresh-start Phase 2 training run using a MuJoCo retail environment and PPO. Recent updates include a bent-knee default pose, foot slip penalty, foot distance penalty, and reward rebalancing to improve stability and forward progress.

## Tech Stack

- MuJoCo
- Stable-Baselines3 (PPO)
- PyTorch
- Unitree G1 humanoid model

## Quick Start

Install dependencies:

    pip install mujoco gymnasium stable-baselines3 torch

Run Phase 2 training:

    python src/training/roboshelf_phase2_train.py --level m2_20m_v22

## Project Structure

    src/
      training/
      envs/
        assets/
    roboshelf-results/

## Roadmap

1. Stabilize retail navigation.
2. Train pick-and-place manipulation.
3. Combine navigation and manipulation with hierarchical control.
4. Build an investor-facing demo pipeline.
