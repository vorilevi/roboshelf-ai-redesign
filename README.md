# Roboshelf AI

MuJoCo- and VLA-based robot learning for humanoid retail tasks: locomotion, in-store navigation, shelf manipulation, and multi-robot vendor-independence validation.

## Overview

Roboshelf AI trains humanoid robots to work in retail store environments. The work covers three locomotion/manipulation tracks plus a vendor-independence track demonstrating that the same software stack runs on robots from different manufacturers.

## Current Status

**Phase 1 — locomotion.** Baseline walking policy trained and validated in MuJoCo. Configs in `configs/locomotion/`, training entry point `src/roboshelf_ai/training/train_loco_v1.py`.

**Phase 2 — in-store navigation.** Hierarchical navigation policy in a MuJoCo retail environment (`src/roboshelf/envs/retail_nav_env.py`), config `configs/navigation/retail_nav_hier_v1.yaml`.

**Phase 3 — manipulation (F3). Closed 2026-05-25.**

The task is a **push primitive**: moving an object into a target zone. This is not full shelf restocking — grasping is not implemented yet.

| Step | Approach | Result |
|---|---|---|
| F3a | MuJoCo push environment | done |
| F3b | PPO baseline, 12 reward/config variants | best ~6% success — insufficient |
| F3c | Behavioural cloning (ACT) from scripted expert | baseline established |
| F3d | UnifoLM-VLA-0 zero-shot | 0% success |
| F3e v1 | VLA fine-tune — 200 demos, 2k steps, LoRA r=16 | 8% (4/50) — rejected |
| F3e v2 | VLA fine-tune — 1,000 demos, 10k steps, LoRA r=32 | **80% (40/50) — accepted** |

Evaluation protocol: 50 independent episodes, randomised object position (x 0.25–0.65 m, y −0.15–0.15 m), max 200 steps per episode, MuJoCo headless with EGL rendering. Training ran on a free-tier Kaggle T4 — no paid compute has been used so far.

**Vendor-independence track — closed 2026-07-25.** The same UnifoLM-VLA-0 pipeline was retrained and evaluated on a **Booster Robotics T1** (different manufacturer from G1). Result: **86% SR (43/50)** — exceeds the G1 result. This directly validates the platform-agnostic positioning.

| Robot | Manufacturer | SR | Eval date |
|---|---|---|---|
| Unitree G1 | Unitree Robotics | 80% (40/50) | 2026-05-25 |
| Booster T1 | Booster Robotics | **86% (43/50)** | 2026-07-25 |

Eval script: `notebooks/kaggle_vla_eval_t1_v2.py`. Kaggle notebook: `roboshelf-t1-eval-v1`.

**Phase 4 — planned.** A/B/C comparison of WALL-OSS, GR00T N1.5 and UnifoLM-VLA-0 on a common benchmark; protocol drafted in `docs/vla_abc_test_protocol.md`. Transfer to physical hardware has not started.

## Tech Stack

- **MuJoCo** — simulation
- **Stable-Baselines3 (PPO)** — RL baselines
- **UnifoLM-VLA-0** — Qwen2.5-VL-7B backbone + DiT action head, LoRA fine-tuning
- **LeRobot v2.1** — dataset format
- **PyTorch**
- **Unitree G1** + **Booster T1** humanoid models (vendor-independence)

## Data and Checkpoints

Datasets, checkpoints and result artefacts are **not** in this repository — see `.gitignore`.

The fine-tuning dataset is public on Kaggle: **`leventevrss/roboshelf-vla-v2`** — 1,000 episodes, 64,622 frames, exported with `tools/lerobot_export.py`.

Demonstrations were produced by a **scripted expert** (`tools/scripted_expert.py`), not by human teleoperation: 1,000 successful episodes out of 11,406 attempts (8.8% scripted success rate). The expert follows a deterministic APPROACH → PUSH → DONE policy and fails often at edge-case object positions; only successful episodes were exported.

## Quick Start

Install dependencies:

    pip install -r requirements.txt

Run Phase 2 navigation training:

    python src/training/roboshelf_phase2_train.py --level m2_20m_v22

VLA fine-tuning and evaluation run as Kaggle notebooks:

    notebooks/kaggle_vla_train.py    # fine-tune
    notebooks/kaggle_vla_eval.py     # 50-episode evaluation

## Project Structure

    configs/          locomotion, navigation, manipulation, bc, demo configs
    src/
      roboshelf/      envs (retail_nav_env, manipulation_env), training, replay
      roboshelf_ai/   locomotion, mujoco, isaac, tasks
      training/       phase 2 entry points
    notebooks/        Kaggle training and evaluation scripts
    tools/            demo collection, LeRobot export, evaluation
    docs/             protocols, retrospectives, known issues

## Roadmap

1. Phase 4 — A/B/C model comparison on a common benchmark (WALL-OSS / GR00T N1.6 / UnifoLM-VLA-0).
2. Transfer the manipulation policy to physical hardware (Unitree G1).
3. Extend beyond the push primitive to grasping.
4. Combine navigation and manipulation under hierarchical control.
