#!/usr/bin/env python3
"""
Debug: mi a kar tényleges workspace-e, és mi a stock optimális z magassága?

Futtatás:
    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/debug_reach.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import mujoco
import yaml

with open("configs/manipulation/shelf_stock_v6.yaml") as f:
    cfg = yaml.safe_load(f)

from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import (
    G1ShelfStockEnv, _JOINT_RANGES, ARM_QPOS_START, N_ARM_DOF, STOCK_QPOS_START
)

env = G1ShelfStockEnv(cfg=cfg)
obs, info = env.reset(seed=42)

names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]

print("\n" + "="*60)
print("  DEBUG — Kar workspace és optimális stock pozíció")
print("="*60)

# 1. Kéz z-tartomány: végigmegyünk pitch és elbow kombókon
print("\n  [1] Kéz z-tartomány (min-max) a teljes joint range-ben:")
z_vals = []
N = 15
for p in np.linspace(_JOINT_RANGES[0,0], _JOINT_RANGES[0,1], N):
    for e in np.linspace(_JOINT_RANGES[3,0], _JOINT_RANGES[3,1], N):
        joints = np.array([p, 0.0, 0.0, e])
        env._data.qpos[ARM_QPOS_START:ARM_QPOS_START+N_ARM_DOF] = joints
        env._data.ctrl[0:N_ARM_DOF] = joints
        mujoco.mj_forward(env._model, env._data)
        hand = env._data.site_xpos[env._hand_site_id].copy()
        z_vals.append((hand[2], hand[0], hand[1], p, e))

z_vals.sort()
print(f"  Kéz z minimum: {z_vals[0][0]:.3f}m  (hand_xyz: [{z_vals[0][1]:.3f}, {z_vals[0][2]:.3f}, {z_vals[0][0]:.3f}])")
print(f"    pitch={z_vals[0][3]:.3f}, elbow={z_vals[0][4]:.3f}")
print(f"  Kéz z maximum: {z_vals[-1][0]:.3f}m  (hand_xyz: [{z_vals[-1][1]:.3f}, {z_vals[-1][2]:.3f}, {z_vals[-1][0]:.3f}])")
print(f"    pitch={z_vals[-1][3]:.3f}, elbow={z_vals[-1][4]:.3f}")

# Melyik z-értékek vannak 0.80-0.90 között? (asztalfelszín közelében)
reachable_z = [(z, x, y, p, e) for z, x, y, p, e in z_vals if 0.75 <= z <= 0.95]
print(f"\n  Kéz z 0.75-0.95m között: {len(reachable_z)} kombináció")

# 2. Grid search: legjobb hand→stock közelítés, ahol stock_xyz-t optimálisan helyezzük
print("\n  [2] Grid search — legjobb hand→stock dist (stock x=0.45, y=0)")
best_dist = 999.0
best_joints = None
best_hand = None
best_stock_z = None

# Stock z értékeket is próbáljuk
for stock_z in [0.755, 0.770, 0.800, 0.820, 0.850, 0.870]:
    stock_xyz = np.array([0.45, 0.0, stock_z])
    local_best = 999.0
    local_joints = None
    local_hand = None

    for p in np.linspace(_JOINT_RANGES[0,0], _JOINT_RANGES[0,1], N):
        for r in np.linspace(_JOINT_RANGES[1,0], _JOINT_RANGES[1,1], 8):
            for y in np.linspace(_JOINT_RANGES[2,0], _JOINT_RANGES[2,1], 8):
                for e in np.linspace(_JOINT_RANGES[3,0], _JOINT_RANGES[3,1], N):
                    joints = np.array([p, r, y, e])
                    env._data.qpos[ARM_QPOS_START:ARM_QPOS_START+N_ARM_DOF] = joints
                    env._data.ctrl[0:N_ARM_DOF] = joints
                    mujoco.mj_forward(env._model, env._data)
                    hand = env._data.site_xpos[env._hand_site_id].copy()
                    d = float(np.linalg.norm(hand - stock_xyz))
                    if d < local_best:
                        local_best = d
                        local_joints = joints.copy()
                        local_hand = hand.copy()

    flag = "✅" if local_best < 0.06 else ("⚠️ " if local_best < 0.15 else "❌")
    print(f"  stock_z={stock_z:.3f}m → best dist={local_best:.4f}m {flag}  hand_z={local_hand[2]:.3f}m")
    if local_best < best_dist:
        best_dist = local_best
        best_joints = local_joints
        best_hand = local_hand
        best_stock_z = stock_z

print(f"\n  → Legjobb stock_z: {best_stock_z:.3f}m, dist={best_dist:.4f}m")
print(f"    Optimális joints: pitch={best_joints[0]:.3f}, roll={best_joints[1]:.3f}, "
      f"yaw={best_joints[2]:.3f}, elbow={best_joints[3]:.3f}")
print(f"    Kéz pozíciója:    {best_hand}")

if best_dist < 0.06:
    print(f"\n  ✅ FIZIKAILAG ELÉRHETŐ — grasp_dist_threshold=0.06m OK")
elif best_dist < 0.15:
    print(f"\n  ⚠️  Ajánlott grasp_dist_threshold: {best_dist + 0.02:.2f}m")
    print(f"     ÉS stock_z={best_stock_z:.3f}m használata")
else:
    print(f"\n  ❌ NEM ELÉRHETŐ — mélyebb analízis kell")

env.close()
print()
