"""
diag_settle_gr1.py — GR1T1 kar settling time mérése.

Megméri hány lépés kell ahhoz, hogy a kar PUSH_ARM_POS-ba stabilizálódjon
a DEFAULT_ARM_POS-ból indulva, kp=150 position control mellett.

Futtatás (repo gyökeréből):
    python3 tools/diag_settle_gr1.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from roboshelf_ai.mujoco.envs.manipulation.gr1_shelf_stock_env import (
    GR1ShelfStockEnv, ARM_QPOS_INDICES, _JOINT_RANGES,
)

PUSH_ARM_POS = np.array([-0.654, 0.111, 1.875, 0.500], dtype=np.float32)
TOL          = 0.05    # rad — "elért" küszöb minden jointra
N_EPISODES   = 5
MAX_STEPS    = 120

# Azonos normalizáció mint a scripted expert _PAP_NORM
_mid      = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
_half     = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0
PAP_NORM  = np.clip((PUSH_ARM_POS - _mid) / (_half + 1e-6), -1.0, 1.0).astype(np.float32)

env = GR1ShelfStockEnv()

print(f"PUSH_ARM_POS : {PUSH_ARM_POS}")
print(f"PAP_NORM     : {PAP_NORM.round(3)}")
print(f"Tolerancia   : ±{TOL} rad minden jointra\n")

settle_steps = []

for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=ep)

    # Kar indulás — közvetlenül qpos-ból (nem obs-ból!)
    arm_start = np.array([env._data.qpos[qi] for qi in ARM_QPOS_INDICES])
    print(f"Ep {ep+1:02d} | Indulás qpos: {arm_start.round(3)}")
    print(f"       | Cél  qpos: {PUSH_ARM_POS}")

    reached = None
    for step in range(1, MAX_STEPS + 1):
        obs, _, terminated, truncated, _ = env.step(PAP_NORM)

        arm_now = np.array([env._data.qpos[qi] for qi in ARM_QPOS_INDICES])
        err     = np.abs(arm_now - PUSH_ARM_POS)

        if step % 5 == 0 or (reached is None and np.all(err < TOL)):
            print(f"       | lépés {step:3d}: qpos={arm_now.round(3)}  err={err.round(3)}")

        if np.all(err < TOL) and reached is None:
            reached = step
            print(f"       ✅ Elérve: {step} lépésnél")
            break

        if terminated or truncated:
            break

    if reached is None:
        arm_final = np.array([env._data.qpos[qi] for qi in ARM_QPOS_INDICES])
        print(f"       ❌ Nem érte el {MAX_STEPS} lépésen belül")
        print(f"       | Végső err: {np.abs(arm_final - PUSH_ARM_POS).round(3)}")
    else:
        settle_steps.append(reached)
    print()

env.close()

if settle_steps:
    print(f"Settling time: min={min(settle_steps)}  max={max(settle_steps)}  átlag={np.mean(settle_steps):.1f} lépés")
    print(f"Javasolt SETTLE_STEPS: {max(settle_steps) + 5}")
else:
    print("❌ Egyik epizódban sem érte el — valószínűleg joint limit probléma.")
    print("   Ellenőrizd: PUSH_ARM_POS értékei a joint range-en belül vannak-e?")
