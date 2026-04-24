#!/usr/bin/env python3
"""
Debug: kéz és termék valódi world-frame pozíciója reset után.
Megmutatja mi a tényleges hand_stock_dist (nem a stock_target_dist).

Futtatás:
    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/debug_hand_pos.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

with open("configs/manipulation/shelf_stock_v6.yaml") as f:
    cfg = yaml.safe_load(f)

from roboshelf_ai.mujoco.envs.manipulation.g1_shelf_stock_env import G1ShelfStockEnv

env = G1ShelfStockEnv(cfg=cfg)
obs, info = env.reset(seed=42)

print("\n" + "="*60)
print("  DEBUG — Kéz / Stock / Target pozíciók (reset után)")
print("="*60)
print(f"  hand_xyz    (right_hand_site): {info['hand_xyz']}")
print(f"  stock_xyz   (stock_1 body):    {info['stock_xyz']}")
print(f"  target_xyz  (target_shelf):    {info['target_xyz']}")
print(f"  hand→stock dist:  {info['hand_stock_dist']:.4f} m")
print(f"  stock→target dist: {info['stock_target_dist']:.4f} m")
print(f"  stock_rise: {info['stock_rise']:.4f} m")
print()

# 10 random action lépés
print("  10 random akció után:")
rng = np.random.default_rng(0)
min_dist = info['hand_stock_dist']
for i in range(10):
    action = rng.uniform(-1, 1, env.action_space.shape)
    obs, reward, term, trunc, info = env.step(action)
    d = info['hand_stock_dist']
    if d < min_dist:
        min_dist = d
    print(f"  step {i+1:2d}: hand→stock={d:.4f}m  reward={reward:.3f}")

print()
print(f"  Legjobb elért hand→stock dist: {min_dist:.4f} m")
print()

# Ellenőrzés: tud-e a kar közelíteni a termékhez?
# Ha min_dist >> 0.06 → a default arm pos rossz vagy a site rossz helyen van
if min_dist > 0.3:
    print("  ❌ PROBLÉMA: kéz nem közelít a termékhez! Vizsgálandó:")
    print("     - right_hand_site pozíciója az XML-ben")
    print("     - _DEFAULT_ARM_POS értékek")
    print("     - stock reset pozíció")
elif min_dist < 0.06:
    print("  ✅ Kéz fizikailag eléri a terméket")
else:
    print(f"  ⚠️  Kéz közelít, de {min_dist:.3f}m még messze van a 0.06m grasphoz")

env.close()
