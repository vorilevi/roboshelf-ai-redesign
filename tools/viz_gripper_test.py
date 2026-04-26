#!/usr/bin/env mjpython
"""
Gripper teszt — csak az ujjak mozognak, a kar áll.
Ciklikusan nyit és zár, hogy lássuk mozognak-e az ujjak.

Futtatás (repo gyökeréből):
    mjpython tools/viz_gripper_test.py
"""

import sys, time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

_ROOT = Path(__file__).resolve().parents[1]

m = mujoco.MjModel.from_xml_path(str(_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

# Gravitáció kikapcsolva a teszt idejére — csak az ujjak mozgása számít
m.opt.gravity[:] = [0.0, 0.0, 0.0]

# Kar alapállásba
ARM_CTRL = [0, 1, 2, 3]
DEFAULT_ARM = [-1.0, 0.2, -0.2, 1.2]
for i, v in zip(ARM_CTRL, DEFAULT_ARM):
    d.ctrl[i] = v

# Gripper ctrl indexek: 4..10
GRIPPER_CTRL_START = 4
N_GRIPPER = 7

# Célpozíciók
CLOSED = np.array([-0.8,  0.5, -1.2,  1.3,  1.5,  1.3,  1.5], dtype=np.float32)
OPEN   = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)

# Kar előre-warm-up
for _ in range(500):
    mujoco.mj_step(m, d)

print("Gripper teszt indul — nyit/zár ciklus. Zárd be az ablakot a kilépéshez.")

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = [0.45, 0.0, 0.92]
    v.cam.distance  = 1.2
    v.cam.azimuth   = 120.0
    v.cam.elevation = -20.0

    cycle = 0
    while v.is_running():
        cycle += 1
        # 2 másodpercig zárva
        print(f"Ciklus {cycle}: ZÁRVA")
        t_end = time.time() + 2.0
        while v.is_running() and time.time() < t_end:
            d.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + N_GRIPPER] = CLOSED
            for i, val in zip(ARM_CTRL, DEFAULT_ARM):
                d.ctrl[i] = val
            for _ in range(50):  # 50 sim lépés = 1 viewer frame (50Hz)
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(0.02)

        # 2 másodpercig nyitva
        print(f"Ciklus {cycle}: NYITVA")
        t_end = time.time() + 2.0
        while v.is_running() and time.time() < t_end:
            d.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + N_GRIPPER] = OPEN
            for i, val in zip(ARM_CTRL, DEFAULT_ARM):
                d.ctrl[i] = val
            for _ in range(50):
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(0.02)

print("Kész.")
