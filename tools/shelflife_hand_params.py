"""
shelflife_hand_params.py — a Menagerie referenciaértékek átvétele és mérése

    python3 tools/shelflife_hand_params.py

────────────────────────────────────────────────────────────────────────────
HONNAN JÖNNEK AZ ÉRTÉKEK
────────────────────────────────────────────────────────────────────────────
A `mujoco_menagerie` négy publikált kezéből (Shadow Hand, Wonik Allegro,
LEAP Hand, Robotiq 2F85), 2026-08-05-én kiolvasva. Ezek NEM tippek: olyan
modellek paraméterei, amelyeket a szerzőik fogásra hangoltak.

                     Shadow      Allegro   LEAP     Robotiq    MI (alap)
    ujj kp           0.4–1.5     1         3.0      ín          20
    forcerange       ±1…±3 Nm    —         —        ±5 N        NINCS
    kv               —           —         0.01     —           nincs
    damping          0.05        0.1       0.03     0.1         0.5
    armature         0.0002      —         —        0.005       0.001
    frictionloss     0.01        —         0.001    —           0
    fogófelület μ    —           —         0.2/0.5  0.7/0.6     1.0 (alap)
    solref           0.005       —         0.0001   0.004       0.02 (alap)
    priority         0           0         0        1           0

A `priority` külön fontos: a MuJoCo a két geom érintkezési paraméterét
MAXIMUMMAL vegyíti — hacsak a prioritás nem különbözik. Emiatt írta felül a
kéz alapértelmezett 1.0-s súrlódása a termékre beállított 0.9-et. A Robotiq
fogópárnái `priority=1`-et kapnak, így az ő értékeik érvényesülnek.

────────────────────────────────────────────────────────────────────────────
MIT MÉR
────────────────────────────────────────────────────────────────────────────
A HITELESÍTETT megközelítéses próbát (`shelflife_grip_test.py --approach`):
a kéz a terv pályáján megy a tárgyhoz, zár, majd emel. A mérce a KÖVETÉS:
a termék hány százalékát teszi meg a kéz emelkedésének.

Báziseset (2026-08-05, hitelesített eszközzel):
    2 ujj · 79.8 N · követés −17%
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
from shelflife_api import Robot                  # noqa: E402
from shelflife_grip_test import GripRig, DIGITS  # noqa: E402


def apply_params(rig: GripRig, kp=None, forcerange=None, kv=None,
                 damping=None, armature=None, frictionloss=None,
                 grip_friction=None, solref=None, solimp=None,
                 priority=None) -> None:
    """Menagerie-stílusú paraméterek az UJJAKRA, futásidőben."""
    m = rig.m
    AN = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    BN = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    acts = [a for a in range(m.nu) if any(d in AN(a) for d in DIGITS)]
    jnts = [int(m.actuator_trnid[a, 0]) for a in acts]
    dofs = [int(m.jnt_dofadr[j]) for j in jnts]
    # fogófelületek: közép- és végperec
    grip = [g for g in range(m.ngeom)
            if BN(m.geom_bodyid[g]).endswith(("medial", "distal"))
            and any(f"r_{d}" in BN(m.geom_bodyid[g]) for d in DIGITS)]

    for a in acts:
        if kp is not None:
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, 1] = -kp
        if kv is not None:
            m.actuator_biasprm[a, 2] = -kv
        if forcerange is not None:
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = [-forcerange, forcerange]
    for dof in dofs:
        if damping is not None:
            m.dof_damping[dof] = damping
        if armature is not None:
            m.dof_armature[dof] = armature
        if frictionloss is not None:
            m.dof_frictionloss[dof] = frictionloss
    for g in grip:
        if grip_friction is not None:
            m.geom_friction[g] = [grip_friction, 0.02, 0.002]
        if solref is not None:
            m.geom_solref[g] = solref
        if solimp is not None:
            m.geom_solimp[g][:3] = solimp
        if priority is not None:
            m.geom_priority[g] = priority


VARIANTS = {
    "0 — alapállapot (a miénk)": {},
    "1 — csak ERŐKORLÁT (±1.5 Nm)": dict(forcerange=1.5),
    "2 — + lágyabb szervó (kp 2, kv 0.01)": dict(forcerange=1.5, kp=2.0, kv=0.01),
    "3 — + Shadow ízületek": dict(forcerange=1.5, kp=2.0, kv=0.01,
                                  damping=0.05, armature=0.0002,
                                  frictionloss=0.01),
    "4 — + Robotiq fogófelület": dict(forcerange=1.5, kp=2.0, kv=0.01,
                                      damping=0.05, armature=0.0002,
                                      frictionloss=0.01, grip_friction=0.7,
                                      solref=[0.004, 1.0],
                                      solimp=[0.95, 0.99, 0.001], priority=1),
}


def main() -> int:
    print("Shelf Life — Menagerie-referenciaértékek a GENE.01 kezén")
    print("A hitelesített MEGKÖZELÍTÉSES próbával mérve.\n")
    print(f"  {'változat':<38}{'ujj':>5}{'közép':>7}{'erő':>9}"
          f"{'követés':>10}")
    print("  " + "─" * 70)
    for name, kw in VARIANTS.items():
        r = Robot()
        r.reset_home()
        rig = GripRig(r)
        if kw:
            apply_params(rig, **kw)
        try:
            res = rig.run_approach()
        except Exception as e:                            # noqa: BLE001
            print(f"  {name:<38}  hiba: {type(e).__name__}")
            continue
        if not res["ok"]:
            print(f"  {name:<38}  {res['why']}")
            continue
        mark = "  ✅" if res["pass"] else ""
        print(f"  {name:<38}{len(res['digits']):5d}{res['medial']:7d}"
              f"{res['force_N']:8.1f} N{res['follow']*100:9.0f}%{mark}")
    print("\n  A követés = a termék emelkedése / a kéz emelkedése.")
    print("  100% = tökéletesen viszi magával · 0% = a helyén marad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
