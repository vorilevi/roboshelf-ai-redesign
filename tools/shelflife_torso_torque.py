"""
shelflife_torso_torque.py — a törzs NYOMATÉKIGÉNYE kp-értékenként (M0 utólagos ellenőrzés)

    python3 tools/shelflife_torso_torque.py

────────────────────────────────────────────────────────────────────────────
MIÉRT KELLETT EZ A FÁJL
────────────────────────────────────────────────────────────────────────────
A projektjegyzetben az M0 táblázatban szerepel egy NYOMATÉK oszlop:

    kp =  30 000 → 257 Nm
    kp = 100 000 → 219 Nm      ← ezzel indokoltuk a választást
    kp = 300 000 → 460 Nm

A `shelflife_torso_tune.py` viszont NEM MÉR NYOMATÉKOT — nincs benne egyetlen
sor sem, ami `actuator_force`-ot vagy `qfrc_actuator`-t olvasna. A lemezen lévő
`results/shelflife_grasp/m0_torso.log` sem tartalmaz nyomaték-oszlopot, sőt a
saját pontozása szerint a NYERTES `kp = 300 000` lenne (a pontozás:
tenyér-hiba, majd `qacc`).

Vagyis a jegyzet táblázata olyan mennyiséget közölt mért adatként, amit
egyetlen futó szkript sem állított elő, és a döntés (100 000) ellentétes a
napló saját ajánlásával.

Ez a fájl a hiányt pótolja: TÉNYLEGESEN megméri a nyomatékot, hogy a döntés
utólag alátámasztható vagy cáfolható legyen. Nem módosít semmit.

────────────────────────────────────────────────────────────────────────────
MIT MÉR
────────────────────────────────────────────────────────────────────────────
Ugyanaz a próbapóz, mint az M0-ban (pre-grasp IK-póz). Minden kp-értéknél:

    · CSÚCSNYOMATÉK   — a mozgás alatti maximum (ezt kell kibírnia a hajtásnak)
    · TARTÓNYOMATÉK   — a beállás utáni állandósult érték (gravitáció ellen)
    · yaw-hiba, tenyér-hiba 2000 lépésnél (a tuner kilépési feltétele)

A nyomaték forrása `data.actuator_force[aid]` — pozíció-aktuátornál ez a
kifejtett általánosított erő, forgó ízületnél nyomaték [Nm].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                     # noqa: E402
import shelflife_grasp as G                       # noqa: E402
import shelflife_torso_tune as T                  # noqa: E402

STEPS = 2000


def measure(r, q: np.ndarray, tgt: np.ndarray) -> dict:
    """Egy parancs kiadása, majd a nyomaték követése végig a mozgáson."""
    m = r.model
    aids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}")
            for n in T.TORSO]
    aids = [a for a in aids if a >= 0]
    yaw_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "torso_yaw")
    yaw_adr = m.jnt_qposadr[yaw_j]
    yaw_cmd = q[r.chain.index("torso_yaw")]

    r.reset()
    r.close_fingers(0.0, settle=1)
    r._cmd = q.copy()
    for a, v in zip(r._arm_a, r._cmd):
        r.data.ctrl[a] = v

    peak = 0.0
    for _ in range(STEPS):
        r.step(1)
        peak = max(peak, float(np.abs(r.data.actuator_force[aids]).max()))

    hold = float(np.abs(r.data.actuator_force[aids]).max())
    return {
        "peak_Nm": peak,
        "hold_Nm": hold,
        "yaw_err": abs(float(r.data.qpos[yaw_adr]) - float(yaw_cmd)),
        "palm_mm": float(np.linalg.norm(r.grasp_point() - tgt)) * 1000,
    }


def main() -> int:
    print("Shelf Life M0 — a törzs NYOMATÉKIGÉNYE (utólagos ellenőrzés)\n")
    r = G.GraspRobot()
    if "torso_yaw" not in r.chain:
        sys.exit("a jelenetben nincs aktív torso_yaw")

    plan = r.plan
    R_des = G.GRASP_POSES[plan["pose"] if plan else "right_thumb_up"]
    box, _ = r.product_box()
    ax, sg = (plan["approach_palm_axis"] if plan else (2, -1))
    pre = box - sg * R_des[:, ax] * 0.12

    q, ep, er = r.ik6_seed(pre, R_des, restarts=16, iters=110)
    tgt = T.kinematic_grasp_point(r, q)
    print(f"  próbapóz: pre-grasp, IK {ep*1000:.1f} mm / {np.degrees(er):.1f}°")
    print(f"  {STEPS} lépés, csillapítás 100, armatúra {T.ARMATURE}\n")

    print(f"{'kp':>9}{'csúcs (Nm)':>13}{'tartó (Nm)':>13}"
          f"{'yaw-hiba':>12}{'tenyér (mm)':>13}")
    print("─" * 62)
    rows = {}
    for kp in T.KP_GRID:
        T.set_torso_gains(r, kp, 100.0, T.ARMATURE)
        t = measure(r, q, tgt)
        rows[kp] = t
        print(f"{kp:9.0f}{t['peak_Nm']:13.0f}{t['hold_Nm']:13.0f}"
              f"{t['yaw_err']:12.4f}{t['palm_mm']:13.1f}")
    print("─" * 62)

    print("\n  ÖSSZEVETÉS a jegyzetben szereplő, forrás nélküli számokkal:")
    for kp, claimed in ((30_000, 257), (100_000, 219), (300_000, 460)):
        if kp in rows:
            got = rows[kp]["peak_Nm"]
            print(f"    kp={kp:>7}:  jegyzet {claimed:>4} Nm  ·  "
                  f"mért csúcs {got:>6.0f} Nm  ·  "
                  f"eltérés {abs(got - claimed) / max(claimed, 1) * 100:5.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
