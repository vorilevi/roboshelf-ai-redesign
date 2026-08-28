"""
shelflife_grasp_run.py — a fogás VÉGREHAJTÁSA az `approach_until` primitívvel

    python3 tools/shelflife_grasp_run.py

Ez a fájl mutatja meg, hogyan néz ki a feladat, ha a platform a megfelelő
primitívet adja. A teljes fogás négy hívás:

    approach_until(pre_grasp,  until="goal")     # odaállás, tiszta térben
    approach_until(grasp_pont, until="contact")  # rá, amíg hozzá nem ér
    close_until(until="grip")                    # zárás szembefogásig
    approach_until(fent,       until="goal")     # emelés

Ezt a négy sort egy ügynök is meg tudja írni. A 6-DoF Jacobian-korrekció, a
gravitációs megereszkedés kezelése és az ütközéskerülés a primitívben van.

A korábbi végrehajtás (diszkrét pályapontok + ízület-interpoláció) ugyanezt
141 mm-es termék-elmozdulással produkálta úgy, hogy egyetlen pályaponton sem
volt kontaktus — a kár a pontok KÖZÖTT keletkezett.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_grasp as G                       # noqa: E402
from shelflife_motion import approach_until, close_until   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standoff", type=float, default=0.12)
    ap.add_argument("--lift", type=float, default=0.10)
    a = ap.parse_args()

    t0 = time.time()
    r = G.GraspRobot()
    if not r.plan:
        sys.exit("nincs fogási terv — futtasd: python3 tools/shelflife_grasp_plan.py")
    plan = r.plan
    R_des = G.GRASP_POSES[plan["pose"]]
    ax, sg = plan["approach_palm_axis"]
    approach_dir = sg * R_des[:, ax]

    box, half = r.product_box()
    prod0 = r.product_pose().copy()
    pre = box - approach_dir * a.standoff

    print("Shelf Life — fogás az approach_until primitívvel\n")
    print(f"  terv     : {plan['approach']} · zárás≈{plan['close_amount']} · "
          f"{plan['digits_in_contact']} ujj · tartalék "
          f"{plan.get('joint_margin_rad')} rad")
    print(f"  fogási pont a tenyérben: {np.round(r._grasp_offset*100,1)} cm")
    print(f"  karton   : közép {np.round(box,3)} · "
          f"{np.round(half*200,1)} cm\n")

    # ── 0. kéz nyitva, kar a pálya elejére ──────────────────────────────────
    r.close_fingers(0.0)
    qs, ep, er = r.ik6_seed(pre, R_des, restarts=16, iters=110)
    print(f"  [0] IK-mag a pre-grasp pontra: {ep*1000:.1f} mm / "
          f"{np.degrees(er):.1f}° · ízülettartalék {r.joint_margin(qs):.2f} rad")
    r.ramp_to(qs, n=20, settle=70)
    print(f"      átvezetés után: termék "
          f"{np.linalg.norm(r.product_pose()-prod0)*1000:.1f} mm")

    # ── 1. a pre-grasp pontra, folytonosan ──────────────────────────────────
    print("\n  [1] odaállás")
    r1 = approach_until(r, pre, R_des, until="goal", verbose=True)

    # ── 2. rá, amíg hozzá nem ér ────────────────────────────────────────────
    print("\n  [2] közelítés kontaktusig")
    r2 = approach_until(r, box, R_des, until="contact", verbose=True)

    # ── 3. zárás szembefogásig ──────────────────────────────────────────────
    print("\n  [3] zárás")
    r3 = close_until(r, until="grip", verbose=True)

    # ── 4. emelés — ez a valódi próba ───────────────────────────────────────
    print("\n  [4] emelés")
    before = r.product_pose().copy()
    up = r.grasp_point() + np.array([0.0, 0.0, a.lift])
    r4 = approach_until(r, up, R_des, until="goal", guard_mm=1e9, verbose=True)
    r.step(500)
    dz = float(r.product_pose()[2] - before[2]) * 1000
    n_end, f_end = r.contact_count()
    held = dz > 30 and n_end > 0

    print("\n" + "─" * 70)
    print(f"  emelkedés          {dz:7.1f} mm  (kéz {a.lift*1000:.0f} mm-t emelt)")
    print(f"  kontaktus utána    {n_end:7d}   {sorted(r.contact_parts())}")
    print(f"  szorítóerő         {f_end:7.1f} N")
    print(f"  MEGTARTOTTA        {'✅ IGEN' if held else '❌ NEM'}")
    print("─" * 70)
    print(f"  futásidő {time.time()-t0:.0f}s")

    out = {"held": held, "lift_mm": dz, "contacts": n_end, "force_N": f_end,
           "parts": sorted(r.contact_parts()),
           "stages": {"odaallas": r1, "kozelites": r2, "zaras": r3, "emeles": r4}}
    p = _REPO / "results/shelflife_grasp/run_approach_until.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"  → {p.relative_to(_REPO)}")
    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
