"""
T1 arm actuator kp gain sweep — vendor-independence track diagnosztika.

Célja: validálni, hogy a G1-ből örökölt kp=150 megfelelő-e a Booster T1 karhoz,
vagy T1-specifikus értékre kell módosítani.

Mér:
  (1) hand_x_mean ≥ 0.30m  — a kéz eléri a tároló asztal x pozícióját
  (2) x_osc_std  < 0.008m  — oszcilláció a measure phase-ban elfogadható
  (3) qacc_max   < 2000    — NaN/Inf veszély guard (settle transient kizárva)

MuJoCo position actuator kp runtime-ban módosítható:
  model.actuator_gainprm[i, 0] = kp
  model.actuator_biasprm[i, 1] = -kp

Futtatás (repo gyökeréből):
  python3 tools/diag_kp_sweep_t1.py
  python3 tools/diag_kp_sweep_t1.py --kp-values 50 100 150 200 300
  python3 tools/diag_kp_sweep_t1.py --quiet

Referenciák:
  G1 kp sweep:   tools/diag_kp_sweep.py  (minta)
  T1 reach diag: results/diag/t1_reach_*.csv  (DEFAULT_ARM_POS forrása)
  Obsidian:      [[multi_robot_strategy_2026-07]]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mujoco

# ---------------------------------------------------------------------------
# Konstansok
# ---------------------------------------------------------------------------

XML_PATH = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_t1_v1.xml"

# T1 arm actuator ctrl indexei (XML sorrendben: pitch, roll, elbow_p, elbow_yaw)
ARM_CTRL_INDICES = [0, 1, 2, 3]

# T1 arm joint qpos indexei (depth-first traversal):
#   0=AAHead_yaw, 1=Head_pitch, 2-5=left arm, 6-9=RIGHT arm, 10+=waist+legs
ARM_QPOS_INDICES = [6, 7, 8, 9]

# Default pozíció (diag_arm_reach_t1.py grid search eredménye):
# hand_x=0.399m, hand_z=0.778m, stable=True
DEFAULT_ARM_POS = np.array([-1.0, 1.5, 1.5, 0.0], dtype=np.float64)

# Teszt pozíciók:
#   pos#0: DEFAULT — validált optimum (hand_x≈0.399m)
#   pos#1: mélyebb nyúlás (elbow_p növelve)
#   pos#2: erősebb könyökhajlítás (elbow_p max közelben)
# KIZÁRVA: [-0.8, 1.3, 1.5, 0.0] — hand_x≈0.21m, kinematikailag nem éri el a tárolót.
#   A T1 karnál a roll≥1.4 és elbow_p≥1.3 kombináció szükséges x≥0.30m eléréséhez.
TEST_POSITIONS = [
    np.array([-1.0,  1.5,  1.5,  0.0], dtype=np.float64),  # pos#0 DEFAULT
    np.array([-1.2,  1.5,  1.8,  0.0], dtype=np.float64),  # pos#1 mélyebb reach
    np.array([-1.0,  1.5,  2.0,  0.0], dtype=np.float64),  # pos#2 max elbow hajlítás
]

# KP értékek, amelyeket sweep-elünk
DEFAULT_KP_VALUES = [10, 30, 50, 100, 150, 200, 300, 400]

# Szimulációs lépések
SETTLE_STEPS  = 300   # stabilizálódás (transient kizárva a mérésből)
MEASURE_STEPS = 50    # mérési fázis (utolsó N lépés)

# Elfogadási küszöbök
X_HAND_MIN = 0.30    # m — kéz kell ≥ tároló asztal x pozíciója (target x=0.30)
OSC_MAX    = 0.008   # m — x-oszcilláció std küszöb (measure phase)
QACC_MAX   = 2000.0  # — NaN/Inf guard (csak measure phase steady-state)


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def get_hand_x(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """right_hand_site x koordinátája (push irány)."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    if site_id < 0:
        # fallback: right_hand_link body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link")
        return float(data.xpos[body_id][0]) if body_id >= 0 else 0.0
    return float(data.site_xpos[site_id][0])


def set_arm_kp(model: mujoco.MjModel, kp: float) -> None:
    """Beállítja a 4 kar actuator kp értékét runtime-ban."""
    for i in ARM_CTRL_INDICES:
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp


def measure_one(
    model: mujoco.MjModel,
    arm_target: np.ndarray,
    settle_steps: int = SETTLE_STEPS,
    measure_steps: int = MEASURE_STEPS,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Egy (kp, pozíció) kombinációhoz méri:
      hand_x_mean, x_osc_std, qacc_max

    A settle fázis (step < settle_steps) tranziense QACC=300-500+ lehet —
    ez normális, nem instabilitás jele. QACC-ot csak a measure fázisban rögzítjük.

    Returns:
        (hand_x_mean, x_osc_std, qacc_max)
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Kar inicializálása
    for i, qi in enumerate(ARM_QPOS_INDICES):
        data.qpos[qi] = arm_target[i]
    for i, ci in enumerate(ARM_CTRL_INDICES):
        data.ctrl[ci] = arm_target[i]
    mujoco.mj_forward(model, data)

    x_history:    List[float] = []
    qacc_history: List[float] = []

    for step in range(settle_steps + measure_steps):
        for i, ci in enumerate(ARM_CTRL_INDICES):
            data.ctrl[ci] = arm_target[i]
        mujoco.mj_step(model, data)

        qacc_cur = float(np.max(np.abs(data.qacc)))

        # NaN/Inf → azonnali leállás
        if np.isnan(qacc_cur) or np.isinf(qacc_cur):
            if verbose:
                print(f"    ⚠ NaN/Inf QACC lépés {step}-ben")
            return 0.0, 999.0, 999.0

        # Csak a measure fázisban rögzítünk (transient kizárva)
        if step >= settle_steps:
            x_history.append(get_hand_x(model, data))
            qacc_history.append(qacc_cur)

    x_arr = np.array(x_history)
    qacc_max = float(np.max(qacc_history)) if qacc_history else 0.0
    return float(np.mean(x_arr)), float(np.std(x_arr)), qacc_max


# ---------------------------------------------------------------------------
# Fő sweep
# ---------------------------------------------------------------------------

def run_sweep(
    kp_values: List[float],
    verbose: bool = True,
) -> List[dict]:
    """
    Végigmegy az összes kp értéken és teszt pozíción.
    """
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    current_kp = float(model.actuator_gainprm[0, 0])

    results = []

    if verbose:
        print(f"\nT1 KP Sweep — arm_motor jelenlegi kp: {current_kp:.0f}")
        print(f"XML: {XML_PATH.name}")
        print(f"Teszt pozíciók: {len(TEST_POSITIONS)}, KP értékek: {kp_values}")
        print(f"Elfogadás: hand_x ≥ {X_HAND_MIN}m, osc < {OSC_MAX}m, qacc < {QACC_MAX}")
        print("─" * 76)
        print(f"{'kp':>6}  {'pos#':>4}  {'hand_x':>8}  {'x_osc':>8}  {'qacc_max':>9}  {'OK?':>5}")
        print("─" * 76)

    for kp in kp_values:
        set_arm_kp(model, kp)

        pos_results = []
        for pos_idx, arm_pos in enumerate(TEST_POSITIONS):
            x_mean, x_osc, qacc_max = measure_one(model, arm_pos, verbose=verbose)
            ok = (x_mean >= X_HAND_MIN) and (x_osc < OSC_MAX) and (qacc_max < QACC_MAX)

            row = {
                "kp":         kp,
                "pos_idx":    pos_idx,
                "hand_x_mean": round(x_mean, 4),
                "x_osc_std":  round(x_osc, 5),
                "qacc_max":   round(qacc_max, 2),
                "ok":         ok,
                "arm_target": arm_pos.tolist(),
            }
            results.append(row)
            pos_results.append(ok)

            if verbose:
                ok_str = "✅" if ok else "❌"
                print(f"{kp:>6}  {pos_idx:>4}  {x_mean:>8.4f}  {x_osc:>8.5f}  {qacc_max:>9.2f}  {ok_str}")

        all_ok = all(pos_results)
        any_ok = any(pos_results)
        if verbose and len(TEST_POSITIONS) > 1:
            summary = "✅ MIND OK" if all_ok else ("⚠ RÉSZBEN" if any_ok else "❌ MIND ROSSZ")
            print(f"  → kp={kp}: {summary} ({sum(pos_results)}/{len(TEST_POSITIONS)} pozíció OK)")
            print()

    return results


# ---------------------------------------------------------------------------
# Analízis és javaslat
# ---------------------------------------------------------------------------

def analyze(results: List[dict], verbose: bool = True) -> dict:
    """
    Meghatározza az ajánlott kp értéket.

    Logika:
      Ha kp=150 (G1 referencia) átmegy → kp=150 VALIDÁLT (nincs XML módosítás).
      Ha kp=150 nem megy át → a legkisebb átmenő kp az ajánlott.

    Megjegyzés: a sweep statikus pozíciótartást mér (szabad tér, nincs kontakt).
    Push task alatt a kéznek kontakt erővel szemben kell tartania — ezért
    ha több kp érték is átmegy, a magasabb (robusztusabb) értéket preferáljuk.
    kp=150 átmenete esetén nincs ok a csökkentésre.
    """
    from collections import defaultdict
    by_kp = defaultdict(list)
    for r in results:
        by_kp[r["kp"]].append(r)

    G1_KP = 150
    g1_kp_ok = False
    min_passing_kp = None

    for kp in sorted(by_kp.keys()):
        rows = by_kp[kp]
        all_ok = all(r["ok"] for r in rows)
        if all_ok:
            if min_passing_kp is None:
                min_passing_kp = kp
            if kp == G1_KP:
                g1_kp_ok = True

    # Ha G1 kp (150) átmegy → azt ajánljuk (nem a minimumot)
    recommended_kp = G1_KP if g1_kp_ok else min_passing_kp

    summary = {
        "recommended_kp": recommended_kp,
        "g1_kp_validated": g1_kp_ok,
        "min_passing_kp": min_passing_kp,
        "x_hand_target":  X_HAND_MIN,
        "osc_threshold":  OSC_MAX,
    }

    if verbose:
        print("─" * 76)
        if g1_kp_ok:
            print(f"\n✅ kp=150 VALIDÁLT T1-en — G1-gyel azonos, nincs XML módosítás.")
            if min_passing_kp and min_passing_kp < G1_KP:
                print(f"   (Megjegyzés: kp={min_passing_kp} is átmegy statikus tesztben,")
                print(f"    de kp=150-et tartjuk — push kontakt alatt robusztusabb.)")
        elif min_passing_kp:
            print(f"\n⚠ kp=150 NEM ment át. Ajánlott: kp={min_passing_kp}")
            print(f"\n   scene_manip_sandbox_t1_v1.xml módosítandó:")
            print(f"   <default class=\"arm_motor\">")
            print(f"     <position kp=\"{min_passing_kp}\" forcerange=\"-15 15\" ctrlrange=\"-3.14 3.14\"/>")
            print(f"   </default>")
        else:
            print("\n❌ Egyik kp érték sem teljesíti az összes feltételt!")
            print("   → Ellenőrizd a forcerange értékeket vagy növeld a sweep tartományát.")
        print()

    return summary


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kp", "pos_idx", "hand_x_mean", "x_osc_std", "qacc_max", "ok"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"CSV mentve: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T1 arm kp gain sweep diagnosztika")
    parser.add_argument("--kp-values", type=float, nargs="+", default=DEFAULT_KP_VALUES,
                        help=f"Tesztelendő kp értékek (default: {DEFAULT_KP_VALUES})")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV kimenet útvonala (opcionális)")
    parser.add_argument("--quiet", action="store_true",
                        help="Csak az ajánlott kp értéket írja ki")
    args = parser.parse_args()

    verbose = not args.quiet

    t0 = time.time()
    results = run_sweep(kp_values=args.kp_values, verbose=verbose)
    summary = analyze(results, verbose=verbose)

    if verbose:
        print(f"Futási idő: {time.time()-t0:.1f}s")

    out_path = Path(args.out) if args.out else \
        _REPO_ROOT / f"results/diag/t1_kp_sweep_{time.strftime('%Y%m%d_%H%M')}.csv"
    save_csv(results, out_path)

    if args.quiet:
        rec = summary["recommended_kp"]
        print(rec if rec else "NONE")
        if not rec:
            sys.exit(1)


if __name__ == "__main__":
    main()
