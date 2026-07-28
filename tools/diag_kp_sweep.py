"""
Arm actuator kp gain sweep — Phase 030 F3b-A diagnosztika.

Célja: meghatározni azt a legkisebb kp értéket, amellyel
  (1) z_hand_max ≥ 0.95m stabilan tartható
  (2) oszcilláció < 0.005m (utolsó 50 step std)
  (3) QACC max < 50 (NaN/Inf veszély elkerülése)

A MuJoCo position actuator kp-je runtime-ban módosítható:
  model.actuator_gainprm[i, 0] = kp
  model.actuator_biasprm[i, 1] = -kp
Így nem kell XML-t módosítani a sweep során.

Futtatás (repo gyökeréből):
  python3 tools/diag_kp_sweep.py
  python3 tools/diag_kp_sweep.py --out results/diag/kp_sweep_$(date +%Y%m%d).csv
  python3 tools/diag_kp_sweep.py --kp-values 50 100 200 300 400 --verbose
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

XML_PATH = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"

# A 4 kar actuator ctrl indexe (XML sorrendben: pitch, roll, yaw, elbow)
ARM_CTRL_INDICES = [0, 1, 2, 3]

# Kar joint qpos indexek a simban (29 loco DOF + index offszet)
ARM_QPOS_START = 29  # g1_shelf_stock_env.py-val konzisztens

# DEFAULT_ARM_POS (v9 config, 20736 kombináció grid search eredménye)
DEFAULT_ARM_POS = np.array([-1.0, 0.2, -0.2, 1.2], dtype=np.float64)

# Teszt pozíciók: DEFAULT + felfelé nyújtott variánsok
# (shoulder_pitch negatívabb → kar magasabbra nyúl)
# KIZÁRVA (kp_sweep_20260430 tanulság):
#   pos [-0.8, 0.3, -0.3, 1.4] — z_max=0.78-0.81m (soha nem éri el 0.87m-t),
#   QACC steady-state is magas → geometriailag rossz konfiguráció, pitch+ irány nem jó.
TEST_POSITIONS = [
    np.array([-1.0,  0.2, -0.2,  1.2], dtype=np.float64),  # pos#0 DEFAULT (z≈0.87m)
    np.array([-1.3,  0.2, -0.2,  1.0], dtype=np.float64),  # pos#1 pitch−, magasabb (z≈1.05m)
    np.array([-1.5,  0.1, -0.1,  0.8], dtype=np.float64),  # pos#2 pitch−−, könyök nyitottabb (z≈1.21m)
    np.array([-1.2,  0.0,  0.0,  1.0], dtype=np.float64),  # pos#3 neutral roll/yaw (z≈1.05m)
]

# KP értékek, amelyeket sweep-elünk
DEFAULT_KP_VALUES = [10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 500]

# Szimulációs lépések egy mérésnél
SETTLE_STEPS   = 300   # ennyi lépés az stabilizálódáshoz
MEASURE_STEPS  = 50    # az utolsó ennyi lépés átlaga/std az oszcilláció méréshez
DT             = 0.001 # model.opt.timestep (scene XML)
DECIMATION     = 50    # policy step = 50 sim step (env-vel konzisztens)

# Elfogadási küszöbök
# FONTOS (kp_sweep_20260430 tanulság):
#   Z_HAND_MIN = 0.87m — a feladat valós igénye (target z=0.85, kéz kell a fölé).
#   Az eredeti 0.95m küszöb téves volt: pos#0 (DEFAULT) max ~0.87m, fizikailag nem
#   éri el 0.95-öt. pos#1/2/4 viszont komfortosan 1.05-1.22m-t ér el kp=150-nél.
#
#   QACC_MAX = 2000 (csak NaN/Inf detekció) — az eredeti 50.0 téves volt:
#   az első ~250 lépés settling tranziense alatt QACC=300-500+ (normális!),
#   steady-state-ben (step>300) QACC=1-15. A qacc_max-ot csak a measure phase-ban
#   mérjük (lásd measure_one() implementáció).
Z_HAND_MIN     = 0.87   # m — kéz kell stock target fölé (target z=0.85)
OSC_MAX        = 0.005  # m — oszcilláció std küszöb (measure phase)
QACC_MAX       = 2000.0 # — csak NaN/Inf veszély guard (measure phase steady-state)


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def get_hand_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """right_hand_site z koordinátája."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    if site_id < 0:
        # fallback: right_wrist_yaw_link body pozíció
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        return float(data.xpos[body_id][2]) if body_id >= 0 else 0.0
    return float(data.site_xpos[site_id][2])


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
      z_hand_max, z_osc_std, qacc_max

    Returns:
        (z_hand_max, z_osc_std, qacc_max)
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Kar beállítása (kinematikai init)
    data.qpos[ARM_QPOS_START:ARM_QPOS_START + 4] = arm_target
    data.ctrl[ARM_CTRL_INDICES] = arm_target
    mujoco.mj_forward(model, data)

    z_history:    List[float] = []
    qacc_history: List[float] = []

    for step in range(settle_steps + measure_steps):
        data.ctrl[ARM_CTRL_INDICES] = arm_target
        mujoco.mj_step(model, data)

        z = get_hand_z(model, data)
        qacc_cur = float(np.max(np.abs(data.qacc)))

        # NaN/Inf check minden lépésen (NaN terjedés ellen)
        if np.isnan(qacc_cur) or np.isinf(qacc_cur):
            if verbose:
                print(f"    ⚠ NaN/Inf QACC lépés {step}-ben")
            return z, 999.0, 999.0

        # z és QACC csak a measure fázisban rögzítjük (transient kizárva)
        # A settling fázis (step < settle_steps) tranziense QACC=300-500+ — normális,
        # nem instabilitás jele. Steady-state-ben (measure fázis) QACC tipikusan 1-20.
        if step >= settle_steps:
            z_history.append(z)
            qacc_history.append(qacc_cur)

    z_arr = np.array(z_history)
    qacc_max = float(np.max(qacc_history)) if qacc_history else 0.0
    return float(np.max(z_arr)), float(np.std(z_arr)), qacc_max


# ---------------------------------------------------------------------------
# Fő sweep
# ---------------------------------------------------------------------------

def run_sweep(
    kp_values: List[float],
    verbose: bool = True,
) -> List[dict]:
    """
    Végigmegy az összes kp értéken és teszt pozíción.
    Visszaadja a mérési eredményeket dict listában.
    """
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    results = []

    if verbose:
        print(f"\nKP Sweep — arm_motor jelenlegi kp: {model.actuator_gainprm[0, 0]:.0f}")
        print(f"XML: {XML_PATH.name}")
        print(f"Teszt pozíciók: {len(TEST_POSITIONS)}, KP értékek: {kp_values}")
        print("─" * 72)
        print(f"{'kp':>6}  {'pos#':>4}  {'z_max':>7}  {'z_osc':>7}  {'qacc_max':>9}  {'OK?':>5}")
        print("─" * 72)

    for kp in kp_values:
        set_arm_kp(model, kp)

        pos_results = []
        for pos_idx, arm_pos in enumerate(TEST_POSITIONS):
            z_max, z_osc, qacc_max = measure_one(model, arm_pos, verbose=verbose)
            ok = (z_max >= Z_HAND_MIN) and (z_osc < OSC_MAX) and (qacc_max < QACC_MAX)

            row = {
                "kp": kp,
                "pos_idx": pos_idx,
                "z_hand_max": round(z_max, 4),
                "z_osc_std":  round(z_osc, 5),
                "qacc_max":   round(qacc_max, 2),
                "ok":         ok,
                "arm_target": arm_pos.tolist(),
            }
            results.append(row)
            pos_results.append(ok)

            if verbose:
                ok_str = "✅" if ok else "❌"
                print(f"{kp:>6}  {pos_idx:>4}  {z_max:>7.4f}  {z_osc:>7.5f}  {qacc_max:>9.2f}  {ok_str}")

        all_ok = all(pos_results)
        any_ok = any(pos_results)
        if verbose and len(TEST_POSITIONS) > 1:
            summary = "✅ MINDEN OK" if all_ok else ("⚠ RÉSZBEN" if any_ok else "❌ MIND ROSSZ")
            print(f"  → kp={kp}: {summary} ({sum(pos_results)}/{len(TEST_POSITIONS)} pozíció OK)")
            print()

    return results


# ---------------------------------------------------------------------------
# Analízis és javaslat
# ---------------------------------------------------------------------------

def analyze(results: List[dict], verbose: bool = True) -> dict:
    """
    Meghatározza az ajánlott kp értéket:
    a legkisebb kp, amelynél az összes teszt pozíció OK.
    """
    from collections import defaultdict
    by_kp = defaultdict(list)
    for r in results:
        by_kp[r["kp"]].append(r)

    recommended_kp = None
    for kp in sorted(by_kp.keys()):
        rows = by_kp[kp]
        n_ok = sum(r["ok"] for r in rows)
        z_max_all = [r["z_hand_max"] for r in rows]
        z_osc_all = [r["z_osc_std"] for r in rows]
        qacc_all  = [r["qacc_max"] for r in rows]

        all_ok = n_ok == len(rows)
        if all_ok and recommended_kp is None:
            recommended_kp = kp

    summary = {
        "recommended_kp": recommended_kp,
        "current_kp":     10,  # XML-ből
        "z_hand_target":  Z_HAND_MIN,
    }

    if verbose:
        print("─" * 72)
        if recommended_kp:
            print(f"\n✅ AJÁNLOTT kp: {recommended_kp}")
            print(f"   (jelenlegi: 10 → szorzó: {recommended_kp/10:.0f}×)")
            print(f"\n   scene_manip_sandbox_v2.xml módosítás:")
            print(f"   <default class=\"arm_motor\">")
            print(f"     <position kp=\"{recommended_kp}\" forcerange=\"-15 15\" ctrlrange=\"-3.14 3.14\"/>")
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
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kp", "pos_idx", "z_hand_max", "z_osc_std", "qacc_max", "ok"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["kp", "pos_idx", "z_hand_max", "z_osc_std", "qacc_max", "ok"]})
    print(f"CSV mentve: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Arm kp gain sweep diagnosztika")
    parser.add_argument("--kp-values", type=float, nargs="+", default=DEFAULT_KP_VALUES,
                        help=f"Tesztelendő kp értékek (default: {DEFAULT_KP_VALUES})")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV kimenet útvonala (opcionális)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true",
                        help="Csak az ajánlott kp értéket írja ki")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    t0 = time.time()
    results = run_sweep(kp_values=args.kp_values, verbose=verbose)
    summary = analyze(results, verbose=verbose)

    if verbose:
        print(f"Futási idő: {time.time()-t0:.1f}s")

    if args.out:
        save_csv(results, Path(args.out))
    elif not args.quiet:
        default_out = _REPO_ROOT / f"results/diag/kp_sweep_{time.strftime('%Y%m%d_%H%M')}.csv"
        save_csv(results, default_out)

    if args.quiet:
        if summary["recommended_kp"]:
            print(summary["recommended_kp"])
        else:
            print("NONE")
            sys.exit(1)


if __name__ == "__main__":
    main()
