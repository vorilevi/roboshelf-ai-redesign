"""
GR1T1 arm reach diagnosztika + kp sweep — Track 3 vendor-independence.

Két fázis:
  1. REACH SEARCH — 4D grid search a jobb kar joint szögein keresztül.
     Megtalálja azokat a konfigurációkat, ahol a right_hand_site eléri
     a storage asztal pozícióját (x≈0.45m, z≈0.77m).

  2. KP SWEEP — A legjobb reach konfiguráción kp stability teszt.
     Validálja, hogy a G1-ről örökölt kp=150 megfelelő-e a GR1T1 karhoz.

Elfogadás (kp sweep):
  hand_x_mean ≥ 0.40m  — eléri a tárolót (target x=0.45m, tolerancia ±0.05m)
  x_osc_std   < 0.008m — stabil tartás
  qacc_max    < 2000   — nincs NaN/Inf veszély

Futtatás (repo gyökeréből):
  python3 tools/diag_arm_reach_gr1.py
  python3 tools/diag_arm_reach_gr1.py --skip-search          # csak kp sweep
  python3 tools/diag_arm_reach_gr1.py --kp-values 50 100 150 200

Robot: Fourier GR1T1 (Track 3 — harmadik gyártó)
  G1 (Unitree, 80% SR) → T1 (Booster, 86% SR) → GR1T1 (Fourier, TBD)

Kapcsolódó fájlok:
  scene:  src/envs/assets/scene_manip_sandbox_gr1_v1.xml
  meshek: mujoco_menagerie/fourier_gr1t1/assets/  (setup_gr1_assets.sh)
  env:    src/roboshelf_ai/mujoco/envs/manipulation/gr1_shelf_stock_env.py  (TODO)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mujoco

# ─── Elérési utak ────────────────────────────────────────────────────────────
XML_PATH = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_gr1_v1.xml"

# ─── Ctrl / qpos indexek ─────────────────────────────────────────────────────
# ctrl[0] → right_shoulder_pitch_joint  → qpos[22]
# ctrl[1] → right_shoulder_roll_joint   → qpos[23]
# ctrl[2] → right_shoulder_yaw_joint    → qpos[24]
# ctrl[3] → right_elbow_pitch_joint     → qpos[25]
ARM_CTRL_INDICES = [0, 1, 2, 3]
ARM_QPOS_INDICES = [22, 23, 24, 25]

# ─── Push task target ────────────────────────────────────────────────────────
# Storage asztal: x=0.45m (GR1 arm ~0.42m, G1-gyel azonos)
# Termék settled: z=0.770m
# Cél: a kéz x ≥ 0.40m (±0.05m tolerancia), z a [0.65, 0.90] ablakban
X_TARGET       = 0.45    # asztal x pozíciója
Z_TARGET       = 0.77    # termék magassága
X_HAND_MIN     = 0.40    # kp sweep elfogadás küszöb (m)
Z_WINDOW       = (0.60, 0.95)  # elfogadható kézmagasság (m)
OSC_MAX        = 0.008   # oszcilláció std küszöb (m)
QACC_MAX       = 2000.0  # qacc guard

# ─── Szimulációs lépések ─────────────────────────────────────────────────────
REACH_SETTLE   = 500     # reach search settle (lépések)
SWEEP_SETTLE   = 300     # kp sweep settle
SWEEP_MEASURE  = 50      # kp sweep mérési fázis

# ─── Grid search tartományok ─────────────────────────────────────────────────
# GR1T1 jobb kar joint tartományok:
#   shoulder_pitch: [-2.79, 1.92] — negatív = előre nyúlás
#   shoulder_roll:  [-3.27, 0.57] — pozitív = befelé
#   shoulder_yaw:   [-2.97, 2.97] — forgás
#   elbow_pitch:    [-2.27, 2.27] — negatív = hajlítás
GRID_PITCH = np.linspace(-2.5,  1.5, 9)    # széles tartomány, GR1 kinematika nem triviális
GRID_ROLL  = np.linspace(-1.5,  0.5, 5)
GRID_YAW   = np.array([-0.5, 0.0, 0.5])
GRID_ELBOW = np.linspace(-2.0,  0.5, 7)   # 7 lépés

# ─── kp értékek ──────────────────────────────────────────────────────────────
DEFAULT_KP_VALUES = [10, 30, 50, 100, 150, 200, 300, 400]


# ─────────────────────────────────────────────────────────────────────────────
# Segédfüggvények
# ─────────────────────────────────────────────────────────────────────────────

def get_hand_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """right_hand_site 3D pozíciója (x=push irány, z=magasság)."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    return data.site_xpos[sid].copy()


def set_arm_kp(model: mujoco.MjModel, kp: float) -> None:
    for i in ARM_CTRL_INDICES:
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp


def simulate_to_target(
    model: mujoco.MjModel,
    arm_target: np.ndarray,
    settle_steps: int,
) -> mujoco.MjData:
    """Resetel és szimulálja a settle_steps lépést a megadott kar célpozícióval."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for i, qi in enumerate(ARM_QPOS_INDICES):
        data.qpos[qi] = arm_target[i]
    for i in ARM_CTRL_INDICES:
        data.ctrl[i] = arm_target[i]
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        for i in ARM_CTRL_INDICES:
            data.ctrl[i] = arm_target[i]
        mujoco.mj_step(model, data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# FÁZIS 1 — Reach search
# ─────────────────────────────────────────────────────────────────────────────

def run_reach_search(verbose: bool = True) -> Optional[np.ndarray]:
    """
    4D grid search: megtalálja a push task-hoz legjobb kar konfigurációt.

    Rangsorolás: min(|hand_x - X_TARGET|) ahol Z_WINDOW[0] ≤ hand_z ≤ Z_WINDOW[1]
    """
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    configs = list(product(GRID_PITCH, GRID_ROLL, GRID_YAW, GRID_ELBOW))
    total = len(configs)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FÁZIS 1 — Reach Search ({total} konfiguráció)")
        print(f"  Target: x≈{X_TARGET}m, z≈{Z_TARGET}m")
        print(f"  Z ablak: {Z_WINDOW[0]}–{Z_WINDOW[1]}m")
        print(f"  Settle: {REACH_SETTLE} lépés/konfiguráció")
        print(f"{'='*70}")

    best_pos   = None
    best_dist  = 1e9
    candidates = []

    t0 = time.time()
    for idx, (pitch, roll, yaw, elbow) in enumerate(configs):
        arm = np.array([pitch, roll, yaw, elbow])
        data = simulate_to_target(model, arm, REACH_SETTLE)
        hp = get_hand_pos(model, data)

        x_dist = abs(hp[0] - X_TARGET)
        z_ok   = Z_WINDOW[0] <= hp[2] <= Z_WINDOW[1]

        if z_ok:
            candidates.append({
                "arm": arm,
                "hand_x": round(float(hp[0]), 4),
                "hand_z": round(float(hp[2]), 4),
                "x_dist": round(x_dist, 4),
            })
            if x_dist < best_dist:
                best_dist = x_dist
                best_pos  = arm.copy()

        if verbose and (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  [{idx+1:4d}/{total}] legjobb x_dist={best_dist:.4f}m  "
                  f"({len(candidates)} jelölt)  ETA {eta:.0f}s")

    # Top 5 kiírása
    if verbose and candidates:
        candidates.sort(key=lambda c: c["x_dist"])
        print(f"\n  Top 5 konfiguráció (rangsorolva x_dist alapján):")
        print(f"  {'pitch':>7} {'roll':>7} {'yaw':>7} {'elbow':>7}  "
              f"{'hand_x':>8} {'hand_z':>8} {'x_dist':>8}")
        print(f"  {'-'*65}")
        for c in candidates[:5]:
            a = c["arm"]
            mark = " ← LEGJOBB" if np.allclose(a, best_pos) else ""
            print(f"  {a[0]:>7.3f} {a[1]:>7.3f} {a[2]:>7.3f} {a[3]:>7.3f}  "
                  f"{c['hand_x']:>8.4f} {c['hand_z']:>8.4f} {c['x_dist']:>8.4f}{mark}")

    if best_pos is not None and verbose:
        print(f"\n  ✅ Legjobb konfiguráció:")
        print(f"     DEFAULT_ARM_POS = np.array([{best_pos[0]:.3f}, {best_pos[1]:.3f}, "
              f"{best_pos[2]:.3f}, {best_pos[3]:.3f}])")
        print(f"     Elért kézpozíció: x={candidates[0]['hand_x']}m, "
              f"z={candidates[0]['hand_z']}m")
    elif verbose:
        print("\n  ❌ Nem találtam Z-ablakban lévő konfigurációt!")
        print("     Próbálkozz szélesebb grid tartománnyal.")

    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# FÁZIS 2 — kp sweep
# ─────────────────────────────────────────────────────────────────────────────

def measure_one(
    model: mujoco.MjModel,
    arm_target: np.ndarray,
    settle_steps: int = SWEEP_SETTLE,
    measure_steps: int = SWEEP_MEASURE,
) -> Tuple[float, float, float, float]:
    """Mér: (hand_x_mean, hand_z_mean, x_osc_std, qacc_max) a measure fázisban."""
    data = simulate_to_target(model, arm_target, settle_steps)

    x_hist: List[float] = []
    z_hist: List[float] = []
    q_hist: List[float] = []

    for _ in range(measure_steps):
        for i in ARM_CTRL_INDICES:
            data.ctrl[i] = arm_target[i]
        mujoco.mj_step(model, data)
        hp = get_hand_pos(model, data)
        qa = float(np.max(np.abs(data.qacc)))
        if np.isnan(qa) or np.isinf(qa):
            return 0.0, 0.0, 999.0, 999.0
        x_hist.append(hp[0])
        z_hist.append(hp[2])
        q_hist.append(qa)

    return (
        float(np.mean(x_hist)),
        float(np.mean(z_hist)),
        float(np.std(x_hist)),
        float(max(q_hist)),
    )


def run_kp_sweep(
    arm_pos: np.ndarray,
    kp_values: List[float],
    verbose: bool = True,
) -> List[dict]:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"FÁZIS 2 — kp Sweep")
        print(f"  Kar konfiguráció: pitch={arm_pos[0]:.3f}  roll={arm_pos[1]:.3f}  "
              f"yaw={arm_pos[2]:.3f}  elbow={arm_pos[3]:.3f}")
        print(f"  Elfogadás: hand_x ≥ {X_HAND_MIN}m, osc < {OSC_MAX}m, "
              f"qacc < {QACC_MAX}")
        print(f"{'='*70}")
        print(f"  {'kp':>6}  {'hand_x':>8}  {'hand_z':>8}  {'x_osc':>8}  "
              f"{'qacc_max':>9}  {'OK?':>5}")
        print(f"  {'─'*60}")

    for kp in kp_values:
        set_arm_kp(model, kp)
        x_mean, z_mean, x_osc, qacc_max = measure_one(model, arm_pos)
        z_ok = Z_WINDOW[0] <= z_mean <= Z_WINDOW[1]
        ok = (x_mean >= X_HAND_MIN) and (x_osc < OSC_MAX) and (qacc_max < QACC_MAX) and z_ok

        row = {
            "kp": kp,
            "hand_x_mean": round(x_mean, 4),
            "hand_z_mean": round(z_mean, 4),
            "x_osc_std":  round(x_osc, 5),
            "qacc_max":   round(qacc_max, 2),
            "ok": ok,
        }
        results.append(row)

        if verbose:
            ok_str = "✅" if ok else "❌"
            print(f"  {kp:>6.0f}  {x_mean:>8.4f}  {z_mean:>8.4f}  "
                  f"{x_osc:>8.5f}  {qacc_max:>9.2f}  {ok_str}")

    return results


def analyze_sweep(results: List[dict], verbose: bool = True) -> dict:
    """Ajánlott kp meghatározása — ha kp=150 átmegy, azt tartjuk."""
    G1_KP = 150
    passing = [r["kp"] for r in results if r["ok"]]
    g1_ok   = G1_KP in passing
    rec_kp  = G1_KP if g1_ok else (min(passing) if passing else None)

    if verbose:
        print(f"\n{'─'*70}")
        if g1_ok:
            print(f"\n✅ kp=150 VALIDÁLT GR1T1-en — G1-gyel azonos, XML nem módosítandó.")
            if passing and min(passing) < G1_KP:
                print(f"   (kp={min(passing)} is átmenne statikus tesztben, "
                      f"de kp=150 push kontaktnál robusztusabb.)")
        elif rec_kp:
            print(f"\n⚠  kp=150 SIKERTELEN. Ajánlott: kp={rec_kp}")
            print(f"\n   scene_manip_sandbox_gr1_v1.xml módosítandó:")
            print(f'   <default class="arm_motor">')
            print(f'     <position kp="{rec_kp}" forcerange="-20 20" ctrlrange="-3.14 3.14"/>')
            print(f'   </default>')
        else:
            print("\n❌ Egyik kp sem teljesíti a feltételeket!")
            print("   → Ellenőrizd a forcerange értékeket vagy a reach konfigurációt.")
        print()

    return {
        "recommended_kp":   rec_kp,
        "g1_kp_validated":  g1_ok,
        "passing_kp_values": passing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kp", "hand_x_mean", "hand_z_mean", "x_osc_std", "qacc_max", "ok"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"CSV mentve: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GR1T1 arm reach diagnosztika + kp sweep (Track 3)"
    )
    parser.add_argument("--skip-search", action="store_true",
                        help="Kihagyja a reach search-öt, csak kp sweep-et futtat")
    parser.add_argument("--arm-pos", type=float, nargs=4,
                        metavar=("PITCH", "ROLL", "YAW", "ELBOW"),
                        help="Kézi kar konfiguráció (ha --skip-search aktív)")
    parser.add_argument("--kp-values", type=float, nargs="+",
                        default=DEFAULT_KP_VALUES)
    parser.add_argument("--out", type=str, default=None,
                        help="CSV kimenet útvonala")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    t0 = time.time()

    if verbose:
        print("\nGR1T1 Arm Reach Diagnosztika — Track 3 (Fourier Intelligence)")
        print(f"XML: {XML_PATH.name}")
        print(f"Target: x={X_TARGET}m, z={Z_TARGET}m")

    # Fázis 1: reach search
    if args.skip_search:
        if args.arm_pos:
            best = np.array(args.arm_pos)
        else:
            # Validált optimum (reach search eredménye, 2026-07-26):
            # hand_x=0.453m (target=0.45m), hand_z=0.879m, kp=150 ✅
            best = np.array([-1.000, 0.000, 0.500, 0.083])
            if verbose:
                print(f"\n  Validált DEFAULT_ARM_POS: {best}")
    else:
        best = run_reach_search(verbose=verbose)
        if best is None:
            print("\n❌ Reach search sikertelen — diag leáll.")
            sys.exit(1)

    # Fázis 2: kp sweep
    sweep_results = run_kp_sweep(best, args.kp_values, verbose=verbose)
    summary = analyze_sweep(sweep_results, verbose=verbose)

    if verbose:
        print(f"Teljes futási idő: {time.time()-t0:.1f}s\n")

    out_path = Path(args.out) if args.out else \
        _REPO_ROOT / f"results/diag/gr1_arm_diag_{time.strftime('%Y%m%d_%H%M')}.csv"
    save_csv(sweep_results, out_path)

    if args.quiet:
        rec = summary["recommended_kp"]
        print(rec if rec else "NONE")
        if not rec:
            sys.exit(1)


if __name__ == "__main__":
    main()
