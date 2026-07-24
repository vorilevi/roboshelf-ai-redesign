"""
Booster T1 — jobb kar előrenyúlási (forward reach) diagnosztika.
Phase: T1 Track v1 (multi-robot stratégia, 2026-07)

Célja:
  1. Megmérni a T1 jobb kar maximális x-irányú előrenyúlását
     (meghatározza, hogy x=0.30m elérhető-e a push task targetként)
  2. Megtalálni a legjobb DEFAULT_ARM_POS értékeket (4 joint szög),
     amelyek kielégítik:
       - hand_x maximális (minél közelebb a push targethez)
       - hand_z ≈ 0.77m ± 0.08m (push target magassága)
       - stabilak: z_osc_std < 0.005m
  3. Validálni: kp=150 (G1-ről örökölve) működik-e T1-en

Koordinátarendszer:
  - Robot törzs (Trunk) z=0.665m, x=0, y=0
  - Push target: x=0.30, y=0, z=0.77m
  - Jobb váll világ pozíciója (kp=0 qpos): (0.0575, -0.1063, 0.665+0.219) = (0.058, -0.106, 0.884)
  - Kar irány rest pozícióban: -y (kar lóg oldalra), pitch rotáció → +x irányba fordul

Joint ↔ ctrl megfeleltetés (ARM_CTRL_INDICES = [0,1,2,3]):
  ctrl[0] = Right_Shoulder_Pitch  (qposadr=6, axis 0 1 0)
  ctrl[1] = Right_Shoulder_Roll   (qposadr=7, axis 1 0 0)
  ctrl[2] = Right_Elbow_Pitch     (qposadr=8, axis 0 1 0)
  ctrl[3] = Right_Elbow_Yaw       (qposadr=9, axis 0 0 1)

Különbség G1-től:
  G1:  Shoulder Pitch/Roll/YAW + Elbow Pitch
  T1:  Shoulder Pitch/Roll + Elbow PITCH/YAW  (yaw a KÖNYÖKNÉL van!)

Futtatás (repo gyökeréből):
  python3 tools/diag_arm_reach_t1.py
  python3 tools/diag_arm_reach_t1.py --out results/diag/t1_reach_$(date +%Y%m%d).csv
  python3 tools/diag_arm_reach_t1.py --top 10 --verbose
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
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

# Kar actuator ctrl indexek (scene XML actuator sorrendben)
ARM_CTRL_INDICES = [0, 1, 2, 3]

# T1 kar joint qpos indexek (Python-ból ellenőrizve: 6,7,8,9)
ARM_QPOS_INDICES = [6, 7, 8, 9]  # Right_Shoulder_Pitch, Roll, Elbow_Pitch, Elbow_Yaw

# Push task cél koordináták
TARGET_X = 0.30   # storage asztal x (scene_manip_sandbox_t1_v1.xml)
TARGET_Z = 0.77   # push target z (G1-gyel megegyező)
Z_TOLERANCE = 0.08  # ±8cm tolerancia a push height körül

# Szimulációs lépések
SETTLE_STEPS  = 300  # stabilizálódási idő
MEASURE_STEPS = 50   # az utolsó N lépés átlaga méréshez

# Stabilitási küszöb
OSC_MAX = 0.008  # m — oszcilláció std (T1-re lazább: +0.003m vs G1 0.005m)

# ---------------------------------------------------------------------------
# Grid keresési tartományok
# ---------------------------------------------------------------------------
# T1 joint ranges:
#   Right_Shoulder_Pitch: -3.31 to 1.22
#   Right_Shoulder_Roll:  -1.57 to 1.74
#   Right_Elbow_Pitch:    -2.27 to 2.27
#   Right_Elbow_Yaw:       0.00 to 2.44
#
# A shoulder_pitch negatív értékei forgatják a kart előre (+x irány felé).
# Az optimum keresési tartomány a kar kinematikája alapján:

GRID = {
    "pitch":      np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5,  0.0]),   # 7 érték
    "roll":       np.array([-0.5,  0.0,  0.5,  1.0,  1.5]),                # 5 érték
    "elbow_p":    np.array([-1.0, -0.5,  0.0,  0.5,  1.0,  1.5,  2.0]),   # 7 érték
    "elbow_yaw":  np.array([ 0.0,  0.5,  1.0,  1.5,  2.0]),               # 5 érték
}
# Összesen: 7 × 5 × 7 × 5 = 1225 kombináció

# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def get_hand_pos(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[float, float, float]:
    """right_hand_site (x, y, z) világ koordinátái."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    if site_id < 0:
        # fallback: right_hand_link body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link")
        if body_id >= 0:
            p = data.xpos[body_id]
            return float(p[0]), float(p[1]), float(p[2])
        return 0.0, 0.0, 0.0
    p = data.site_xpos[site_id]
    return float(p[0]), float(p[1]), float(p[2])


def simulate_arm(
    model: mujoco.MjModel,
    arm_target: np.ndarray,
) -> dict:
    """
    Adott arm joint target-hez méri:
      hand_x, hand_z (settle utáni átlag), x_osc_std, z_osc_std, qacc_max, stable

    Returns:
        dict with keys: hand_x, hand_z, x_osc, z_osc, qacc_max, stable
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Kar inicializálása a target pozícióba
    for i, idx in enumerate(ARM_QPOS_INDICES):
        data.qpos[idx] = arm_target[i]
    data.ctrl[ARM_CTRL_INDICES] = arm_target
    mujoco.mj_forward(model, data)

    x_hist: List[float] = []
    z_hist: List[float] = []
    qacc_hist: List[float] = []

    for step in range(SETTLE_STEPS + MEASURE_STEPS):
        data.ctrl[ARM_CTRL_INDICES] = arm_target
        mujoco.mj_step(model, data)

        qacc_cur = float(np.max(np.abs(data.qacc)))
        if np.isnan(qacc_cur) or np.isinf(qacc_cur):
            return {"hand_x": 0.0, "hand_z": 0.0, "x_osc": 9.9,
                    "z_osc": 9.9, "qacc_max": 9999.0, "stable": False}

        if step >= SETTLE_STEPS:
            hx, _, hz = get_hand_pos(model, data)
            x_hist.append(hx)
            z_hist.append(hz)
            qacc_hist.append(qacc_cur)

    x_arr = np.array(x_hist)
    z_arr = np.array(z_hist)
    hand_x   = float(np.mean(x_arr))
    hand_z   = float(np.mean(z_arr))
    x_osc    = float(np.std(x_arr))
    z_osc    = float(np.std(z_arr))
    qacc_max = float(np.max(qacc_hist)) if qacc_hist else 0.0
    stable   = (x_osc < OSC_MAX) and (z_osc < OSC_MAX) and (qacc_max < 2000.0)

    return {
        "hand_x":   hand_x,
        "hand_z":   hand_z,
        "x_osc":    x_osc,
        "z_osc":    z_osc,
        "qacc_max": qacc_max,
        "stable":   stable,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(verbose: bool = True) -> List[dict]:
    """
    Végigmegy az összes joint kombináción és megméri az elérési adatokat.
    """
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    combinations = list(product(
        GRID["pitch"],
        GRID["roll"],
        GRID["elbow_p"],
        GRID["elbow_yaw"],
    ))
    n_total = len(combinations)

    if verbose:
        print(f"\n{'='*72}")
        print(f"  Booster T1 — Kar előrenyúlás diagnosztika")
        print(f"  Push target: x={TARGET_X:.2f}m, z={TARGET_Z:.2f}m (±{Z_TOLERANCE:.2f}m)")
        print(f"  Grid kombinációk: {n_total}")
        print(f"  kp=150 (G1-ről örökölve — validálás folyamatban)")
        print(f"{'='*72}")
        print(f"  {'#':>5} {'pitch':>7} {'roll':>7} {'elb_p':>7} {'elb_y':>7} "
              f"{'hand_x':>8} {'hand_z':>8} {'x_osc':>7} {'OK?':>5}")
        print(f"  {'-'*72}")

    results = []
    t0 = time.time()

    for i, (pitch, roll, elbow_p, elbow_yaw) in enumerate(combinations):
        arm_target = np.array([pitch, roll, elbow_p, elbow_yaw], dtype=np.float64)
        m = simulate_arm(model, arm_target)

        # z a push target közelében van-e?
        z_ok = abs(m["hand_z"] - TARGET_Z) < Z_TOLERANCE
        # x elég közel van-e a push targethez?
        x_ok = m["hand_x"] >= (TARGET_X - 0.05)  # legalább 5cm-en belül
        candidate = m["stable"] and z_ok and x_ok

        row = {
            "pitch":    round(pitch, 2),
            "roll":     round(roll, 2),
            "elbow_p":  round(elbow_p, 2),
            "elbow_yaw": round(elbow_yaw, 2),
            "hand_x":   round(m["hand_x"], 4),
            "hand_z":   round(m["hand_z"], 4),
            "x_osc":    round(m["x_osc"], 5),
            "z_osc":    round(m["z_osc"], 5),
            "qacc_max": round(m["qacc_max"], 2),
            "stable":   m["stable"],
            "z_ok":     z_ok,
            "x_ok":     x_ok,
            "candidate": candidate,
        }
        results.append(row)

        if verbose and (candidate or (i % 100 == 0)):
            flag = "✅" if candidate else ("⚠" if (m["stable"] and x_ok) else "  ")
            print(f"  {i:>5} {pitch:>7.1f} {roll:>7.1f} {elbow_p:>7.1f} {elbow_yaw:>7.1f} "
                  f"{m['hand_x']:>8.4f} {m['hand_z']:>8.4f} {m['x_osc']:>7.5f} {flag}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Futási idő: {elapsed:.1f}s ({n_total/elapsed:.0f} kombo/s)")

    return results


# ---------------------------------------------------------------------------
# Analízis
# ---------------------------------------------------------------------------

def analyze(results: List[dict], top_n: int = 5, verbose: bool = True) -> dict:
    """
    Összefoglalja a grid search eredményeit:
    - Maximum x elérés (bármely z-nél)
    - Legjobb kandidátusok (stable, z≈0.77, max x)
    - DEFAULT_ARM_POS ajánlás
    """
    candidates = [r for r in results if r["candidate"]]
    stable_any = [r for r in results if r["stable"]]
    max_x_ever = max((r["hand_x"] for r in results), default=0.0)
    max_x_stable = max((r["hand_x"] for r in stable_any), default=0.0)

    candidates_sorted = sorted(candidates, key=lambda r: r["hand_x"], reverse=True)

    if verbose:
        print(f"\n{'='*72}")
        print(f"  EREDMÉNY ÖSSZEFOGLALÓ")
        print(f"{'='*72}")
        print(f"  Grid kombinációk:          {len(results)}")
        print(f"  Stabil konfigurációk:      {len(stable_any)}")
        print(f"  Jelölt konfigurációk:      {len(candidates)}  "
              f"(stabil + z≈{TARGET_Z}±{Z_TOLERANCE} + x≥{TARGET_X-0.05:.2f})")
        print(f"")
        print(f"  Max x elérés (bármely z):  {max_x_ever:.4f}m")
        print(f"  Max x elérés (stabil):     {max_x_stable:.4f}m")
        print(f"  Push target x:             {TARGET_X:.2f}m")

        if max_x_stable < TARGET_X - 0.05:
            print(f"\n  ⚠ FIGYELEM: A kar maximális stabil előrenyúlása ({max_x_stable:.3f}m) "
                  f"kevesebb, mint a push target ({TARGET_X:.2f}m) - 5cm toleranciával!")
            print(f"  → Fontold meg a storage asztal x={max_x_stable - 0.03:.2f}m-re való közelítését.")
        else:
            print(f"\n  ✅ x={TARGET_X:.2f}m elérhető! (max stabil: {max_x_stable:.4f}m)")

        print(f"\n  TOP {min(top_n, len(candidates_sorted))} JELÖLT "
              f"(x csökkenő sorrendben, stabil + z≈{TARGET_Z}m):")
        print(f"  {'#':>3} {'pitch':>7} {'roll':>7} {'elb_p':>7} {'elb_y':>7} "
              f"{'hand_x':>8} {'hand_z':>8}")
        print(f"  {'-'*60}")
        for j, r in enumerate(candidates_sorted[:top_n]):
            print(f"  {j+1:>3} {r['pitch']:>7.1f} {r['roll']:>7.1f} {r['elbow_p']:>7.1f} "
                  f"{r['elbow_yaw']:>7.1f} {r['hand_x']:>8.4f} {r['hand_z']:>8.4f}")

        if candidates_sorted:
            best = candidates_sorted[0]
            print(f"\n  ✅ AJÁNLOTT DEFAULT_ARM_POS (T1):")
            print(f"  [{best['pitch']}, {best['roll']}, {best['elbow_p']}, {best['elbow_yaw']}]")
            print(f"  → hand_x={best['hand_x']:.4f}m, hand_z={best['hand_z']:.4f}m")
            print(f"\n  g1_shelf_stock_t1_env.py-ban:")
            print(f"  DEFAULT_ARM_POS = np.array([{best['pitch']}, {best['roll']}, "
                  f"{best['elbow_p']}, {best['elbow_yaw']}], dtype=np.float64)")
        else:
            print(f"\n  ❌ Nem találtunk jelölt konfigurációt.")
            print(f"     Legjobb stabil x elérés: {max_x_stable:.4f}m")
            print(f"     Javasolt: csökkentsd a storage asztal x értékét "
                  f"≤ {max_x_stable - 0.02:.2f}m-re a scene XML-ben.")

        print(f"{'='*72}\n")

    return {
        "max_x_ever":       max_x_ever,
        "max_x_stable":     max_x_stable,
        "n_candidates":     len(candidates),
        "top_candidate":    candidates_sorted[0] if candidates_sorted else None,
        "recommended_pos":  (
            [candidates_sorted[0]["pitch"], candidates_sorted[0]["roll"],
             candidates_sorted[0]["elbow_p"], candidates_sorted[0]["elbow_yaw"]]
            if candidates_sorted else None
        ),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pitch", "roll", "elbow_p", "elbow_yaw",
              "hand_x", "hand_z", "x_osc", "z_osc", "qacc_max",
              "stable", "z_ok", "x_ok", "candidate"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"CSV mentve: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Booster T1 kar előrenyúlás diagnosztika (DEFAULT_ARM_POS meghatározás)"
    )
    parser.add_argument("--out", type=str, default=None,
                        help="CSV kimenet útvonala (default: results/diag/t1_reach_YYYYMMDD.csv)")
    parser.add_argument("--top", type=int, default=5,
                        help="Hány top jelöltet mutasson (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Csak az ajánlott pozíciót írja ki")
    args = parser.parse_args()

    verbose = not args.quiet

    results = run_grid_search(verbose=verbose)
    summary = analyze(results, top_n=args.top, verbose=verbose)

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / f"results/diag/t1_reach_{time.strftime('%Y%m%d_%H%M')}.csv"
    )
    save_csv(results, out_path)

    if args.quiet:
        pos = summary.get("recommended_pos")
        if pos:
            print(pos)
        else:
            print(f"NO_CANDIDATE — max_x_stable={summary['max_x_stable']:.4f}")
            sys.exit(1)


if __name__ == "__main__":
    main()
