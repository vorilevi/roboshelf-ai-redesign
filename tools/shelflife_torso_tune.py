"""
shelflife_torso_tune.py — M0: a törzs vezérlési paramétereinek bemérése

    python3 tools/shelflife_torso_tune.py

────────────────────────────────────────────────────────────────────────────
A MÉRT PROBLÉMA
────────────────────────────────────────────────────────────────────────────
A törzs bevonása a vezérlési láncba feloldotta az ízülethatár-problémát
(tartalék 0.05–0.14 rad → 0.33 rad), de új hibát hozott:

    ízület             parancsolt    beállt     eltérés
    torso_yaw            −0.550      −0.413     +0.138 rad
    r_shoulder_pitch     +0.169      +0.170     +0.001 rad
    r_elbow              −1.687      −1.686     +0.002 rad

A KARÍZÜLETEK ezredrad pontosan követnek, a TÖRZS 0.138 rad-ot hibázik — és
mivel a törzs a teljes felsőtestet forgatja, ez a kéznél **16 cm**. A hiba
8000 lépés után is csak 46 mm-re csökkent, tehát nem lengés, hanem tartós
alul-vezérlés.

Ok: a `TORSO_KP = 2000` a felsőtest tehetetlenségéhez képest gyenge, és a
törzs a `gene_arm` osztály csillapítását/armatúráját örökölte, ami karra való.

────────────────────────────────────────────────────────────────────────────
MIÉRT IN-PROCESS SWEEP
────────────────────────────────────────────────────────────────────────────
A jelenet újraépítése paraméterenként ~10 s, és a mesh-másolás miatt zajos.
A pozíció-aktuátor erősítése viszont futásidőben állítható:

    actuator_gainprm[i, 0] =  kp
    actuator_biasprm[i, 1] = -kp        (affin bias: −kp·q)
    dof_damping[dof], dof_armature[dof]

Így egyetlen betöltött modellen végigmehet az egész rács. A nyertes értékek
utána kerülnek be a builderbe — a sweep NEM módosít fájlt.

────────────────────────────────────────────────────────────────────────────
KILÉPÉSI FELTÉTEL (a projektterv M0 pontja)
────────────────────────────────────────────────────────────────────────────
  · torso_yaw követési hiba          < 0.010 rad
  · tenyér-hiba a kinematikaihoz     < 5 mm 2000 lépésen belül
  · max |qacc| nem robban            (a kiindulási nagyságrendben marad)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                     # noqa: E402
import shelflife_grasp as G                       # noqa: E402

TORSO = ["torso_yaw", "torso_roll"]

KP_GRID = (2_000.0, 10_000.0, 30_000.0, 100_000.0, 300_000.0)
DAMP_GRID = (10.0, 100.0, 500.0)
ARMATURE = 0.5            # törzsre való tehetetlenség-kiegészítés
CHECKPOINTS = (500, 1000, 2000, 4000)

TOL_RAD = 0.010
TOL_MM = 5.0


def set_torso_gains(r, kp: float, damping: float, armature: float) -> None:
    m = r.model
    for n in TORSO:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}")
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if aid < 0 or jid < 0:
            continue
        m.actuator_gainprm[aid, 0] = kp
        m.actuator_biasprm[aid, 1] = -kp
        dof = m.jnt_dofadr[jid]
        m.dof_damping[dof] = damping
        m.dof_armature[dof] = armature


def kinematic_grasp_point(r, q: np.ndarray) -> np.ndarray:
    s = r._scratch
    mujoco.mj_resetData(r.model, s)
    s.qpos[np.array(r._arm_q)] = q
    r.set_hand_qpos(s, 0.0)
    mujoco.mj_forward(r.model, s)
    P = s.xpos[r._palm]
    R = s.xmat[r._palm].reshape(3, 3)
    return P + R @ r._grasp_offset


def trial(r, q: np.ndarray, target_kin: np.ndarray) -> dict:
    """Parancs kiadása, majd a beállás követése ellenőrzőpontokon."""
    r.reset()
    r.close_fingers(0.0, settle=1)
    r._cmd = q.copy()
    for a, v in zip(r._arm_a, r._cmd):
        r.data.ctrl[a] = v

    yaw_j = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_JOINT, "torso_yaw")
    yaw_adr = r.model.jnt_qposadr[yaw_j]
    yaw_cmd = q[r.chain.index("torso_yaw")]

    out, done = {}, 0
    qacc_max = 0.0
    for cp in CHECKPOINTS:
        r.step(cp - done); done = cp
        qacc_max = max(qacc_max, float(np.abs(r.data.qacc).max()))
        out[cp] = {
            "yaw_err": abs(float(r.data.qpos[yaw_adr]) - float(yaw_cmd)),
            "palm_mm": float(np.linalg.norm(
                r.grasp_point() - target_kin)) * 1000,
        }
    out["qacc_max"] = qacc_max
    return out


def main() -> int:
    print("Shelf Life M0 — a törzs vezérlésének bemérése\n")
    r = G.GraspRobot()
    if "torso_yaw" not in r.chain:
        sys.exit("a jelenetben nincs aktív torso_yaw — futtasd a "
                 "shelflife_build_scene_sku.py-t")

    plan = r.plan
    R_des = G.GRASP_POSES[plan["pose"] if plan else "right_thumb_up"]
    box, _ = r.product_box()
    ax, sg = (plan["approach_palm_axis"] if plan else (2, -1))
    pre = box - sg * R_des[:, ax] * 0.12

    q, ep, er = r.ik6_seed(pre, R_des, restarts=16, iters=110)
    tgt = kinematic_grasp_point(r, q)
    print(f"  próbapóz: pre-grasp, IK {ep*1000:.1f} mm / {np.degrees(er):.1f}°")
    print(f"  kinematikai fogási pont: {np.round(tgt, 4)}")
    print(f"  ízülettartalék: {r.joint_margin(q):.2f} rad\n")

    print(f"{'kp':>9}{'csill.':>8}  "
          f"{'yaw-hiba (rad) 500/1k/2k/4k':^34}  "
          f"{'tenyér (mm) 500/1k/2k/4k':^30}{'qacc':>10}")
    print("─" * 96)

    best = None
    for kp in KP_GRID:
        for damp in DAMP_GRID:
            set_torso_gains(r, kp, damp, ARMATURE)
            t = trial(r, q, tgt)
            ys = "/".join(f"{t[c]['yaw_err']:.3f}" for c in CHECKPOINTS)
            ps = "/".join(f"{t[c]['palm_mm']:5.1f}" for c in CHECKPOINTS)
            ok = (t[2000]["yaw_err"] < TOL_RAD and t[2000]["palm_mm"] < TOL_MM)
            print(f"{kp:9.0f}{damp:8.0f}  {ys:^34}  {ps:^30}"
                  f"{t['qacc_max']:10.0f}  {'✅' if ok else ''}")
            if ok:
                score = (t[2000]["palm_mm"], t["qacc_max"])
                if best is None or score < best[0]:
                    best = (score, kp, damp, t)

    print("─" * 96)
    if best is None:
        print("\n❌ Egyetlen kombináció sem teljesíti a kilépési feltételt.")
        print("   Ez NEM hangolási kudarc, hanem jelzés: a törzs kp-jével nem")
        print("   érhető el a szükséges pontosság. Következő hipotézisek:")
        print("     · a törzs armatúrája/tehetetlensége a modellben irreális")
        print("     · gravitáció-kompenzáció kell (feedforward nyomaték)")
        print("     · a robotot kell közelebb/elfordítva állítani a polchoz")
        return 1

    (score, kp, damp, t) = best
    print(f"\n✅ NYERTES: TORSO_KP={kp:.0f}  csillapítás={damp:.0f}  "
          f"armatúra={ARMATURE}")
    print(f"   2000 lépésnél: yaw-hiba {t[2000]['yaw_err']:.4f} rad · "
          f"tenyér {t[2000]['palm_mm']:.1f} mm · qacc {t['qacc_max']:.0f}")
    print(f"   4000 lépésnél: yaw-hiba {t[4000]['yaw_err']:.4f} rad · "
          f"tenyér {t[4000]['palm_mm']:.1f} mm")
    print("\n   Ezt kell beírni a shelflife_build_scene_sku.py-be, majd a")
    print("   jelenetet újraépíteni és ezt a mérést megismételni ellenőrzésként.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
