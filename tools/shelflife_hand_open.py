"""
shelflife_hand_open.py — a NYITOTT KÉZFORMA újratervezése hengeres tárgyra

    python3 tools/shelflife_hand_open.py

────────────────────────────────────────────────────────────────────────────
MIÉRT
────────────────────────────────────────────────────────────────────────────
A jelenlegi `HAND_OPEN` a tejesdobozhoz készült, és a négy ujjat NYÚJTVA,
SZÉTTERPESZTVE tartja:

    r_index_1_add  +0.35 · r_ring_1_add −0.20 · r_little_1_add −0.35
    minden 1_flex / _2 / _3 ízület:  0.00   (teljesen nyújtott)

Egy nyújtott, szétterpesztett kéz SÍKOT formál, nem üreget. Ezért:

  · a fogófelületek 22–28 mm-re állnak egy 58 mm-es hengertől, akárhogy
    pozicionálunk (mérve, a tenyér-küszöbtől függetlenül);
  · záráskor csak az UJJBEGYEK érnek hozzá, középperec soha;
  · a fogás csipesz marad, nem átfogás — és nem tart meg.

A Generative Bionics saját videóján (2026-08) a GENE.01 egy kulacsot NAGY
FELÜLETTEL fog: a tárgy a tenyéren fekszik, a hüvelyk lapos párnaként rajta,
az ujjak körbehajlanak. Ehhez az ujjaknak ELŐRE hajlítottnak kell lenniük.

────────────────────────────────────────────────────────────────────────────
A MÓDSZER: SÖPRÉS, NEM TIPP
────────────────────────────────────────────────────────────────────────────
A kézformát négy számmal paraméterezzük, és VÉGIGMÉRJÜK. Minden jelöltnél a
tárgyat a kézhez képest optimálisan helyezzük el (a min-rést maximalizálva,
áthatolás nélkül), és megnézzük, hány ujj kerül fogótávolságba és mely
perecekkel.

────────────────────────────────────────────────────────────────────────────
A KRITÉRIUM — ELŐRE KIMONDVA
────────────────────────────────────────────────────────────────────────────
    · legalább NÉGY ujj a tárgytól ≤ 15 mm-re
    · közülük legalább KETTŐ a KÖZÉPPERECÉVEL is (nem csak begy)
    · a kéz többi része ne hatoljon a tárgyba

Az így kapott jelölteket utána a rögzített-elengedett fogáspróba minősíti
(csúszás < 20 mm, erő < 50 N) — azt a `shelflife_hand_open_test.py` végzi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
import shelflife_grasp as G                      # noqa: E402

NEAR_MM = 15.0            # „fogótávolság"
NEED_DIGITS = 4
NEED_MEDIAL = 2
PENETRATION_TOL = -0.001  # 1 mm numerikus tűrés

DIGITS = ("thumb", "index", "middle", "ring", "little")


def hand_open_variant(f_flex: float, f_spread: float,
                      t_rot: float, t_flex: float) -> dict:
    """Paraméterezett nyitott kézforma.

    f_flex   a négy ujj ELŐHAJLÍTÁSA (0 = nyújtott, mint most)
    f_spread a terpesztés szorzója (1 = a mostani, 0 = zárt ujjak)
    t_rot    a hüvelyk oppozíciós forgatása
    t_flex   a hüvelyk hajlítása
    """
    d = {
        "r_thumb_1_rot": t_rot,
        "r_thumb_1_add": -0.52,
        "r_thumb_1_flex": t_flex,
        "r_thumb_2": 0.0,
        "r_thumb_3": 0.0,
        "r_index_1_add": 0.35 * f_spread,
        "r_ring_1_add": -0.20 * f_spread,
        "r_little_1_add": -0.35 * f_spread,
    }
    for f in ("index", "middle", "ring", "little"):
        d[f"r_{f}_1_flex"] = f_flex
        d[f"r_{f}_2"] = f_flex * 1.2
        d[f"r_{f}_3"] = f_flex * 0.8
    return d


def main() -> int:
    print("Shelf Life — a NYITOTT KÉZFORMA újratervezése\n")
    G.GRASP_TWEAK_CM = np.zeros(3)
    _rp, G.PLAN_PATH = G.PLAN_PATH, Path("/nonexistent")
    r = G.GraspRobot()
    G.PLAN_PATH = _rp
    m, s = r.model, r._scratch
    box, half = r.product_box()
    R = G.GRASP_POSES["right_thumb_up"]
    q, _, _ = r.ik6_seed(box, R, restarts=16, iters=110)

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    adr = m.jnt_qposadr[jid]
    dorig = r.data.qpos[adr:adr + 3].copy() - box
    prod = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")
    bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    ARM = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
           "r_thumb", "r_index", "r_middle", "r_ring", "r_little")
    HAND_G = [g for g in range(m.ngeom) if bn(m.geom_bodyid[g]).startswith(ARM)]

    # perecenként külön geomcsoport, hogy lássuk, MIVEL ér hozzá
    def geoms(dg: str, part: str) -> list[int]:
        return [g for g in range(m.ngeom)
                if bn(m.geom_bodyid[g]) == f"r_{dg}_{part}"]

    PARTS = {(dg, p): geoms(dg, p) for dg in DIGITS
             for p in ("distal", "medial")}
    ft = np.zeros(6)

    def setup(pose: dict, centre: np.ndarray) -> None:
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(r._arm_q)] = q
        for act, (o, c) in r._pose.items():
            jid2 = m.actuator_trnid[act, 0]
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid2) or ""
            lo, hi = m.jnt_range[jid2]
            s.qpos[m.jnt_qposadr[jid2]] = float(
                np.clip(pose.get(nm, 0.0), lo, hi))
        s.qpos[adr:adr + 3] = centre + dorig
        s.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, s)

    def gap(gs) -> float:
        if not gs:
            return 1e3
        return min(float(mujoco.mj_geomDistance(m, s, g, prod, 0.4, ft))
                   for g in gs)

    setup(G.HAND_OPEN, box)
    Rp = s.xmat[r._palm].reshape(3, 3)

    def best_placement(pose: dict):
        """A tárgy legjobb helye ehhez a kézformához (áthatolás nélkül)."""
        bestc, bestk = None, None
        rng = np.arange(-0.02, 0.0201, 0.005)
        for du in rng:
            for dv in rng:
                for dw in rng:
                    c = box + Rp @ np.array([du, dv, dw])
                    setup(pose, c)
                    if gap(HAND_G) < PENETRATION_TOL:
                        continue
                    near = [(dg, p) for (dg, p), gs in PARTS.items()
                            if gap(gs) * 1000 <= NEAR_MM]
                    nd = len({dg for dg, _ in near})
                    nm_ = len([1 for _, p in near if p == "medial"])
                    k = (nd, nm_)
                    if bestk is None or k > bestk:
                        bestk, bestc = k, c
        return bestc, bestk

    print(f"  {'előhajl.':>9}{'terp.':>7}{'hü.rot':>8}{'hü.flex':>9}"
          f"{'ujj ≤15mm':>11}{'ebből középperec':>18}")
    print("  " + "─" * 64)
    rows = []
    for f_flex in (0.0, 0.25, 0.5, 0.75):
        for f_spread in (1.0, 0.5, 0.0):
            for t_rot, t_flex in ((0.70, -0.79), (1.20, -0.30), (1.20, 0.30)):
                pose = hand_open_variant(f_flex, f_spread, t_rot, t_flex)
                c, k = best_placement(pose)
                if k is None:
                    continue
                nd, nm_ = k
                ok = nd >= NEED_DIGITS and nm_ >= NEED_MEDIAL
                print(f"  {f_flex:9.2f}{f_spread:7.1f}{t_rot:8.2f}{t_flex:9.2f}"
                      f"{nd:11d}{nm_:18d}{'  ✅' if ok else ''}")
                rows.append((nd, nm_, f_flex, f_spread, t_rot, t_flex))
    if not rows:
        print("\n  ❌ egyetlen kézforma sem fér el a tárgy körül")
        return 1
    rows.sort(reverse=True)
    nd, nm_, ff, fs, tr, tf = rows[0]
    print(f"\n  LEGJOBB: előhajlítás {ff:.2f} · terpesztés {fs:.1f} · "
          f"hüvelyk rot {tr:.2f} / flex {tf:.2f}")
    print(f"           {nd} ujj fogótávolságban, {nm_} középpereccel")
    print(f"  (a mostani `HAND_OPEN` = előhajlítás 0.00, terpesztés 1.0, "
          f"rot 0.70, flex −0.79)")
    if nd < NEED_DIGITS or nm_ < NEED_MEDIAL:
        print(f"\n  ⚠️ A KRITÉRIUM ({NEED_DIGITS} ujj / {NEED_MEDIAL} középperec) "
              f"NEM teljesül. Ez jelentés.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
