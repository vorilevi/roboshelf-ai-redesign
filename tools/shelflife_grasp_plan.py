"""
shelflife_grasp_plan.py — fogási terv KERESÉSE (egyszer fut, JSON-t ír)

    python3 tools/shelflife_grasp_plan.py --sku alpro_barista_coconut_1l

Kimenet: models/shelflife_sku/<sku>/grasp_plan.json

────────────────────────────────────────────────────────────────────────────
MIÉRT KÜLÖN, ELŐRE FUTÓ KERESÉS
────────────────────────────────────────────────────────────────────────────
A fogás három feltétele KÜLÖN-KÜLÖN teljesíthető, EGYÜTT viszont nehezen, és
eddig mindig azért buktunk el, mert egyet optimalizáltunk a másik rovására:

  1. GEOMETRIA — a nyitott kéz ne metssze a terméket, záráskor viszont a
     hüvelyk ÉS az ujjak SZEMBEN fogják.
  2. ÚT — a közelítés teljes hossza legyen ütközésmentes (a kinyitott hüvelyk
     9 cm-re áll ki a tenyér síkjából, és oldalirányú közelítésnél
     gereblyeként söpör végig a terméken).
  3. ELÉRHETŐSÉG — a 7-DoF kar a fogási ORIENTÁCIÓT is fel tudja venni, ne
     csak a pozíciót. Mérve: a négy jelölt orientációból egy működik, és azt
     is csak a termék szűk körzetében.

Amikor a 2. feltétel miatt átálltunk felülről közelítésre, elromlott a 3.;
amikor a fogási pontot az 1. szerint korrigáltuk, megint elromlott a 3.

Ezért a keresés MINDHÁRMAT egyszerre szűri, és a nyertes paramétereket
fájlba írja. A futásidejű kód (`shelflife_grasp.py`) ezt csak beolvassa —
így az eval nem függ egy keresés véletlenétől, és a terv auditálható.

A kimenet közvetlenül az SKU-bejegyzés `grasp.recommended` mezőjének felel meg
(2. pillér: „SKU-nkénti fogástechnika").
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

import mujoco                                    # noqa: E402
import shelflife_grasp as G                      # noqa: E402

# A közelítési út mintavételi pontjai — 1 cm-enként.
#
# MIÉRT ILYEN SŰRŰN: 3 cm-es lépésekkel a terv „tisztának" jelentett egy utat,
# amin a dinamikus futás 109 mm-t lökött a terméken. A kéz 20 cm hosszú és
# szűk résekben mozog; 3 cm-es rácson ÁTLÉP az ütközések fölött.
STANDOFFS = tuple(round(0.01 * k, 2) for k in range(1, 15))
# 14 cm-nél messzebb nem érdemes: MÉRVE, a kar a fogási ORIENTÁCIÓT 17 cm-től
# már nem tudja tartani. A gravitációs megereszkedést tehát nem lehet
# „távolabb kezdve" ledolgozni — a pálya első pontján kell, ott viszont
# egyenes vonalon (Cartesian szeletelve), hogy a korrekció ne ívben menjen.
REACH_MM, REACH_DEG = 8.0, 4.0           # elérhetőségi küszöb
MAX_STEP_RAD = 0.20                      # két szomszédos pályapont közti
# legnagyobb ízületváltozás. 0.30 rad-nál a végrehajtás átcsapott: 1 cm-es
# CARTESIAN lépés mögött 0.3 rad ÍZÜLETI ugrás volt, amit a kar 240 lépés
# alatt tett meg — a kéz átsöpört a terméken (169 mm) anélkül, hogy a
# pályapontokon bármi kontaktus látszott volna. A baj a pontok KÖZÖTT történt.
MIN_MARGIN = 0.15                        # rad — ízülethatár-tartalék a
                                         # gravitációs korrekciónak
CLOSE_LEVELS = (0.25, 0.35, 0.45, 0.55, 0.65)

# ── ERŐZÁRÁS (M2, 2026-08-04) ───────────────────────────────────────────────
#
# A korábbi geometriai feltétel csak azt kérte, hogy „a hüvelyk és legalább
# három ujj érintkezzen". Ez PROXY volt, és rossz: a nyertes fogásban a mutató
# és a középső ugyanazon a lapon nyomott, a hüvelyk egy MERŐLEGESEN, a túlsó
# lapon semmi. Az eredő 14.1 N lett — NAGYOBB, mint a doboz súlya (10.1 N).
# A kéz tolta a kartont, nem fogta.
#
# A felhasználó fogalmazta meg a helyes feltételt:
#   „az ember is úgy fog meg egy tárgyat, hogy az ujjak EGYSZERRE érnek
#    a tárgyhoz, egymással ELLENTÉTES oldalon"
#
# Ebből három mérhető követelmény:
GAP_MIN, GAP_MAX = 0.005, 0.025   # nyitott kézben mindkét oldal ilyen közel
GAP_BALANCE = 0.010               # és közel egyformán — különben az egyik ELŐBB
                                  # ér oda és eltolja (mérve: 46 vs 69 mm-nél a
                                  # hüvelyk 0.13-nál, az ujjak 0.28-nál értek oda)
OPPOSITION_DEG = 120.0            # a hüvelyk és az ujjak a doboz szemközti oldalán
MAX_IMBALANCE = 0.55              # a kontaktus-normálisok eredője / összerő.
                                  # Ez a valódi erőzárás-mérőszám: ha nagy, TOL.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="alpro_barista_coconut_1l")
    ap.add_argument("--pose", default="right_thumb_up")
    a = ap.parse_args()

    t0 = time.time()
    # A keresésnek a NYERS ujjbegy-centroidból kell indulnia.
    #
    # BUG, amibe belefutottunk: a `GraspRobot` konstruktora beolvassa a MÁR
    # MEGLÉVŐ `grasp_plan.json`-t, tehát a második futás a saját előző
    # eredményére rakta rá az új korrekciót — visszacsatolás. A kimenet
    # látszólag rendben volt (0.0 mm / 0.0°), de a fogási pont körönként
    # vándorolt. Ezért itt a terv-beolvasást KIKAPCSOLJUK.
    G.GRASP_TWEAK_CM = np.zeros(3)
    _real_plan, G.PLAN_PATH = G.PLAN_PATH, Path("/nonexistent")
    r = G.GraspRobot()
    G.PLAN_PATH = _real_plan
    assert r.plan is None, "a tervező nem indulhat korábbi tervről"
    m, s = r.model, r._scratch
    R_des = G.GRASP_POSES[a.pose]
    box, half = r.product_box()
    raw = r._grasp_offset.copy()

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    adr = m.jnt_qposadr[jid]
    dorig = r.data.qpos[adr:adr + 3].copy() - box
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""

    q_nom, _, _ = r.ik6_seed(box, R_des, restarts=16, iters=110)

    def setup(amount, centre):
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(r._arm_q)] = q_nom
        # `phased=False`: a tervező GEOMETRIAI kérdést tesz fel („ha a kéz
        # X szintig zár, mi történik"), az ujjankénti fáziskésleltetés viszont
        # VÉGREHAJTÁSI részlet. Késleltetéssel értékelve 0.65-nél még alig zár
        # be valami — az első futás emiatt adott NULLA jelöltet.
        r.set_hand_qpos(s, amount, phased=False)
        s.qpos[adr:adr + 3] = centre + dorig
        s.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, s)

    # ── FELÜLETI távolság, nem origó-távolság ───────────────────────────────
    #
    # ⚠️ Az első változat az ujjperecek TEST-ORIGÓJÁTÓL mérte a rést a doboz
    # felületéig. Az ujjperec viszont 1.5–2 cm vastag: egy „6 mm-es rés" az
    # origótól azt jelenti, hogy a link FELÜLETE már a dobozban van. Mérve:
    # 424 jelölt ment át a rés- és oppozíció-szűrőn, és NULLA volt közülük
    # ütközésmentes nyitott kézzel.
    #
    # A `mj_geomDistance` a tényleges geometriák közti távolságot adja —
    # nincs mit közelíteni.
    prod_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")
    bnm = lambda g: bn(m.geom_bodyid[g])

    def geoms_of(tokens) -> list[int]:
        return [g for g in range(m.ngeom)
                if any(tk in bnm(g) for tk in tokens)]

    THUMB_G = geoms_of(("r_thumb",))
    FINGER_G = geoms_of(("r_index", "r_middle", "r_ring", "r_little"))

    def surf_gap(geoms) -> float:
        """A legkisebb FELÜLETI távolság a megadott geomok és a termék közt."""
        best = 1e3
        ft = np.zeros(6)
        for g in geoms:
            dist = mujoco.mj_geomDistance(m, s, g, prod_geom, 0.4, ft)
            best = min(best, float(dist))
        return best

    def centroid(geoms) -> np.ndarray:
        return np.mean([s.geom_xpos[g] for g in geoms], axis=0)

    def normal_imbalance() -> float:
        """A termékre ható kontaktus-normálisok eredője / darabszám.

        0 közelében erőzárás (az erők kioltják egymást), 1 közelében tiszta
        tolás. Ez váltja ki a régi „hány ujj ér hozzá" proxyt.
        """
        F, tot = np.zeros(3), 0.0
        for k in range(s.ncon):
            c = s.contact[k]
            for x, y, sg in ((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2], 1.0),
                             (m.geom_bodyid[c.geom2], m.geom_bodyid[c.geom1], -1.0)):
                if y == pb and bn(x).startswith("r_"):
                    F += c.frame[:3] * sg
                    tot += 1.0
        return float(np.linalg.norm(F) / max(tot, 1.0))

    def digits():
        out = set()
        for k in range(s.ncon):
            c = s.contact[k]
            for x, y in ((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]),
                         (m.geom_bodyid[c.geom2], m.geom_bodyid[c.geom1])):
                if y == pb and bn(x).startswith("r_"):
                    out.add(bn(x).split("_")[1])
        return out

    setup(0.0, box)
    Rp = s.xmat[r._palm].reshape(3, 3)
    # (név, tengelyindex a tenyér frame-jében, előjel) — a JSON-be a tengely
    # kerül, nem a világvektor, hogy a terv az orientációtól függetlenül
    # visszaolvasható legyen.
    DIRS = {"felülről": (0, -1), "alulról": (0, +1),
            "elölről": (2, -1), "oldalról": (1, -1)}

    # ── 1. GEOMETRIA + ERŐZÁRÁS ─────────────────────────────────────────────
    print("[1/3] geometria: hol áll a kéz SZEMBEN, közel és kiegyenlítetten")
    setup(0.0, box)
    # a SCRATCH állapotból olvasunk (a `setup()` ezt tölti fel), nem az élőből
    geo = []
    cnt = {"osszes": 0, "res_ok": 0, "egyenlo": 0, "szemben": 0,
           "nyitva_tiszta": 0, "zarva_fog": 0, "erozaras": 0}
    best_ang = 0.0
    print(f"      kiindulás (FELÜLETI távolság): hüvelyk "
          f"{surf_gap(THUMB_G)*1000:.0f} mm · ujjak {surf_gap(FINGER_G)*1000:.0f} mm")
    grid = np.arange(-0.12, 0.121, 0.005)   # 5 mm: az elfogadási sáv szűk
    for du in grid:
        for dv in grid:
            for dw in grid:
                dd = np.array([du, dv, dw])
                c = box + Rp @ dd
                cnt["osszes"] += 1
                # (a) mindkét oldal KÖZEL és KIEGYENLÍTETTEN, nyitott kézzel
                setup(0.0, c)
                gt, gf = surf_gap(THUMB_G), surf_gap(FINGER_G)
                if not (GAP_MIN <= gt <= GAP_MAX and GAP_MIN <= gf <= GAP_MAX):
                    continue
                cnt["res_ok"] += 1
                if abs(gt - gf) > GAP_BALANCE:
                    continue
                cnt["egyenlo"] += 1
                # (b) SZEMBEN: a doboz közepéből nézve ellentétes irányban
                vt = centroid(THUMB_G) - c
                vf = centroid(FINGER_G) - c
                ang = float(np.degrees(np.arccos(np.clip(
                    (vt / np.linalg.norm(vt)) @ (vf / np.linalg.norm(vf)), -1, 1))))
                best_ang = max(best_ang, ang)
                if ang < OPPOSITION_DEG:
                    continue
                cnt["szemben"] += 1
                # (c) nyitva ne érjen hozzá  (a `setup` fent már megtörtént)
                if digits():
                    continue
                cnt["nyitva_tiszta"] += 1
                # (d) záráskor jöjjön létre ERŐZÁRÁS, ne tolás
                for lvl in CLOSE_LEVELS:
                    setup(lvl, c)
                    dg = digits()
                    if "thumb" in dg and len(dg - {"thumb"}) >= 2:
                        cnt["zarva_fog"] += 1
                        imb = normal_imbalance()
                        if imb <= MAX_IMBALANCE:
                            cnt["erozaras"] += 1
                            geo.append((dd, lvl, len(dg), gt, gf, ang, imb))
                        break
    geo.sort(key=lambda x: (x[6], max(x[3], x[4])))
    print("      szűrők:", " → ".join(f"{k} {v}" for k, v in cnt.items()))
    print(f"      a rácson elérhető legjobb oppozíció (rés-szűrő után): "
          f"{best_ang:.0f}° (küszöb {OPPOSITION_DEG:.0f}°)")
    print(f"      {len(geo)} fogási középpont ad ERŐZÁRÁST")
    if geo:
        dd, lvl, n, gt, gf, ang, imb = geo[0]
        print(f"      legjobb: rés {gt*1000:.1f}/{gf*1000:.1f} mm · "
              f"oppozíció {ang:.0f}° · {n} ujj · eredő/össz {imb:.2f}")
    if not geo:
        sys.exit("nincs geometriailag működő fogás — ezt jelenteni kell")

    # ── 2. ELÉRHETŐSÉG a célpontban ─────────────────────────────────────────
    print("[2/3] elérhetőség: fel tudja-e venni a kar az orientációt")
    reach = []
    for d, lvl, n, *_rest in geo[:60]:
        r._grasp_offset = raw + d
        qk, ep, er = r.ik6_seed(box, R_des, restarts=8, iters=70)
        mg = r.joint_margin(qk)
        # Az ÍZÜLETTARTALÉK is feltétel, nem csak a pontosság: határon ülő
        # pózban a zárt hurok nem tudja korrigálni a gravitációs
        # megereszkedést, és 33 mm-en beragad (mérve).
        if ep * 1000 < REACH_MM and np.degrees(er) < REACH_DEG and mg > MIN_MARGIN:
            reach.append((d, lvl, n, ep * 1000, float(np.degrees(er)), mg))
    reach.sort(key=lambda x: (-x[2], -x[5]))
    print(f"      {len(reach)}/{len(geo)} elérhető  ({time.time()-t0:.0f}s)")
    if not reach:
        sys.exit("a kar egyik jó fogási pontot sem éri el ezzel az orientációval")

    # ── 3. AZ ÚT: ütközésmentes ÉS végig elérhető ───────────────────────────
    #
    # ⚠️ AMI ELŐSZÖR KIMARADT: a POLC. Az első keresés csak a kéz és a TERMÉK
    # ütközését nézte, és emiatt az „alulról" közelítést hozta ki nyertesnek —
    # a 12 cm-es pre-grasp pont z=1.044-en van, a polclap teteje 1.062-n:
    # a terv a POLCLAPON KERESZTÜL közelített volna. Kinematikailag hibátlan
    # (0.0 mm / 0.0°), fizikailag lehetetlen.
    #
    # Ezért itt a kart TÉNYLEGESEN a közbenső pózba állítjuk, a terméket a
    # helyén hagyjuk, és MINDEN robot–környezet kontaktust nézünk.
    print("[3/3] a közelítési út ellenőrzése (termék ÉS polc)")

    def pose_arm(q_arm, amount):
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(r._arm_q)] = q_arm
        # `phased=False`: a tervező GEOMETRIAI kérdést tesz fel („ha a kéz
        # X szintig zár, mi történik"), az ujjankénti fáziskésleltetés viszont
        # VÉGREHAJTÁSI részlet. Késleltetéssel értékelve 0.65-nél még alig zár
        # be valami — az első futás emiatt adott NULLA jelöltet.
        r.set_hand_qpos(s, amount, phased=False)
        mujoco.mj_forward(m, s)

    # A JOBB KAR testei. Nem elég az "r_" prefix: a jobb LÁB is azzal kezdődik
    # (r_hip, r_upper_leg, r_ankle), és az természetesen a padlón áll —
    # az első futás ezért jelzett minden jelöltnél „padló" ütközést.
    ARM_BODIES = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
                  "r_thumb", "r_index", "r_middle", "r_ring", "r_little")

    def obstacles():
        """Mihez ér hozzá a jobb KAR/KÉZ: {'termék', 'polc', 'padló'}"""
        hit = set()
        for k in range(s.ncon):
            c = s.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            n1, n2 = bn(b1), bn(b2)
            for x, y in ((n1, n2), (n2, n1)):
                if not x.startswith(ARM_BODIES):
                    continue
                if y == "product_0":
                    hit.add("termék")
                elif y in ("world", ""):
                    g = c.geom1 if bn(b1) != x else c.geom2
                    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                    if gn.startswith("shelf"):
                        hit.add("polc")
                    elif gn == "floor":
                        hit.add("padló")
        return hit

    best = None
    for d, lvl, n, mm, deg, mg in reach[:14]:
        r._grasp_offset = raw + d
        for dname, (ax, sg) in DIRS.items():
            dvec = sg * Rp[:, ax]
            # A pályát TÁVOLRÓL A CÉL FELÉ, FOLYAMATOSAN oldjuk meg: minden
            # lépés az előző konfigurációból indul (meleg indítás), és ha a
            # következő póz nagyot ugrik az előzőhöz képest, a jelöltet
            # elvetjük. Enélkül a szomszédos állomások külön IK-ágakra
            # eshetnek, és a kar köztük átcsapva végigsöpör a polcon.
            ok, worst, why, way = True, 0.0, "", []
            q_prev = None
            for so in sorted(STANDOFFS + (0.0,), reverse=True):
                if q_prev is not None:
                    r._cmd = q_prev.copy()          # meleg indítás
                q_so, ep, er = r.ik6_seed(box - dvec * so, R_des,
                                          restarts=(6 if q_prev is None else 3),
                                          iters=60)
                if q_prev is not None:
                    jump = float(np.max(np.abs(q_so - q_prev)))
                    if jump > MAX_STEP_RAD:
                        ok, why = False, (f"{so*100:.0f} cm-nél {jump:.2f} rad "
                                          f"ugrás a pályán"); break
                if ep * 1000 > REACH_MM or np.degrees(er) > REACH_DEG:
                    ok, why = False, f"{so*100:.0f} cm-en nem éri el"; break
                if r.joint_margin(q_so) < MIN_MARGIN:
                    ok, why = False, (f"{so*100:.0f} cm-en nincs ízülettartalék "
                                      f"({r.joint_margin(q_so):.2f} rad)"); break
                pose_arm(q_so, 0.0)
                h = obstacles()
                if h:
                    ok, why = False, f"{so*100:.0f} cm-en ütközik: {sorted(h)}"; break
                worst = max(worst, ep * 1000)
                way.append({"standoff_cm": round(so * 100, 1),
                            "q": [round(float(x), 5) for x in q_so]})
                q_prev = q_so.copy()
            if not ok:
                print(f"      ✗ {dname:<9} {why}")
                continue
            score = (n, -worst)
            if best is None or score > best[0]:
                best = (score, d, lvl, n, (dname, ax, sg), mm, deg, worst, mg, way)
                print(f"      ✅ {dname:<9} {n} ujj · zárás {lvl:.2f} · "
                      f"cél {mm:.1f} mm/{deg:.1f}° · út max {worst:.1f} mm · "
                      f"offset {np.round(d*100,1)} cm")

    if best is None:
        sys.exit("van jó fogás és van elérhető póz, de nincs tiszta ÚT hozzá")

    _, d, lvl, n, (dname, ax, sg), mm, deg, worst, mg, way = best
    out = {
        "sku": a.sku, "pose": a.pose, "approach": dname,
        "approach_palm_axis": [int(ax), int(sg)],
        "grasp_offset_palm_cm": [float(x) for x in np.round((raw + d) * 100, 2)],
        "tweak_cm": [float(x) for x in np.round(d * 100, 1)],
        "close_amount": float(lvl), "digits_in_contact": int(n),
        "reach_err_mm": round(mm, 2), "reach_err_deg": round(deg, 2),
        "path_worst_mm": round(worst, 2), "joint_margin_rad": round(float(mg), 3),
        "standoffs_checked_cm": [round(s * 100, 1) for s in STANDOFFS],
        # A TELJES PÁLYA ízületkonfigurációi, 14 cm-től a célig.
        #
        # MIÉRT NEM ELÉG A VÉGPONT: a futásidejű kód eddig a pre-grasp pontra
        # ugrott, onnan szeletelt a célig — és a köztes ív végigsöpört a
        # terméken (109–454 mm). A tervező viszont MINDEN 1 cm-es lépésre
        # ellenőrzött pózt talált; ezeket most el is mentjük, és a futás
        # ezeken halad végig. A pálya a terv része, nem a végrehajtásé.
        "waypoints": way,
        "_note": ("Kereséssel előállítva: geometria + út + elérhetőség együtt. "
                  "A futásidejű kód ezt olvassa be, nem keres újra."),
    }
    p = _REPO / "models/shelflife_sku" / a.sku / "grasp_plan.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n  terv: {p.relative_to(_REPO)}   ({time.time()-t0:.0f}s)")
    for k, v in out.items():
        if not k.startswith("_"):
            print(f"    {k:<22} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
