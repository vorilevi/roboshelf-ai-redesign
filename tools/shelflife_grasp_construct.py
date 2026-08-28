"""
shelflife_grasp_construct.py — a fogási terv MEGSZERKESZTÉSE (M2 lezárás)

    python3 tools/shelflife_grasp_construct.py            # mérés + javaslat
    python3 tools/shelflife_grasp_construct.py --write    # beírja a tervfájlba

────────────────────────────────────────────────────────────────────────────
MIÉRT ÚJ FÁJL, ÉS MIÉRT NEM A RÁCS
────────────────────────────────────────────────────────────────────────────
A `shelflife_grasp_plan.py` mindhárom feltételt (geometria + erőzárás +
elérhetőség + út) EGYÜTT szűri — a kritérium tehát jó. Csak a KERESÉSI MÓD
rossz: 49³ = 117 649 rácspont, mindegyiknél ütközésszámítás. Ez a fejlesztői
sandbox 45 másodperces ablakában nem fut le, és háromszor halt el némán
háttérben.

De nem is kell keresni. Az állkapocs-mérés (`shelflife_jaw.py`) ANALITIKUSAN
megadja, hol van a fogási pont: a hüvelyk és az ujjak fogófelületei közti
legközelebbi pontpár felezőpontjában. Egy pont, nulla keresés.

Ami abból még hiányzott:

  · nyitott kézzel is volt 11 kontaktus — a pont túl MÉLYEN van a kézben;
  · nem volt ellenőrizve az elérhetőség és a közelítési út.

Mindkettő megoldható SZERKESZTÉSSEL, nem rácson:

  1. a pontot az állkapocs TENGELYE mentén és a tenyér normálisa mentén
     kifelé toljuk, amíg a nyitott kéz tisztán elfér — ez 1-D söprés,
     nem 3-D rács;
  2. a kapott pontra lefuttatjuk ugyanazt az elérhetőség- és
     út-ellenőrzést, amit a rácsos tervező használ.

Nagyságrend: ~500 kiértékelés 117 649 helyett.

────────────────────────────────────────────────────────────────────────────
A KRITÉRIUMOK — SZÓ SZERINT A RÁCSOS TERVEZŐBŐL
────────────────────────────────────────────────────────────────────────────
Szándékosan ugyanazok az értékek, hogy a két út eredménye összevethető legyen.
Ha itt lazítunk, az eredmény nem hasonlítható a korábbiakhoz.
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
import shelflife_jaw as J                        # noqa: E402

# ── a rácsos tervezőből átvett küszöbök (shelflife_grasp_plan.py) ───────────
GAP_MIN = 0.005
GAP_MAX = 0.025
GAP_BALANCE = 0.010
OPPOSITION_DEG = 120.0
MAX_IMBALANCE = 0.55
CLOSE_LEVELS = (0.25, 0.35, 0.45, 0.55, 0.65)
STANDOFFS = tuple(round(0.01 * k, 2) for k in range(1, 15))
REACH_MM, REACH_DEG = 8.0, 4.0
MAX_STEP_RAD = 0.20
MIN_MARGIN = 0.15

ARM_BODIES = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
              "r_thumb", "r_index", "r_middle", "r_ring", "r_little")

# A kifelé tolás söprése: 0-tól 6 cm-ig, 2 mm-enként.
SWEEP = np.arange(0.0, 0.0601, 0.002)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="alpro_barista_coconut_1l")
    ap.add_argument("--pose", default="right_thumb_up")
    ap.add_argument("--write", action="store_true")
    # ── MIÉRT KÜLÖN KAPCSOLÓ, ÉS MIÉRT NEM ÍRTAM ÁT A KÜSZÖBÖT ──────────────
    # Az eredmény: a tenyér tisztán tartása mellett a legkisebb kiegyenlített
    # fogófelület-rés 25.3 mm — épp a 25 mm-es küszöb FÖLÖTT. A csábítás az,
    # hogy a küszöböt írjam át 30-ra és „sikerüljön". Egyszer már megtettem
    # (az M1 `close_until` sorát a mérés lefutása UTÁN pontosítottam), és az
    # rossz szokás: így a kritérium az eredményt követi, nem fordítva.
    #
    # Ezért az alapértelmezés VÁLTOZATLAN marad, a tágítás pedig külön
    # kapcsoló, ami a kimenetben és a tervfájlban is nyomot hagy.
    ap.add_argument("--explore-gap-max-mm", type=float, default=None,
                    help="FELTÁRÓ mód: tágabb rés-sáv, jelölve a kimenetben")
    a = ap.parse_args()
    global GAP_MAX
    explore = a.explore_gap_max_mm is not None
    if explore:
        GAP_MAX = a.explore_gap_max_mm / 1000.0
        print(f"⚠️  FELTÁRÓ MÓD: a rés felső határa {GAP_MAX*1000:.0f} mm "
              f"(alapértelmezés 25 mm). Az eredmény NEM összevethető a "
              f"korábbi futásokkal.\n")
    t0 = time.time()

    print("Shelf Life M2 — a fogási terv MEGSZERKESZTÉSE\n")

    # ⚠️ A tervezőnek NYERS offsetről kell indulnia. A `GraspRobot` beolvassa a
    # meglévő tervet, tehát a második futás a saját előző eredményére rakná rá
    # az új korrekciót — visszacsatolás. (Ugyanez a hiba megvolt a rácsos
    # tervezőben, onnan a megoldás is.)
    G.GRASP_TWEAK_CM = np.zeros(3)
    _real_plan, G.PLAN_PATH = G.PLAN_PATH, Path("/nonexistent")
    r = G.GraspRobot()
    G.PLAN_PATH = _real_plan
    assert r.plan is None, "a szerkesztő nem indulhat korábbi tervről"

    m, s = r.model, r._scratch
    R_des = G.GRASP_POSES[a.pose]
    box, half = r.product_box()
    raw = r._grasp_offset.copy()

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    adr = m.jnt_qposadr[jid]
    dorig = r.data.qpos[adr:adr + 3].copy() - box
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    prod_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")

    q_nom, ep0, er0 = r.ik6_seed(box, R_des, restarts=16, iters=110)
    print(f"  névleges póz: IK {ep0*1000:.1f} mm / {np.degrees(er0):.1f}° · "
          f"ízülettartalék {r.joint_margin(q_nom):.2f} rad")

    def setup(amount: float, centre: np.ndarray) -> None:
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(r._arm_q)] = q_nom
        r.set_hand_qpos(s, amount, phased=False)     # geometriai kérdés
        s.qpos[adr:adr + 3] = centre + dorig
        s.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, s)

    THUMB_G = J.side_geoms(m, J.THUMB)
    FINGER_G = J.side_geoms(m, J.FINGERS)

    def surf_gap(geoms) -> float:
        best, ft = 1e3, np.zeros(6)
        for g in geoms:
            best = min(best, float(
                mujoco.mj_geomDistance(m, s, g, prod_geom, 0.4, ft)))
        return best

    def centroid(geoms) -> np.ndarray:
        return np.mean([s.geom_xpos[g] for g in geoms], axis=0)

    # ⚠️ A NÉV SZERINTI UJJ-FELISMERÉS. Az első változat így szűrt:
    #       if bn(x).startswith("r_"): out.add(bn(x).split("_")[1])
    # — ez az `r_wrist_3`-ból „wrist"-et csinált, és beszámította ujjnak.
    # A jelentésben ezért állt **6 ujj**, ami öt ujjú kéznél lehetetlen; ez
    # volt az árulkodó jel. Egy „ujj" csak az öt digit egyike lehet.
    DIGIT_NAMES = ("thumb", "index", "middle", "ring", "little")

    def digits() -> set[str]:
        out = set()
        for k in range(s.ncon):
            c = s.contact[k]
            for x, y in ((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]),
                         (m.geom_bodyid[c.geom2], m.geom_bodyid[c.geom1])):
                if y != pb:
                    continue
                nm = bn(x)
                for dg in DIGIT_NAMES:
                    if f"_{dg}_" in nm or nm.endswith(f"_{dg}"):
                        out.add(dg)
        return out

    def imbalance() -> float:
        F, tot = np.zeros(3), 0.0
        for k in range(s.ncon):
            c = s.contact[k]
            for x, y, sg in ((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2], 1.0),
                             (m.geom_bodyid[c.geom2], m.geom_bodyid[c.geom1], -1.0)):
                if y == pb and bn(x).startswith("r_"):
                    F += c.frame[:3] * sg
                    tot += 1.0
        return float(np.linalg.norm(F) / max(tot, 1.0))

    # ── 1. AZ ÁLLKAPOCS BEMÉRÉSE ────────────────────────────────────────────
    setup(0.0, box)
    Rp = s.xmat[r._palm].reshape(3, 3)
    ft = np.zeros(6)
    best, p_th, p_fi = 1e3, None, None
    for gt in THUMB_G:
        for gf in FINGER_G:
            dist = float(mujoco.mj_geomDistance(m, s, gt, gf, 1.0, ft))
            if dist < best:
                best, p_th, p_fi = dist, ft[:3].copy(), ft[3:].copy()
    axis = p_fi - p_th
    opening = float(np.linalg.norm(axis))
    axis /= opening
    jaw_centre = 0.5 * (p_th + p_fi)
    need = float(2 * min(half[0], half[1]))
    print(f"  állkapocs: nyílás {opening*1000:.1f} mm · karton {need*1000:.0f} mm"
          f" · tartalék {(opening-need)*1000:.0f} mm")
    if opening < need:
        sys.exit("a nyílás kisebb, mint a termék — a KÉZFORMÁT kell újratervezni")

    # ── 2. HELYI KERESÉS AZ ÁLLKAPOCS KÖRÜL ─────────────────────────────────
    #
    # ⚠️ AZ ELSŐ VÁLTOZAT ITT ELBUKOTT, és a hiba tanulságos.
    #
    # Azt hittem, az állkapocs felezőpontja MÁR jó hely, csak kicsit kifelé
    # kell tolni. Mérve viszont a felezőpontban a doboz BELEMETSZ az ujjakba:
    #
    #     hüvelyk-rés  +3.7 mm      ujj-rés  −5.1 mm
    #
    # Az ok: a „nyílás" EGYETLEN pontpár távolsága (a hüvelyk legközelebbi
    # fogófelülete a legközelebbi ujj-fogófelülethez), a KORLÁT viszont az
    # ÖSSZES fogófelület MINIMUMA. A nyitott kéz nem párhuzamos pofájú fogó:
    # az ujjak szétterpesztve, különböző mélységben állnak, így a legszűkebb
    # hely nem ott van, ahol a felezőpont.
    #
    # Emiatt a tengely menti 1-D tolás sem segít: ami az egyik oldalon nyit,
    # a másikon ugyanannyit zár (mérve: ±0.77 mm/mm), és a legjobb elérhető
    # kiegyenlített állapot is −0.7 / −0.7 mm, vagyis még mindig metszés.
    #
    # A helyes kérdés tehát 3-D, de NEM a teljes rács: az állkapocs körüli
    # ±5 cm bőven elég, mert a kéz mérete ezt kijelöli. Két menetben:
    # durva (5 mm) → finom (1 mm) a nyertes körül. ~10 600 kiértékelés,
    # 0.49 ms/db → néhány másodperc, a 117 649-es vak rács helyett.
    print("\n[1/3] helyi keresés az állkapocs körül")

    # ── A TENYÉR IS SZÁMÍT ──────────────────────────────────────────────────
    #
    # ⚠️ A MÁSODIK VÁLTOZAT ITT BUKOTT EL. A keresés csak a FOGÓFELÜLETEKTŐL
    # (medial + distal) mért rést nézte, és talált egy helyet, ahol mindkét
    # oldal szép 8.0 mm-re volt. Csakhogy ott a doboz **38 mm-rel belemetszett
    # a tenyérbe** — a fogófelületek tisztán álltak, a kéz többi része nem.
    #
    # Emiatt az ÚT minden irányból bukott: a végpontban maga a póz lehetetlen
    # volt, tehát nem az útvonallal volt baj.
    #
    # Ugyanaz a hibaosztály, mint korábban: NEM AZT MÉRNI, AMI SZÁMÍT.
    # A fogás feltétele nem az, hogy a fogófelületek jó távolságra legyenek,
    # hanem hogy közben a kéz TÖBBI RÉSZE se legyen a dobozban.
    HAND_G = [g for g in range(m.ngeom)
              if bn(m.geom_bodyid[g]).startswith(ARM_BODIES)]
    PALM_CLEAR = 0.002        # a nem-fogó részek legalább ennyire maradjanak el

    def evaluate(c: np.ndarray):
        """(min-rés, |eltérés|, hüvelyk-rés, ujj-rés, kéz-min-rés, oppozíció°,
        a hüvelyk érintési MAGASSÁGA a termék talpától)."""
        setup(0.0, c)
        gt, gf = surf_gap(THUMB_G), surf_gap(FINGER_G)
        vt, vf = centroid(THUMB_G) - c, centroid(FINGER_G) - c
        ang = float(np.degrees(np.arccos(np.clip(
            (vt / np.linalg.norm(vt)) @ (vf / np.linalg.norm(vf)), -1, 1))))
        # ⚠️ EZ A FELTÉTEL TELJESEN HIÁNYZOTT A KERESÉSBŐL.
        #
        # A renderelt képeken (2026-08-05) látszik, hogy a kéz a doboz
        # TETEJÉT markolja körbe, mint egy fedelet: a négy ujj gyűrűt formál
        # a felső perem körül, a hüvelyk FÖLÖTTE van, nem szemben a
        # mutatóval. A felhasználó mondta ki: „a hüvelykujj nincs szemben a
        # mutatóujjal és a doboz felett van."
        #
        # Mérve is ez volt: a hüvelyk érintése 146 mm-en, a doboz 145 mm
        # magas. A kereséstől viszont SOHA nem kértük, hogy a PALÁSTON
        # fogjon — csak réseket, oppozíciót és tenyér-tisztaságot néztünk,
        # és mindhárom teljesíthető a tetőn is.
        #
        # Új mérőszám: a hüvelyk legközelebbi pontjának magassága a termék
        # talpától. Ha ez a henger magassága fölé megy, az fedélfogás.
        ft = np.zeros(6)
        bz, bd = 0.0, 1e3
        for g in THUMB_G:
            dist = float(mujoco.mj_geomDistance(m, s, g, prod_geom, 0.5, ft))
            if dist < bd:
                bd, bz = dist, float(ft[2])
        th_h = (bz - (c[2] - half[2])) * 1000        # mm a talptól
        return min(gt, gf), abs(gt - gf), gt, gf, surf_gap(HAND_G), ang, th_h

    def scan(centre: np.ndarray, reach_m: float, step_m: float):
        rng = np.arange(-reach_m, reach_m + 1e-9, step_m)
        best_c, best_key = None, None
        for du in rng:
            for dv in rng:
                for dw in rng:
                    c = centre + Rp @ np.array([du, dv, dw])
                    mn, bal, gt, gf, hand, ang, th_h = evaluate(c)
                    # ── MIÉRT PONTOZÁS ÉS NEM SORREND ───────────────────────
                    # Az előző változat lexikografikusan rendezett: előbb
                    # „elfér-e a kéz", aztán „sávban van-e a rés". Mivel a
                    # sávot EGYETLEN pont sem teljesítette, a döntés a harmadik
                    # kulcsra esett (maximális kéz-tartalék), és az a dobozt
                    # kitolta az ujjak végére: rés 48.5 / 24.9 mm.
                    #
                    # A két követelmény ugyanis HÚZZA EGYMÁST: a tenyér csak
                    # akkor marad kint, ha a doboz kijjebb ül — ott viszont
                    # nagyobb a rés. Ilyenkor a sorrend rossz eszköz; kell egy
                    # KÖZÖS KÖLTSÉG, amiben a két hiba összemérhető.
                    #
                    # Az OPPOZÍCIÓ is a költség része, nem utólagos szűrő.
                    # Az előző menetben a keresés csak résre és
                    # kiegyenlítettségre optimalizált, és 36.4/36.0 mm-es
                    # gyönyörű helyet talált — 115°-os oppozícióval, ami a
                    # 120°-os küszöb alatt van. A keresés azt hozza, amit
                    # kérünk tőle; ha egy feltétel nincs a célfüggvényben,
                    # akkor a nyertes véletlenül teljesíti vagy sem.
                    # Skálázás: 10° ≈ 5 mm rés-hiba.
                    clear = hand >= PALM_CLEAR
                    band_err = max(0.0, GAP_MIN - mn) + max(0.0, mn - GAP_MAX)
                    opp_err = max(0.0, OPPOSITION_DEG - ang) * 0.0005
                    # A KIS RÉS ÖNMAGÁBAN ÉRTÉK, nem csak a sávon belül-lét.
                    # A feltáró futásnál (sáv 45 mm) a keresés 43.5 mm-es rést
                    # választott, mert a sávon belül már semmi nem büntette —
                    # pedig a rés az a távolság, amit minden ujj MEGTESZ a
                    # kontaktusig, és amelyik elsőnek odaér, az tol. Enyhe
                    # húzás a kisebb rés felé (0.2 súly: a kiegyenlítettségnél
                    # gyengébb, mert a KIEGYENLÍTETLENSÉG a rosszabb hiba).
                    # A PALÁSTON kell fogni, nem a tetőn. A cél a henger
                    # FÉLMAGASSÁGA; a tetőn túl (vagy a talp alatt) tiltás.
                    h_mm = half[2] * 2000.0
                    if not (0.15 * h_mm <= th_h <= 0.85 * h_mm):
                        continue                     # fedél- vagy talpfogás
                    top_err = abs(th_h - 0.5 * h_mm) / 1000.0
                    cost = (band_err + 0.5 * bal + opp_err + 0.2 * mn
                            + 0.6 * top_err)
                    key = (clear, -cost if clear else hand)
                    if best_key is None or key > best_key:
                        best_key, best_c = key, c
        return best_c, best_key

    c1, _ = scan(jaw_centre, 0.05, 0.005)
    mn, bal, gt, gf, hand, ang, th1 = evaluate(c1)
    print(f"      durva (±5 cm / 5 mm): fogófelület {gt*1000:5.1f}/{gf*1000:.1f} mm"
          f" · eltérés {bal*1000:4.1f} mm · KÉZ-min {hand*1000:5.1f} mm"
          f" · oppozíció {ang:3.0f}° · hüvelyk {th1:.0f} mm")
    c2, _ = scan(c1, 0.005, 0.001)
    mn, bal, gt, gf, hand, ang, th2 = evaluate(c2)
    print(f"      finom (±5 mm / 1 mm): fogófelület {gt*1000:5.1f}/{gf*1000:.1f} mm"
          f" · eltérés {bal*1000:4.1f} mm · KÉZ-min {hand*1000:5.1f} mm"
          f" · oppozíció {ang:3.0f}° · hüvelyk {th2:.0f} mm")

    if hand < PALM_CLEAR:
        print(f"\n      ✗ A nyitott kéz SEHOL nem fér el a doboz körül: a "
              f"legjobb helyen is {(-hand)*1000:.0f} mm-rel belemetsz "
              f"(a fogófelületeken kívüli rész).")
        print("        Ez JELENTÉS: a NYITOTT KÉZFORMÁT (`HAND_OPEN`) kell "
              "tágítani — nem a pontot tologatni, és nem az utat keresni.")
        return 1
    if not (GAP_MIN <= mn <= GAP_MAX):
        print(f"\n      ✗ A kéz elfér, de a fogófelületek rése nincs a "
              f"{GAP_MIN*1000:.0f}–{GAP_MAX*1000:.0f} mm-es sávban "
              f"(legjobb {mn*1000:.1f} mm).")
        return 1

    # a nyertes hely körül végigpróbáljuk a zárási szinteket
    cands = []
    for lvl in CLOSE_LEVELS:
        setup(lvl, c2)
        dg = digits()
        if not ("thumb" in dg and len(dg - {"thumb"}) >= 2):
            continue
        imb = imbalance()
        setup(0.0, c2)
        vt, vf = centroid(THUMB_G) - c2, centroid(FINGER_G) - c2
        ang = float(np.degrees(np.arccos(np.clip(
            (vt / np.linalg.norm(vt)) @ (vf / np.linalg.norm(vf)), -1, 1))))
        cands.append(dict(c=c2, d=Rp.T @ (c2 - box), lvl=lvl, n=len(dg),
                          gt=gt, gf=gf, ang=ang, imb=imb,
                          dir="helyi keresés", t=float(np.linalg.norm(c2 - jaw_centre))))
        print(f"      zárás {lvl:.2f}: {len(dg)} ujj · oppozíció {ang:.0f}° · "
              f"eredő/össz {imb:.2f}")

    cands = [c for c in cands if c["ang"] >= OPPOSITION_DEG
             and c["imb"] <= MAX_IMBALANCE]
    if not cands:
        print("\n      ✗ A hely geometriailag jó, de a ZÁRÁS nem ad erőzárást "
              f"(oppozíció ≥ {OPPOSITION_DEG:.0f}° és eredő/össz ≤ {MAX_IMBALANCE}).")
        return 1

    cands.sort(key=lambda x: (x["imb"], -x["n"]))
    print(f"      ✅ legjobb: zárás {cands[0]['lvl']:.2f} · {cands[0]['n']} ujj "
          f"· oppozíció {cands[0]['ang']:.0f}° · eredő/össz {cands[0]['imb']:.2f}")

    # ── 3. ELÉRHETŐSÉG ──────────────────────────────────────────────────────
    print("[2/3] elérhetőség — fel tudja-e venni a kar az orientációt")
    reach = []
    for cd in cands[:12]:
        r._grasp_offset = raw + cd["d"]
        qk, ep, er = r.ik6_seed(box, R_des, restarts=8, iters=70)
        mg = r.joint_margin(qk)
        ok = (ep * 1000 < REACH_MM and np.degrees(er) < REACH_DEG
              and mg > MIN_MARGIN)
        print(f"      {'✅' if ok else '✗ '} {cd['dir']:<17}+{cd['t']*1000:3.0f} mm"
              f" · IK {ep*1000:5.1f} mm/{np.degrees(er):4.1f}° · "
              f"tartalék {mg:.2f} rad · eredő/össz {cd['imb']:.2f}")
        if ok:
            reach.append((cd, ep * 1000, float(np.degrees(er)), mg))
    if not reach:
        sys.exit("van jó geometria, de a kar egyik pontot sem éri el")

    # ── 4. AZ ÚT ────────────────────────────────────────────────────────────
    print("[3/3] a közelítési út — termék ÉS polc")

    def pose_arm(q_arm, amount):
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(r._arm_q)] = q_arm
        r.set_hand_qpos(s, amount, phased=False)
        mujoco.mj_forward(m, s)

    def obstacles() -> set[str]:
        hit = set()
        for k in range(s.ncon):
            c = s.contact[k]
            n1, n2 = bn(m.geom_bodyid[c.geom1]), bn(m.geom_bodyid[c.geom2])
            for x, y, g in ((n1, n2, c.geom2), (n2, n1, c.geom1)):
                if not x.startswith(ARM_BODIES):
                    continue
                if y == "product_0":
                    hit.add("termék")
                elif y in ("world", ""):
                    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                    if gn.startswith("shelf"):
                        hit.add("polc")
                    elif gn == "floor":
                        hit.add("padló")
        return hit

    # MIND A HAT tenyér-irány. A rácsos tervező csak négyet nézett — hiányzott
    # a `palm+y` és a `palm+z`. Mérve (behatolás mm-ben, 14 cm-től a célig)
    # épp a `palm+y` volt a legtisztább közelítés: 14 cm-től 6 cm-ig nulla
    # ütközés. Egy irányt kihagyni ugyanolyan mérési hiba, mint rosszat mérni.
    DIRS = {"felülről": (0, -1), "alulról": (0, +1),
            "elölről": (2, -1), "hátulról": (2, +1),
            "oldalról": (1, -1), "másik oldalról": (1, +1)}
    # ── KÉT VÁLTOZTATÁS AZ ÚT-ELLENŐRZÉSBEN, MINDKETTŐ MÉRÉSBŐL ─────────────
    #
    # (1) A PÁLYÁT A CÉLBÓL KIFELÉ oldjuk meg, nem kívülről befelé.
    #     A rácsos tervező 14 cm-től indult, és minden lépést az előzőből
    #     melegen. Csakhogy a LEGKÖTÖTTEBB pont a cél (ott az orientáció és a
    #     pozíció is fix, szűk ízülettartalékkal) — ha onnan indulunk, minden
    #     további pont annak a megoldási ágnak a folytatása. Mérve, ugyanarra
    #     a fogási pontra:
    #         kívülről befelé:  „elölről 13 cm-nél ízületi ugrás"
    #         célból kifelé:    tiszta 0-tól 9 cm-ig, max ugrás 0.11 rad
    #     Ugyanaz a geometria, ugyanaz a küszöb — csak a bejárás iránya más.
    #
    # (2) NEM KÖVETELJÜK MEG A TELJES 14 cm-t. A `STANDOFFS` felső határa
    #     eredetileg azt jelentette, hogy „ennél messzebb nincs értelme"
    #     (17 cm-től a kar nem tartja az orientációt) — nem azt, hogy 14 cm-ig
    #     mindennek tisztának kell lennie. A pre-grasp pont ott van, ameddig
    #     az út tiszta. Kikötés: legalább MIN_PATH_CM, hogy a kéz elférjen.
    #
    # Ez nem a kritérium lazítása az eredmény kedvéért: a régi feltétel a
    # pre-grasp pont HELYÉT rögzítette előre, ami sosem volt fizikai indok.
    MIN_PATH_CM = 8.0

    best_plan = None
    for cd, mm, deg, mg in reach[:6]:
        r._grasp_offset = raw + cd["d"]
        for dname, (ax, sgn) in DIRS.items():
            dvec = sgn * Rp[:, ax]
            worst, why, way, q_prev = 0.0, "", [], None
            for so in (0.0,) + STANDOFFS:            # CÉLBÓL KIFELÉ
                if q_prev is not None:
                    r._cmd = q_prev.copy()
                q_so, ep, er = r.ik6_seed(box - dvec * so, R_des,
                                          restarts=(6 if q_prev is None else 3),
                                          iters=60)
                if q_prev is not None and float(np.max(np.abs(q_so - q_prev))) > MAX_STEP_RAD:
                    why = f"{so*100:.0f} cm-nél ízületi ugrás"; break
                if ep * 1000 > REACH_MM or np.degrees(er) > REACH_DEG:
                    why = f"{so*100:.0f} cm-en nem éri el"; break
                if r.joint_margin(q_so) < MIN_MARGIN:
                    why = f"{so*100:.0f} cm-en nincs tartalék"; break
                pose_arm(q_so, 0.0)
                h = obstacles()
                if h:
                    why = f"{so*100:.0f} cm-en ütközik: {sorted(h)}"; break
                worst = max(worst, ep * 1000)
                way.append({"standoff_cm": round(so * 100, 1),
                            "q": [round(float(x), 5) for x in q_so]})
                q_prev = q_so.copy()

            reach_cm = way[-1]["standoff_cm"] if way else 0.0
            if reach_cm < MIN_PATH_CM:
                print(f"      ✗ {dname:<15} csak {reach_cm:.0f} cm tiszta "
                      f"({why})")
                continue
            # a JSON-t a szokott sorrendben, kívülről befelé kérjük
            way = list(reversed(way))
            score = (-cd["imb"], reach_cm, -worst)
            if best_plan is None or score > best_plan[0]:
                best_plan = (score, cd, dname, ax, sgn, mm, deg, worst, mg,
                             way, reach_cm)
                print(f"      ✅ {dname:<15} tiszta {reach_cm:.0f} cm-ig · "
                      f"{cd['n']} ujj · zárás {cd['lvl']:.2f} · "
                      f"eredő/össz {cd['imb']:.2f} · út max {worst:.1f} mm")

    if best_plan is None:
        print(f"\n  ✗ van jó fogás és elérhető póz, de nincs legalább "
              f"{MIN_PATH_CM:.0f} cm-es tiszta ÚT hozzá.")
        print("    Ez JELENTÉS, nem megkerülendő akadály.")
        return 1

    _, cd, dname, ax, sgn, mm, deg, worst, mg, way, reach_cm = best_plan
    out = {
        "sku": a.sku, "pose": a.pose, "approach": dname,
        "approach_palm_axis": [int(ax), int(sgn)],
        "grasp_offset_palm_cm": [float(x) for x in np.round((raw + cd["d"]) * 100, 2)],
        "tweak_cm": [float(x) for x in np.round(cd["d"] * 100, 2)],
        "close_amount": float(cd["lvl"]), "digits_in_contact": int(cd["n"]),
        "reach_err_mm": round(mm, 2), "reach_err_deg": round(deg, 2),
        "path_worst_mm": round(worst, 2), "joint_margin_rad": round(float(mg), 3),
        "standoffs_checked_cm": [w["standoff_cm"] for w in way],
        "path_clean_to_cm": reach_cm,
        "waypoints": way,
        "_note": ("SZERKESZTVE, nem keresve: az állkapocs felezőpontjából "
                  "1-D kifelé tolással, majd ugyanazokkal a küszöbökkel "
                  "szűrve, mint a rácsos tervező (shelflife_grasp_plan.py)."),
        "_construct": {
            "tool": "shelflife_grasp_construct.py",
            "jaw_opening_mm": round(opening * 1000, 1),
            "product_min_width_mm": round(need * 1000, 1),
            "push_direction": cd["dir"],
            "push_mm": round(cd["t"] * 1000, 1),
            "explore_gap_max_mm": (a.explore_gap_max_mm if explore else None),
            "palm_clearance_mm": round(PALM_CLEAR * 1000, 1),
            "gap_thumb_mm": round(cd["gt"] * 1000, 1),
            "gap_fingers_mm": round(cd["gf"] * 1000, 1),
            "opposition_deg": round(cd["ang"], 1),
            "normal_imbalance": round(cd["imb"], 3),
        },
    }
    print(f"\n  EREDMÉNY  ({time.time()-t0:.0f}s)")
    for k, v in out.items():
        if not k.startswith("_") and k != "waypoints":
            print(f"    {k:<24} {v}")
    print(f"    {'waypoints':<24} {len(way)} pont")

    if a.write:
        p = _REPO / "models/shelflife_sku" / a.sku / "grasp_plan.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n  → beírva: {p.relative_to(_REPO)}")
    else:
        print("\n  (a beíráshoz: --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
