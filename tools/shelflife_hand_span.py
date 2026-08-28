"""
shelflife_hand_span.py — MEKKORA TÁRGYAT fog ez a kéz, és HOL fogja meg?

    python3 tools/shelflife_hand_span.py            # a jelenlegi doboz
    python3 tools/shelflife_hand_span.py --sweep    # átmérő-söprés

────────────────────────────────────────────────────────────────────────────
MIÉRT
────────────────────────────────────────────────────────────────────────────
2026-08-06, a fogásfelvétel megnézése után a felhasználó két dolgot állapított
meg a képekről:

    1. „túl magasan próbálod megfogni a dobozt"
    2. „nincsenek szemben a hüvelyk és a mutatóujj"

…és feltette a kérdést, hogy nem kellene-e NAGYOBB tárgy (0,5 literes doboz).

Mindhárom eldönthető méréssel, és egyiket sem szabad tippelni. Ez a modul
azt méri meg, amit a szem lát:

  · OPPOZÍCIÓS SZÖG — a hüvelyk és a négy ujj érintkezési pontja a henger
    tengelye körül hány fokra van egymástól. 180° = tökéletesen szemben,
    0° = ugyanazon az oldalon. Ez a 2. állítás számszerű alakja.

  · FOGÁSMAGASSÁG — az érintkezési pontok magassága a doboz talpától, a
    doboz magasságának százalékában. 50% = a közepén fogja. Ez az 1. állítás.

  · MEKKORA ÁTMÉRŐRE ZÁR — ugyanez végigmérve 45…95 mm átmérőn. Erre a
    kérdésre („nagyobb tárgy kell?") csak így lehet felelni.

────────────────────────────────────────────────────────────────────────────
MÓDSZER
────────────────────────────────────────────────────────────────────────────
A henger ütközőgeometriáját FUTÁSIDŐBEN átméretezzük (`geom_size`, és vele
`geom_rbound`, különben a széles fázis eltéved), a testet a polcra állítjuk,
a tömeget a térfogattal arányosan skálázzuk a MÉRT 343 g / 330 ml-ből.
Minden átmérőhöz ÚJ inverz kinematika készül, hogy a fogáspont a tárgyhoz
igazodjon — különben a méret hatását összekevernénk a fogáspontéval.

⚠️ Ez a modul SEMMILYEN SKU-fájlt nem ír. A söprés hipotetikus hengereken
   dolgozik; ha egy méret jónak bizonyul, a valódi terméket külön kell
   felvenni, mért adatokkal (l. `docs/roboshelf_sku_sema.md`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
import shelflife_grasp as G                      # noqa: E402
from shelflife_api import Robot                  # noqa: E402

DIGITS = ("thumb", "index", "middle", "ring", "little")
HU = {"thumb": "hüvelyk", "index": "mutató", "middle": "középső",
      "ring": "gyűrűs", "little": "kisujj"}
FINGERS = ("index", "middle", "ring", "little")

# a MÉRT referencia, amiből a hipotetikus méretek tömegét skálázzuk
REF_ML, REF_G = 330.0, 343.0

# ELŐRE KIMONDOTT kritérium
MAX_RESULTANT = 0.50            # a befelé mutató normálisok eredője (0 = kiegyensúlyozott)
MAX_GAP_DEG = 180.0             # a legnagyobb szögrés két szomszédos kontaktus közt
MAX_THUMB_OFFSET_MM = 20.0      # a hüvelyk és az ujjak ÁTLAGA közti magasságkülönbség
HEIGHT_BAND = (0.25, 0.75)      # az átlagos fogásmagasság sávja
MAX_THUMB_FRAC = 0.90           # a hüvelyk ne a fedélen legyen

# ⚠️ A KRITÉRIUM KÉTSZER VOLT ROSSZ EGY NAPON. MINDKETTŐT ÉRDEMES MEGŐRIZNI.
#
# (1) Az első változat csak az ÁTLAGOS fogásmagasságot nézte: 61% → „✅ a
#     középső sávban". Közben a hüvelyk a doboz 99%-ánál volt (a FEDÉLEN), a
#     kisujj 21%-nál. Az átlag azért stimmelt, mert a szélsőségek kioltották
#     egymást. Ugyanaz a hiba, amit a `shelflife_pose_cylinder.py` már
#     egyszer megtalált és feljegyzett — és mégis újra elkövettem.
#
# (2) A javításom viszont az UJJAK SZÓRÁSÁT korlátozta 40%-ra. Ez fizikailag
#     téves: egy hengert az ember is úgy fog, hogy a négy ujja a tengely
#     mentén EGYMÁS ALATT sorakozik — a szórás ott is 45–60%. A szórás nem
#     hiba.
#
#     Ami hiba, az a HÜVELYK helye: ha a hüvelyk az ujjsor FÖLÖTT van, a
#     szorítás nyomatékot ad, és a doboz eldől — pontosan ez látszott a
#     2026-08-06-i felvételen. A hüvelyknek az ujjsor KÖZEPÉVEL kell
#     szemben lennie. Ezért a mérce: |hüvelyk − az ujjak átlaga| ≤ 20 mm.
#
# (3) A SZÖGRE pedig „oppozíciós szöget" használtam: a hüvelyk szöge mínusz a
#     négy ujj KÖRKÖRÖS ÁTLAGA. Ez akkor is 164°-ot ad, amikor a hüvelyk és a
#     mutató PONTOSAN ugyanott van (mindkettő 80°-nál), mert a másik három ujj
#     körbeér a hengeren, és az átlaguk a túloldalra esik. Egy körbeérő
#     ponthalmaznak nincs értelmes „átlagiránya".
#
#     A helyes, szabványos mérce két szám:
#       · EREDŐ  = |Σ egységvektor| / N — 0, ha a kontaktusok kiegyenlítik
#         egymást; 1, ha mind ugyanazon az oldalon van. Ez a tényleges
#         erőzárás feltétele.
#       · LEGNAGYOBB SZÖGRÉS két szomszédos kontaktus közt. Ha ez > 180°,
#         minden kontaktus egy félsíkba esik, és a tárgy kicsúszik a rés felé.


class Span:
    def __init__(self, robot: Robot):
        self.r = robot
        self.g = robot._r
        self.m, self.d = self.g.model, self.g.data
        self.bn = lambda b: mujoco.mj_id2name(          # noqa: E731
            self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self.gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                     "product_0_col")
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                     "product_0")
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                "product_0_free")
        self.adr = self.m.jnt_qposadr[jid]
        self.dg = {dg: [g for g in range(self.m.ngeom)
                        if f"r_{dg}_" in self.bn(self.m.geom_bodyid[g])]
                   for dg in DIGITS}
        self._ft = np.zeros(6)

    # ── geometria ────────────────────────────────────────────────────────────

    def resize(self, diam_mm: float, height_mm: float) -> None:
        """A henger átméretezése futásidőben, tömeggel és rbounddal együtt."""
        rad, half = diam_mm / 2000.0, height_mm / 2000.0
        self.m.geom_size[self.gid][:2] = [rad, half]
        # ⚠️ rbound NÉLKÜL a széles fázis a RÉGI mérettel dolgozna, és a
        #    kontaktusok egy része néma maradna. Hengerre: √(r² + h²).
        self.m.geom_rbound[self.gid] = float(np.hypot(rad, half))
        ml = np.pi * rad ** 2 * (2 * half) * 1e6          # m³ → ml
        self.m.body_mass[self.bid] = REF_G / 1000.0 * ml / REF_ML
        mujoco.mj_setConst(self.m, self.d)

    def stand_on_shelf(self, shelf_z: float) -> None:
        mujoco.mj_forward(self.m, self.d)
        half = float(self.m.geom_size[self.gid][1])
        off = self.d.geom_xpos[self.gid] - self.d.qpos[self.adr:self.adr + 3]
        self.d.qpos[self.adr + 2] = shelf_z + half - off[2]
        self.d.qpos[self.adr + 3:self.adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    # ── mérés ────────────────────────────────────────────────────────────────

    def touch_points(self) -> dict[str, np.ndarray]:
        """Ujjanként a TÁRGY felszínén az a pont, amelyik a legközelebb van.

        Ez akkor is értelmes, ha még nincs tényleges kontaktus — így a fogás
        IRÁNYA a záródás előtt is mérhető, nem csak utána.
        """
        out = {}
        for dg, gs in self.dg.items():
            best, bd = None, 1e9
            for g in gs:
                dist = float(mujoco.mj_geomDistance(self.m, self.d, g,
                                                    self.gid, 0.6, self._ft))
                if dist < bd:
                    bd, best = dist, self._ft[3:6].copy()   # a TÁRGY-oldali pont
            if best is not None:
                out[dg] = best
        return out

    def geometry(self) -> dict:
        """Oppozíciós szög és fogásmagasság a jelenlegi állásban."""
        c = self.d.geom_xpos[self.gid]
        half = float(self.m.geom_size[self.gid][1])
        base = float(c[2]) - half
        tp = self.touch_points()

        def ang(p):                       # a tengely körüli szög, fokban
            return float(np.degrees(np.arctan2(p[1] - c[1], p[0] - c[0])))

        def hgt(p):                       # magasság a talptól, mm
            return (float(p[2]) - base) * 1000

        angs = {k: ang(v) for k, v in tp.items()}
        a = np.radians(np.array(list(angs.values())))
        resultant = float(abs(np.mean(np.exp(1j * a))))
        srt = np.sort(np.degrees(a) % 360)
        gaps = np.diff(np.concatenate([srt, [srt[0] + 360]]))
        max_gap = float(gaps.max())
        hs = {k: hgt(v) for k, v in tp.items()}
        h_all = np.array(list(hs.values()))
        H = half * 2000
        fh = float(np.mean([hs[f] for f in FINGERS if f in hs]))
        return {
            "resultant": resultant,
            "max_gap_deg": max_gap,
            "angles": angs,
            "heights_mm": hs,
            "height_mean_frac": float(np.mean(h_all) / H),
            "height_spread_mm": float(h_all.max() - h_all.min()),
            "spread_frac": float((h_all.max() - h_all.min()) / H),
            "finger_mean_mm": fh,
            # a DÖNTŐ szám: a hüvelyk az ujjsor közepe fölött/alatt hány mm-rel
            "thumb_offset_mm": float(hs.get("thumb", 0.0) - fh),
            "thumb_frac": float(hs["thumb"] / H) if "thumb" in hs else 1.0,
            "product_h_mm": H,
        }


def verdict(gm: dict) -> tuple[bool, list[str]]:
    """A négy feltétel EGYÜTT. Egy is elbukik → nincs fogás."""
    bad = []
    if gm["resultant"] > MAX_RESULTANT:
        bad.append(f"eredő {gm['resultant']:.2f} (> {MAX_RESULTANT:.2f}) — "
                   f"a kontaktusok egy irányba húznak")
    if gm["max_gap_deg"] > MAX_GAP_DEG:
        bad.append(f"szögrés {gm['max_gap_deg']:.0f}° (> {MAX_GAP_DEG:.0f}°) — "
                   f"a tárgy kicsúszik a rés felé")
    if abs(gm["thumb_offset_mm"]) > MAX_THUMB_OFFSET_MM:
        bad.append(f"a hüvelyk {gm['thumb_offset_mm']:+.0f} mm-rel az ujjsor "
                   f"közepéhez képest (max ±{MAX_THUMB_OFFSET_MM:.0f})")
    if not HEIGHT_BAND[0] <= gm["height_mean_frac"] <= HEIGHT_BAND[1]:
        bad.append(f"átlagmagasság {gm['height_mean_frac']*100:.0f}%")
    if gm["thumb_frac"] > MAX_THUMB_FRAC:
        bad.append(f"a hüvelyk {gm['thumb_frac']*100:.0f}%-nál (a fedélen)")
    return (not bad), bad


def _pose_to_product(r: Robot) -> tuple[bool, float]:
    """A kart a TÁRGY köré visszük, friss IK-val (nem a rögzített tervvel)."""
    g = r._r
    box, _ = g.product_box()
    R = G.GRASP_POSES[(g.plan or {}).get("pose", "right_thumb_up")]
    q, ep, _ = g.ik6_seed(box, R, restarts=16, iters=110)
    g.ramp_to(q, n=18, settle=120)
    return ep * 1000 < 15.0, ep * 1000


def report_current() -> int:
    print("Shelf Life — HOL és MIVEL fogja meg a kéz a dobozt\n")
    r = Robot(); r.reset_home()
    s = Span(r)
    fp = r.follow_plan(guard_mm=1e9)
    if not fp.ok:
        print(f"  a pálya nem járható: {fp.detail}")
        return 1
    gm = s.geometry()

    print(f"  a doboz magassága: {gm['product_h_mm']:.0f} mm\n")
    print(f"  {'ujj':<10}{'szög a tengely körül':>22}{'magasság a talptól':>21}")
    print("  " + "─" * 53)
    for k in DIGITS:
        if k in gm["angles"]:
            h = gm["heights_mm"][k]
            print(f"  {HU[k]:<10}{gm['angles'][k]:19.0f}°"
                  f"{h:16.0f} mm{h / gm['product_h_mm'] * 100:7.0f}%")


    print(f"\n  ERŐZÁRÁS: eredő {gm['resultant']:.2f} (0 = kiegyensúlyozott) · "
          f"legnagyobb szögrés {gm['max_gap_deg']:.0f}°")
    print(f"\n  FOGÁSMAGASSÁG: átlag a doboz {gm['height_mean_frac']*100:.0f}%-a"
          f" · szórás {gm['height_spread_mm']:.0f} mm "
          f"({gm['spread_frac']*100:.0f}%) · a hüvelyk "
          f"{gm['thumb_frac']*100:.0f}%-nál")

    ok, bad = verdict(gm)
    print("\n  " + ("✅ MINDEN FELTÉTEL TELJESÜL" if ok else
                    "❌ ELBUKIK: " + " · ".join(bad)))
    return 0 if ok else 1


DIAMS = (45, 52, 58, 66, 75, 85, 95)
HEIGHT_OF = {45: 130, 52: 140, 58: 145, 66: 168, 75: 180, 85: 190, 95: 200}


def sweep() -> int:
    """Mekkora átmérőnél lesz szembefogás? Ez a felelet a „nagyobb tárgy" kérdésre."""
    print("Shelf Life — MEKKORA TÁRGYAT fog meg ez a kéz\n")
    print("  Minden átmérőhöz ÚJ inverz kinematika készül, hogy a fogáspont")
    print("  a tárgyhoz igazodjon. A tömeg a mért 343 g / 330 ml-ből skálázva.\n")
    print(f"  {'Ø mm':>6}{'magas':>7}{'tömeg':>8}{'IK':>7}"
          f"{'eredő':>9}{'szögrés':>8}{'átlagmag.':>11}{'hü−ujjak':>11}")
    print("  " + "─" * 68)

    best = None
    for dia in DIAMS:
        h = HEIGHT_OF[dia]
        r = Robot(); r.reset_home()
        s = Span(r)
        shelf_z = float(s.d.geom_xpos[s.gid][2]
                        - s.m.geom_size[s.gid][1])       # a mostani talp
        s.resize(dia, h)
        s.stand_on_shelf(shelf_z)
        ok_ik, ep = _pose_to_product(r)
        if not ok_ik:
            print(f"  {dia:6d}{h:7d}{s.m.body_mass[s.bid]*1000:7.0f} g"
                  f"{ep:6.1f}   nem éri el")
            continue
        gm = s.geometry()
        okay, _ = verdict(gm)
        print(f"  {dia:6d}{h:7d}{s.m.body_mass[s.bid]*1000:7.0f} g"
              f"{ep:6.1f}{gm['resultant']:9.2f}{gm['max_gap_deg']:8.0f}°"
              f"{gm['height_mean_frac']*100:10.0f}%"
              f"{gm['thumb_offset_mm']:9.0f} mm"
              f"{'  ✅' if okay else ''}")
        k = (-gm["resultant"], -abs(gm["thumb_offset_mm"]))
        if best is None or k > best[0]:
            best = (k, dia, gm)

    if best is None:
        print("\n  ❌ egyetlen méret sem elérhető — a hiba nem a méretben van")
        return 1
    _, dia, gm = best
    print(f"\n  A LEGJOBB: Ø{dia} mm · eredő {gm['resultant']:.2f} · "
          f"fogásmagasság {gm['height_mean_frac']*100:.0f}%")
    if gm["resultant"] > MAX_RESULTANT:
        print(f"  ⚠️ EZ IS a küszöb FÖLÖTT van: a "
              f"nagyobb tárgy ÖNMAGÁBAN nem oldja meg\n     a szembefogást. "
              f"Ez jelentés, nem hiba.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="átmérő-söprés: mekkora tárgyra zár szembe a kéz")
    a = ap.parse_args()
    return sweep() if a.sweep else report_current()


if __name__ == "__main__":
    raise SystemExit(main())
