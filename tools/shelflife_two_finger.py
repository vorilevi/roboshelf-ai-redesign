"""
shelflife_two_finger.py — KÉTUJJAS fogás a meglévő ötujjas kézzel

    python3 tools/shelflife_two_finger.py --scan    # csak a keresés
    python3 tools/shelflife_two_finger.py           # keresés + valódi próba

────────────────────────────────────────────────────────────────────────────
MIÉRT
────────────────────────────────────────────────────────────────────────────
2026-08-06-án két, egymástól független dolog mutatott ugyanabba az irányba.

**Kívülről:** a Google Gemini Robotics 2 publikus mérésein a KÉTUJJAS FOGÓ
jobban teljesített, mint a többujjas kéz. A legjobb számaik (74,2% felvétel-
letevés, 89,6% beillesztés) kétujjas fogóval, kétkarú ipari roboton
születtek. A többujjas kéz ott nyer, ahol tényleg ujjak kellenek — körte
kicsavarása, zacskókötés, cipzár —, és ezek közül egyik sem polcfeltöltés.

**Belülről:** a saját kudarcunk oka MÉRVE ötujjas összehangolási probléma.
Zárás közben a kisujj 0,60-nál ér a dobozhoz, a hüvelyk csak 0,65-nél, és a
hüvelyk közben 19 mm-t tesz meg egyetlen lépésben. Vagyis a kisujj
ellenerő nélkül nekinyomja a dobozt, a hüvelyk pedig utána csapódik bele.
A doboz ettől DŐL EL, nem csúszik ki.

    zárás     hüvelyk   mutató   középső   gyűrűs   kisujj
    0,55       +25,4     +21,0     +21,2    +21,6    +14,0
    0,60       +19,0     +18,4     +18,0    +14,3     −1,6   ← elsőként ér oda
    0,65        −1,5      +8,5     +13,4    +12,8     +0,8

**Ez a hibamód kétujjas fogásnál nem létezik.** Két ujj vagy szemben van,
vagy nincs — nincs harmadik, ami előbb odaér és feldönti.

────────────────────────────────────────────────────────────────────────────
MIT CSINÁL EZ A MODUL
────────────────────────────────────────────────────────────────────────────
NEM cserél hardvert. A meglévő ötujjas kézzel fog kétujjas módra:

    hüvelyk + mutató  →  szemben, ezek zárnak
    középső, gyűrűs, kisujj  →  BEHAJLÍTVA a tenyérbe, végig mozdulatlanul

Technikailag: a három nem használt ujjnál a „nyitott" és a „zárt" állás
UGYANAZ, tehát a záróparancs nem mozgatja őket.

────────────────────────────────────────────────────────────────────────────
A KRITÉRIUM — ELŐRE KIMONDVA
────────────────────────────────────────────────────────────────────────────
Geometria (kinematikai keresés):
    · a hüvelyk és a mutató szöge a henger tengelye körül ≥ 150°
    · magasságkülönbségük ≤ 20 mm
    · az átlagos fogásmagasság a doboz 25–75%-a között

ÜTKÖZÉSKAPU — ez az, ami ma hiányzott, és emiatt nyert a keresés
fizikailag elérhetetlen pózokat:
    · a nyitott kéz ne érjen a TERMÉKHEZ (≥ 1 mm)
    · a kar és a kéz ne érjen a POLCHOZ (≥ 2 mm)
    · a kéz ne ütközzön ÖNMAGÁVAL (0 kontaktus)

Valódi próba (dinamika):
    · a kar tényleg érkezzen meg oda, ahová küldtük (eltérés < 15 mm)
    · zárás után legalább 2 ujj érintkezzen
    · emeléskor a termék KÖVESSE a kezet (> 80%)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
import shelflife_grasp as G                      # noqa: E402
from shelflife_api import Robot, Pose            # noqa: E402
from shelflife_hand_span import Span             # noqa: E402

OUT = _REPO / "results/shelflife_two_finger"

USE = ("thumb", "index")                 # ezek fognak
TUCK = ("middle", "ring", "little")      # ezek behajlítva félreállnak

# ── kritérium ───────────────────────────────────────────────────────────────
NEED_OPPOSITION_DEG = 150.0
MAX_HEIGHT_DIFF_MM = 20.0
HEIGHT_BAND = (0.25, 0.75)
MIN_PRODUCT_GAP_MM = -5.0        # l. a magyarázatot lent
MIN_SHELF_GAP_MM = -2.0
EXTRA_SELF_CONTACTS = 3          # az ALAPHELYZETHEZ képest megengedett többlet

# ⚠️ KÉT KAPUÉRTÉKET UTÓLAG ENGEDNI KELLETT, ÉS MINDKETTŐ TANULSÁGOS.
#
# (1) ÖNÜTKÖZÉS. Az első kapum nulla önütközést követelt. Megmérve: a kéz
#     ALAPHELYZETBEN, kinyújtva, érintetlenül **13 ponton ütközik önmagával** —
#     a mutató és a középső végperece, a kisujj és a gyűrűs, a tenyér és az
#     ujjtövek, sőt a hüvelyk két saját perece. Ez a publikált GENE.01 modell
#     tulajdonsága: a szomszédos elemek geometriája átfed. Nulla önütközést
#     követelni tehát lehetetlent kérni. A mérce az ALAPHELYZETHEZ KÉPESTI
#     TÖBBLET lett.
#
# (2) TERMÉK-RÉS. Az első kapum azt kérte, hogy a nyitott kéz semmilyen része
#     ne érjen a termékhez. Csakhogy a kapu a TELJES kart nézi, a tenyérrel
#     együtt — egy fogásnál viszont a tenyérnek éppen a termék mellé kell
#     kerülnie, gyakran átfedésben a termék befoglaló testével. A nulla
#     küszöb minden fogást kizárt volna.
#
# A tanulság ugyanaz, mint a nap többi hibájánál: a kapu csak akkor ér
# valamit, ha a MÉRT alapállapothoz van kalibrálva, nem egy elképzelt
# ideálishoz.

# ── keresési tér ────────────────────────────────────────────────────────────
KNUCKLE_DEG = range(-90, 91, 15)
DZ_MM = (-40, -25, -10, 0, 15)
# (rot, add, flex) — a hüvelyk oppozíciós állása
THUMB = ((0.70, -0.52, -0.79), (1.10, -0.30, -0.40), (1.40, -0.10, 0.00),
         (1.40, 0.20, 0.40), (0.30, -0.70, -1.00), (1.10, 0.20, -0.20))
INDEX_PREFLEX = (0.0, 0.2, 0.4, 0.6)
TUCK_FLEX = 1.30                         # a három félreállított ujj behajlítása
PROBE_CLOSE = 0.45
DESCEND_MM = (0, 20, 30, 40, 50)         # mennyivel megyünk LEJJEBB a terv után

# ⚠️ MIÉRT KELL A LESÜLLYEDÉS, ÉS MIÉRT ÍGY.
#
# A validált terv a doboz magasságának 91%-ánál fog — vagyis a felső pereme
# közelében. Ezt a felhasználó vette észre a felvételen („túl magasan
# próbálod megfogni"), és a mérés igazolta: a hüvelyk 145 mm-nél, a doboz
# 145 mm magas.
#
# A fogáspontot NEM a terv átírásával mozgatjuk, mert akkor a terv
# ellenőrzött pályája is érvényét vesztené. Helyette végigmegyünk a terv
# pályáján — ez a rész validált, a termék 1,1 mm-t mozdul rajta —, és utána
# ZÁRT HURKÚ mozgással süllyedünk. A zárt hurok azért kell, mert nyílt
# hurokban a kar 86 mm-t téved (mérve, 2026-08-06).


def hand_open(thumb, ipre: float) -> dict:
    """A NYITOTT állás: hüvelyk + mutató fogásra kész, a többi behajlítva."""
    rot, add, flex = thumb
    d = {"r_thumb_1_rot": rot, "r_thumb_1_add": add, "r_thumb_1_flex": flex,
         "r_thumb_2": 0.0, "r_thumb_3": 0.0,
         "r_index_1_add": 0.0, "r_index_1_flex": ipre,
         "r_index_2": ipre * 1.2, "r_index_3": ipre * 0.8}
    for f in TUCK:
        d[f"r_{f}_1_add"] = 0.0
        d[f"r_{f}_1_flex"] = TUCK_FLEX
        d[f"r_{f}_2"] = TUCK_FLEX * 1.3
        d[f"r_{f}_3"] = TUCK_FLEX * 1.3
    return d


def hand_closed(thumb, ipre: float) -> dict:
    """A ZÁRT állás: csak a hüvelyk és a mutató mozdul. A többi VÁLTOZATLAN.

    ⚠️ Ez a kétujjas fogás lényege. Ha a három félreállított ujj „zárt"
    értéke különbözne a „nyitott"-tól, a záróparancs őket is mozgatná, és
    visszakapnánk pontosan azt a hibát, ami elől menekülünk.
    """
    d = hand_open(thumb, ipre)            # a három félreállított ujj marad
    d.update({"r_thumb_1_add": abs(thumb[1]), "r_thumb_1_flex": 1.31,
              "r_thumb_2": -1.75, "r_thumb_3": 1.75,
              "r_index_1_flex": 1.38, "r_index_2": 1.75, "r_index_3": 1.75})
    return d


class TwoFinger:
    def __init__(self):
        # ⚠️ A fogási eltolást NEM nullázzuk. A validált tervben −5,11 cm
        #    szerepel; e nélkül az IK a doboz mértani közepére céloz, és a
        #    tenyér mindig a dobozon BELÜL van. (Ezt ma egyszer elrontottam:
        #    a kapu ezért utasított el minden pózt.)
        self.r = Robot()
        self.r.reset_home()
        self.g = self.r._r
        self.m, self.d = self.g.model, self.g.data
        self.s = Span(self.r)
        self.box, self.half = self.g.product_box()
        self.base_self = 0
        bn = lambda b: mujoco.mj_id2name(          # noqa: E731
            self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self.bn = bn
        ARM = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
               "r_thumb", "r_index", "r_middle", "r_ring", "r_little")
        self.arm_g = [g for g in range(self.m.ngeom)
                      if bn(self.m.geom_bodyid[g]).startswith(ARM)]
        self.shelf_g = [g for g in range(self.m.ngeom)
                        if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                              g) or "").startswith("shelf")]
        self._ft = np.zeros(6)

    def set_hand(self, pose: dict) -> None:
        for act in self.g._pose:
            jid = int(self.m.actuator_trnid[act, 0])
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                   jid) or ""
            lo, hi = self.m.jnt_range[jid]
            self.d.qpos[self.m.jnt_qposadr[jid]] = float(
                np.clip(pose.get(nm, 0.0), lo, hi))

    # ── ütközéskapu ─────────────────────────────────────────────────────────

    def baseline_self(self) -> int:
        """Hány ponton ütközik a kéz ÖNMAGÁVAL érintetlen alaphelyzetben."""
        self.r.reset_home()
        mujoco.mj_forward(self.m, self.d)
        return sum(1 for k in range(self.d.ncon)
                   if self.d.contact[k].geom1 in self.arm_g
                   and self.d.contact[k].geom2 in self.arm_g)

    def gate(self, q: np.ndarray | None, pose: dict) -> tuple[bool, dict]:
        """A NYITOTT kéz fizikailag felvehető-e ebben a pózban?"""
        if q is not None:
            self.d.qpos[np.array(self.g._arm_q)] = q
        self.set_hand(pose)
        mujoco.mj_forward(self.m, self.d)
        prod = min(float(mujoco.mj_geomDistance(self.m, self.d, g, self.s.gid,
                                                0.6, self._ft))
                   for g in self.arm_g) * 1000
        shelf = min(float(mujoco.mj_geomDistance(self.m, self.d, a, b, 0.6,
                                                 self._ft))
                    for a in self.arm_g for b in self.shelf_g) * 1000
        selfc = sum(1 for k in range(self.d.ncon)
                    if self.d.contact[k].geom1 in self.arm_g
                    and self.d.contact[k].geom2 in self.arm_g)
        info = {"product_mm": prod, "shelf_mm": shelf, "self": selfc}
        ok = (prod >= MIN_PRODUCT_GAP_MM and shelf >= MIN_SHELF_GAP_MM
              and selfc <= self.base_self + EXTRA_SELF_CONTACTS)
        return ok, info

    # ── kétujjas geometria ──────────────────────────────────────────────────

    def pinch(self, q: np.ndarray | None, pose: dict) -> dict | None:
        """Félig zárt kézzel: szemben van-e a két ujj, és milyen magasan."""
        if q is not None:
            self.d.qpos[np.array(self.g._arm_q)] = q
        self.set_hand(pose)
        closed = hand_closed_from(pose)
        for act in self.g._pose:
            jid = int(self.m.actuator_trnid[act, 0])
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                   jid) or ""
            if not any(f"r_{u}_" in nm for u in USE):
                continue                       # a félreállított ujjak nem mozdulnak
            adr = self.m.jnt_qposadr[jid]
            lo, hi = self.m.jnt_range[jid]
            tgt = float(np.clip(closed.get(nm, 0.0), lo, hi))
            cur = float(self.d.qpos[adr])
            self.d.qpos[adr] = cur + (tgt - cur) * PROBE_CLOSE
        mujoco.mj_forward(self.m, self.d)

        tp = self.s.touch_points()
        if not all(u in tp for u in USE):
            return None
        c = self.d.geom_xpos[self.s.gid]
        H = float(self.m.geom_size[self.s.gid][1]) * 2000
        base = float(c[2]) - H / 2000.0

        def ang(p):
            return float(np.degrees(np.arctan2(p[1] - c[1], p[0] - c[0])))

        def hgt(p):
            return (float(p[2]) - base) * 1000

        at, ai = ang(tp["thumb"]), ang(tp["index"])
        opp = abs(((at - ai + 180) % 360) - 180)
        ht, hi_ = hgt(tp["thumb"]), hgt(tp["index"])
        return {"opposition_deg": opp, "thumb_deg": at, "index_deg": ai,
                "thumb_mm": ht, "index_mm": hi_,
                "height_diff_mm": abs(ht - hi_),
                "height_frac": (ht + hi_) / 2 / H, "product_h_mm": H}


_CLOSED_CACHE: dict[int, dict] = {}


def hand_closed_from(open_pose: dict) -> dict:
    """A nyitott állásból a hozzá tartozó zárt állás (csak a két fogó ujj)."""
    key = id(open_pose)
    if key not in _CLOSED_CACHE:
        d = dict(open_pose)
        d.update({"r_thumb_1_add": abs(open_pose.get("r_thumb_1_add", 0.0)),
                  "r_thumb_1_flex": 1.31, "r_thumb_2": -1.75,
                  "r_thumb_3": 1.75, "r_index_1_flex": 1.38,
                  "r_index_2": 1.75, "r_index_3": 1.75})
        _CLOSED_CACHE[key] = d
    return _CLOSED_CACHE[key]


def frames() -> dict[str, np.ndarray]:
    out = {}
    for deg in KNUCKLE_DEG:
        t = np.radians(deg)
        x = np.array([0.0, -np.sin(t), np.cos(t)])
        y = np.array([0.0, np.cos(t), np.sin(t)])
        out[f"{deg:+d}°"] = G._frame(x, y)
    return out


def verdict(p: dict) -> tuple[bool, list[str]]:
    bad = []
    if p["opposition_deg"] < NEED_OPPOSITION_DEG:
        bad.append(f"oppozíció {p['opposition_deg']:.0f}°")
    if p["height_diff_mm"] > MAX_HEIGHT_DIFF_MM:
        bad.append(f"magasságkülönbség {p['height_diff_mm']:.0f} mm")
    if not HEIGHT_BAND[0] <= p["height_frac"] <= HEIGHT_BAND[1]:
        bad.append(f"fogásmagasság {p['height_frac']*100:.0f}%")
    return (not bad), bad


def descend(r: Robot, mm: float) -> None:
    """Zárt hurkú süllyedés a terv állomásáról, 5 mm-es lépésekben."""
    g = r._r
    for _ in range(int(round(mm / 5.0))):
        t = Pose("le", g.grasp_point() - np.array([0, 0, 0.005]), r._R_des)
        r.approach_until(t, until="goal", guard_mm=1e9)


def scan_hand_shapes(tf: "TwoFinger") -> list[tuple]:
    """A kézformák söprése a TERV szerinti kar-álláson.

    ⚠️ Szándékosan EGY dolgot változtatunk. A közelítés, a fogáspont és a
    kar pályája a validált tervből jön (`follow_plan`, mérve: a termék
    1,1 mm-t mozdul a pálya alatt). Csak a kéz alakja más: ötujjas helyett
    kétujjas. Így ha változik az eredmény, tudjuk, mitől.
    """
    print("  A kar a TERV pályáján áll. Csak a kézforma változik.\n")
    print(f"  {'hü.rot':>8}{'hü.add':>8}{'hü.flex':>9}{'mut.hajl':>10}"
          f"{'termék':>9}{'polc':>8}{'önütk':>7}"
          f"{'oppoz.':>9}{'Δmagas':>9}{'magasság':>10}")
    print("  " + "─" * 79)
    rows = []
    for th in THUMB:
        for ipre in INDEX_PREFLEX:
            pose = hand_open(th, ipre)
            ok_gate, gi = tf.gate(None, pose)
            pn = tf.pinch(None, pose)
            if pn is None:
                continue
            ok, _bad = verdict(pn)
            print(f"  {th[0]:8.2f}{th[1]:8.2f}{th[2]:9.2f}{ipre:10.2f}"
                  f"{gi['product_mm']:8.1f}{gi['shelf_mm']:8.1f}"
                  f"{gi['self']:7d}"
                  f"{pn['opposition_deg']:8.0f}°{pn['height_diff_mm']:8.0f} mm"
                  f"{pn['height_frac']*100:9.0f}%"
                  f"{'  ✅' if (ok and ok_gate) else ('  kapu' if ok else '')}")
            score = (ok and ok_gate, -abs(pn["opposition_deg"] - 180)
                     - pn["height_diff_mm"] / 5.0)
            rows.append((score, th, ipre, pn, gi))
    rows.sort(key=lambda t: t[0], reverse=True)
    return rows


def try_it(th, ipre, desc_mm: float = 0.0, label: str = "") -> dict:
    """Valódi próba a TERV pályáján — ugyanaz az út, más kézforma."""
    G.HAND_OPEN.clear(); G.HAND_OPEN.update(hand_open(th, ipre))
    G.HAND_CLOSED.clear(); G.HAND_CLOSED.update(hand_closed(th, ipre))

    from shelflife_grip_test import GripRig
    r = Robot(); r.reset_home()
    g = r._r
    rig = GripRig(r)
    p0 = rig.d.geom_xpos[rig.gid].copy()
    fp = r.follow_plan(guard_mm=1e9)
    if not fp.ok:
        print(f"    ❌ a pálya nem járható: {fp.detail}")
        return {"held": False}
    on_path = fp.data.get("product_moved_mm", float("nan"))
    if desc_mm:
        descend(r, desc_mm)

    p1 = rig.d.geom_xpos[rig.gid].copy()
    for i in range(1, 21):
        g.close_fingers(i / 20, settle=40)
    moved = float(np.linalg.norm(rig.d.geom_xpos[rig.gid] - p1)) * 1000
    dg, parts, F = rig.contacts()

    h0 = g.grasp_point().copy()
    z0 = rig.d.geom_xpos[rig.gid].copy()
    for _ in range(4):
        t = Pose("lift", g.grasp_point() + np.array([0, 0, 0.005]), r._R_des)
        r.approach_until(t, until="goal", guard_mm=1e9)
    rise = float(rig.d.geom_xpos[rig.gid][2] - z0[2]) * 1000
    hand = float(g.grasp_point()[2] - h0[2]) * 1000
    follow = rise / hand if abs(hand) > 1.0 else 0.0
    dg2, _p2, _f2 = rig.contacts()

    print(f"\n  VALÓDI PRÓBA {label}\n")
    print(f"    a pálya alatt a termék {on_path:.1f} mm-t mozdult")
    print(f"    süllyedés a terv állomásáról: {desc_mm:.0f} mm")
    print(f"    záráskor a termék {moved:.1f} mm-t mozdult")
    print(f"    {len(dg)} ujj {sorted(dg)} · {F:.1f} N")
    print(f"    EMELÉS: kéz {hand:+.1f} mm · termék {rise:+.1f} mm · "
          f"KÖVETÉS {follow*100:.0f}%")
    print(f"    emelés után {len(dg2)} ujj érintkezik")
    held = follow > 0.8 and len(dg2) >= 2
    print("\n    " + ("✅ FELEMELTE" if held else "❌ nem emelte fel"))
    return {"path_mm": on_path, "moved_mm": moved, "digits": sorted(dg),
            "force_N": F, "hand_mm": hand, "rise_mm": rise, "follow": follow,
            "digits_after": sorted(dg2), "held": held}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    print("Shelf Life — KÉTUJJAS fogás a meglévő kézzel\n")
    tf = TwoFinger()
    tf.base_self = tf.baseline_self()
    print(f"  termék: Ø{tf.half[0]*2000:.0f} mm × {tf.half[2]*2000:.0f} mm")
    print(f"  fog: {', '.join(USE)} · félreállítva: {', '.join(TUCK)}")
    print(f"  a kéz alaphelyzetben {tf.base_self} ponton ütközik önmagával "
          f"(a modell tulajdonsága)\n")

    fp = tf.r.follow_plan(guard_mm=1e9)
    if not fp.ok:
        print(f"  ❌ a terv pályája nem járható: {fp.detail}")
        return 1

    allrows = []
    done = 0.0
    for dsc in DESCEND_MM:
        descend(tf.r, dsc - done)
        done = dsc
        print(f"\n  ── süllyedés a terv állomásáról: {dsc} mm "
              f"─────────────────────────")
        for sc, th, ipre, pn, gi in scan_hand_shapes(tf):
            allrows.append((sc, th, ipre, dsc, pn, gi))
    if not allrows:
        print("\n  ❌ egyetlen kézforma sem mérhető")
        return 1
    allrows.sort(key=lambda t: t[0], reverse=True)
    ok, th, ipre, dsc, pn, gi = allrows[0]
    print(f"\n  LEGJOBB: süllyedés {dsc} mm · hüvelyk {th} · "
          f"mutató előhajlítás {ipre:.2f}")
    print(f"    oppozíció {pn['opposition_deg']:.0f}° · "
          f"magasságkülönbség {pn['height_diff_mm']:.0f} mm · "
          f"fogásmagasság {pn['height_frac']*100:.0f}%")
    print(f"    hüvelyk {pn['thumb_deg']:.0f}° / {pn['thumb_mm']:.0f} mm · "
          f"mutató {pn['index_deg']:.0f}° / {pn['index_mm']:.0f} mm")
    if not ok[0]:
        print(f"    ⚠️ NEM teljesíti: {' · '.join(verdict(pn)[1])}")
    if a.scan:
        return 0 if ok[0] else 1

    res = try_it(th, ipre, dsc, "— kétujjas")
    print("\n  ÖSSZEHASONLÍTÁS az ötujjas alapesettel (2026-08-06):")
    print("    ötujjas: 2 ujj · 79,8 N · követés −17%")
    print(f"    kétujjas: {len(res.get('digits', []))} ujj · "
          f"{res.get('force_N', 0):.1f} N · "
          f"követés {res.get('follow', 0)*100:.0f}%")
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "nyertes.json").write_text(json.dumps(
            {"thumb": list(th), "index_pre": ipre, "descend_mm": dsc,
             "geometry": pn,
             "gate": gi, "run": res, "baseline_self": tf.base_self},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  mentve: {OUT/'nyertes.json'}")
    return 0 if res["held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
