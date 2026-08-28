"""
shelflife_grasp_redesign.py — a fogás ÚJRATERVEZÉSE mért kritériumra

    python3 tools/shelflife_grasp_redesign.py            # söprés + a legjobb kipróbálása
    python3 tools/shelflife_grasp_redesign.py --scan     # csak a söprés
    python3 tools/shelflife_grasp_redesign.py --write    # a nyertes mentése

────────────────────────────────────────────────────────────────────────────
MIT JAVÍT
────────────────────────────────────────────────────────────────────────────
2026-08-06-án mérve, a jelenlegi fogásnál (`shelflife_hand_span.py`):

    hüvelyk   178° · 144 mm a talptól  ← a doboz 145 mm magas: A FEDÉLEN
    mutató    −75° · 124 mm
    középső   −68° ·  86 mm
    gyűrűs    −82° ·  55 mm
    kisujj    −89° ·  31 mm

    oppozíciós szög: 104°   (180° = szemben)
    magasságszórás:  113 mm = a doboz 78%-a

A felhasználó ugyanezt a két dolgot olvasta le a felvételről: „túl magasan
fogod" és „nincs szemben a hüvelyk és a mutatóujj". Az átmérő-söprés szerint
NAGYOBB TÁRGY nem segít: 45-től 95 mm-ig az oppozíció végig 51–56°.

Tehát a hiba a KÉZ ÁLLÁSÁBAN van, nem a termékben.

────────────────────────────────────────────────────────────────────────────
A KERESÉSI TÉR — és miért pont ez
────────────────────────────────────────────────────────────────────────────
1. BÜTYÖKSOR ÁLLÁSA. A tenyér Z tengelye maradjon −x (a robot felé nézzen,
   különben az IK a polc túloldaláról kérne fogást — ezt a
   `shelflife_pose_cylinder.py` már megmérte). Csak a tenyér X tengelyét
   forgatjuk, azaz a bütyöksort függőlegesből vízszintesbe.

2. FÜGGŐLEGES FOGÁSPONT. A mostani IK a doboz mértani középpontjára céloz,
   és onnan a hüvelyk a fedélre kerül. A célpont z-eltolása közvetlenül ezt
   a hibát mozgatja.

3. HÜVELYK-OPPOZÍCIÓ. Az `r_thumb_1_rot` / `_add` / `_flex` hármas dönti el,
   hogy a hüvelyk szembefordul-e az ujjakkal. A mostani érték (0.70 / −0.52 /
   −0.79) a TEJESDOBOZHOZ készült.

4. UJJAK ELŐHAJLÍTÁSA. Nyújtott ujjak síkot adnak, nem üreget.

────────────────────────────────────────────────────────────────────────────
A KRITÉRIUM — ELŐRE KIMONDVA
────────────────────────────────────────────────────────────────────────────
A `shelflife_hand_span.verdict()` négy feltétele, változtatás nélkül:

    oppozíció ≥ 120° · átlagmagasság 25–75% · szórás ≤ 40% ·
    a hüvelyk ≤ 85% (ne a fedélen legyen)

…és a kar érje el: IK < 8 mm, ízülettartalék > 0.15 rad.

⚠️ A söprés KINEMATIKAI (mj_forward, dinamika nélkül) — gyors, de csak
   IRÁNYT ad. A verdiktet a nyertesen lefuttatott VALÓDI fogás mondja ki:
   rásimuló zárás + emelés, és a mérce a termék követése.
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
from shelflife_hand_span import Span, verdict, HU, DIGITS   # noqa: E402

REACH_MM, MIN_MARGIN = 8.0, 0.15
MIN_OPEN_GAP_MM = 1.0       # a NYITOTT kéz ne lógjon bele a termékbe

# ⚠️ A NEGYEDIK KRITÉRIUM-HIBA EGY NAPON, ÉS EZ VOLT A LEGDRÁGÁBB.
#
# A söprés első változata csak a FÉLIG ZÁRT kéz érintkezési pontjait mérte, és
# a nyertes szép számokat adott: eredő 0.12, szögrés 118°, a hüvelyk az ujjsor
# közepén. A valódi futásban viszont NULLA ujj ért a dobozhoz, a doboz pedig
# lekerült a polcról.
#
# Az ok: a nyertes pózban a NYITOTT kéz már a dobozban van — a mutató −3.9,
# a középső −13.2, a gyűrűs −8.9 mm-rel. Olyan fogást kerestem, amit a kéz
# csak a terméken ÁTHALADVA tudna felvenni. A kar odafelé menet egyszerűen
# lelökte a dobozt.
#
# Ezért a nyitott kéz áthatolásmentessége HARD KAPU: ami ezen elbukik, azt
# meg sem mérjük tovább.
PROBE_CLOSE = 0.45          # félig zárt kéz a bemérésnél
OUT = _REPO / "results/shelflife_grasp_redesign"

# 1. bütyöksor: −90° (vízszintes) … +90°, a mostani a 0°-hoz van közel
KNUCKLE_DEG = range(-90, 91, 15)
# 2. függőleges fogáspont-eltolás a doboz KÖZEPÉHEZ képest
DZ_MM = (-40, -25, -10, 0, +10)
# 3. hüvelyk-oppozíció: (rot, add, flex)
THUMB = ((0.70, -0.52, -0.79),      # a mostani
         (1.10, -0.30, -0.40),
         (1.40, -0.10, 0.00),
         (1.40, 0.20, 0.40),
         (0.30, -0.70, -1.00))
# 4. ujjak előhajlítása
PREFLEX = (0.0, 0.25, 0.50, 0.75)


def frames() -> dict[str, np.ndarray]:
    """A bütyöksor-család. A tenyér Z tengelye VÉGIG −x marad."""
    out: dict[str, np.ndarray] = {}
    for deg in KNUCKLE_DEG:
        t = np.radians(deg)
        x = np.array([0.0, -np.sin(t), np.cos(t)])
        y = np.array([0.0, np.cos(t), np.sin(t)])
        R = G._frame(x, y)
        assert np.allclose(R[:, 2], [-1, 0, 0], atol=1e-6), deg
        out[f"{deg:+d}°"] = R
    return out


def hand_pose(thumb: tuple[float, float, float], preflex: float) -> dict:
    rot, add, flex = thumb
    d = {"r_thumb_1_rot": rot, "r_thumb_1_add": add, "r_thumb_1_flex": flex,
         "r_thumb_2": 0.0, "r_thumb_3": 0.0,
         "r_index_1_add": 0.35 * (1 - preflex),
         "r_ring_1_add": -0.20 * (1 - preflex),
         "r_little_1_add": -0.35 * (1 - preflex)}
    for f in ("index", "middle", "ring", "little"):
        d[f"r_{f}_1_flex"] = preflex
        d[f"r_{f}_2"] = preflex * 1.2
        d[f"r_{f}_3"] = preflex * 0.8
    return d


class Scanner:
    def __init__(self):
        # ⚠️ A tervet NEM lehet kikapcsolni: a `Robot.__init__` megköveteli
        #    (a terméktudásbázis épsége szándékosan kötelező). Csak a fogási
        #    finomeltolást nullázzuk, a fogáspontot magunk célozzuk meg.
        G.GRASP_TWEAK_CM = np.zeros(3)
        self.r = Robot()
        self.r.reset_home()
        self.g = self.r._r
        self.m, self.d = self.g.model, self.g.data
        self.s = Span(self.r)
        self.box, self.half = self.g.product_box()

    def set_hand(self, pose: dict) -> None:
        for act, _oc in self.g._pose.items():
            jid = int(self.m.actuator_trnid[act, 0])
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            lo, hi = self.m.jnt_range[jid]
            self.d.qpos[self.m.jnt_qposadr[jid]] = float(
                np.clip(pose.get(nm, 0.0), lo, hi))

    def open_gap(self, q: np.ndarray, pose: dict) -> float:
        """A NYITOTT kéz legkisebb távolsága a terméktől, mm. Negatív = benne van."""
        self.d.qpos[np.array(self.g._arm_q)] = q
        self.set_hand(pose)
        mujoco.mj_forward(self.m, self.d)
        ft = np.zeros(6)
        return min(float(mujoco.mj_geomDistance(self.m, self.d, gg,
                                                self.s.gid, 0.6, ft))
                   for gs in self.s.dg.values() for gg in gs) * 1000

    def probe(self, q: np.ndarray, pose: dict) -> dict:
        """Kinematikai bemérés: kar q-ban, kéz `pose`-ban, félig zárva."""
        og = self.open_gap(q, pose)
        self.d.qpos[np.array(self.g._arm_q)] = q
        self.set_hand(pose)
        # a félig zárt állapot a nyitott és a zárt közti interpoláció
        for act, (o, c) in self.g._pose.items():
            jid = int(self.m.actuator_trnid[act, 0])
            adr = self.m.jnt_qposadr[jid]
            cur = float(self.d.qpos[adr])
            self.d.qpos[adr] = cur + (c - cur) * PROBE_CLOSE
        mujoco.mj_forward(self.m, self.d)
        gm = self.s.geometry()
        gm["open_gap_mm"] = og
        return gm

    # ── 1. menet: hova álljon a kéz ─────────────────────────────────────────

    def stage_a(self) -> list[tuple]:
        print("  1. MENET — bütyöksor állása × függőleges fogáspont\n")
        print(f"  {'bütyök':>8}{'dz mm':>7}{'IK mm':>8}{'tartalék':>10}"
              f"{'nyitott rés':>13}{'eredő':>8}{'szögrés':>9}{'átlagmag.':>11}{'hü−ujjak':>11}")
        print("  " + "─" * 78)
        base = hand_pose(THUMB[0], 0.0)
        rows = []
        for nm, R in frames().items():
            for dz in DZ_MM:
                tgt = self.box + np.array([0.0, 0.0, dz / 1000.0])
                q, ep, _er = self.g.ik6_seed(tgt, R, restarts=8, iters=70)
                mg = self.g.joint_margin(q)
                if ep * 1000 > REACH_MM or mg < MIN_MARGIN:
                    continue
                gm = self.probe(q, base)
                if gm["open_gap_mm"] < MIN_OPEN_GAP_MM:
                    continue                      # a nyitott kéz beleérne
                ok, _ = verdict(gm)
                print(f"  {nm:>8}{dz:7d}{ep*1000:8.1f}{mg:10.2f}"
                      f"{gm['open_gap_mm']:10.1f} mm"
                      f"{gm['resultant']:8.2f}{gm['max_gap_deg']:8.0f}°"
                      f"{gm['height_mean_frac']*100:10.0f}%"
                      f"{gm['thumb_offset_mm']:9.0f} mm"
                      f"{'  ✅' if ok else ''}")
                # rangsor: elsődlegesen a hüvelyk helyessége, aztán az oppozíció
                score = (-gm["resultant"] - abs(gm["thumb_offset_mm"]) / 40.0
                         - max(0.0, gm["max_gap_deg"] - 180.0) / 90.0)
                rows.append((score, nm, dz, R, q, gm))
        if not rows:
            print("\n  ❌ egyetlen kar-állás sem érhető el")
            return []
        rows.sort(key=lambda t: -t[0])
        print("\n  a négy legjobb bütyök/dz: "
              + " · ".join(f"{r[1]}/{r[2]:+d}mm "
                           f"(eredő {r[5]['resultant']:.2f}, hü "
                           f"{r[5]['thumb_offset_mm']:+.0f} mm)"
                           for r in rows[:4]))
        return rows[:4]

    # ── 2. menet: hogyan álljon a kéz ───────────────────────────────────────

    def stage_b(self, tops: list[tuple]) -> tuple | None:
        print("\n  2. MENET — hüvelyk-oppozíció × ujjak előhajlítása\n")
        print(f"  {'bütyök':>8}{'dz':>6}{'hü.rot':>8}{'hü.add':>8}"
              f"{'hü.flex':>9}{'előhajl':>9}{'oppoz.':>8}{'átlagm.':>9}"
              f"{'hü−ujjak':>11}")
        print("  " + "─" * 78)
        best, shown = None, 0
        for _opp0, nm, dz, R, q, _gm0 in tops:
            for th in THUMB:
                for pf in PREFLEX:
                    pose = hand_pose(th, pf)
                    gm = self.probe(q, pose)
                    if gm["open_gap_mm"] < MIN_OPEN_GAP_MM:
                        continue                  # a nyitott kéz beleérne
                    ok, _bad = verdict(gm)
                    if ok or shown < 18:
                        shown += 1
                        print(f"  {nm:>8}{dz:6d}{th[0]:8.2f}{th[1]:8.2f}"
                              f"{th[2]:9.2f}{pf:9.2f}"
                              f"{gm['open_gap_mm']:8.1f} mm"
                              f"{gm['resultant']:7.2f}{gm['max_gap_deg']:8.0f}°"
                              f"{gm['height_mean_frac']*100:8.0f}%"
                              f"{gm['thumb_offset_mm']:9.0f} mm"
                              f"{'  ✅' if ok else ''}")
                    k = (ok, -gm["resultant"], -abs(gm["thumb_offset_mm"]))
                    if best is None or k > best[0]:
                        best = (k, nm, dz, R, q, th, pf, gm)
        return best


def try_it(nm: str, dz: int, R: np.ndarray, th, pf, verbose=True) -> dict:
    """A nyertes VALÓDI kipróbálása: rásimuló zárás, majd emelés."""
    from shelflife_conform import close_conforming
    from shelflife_grip_test import GripRig

    G.HAND_OPEN.update(hand_pose(th, pf))     # az új nyitott kézforma
    G.GRASP_TWEAK_CM = np.zeros(3)
    r = Robot(); r.reset_home()
    g = r._r
    box, _half = g.product_box()
    tgt = box + np.array([0.0, 0.0, dz / 1000.0])
    q, ep, _ = g.ik6_seed(tgt, R, restarts=16, iters=110)
    g.ramp_to(q, n=20, settle=140)

    rig = GripRig(r)
    p0 = rig.d.geom_xpos[rig.gid].copy()
    out = close_conforming(r)
    dg, parts, F = rig.contacts()
    med = len([x for x in parts if "medial" in x])
    moved = float(np.linalg.norm(rig.d.geom_xpos[rig.gid] - p0)) * 1000

    h0 = g.grasp_point().copy()
    q0 = rig.d.geom_xpos[rig.gid].copy()
    for _ in range(4):
        t = Pose("lift", g.grasp_point() + np.array([0, 0, 0.005]), r._R_des)
        r.approach_until(t, until="goal", guard_mm=1e9)
    rise = float(rig.d.geom_xpos[rig.gid][2] - q0[2]) * 1000
    hand = float(g.grasp_point()[2] - h0[2]) * 1000
    follow = rise / hand if abs(hand) > 1.0 else 0.0
    dg2, _p2, _F2 = rig.contacts()

    if verbose:
        print(f"\n  A NYERTES VALÓDI KIPRÓBÁLÁSA — {nm} / {dz:+d} mm\n")
        print(f"    IK {ep*1000:.1f} mm · záráskor a termék {moved:.1f} mm-t "
              f"mozdult")
        print(f"    {len(dg)} ujj {sorted(dg)} · {med} középperec · {F:.1f} N")
        print(f"    EMELÉS: kéz {hand:+.1f} mm · termék {rise:+.1f} mm · "
              f"KÖVETÉS {follow*100:.0f}%")
        print(f"    emelés után {len(dg2)} ujj érintkezik")
        good = follow > 0.8 and len(dg2) >= 2
        print("\n    " + ("✅ FELEMELTE" if good else "❌ nem emelte fel"))
    return {"digits": sorted(dg), "medial": med, "force_N": F,
            "moved_mm": moved, "follow": follow, "hand_mm": hand,
            "rise_mm": rise, "digits_after": sorted(dg2),
            "held": bool(follow > 0.8 and len(dg2) >= 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="csak a söprés")
    ap.add_argument("--write", action="store_true",
                    help="a nyertes mentése JSON-be")
    a = ap.parse_args()

    print("Shelf Life — a fogás ÚJRATERVEZÉSE\n")
    sc = Scanner()
    print(f"  termék: Ø{sc.half[0]*2000:.0f} mm × {sc.half[2]*2000:.0f} mm\n")
    tops = sc.stage_a()
    if not tops:
        return 1
    best = sc.stage_b(tops)
    if best is None:
        print("\n  ❌ a 2. menet nem adott jelöltet")
        return 1
    (ok, opp, _), nm, dz, R, _q, th, pf, gm = best
    print(f"\n  LEGJOBB JELÖLT: bütyöksor {nm} · fogáspont {dz:+d} mm · "
          f"hüvelyk {th} · előhajlítás {pf:.2f}")
    print(f"    nyitott rés {gm['open_gap_mm']:.1f} mm · "
          f"eredő {gm['resultant']:.2f} · szögrés "
          f"{gm['max_gap_deg']:.0f}° · átlagmagasság "
          f"{gm['height_mean_frac']*100:.0f}% · a hüvelyk az ujjsor közepéhez "
          f"képest {gm['thumb_offset_mm']:+.0f} mm")
    if not ok:
        _o, bad = verdict(gm)
        print(f"    ⚠️ a kritériumot NEM teljesíti: {' · '.join(bad)}")
    for k in DIGITS:
        if k in gm["angles"]:
            print(f"      {HU[k]:<10}{gm['angles'][k]:7.0f}°"
                  f"{gm['heights_mm'][k]:8.0f} mm")
    if a.scan:
        return 0 if ok else 1

    res = try_it(nm, dz, R, th, pf)
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "nyertes.json").write_text(json.dumps({
            "knuckle": nm, "dz_mm": dz, "thumb": list(th), "preflex": pf,
            "geometry": {k: v for k, v in gm.items() if k != "angles"},
            "run": res, "_note": "kinematikai söprés + egy valódi futás",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  mentve: {OUT/'nyertes.json'}")
    return 0 if res["held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
