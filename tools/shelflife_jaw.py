"""
shelflife_jaw.py — a fogási pont MEGSZERKESZTÉSE (nem keresése)

    python3 tools/shelflife_jaw.py            # mérés + javaslat
    python3 tools/shelflife_jaw.py --write    # beírja a fogási tervbe

────────────────────────────────────────────────────────────────────────────
MIÉRT SZERKESZTÉS ÉS NEM KERESÉS
────────────────────────────────────────────────────────────────────────────
Eddig kitaláltam egy fogási pontot, majd RÁCSON KERESTEM hozzá jó pozíciót —
117 649 ponton, egyre drágább ütközésszámítással, és minden kör egy újabb
korrekció volt egy rossz kiinduláson.

Ez rossz kérdés volt. A kéz mérete alapján a fogás LÉTEZIK:

    ujjhossz 8.2 cm · hüvelyk 11.0 cm · tenyérszélesség 9.0 cm
    hüvelyk–mutató nyílás 18.2 cm nyitva
    karton 7.9 × 8.0 × 20.4 cm

Ha tudjuk, hogy létezik, akkor nem keresni kell, hanem MEGSZERKESZTENI.

────────────────────────────────────────────────────────────────────────────
A SZERKESZTÉS
────────────────────────────────────────────────────────────────────────────
Nyitott kézben a hüvelyk belső felülete és a négy ujj belső felülete egy
„állkapcsot" határoz meg. Ennek van

    · egy TENGELYE      — a hüvelyk felőli és az ujjak felőli pont közti irány
    · egy KÖZÉPPONTJA   — a két pont felezőpontja
    · egy NYÍLÁSA       — a két felület távolsága

A fogási pont ebből adódik: a termék középpontja a felezőpontba kerül. Nincs
mit keresni — legfeljebb ellenőrizni, hogy a nyílás elég-e, és hogy a záródás
tényleg erőzárást ad.

Ez egy számítás, nem egy rács.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                       # noqa: E402
import shelflife_grasp as _G                        # noqa: E402
from shelflife_api import Robot                     # noqa: E402

THUMB = ("r_thumb",)
FINGERS = ("r_index", "r_middle", "r_ring", "r_little")


# Csak a VALÓDI FOGÓFELÜLETEK számítanak: a középső és a végperec.
#
# ⚠️ Az első változat MINDEN hüvelyk- és ujj-geomot nézett, és 57.1 mm-es
# „nyílást" mért — csakhogy az a `thumb_prox_1` és az `index_prox_1` közt van,
# vagyis a hüvelyk és a mutatóujj TÖVE közti hártyánál. Ott soha nem fogunk
# meg semmit. Ebből majdnem azt a hamis következtetést vontam le, hogy a kéz
# nem képes megfogni egy 8 cm-es kartont.
#
#     minden geom                  57.1 mm   ← a hártya, félrevezető
#     medial + distal (fogófelület) 117.7 mm  ← EZ a nyílás
#     csak az ujjbegyek            169.8 mm
#
# Ugyanaz a hibafajta, mint korábban ötször: nem azt mérni, ami számít.
GRIP_LINKS = ("medial", "distal")


def side_geoms(m, tokens) -> list[int]:
    bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                     m.geom_bodyid[g]) or ""
    return [g for g in range(m.ngeom)
            if bn(g).startswith(tokens) and bn(g).endswith(GRIP_LINKS)]


def closest_points(m, d, geoms_a: list[int], geom_b: int
                   ) -> tuple[float, np.ndarray, np.ndarray]:
    """A legközelebbi pontpár a geomcsoport és egy geom között.

    A `mj_geomDistance` a `fromto` tömbben visszaadja a két legközelebbi
    pontot is — ebből lesz az állkapocs tengelye.
    """
    best, pa, pb = 1e3, None, None
    ft = np.zeros(6)
    for g in geoms_a:
        dist = float(mujoco.mj_geomDistance(m, d, g, geom_b, 1.0, ft))
        if dist < best:
            best, pa, pb = dist, ft[:3].copy(), ft[3:].copy()
    return best, pa, pb


def main() -> int:
    ap_write = "--write" in sys.argv
    print("Shelf Life M2 — a fogási pont MEGSZERKESZTÉSE\n")

    r = Robot()
    g = r._r
    m, d = g.model, g.data

    # a kart a jelenlegi fogási pózba visszük, kéz NYITVA
    r.approach_until(r.preset("pre_grasp"), until="goal")
    r.approach_until(r.preset("grasp"), until="goal")
    r.open_hand()
    box, half = g.product_box()
    Rp = g.palm_R()

    TG, FG = side_geoms(m, THUMB), side_geoms(m, FINGERS)
    prod = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")

    # ── 1. az állkapocs bemérése ────────────────────────────────────────────
    # A hüvelyk és az ujjak EGYMÁSTÓL mért legközelebbi pontpárja adja a
    # tengelyt. A termékkel most nem foglalkozunk — ez a KÉZ tulajdonsága.
    ft = np.zeros(6)
    best, p_thumb, p_finger = 1e3, None, None
    for gt in TG:
        for gf in FG:
            dist = float(mujoco.mj_geomDistance(m, d, gt, gf, 1.0, ft))
            if dist < best:
                best, p_thumb, p_finger = dist, ft[:3].copy(), ft[3:].copy()

    axis = p_finger - p_thumb
    opening = float(np.linalg.norm(axis))
    axis /= opening
    centre = 0.5 * (p_thumb + p_finger)

    print(f"  ÁLLKAPOCS (nyitott kéz, csak a fogófelületek)")
    print(f"    nyílás           {opening*1000:6.1f} mm")
    print(f"    hüvelyk-oldali pont  {np.round(p_thumb, 4)}")
    print(f"    ujj-oldali pont      {np.round(p_finger, 4)}")
    print(f"    tengely (világ)      {np.round(axis, 3)}")
    print(f"    tengely (tenyér)     {np.round(Rp.T @ axis, 3)}")
    print(f"    középpont            {np.round(centre, 4)}")

    # a karton legkisebb keresztmetszete: a két rövidebb él közül a kisebb
    need = float(2 * min(half[0], half[1]))
    print(f"\n  KARTON legkisebb szélessége {need*1000:.0f} mm "
          f"(élek: {np.round(half*200, 1)} cm)")
    if opening < need:
        print(f"\n  ❌ A NYÍLÁS KEVESEBB, mint a termék ({opening*1000:.0f} < "
              f"{need*1000:.0f} mm).")
        print("     Ekkor nem a pozíció a hibás, hanem a NYITOTT KÉZ FORMÁJA")
        print("     (`HAND_OPEN`) — azt kell újratervezni.")
        return 1
    print(f"  ✅ befér, {(opening - need)*1000:.0f} mm tartalékkal")

    # ── 2. a fogási pont: a termék középpontja az állkapocs közepére ────────
    # A fogási offset a TENYÉR frame-jében: a jelenlegi offsethez képest annyit
    # kell módosítani, amennyivel az állkapocs közepe eltér attól a ponttól,
    # ahol most a termék van.
    cur_grasp = g.grasp_point()
    delta_world = centre - cur_grasp
    delta_palm = Rp.T @ delta_world
    plan = dict(r._plan)
    new_tweak = np.array(plan["tweak_cm"], float) + delta_palm * 100

    print(f"\n  SZERKESZTETT FOGÁSI PONT")
    print(f"    jelenlegi fogási pont {np.round(cur_grasp, 4)}")
    print(f"    állkapocs közepe      {np.round(centre, 4)}")
    print(f"    eltérés (tenyér, cm)  {np.round(delta_palm*100, 1)}")
    print(f"    tweak_cm  {plan['tweak_cm']}  →  {np.round(new_tweak, 1).tolist()}")

    # ── 3. ellenőrzés: ide téve a terméket, mit ad a zárás? ─────────────────
    print(f"\n  ELLENŐRZÉS (a terméket az állkapocs közepére téve)")
    s = g._scratch
    qarm = d.qpos[np.array(g._arm_q)].copy()
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    adr = m.jnt_qposadr[jid]
    dorig = d.qpos[adr:adr + 3].copy() - box
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""

    def probe(level: float):
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(g._arm_q)] = qarm
        g.set_hand_qpos(s, level, phased=False)
        s.qpos[adr:adr + 3] = centre + dorig
        s.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, s)
        dig, F, n = set(), np.zeros(3), 0
        for k in range(s.ncon):
            c = s.contact[k]
            for x, y, sg in ((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2], 1.0),
                             (m.geom_bodyid[c.geom2], m.geom_bodyid[c.geom1], -1.0)):
                if y == pb and bn(x).startswith("r_"):
                    for t in ("thumb", "index", "middle", "ring", "little"):
                        if t in bn(x):
                            dig.add(t)
                    F += c.frame[:3] * sg
                    n += 1
        return dig, (float(np.linalg.norm(F) / n) if n else 1.0), n

    chosen = None
    print(f"    {'zárás':>7}{'kontakt':>9}{'eredő/össz':>12}   ujjak")
    for lvl in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80):
        dig, imb, n = probe(lvl)
        print(f"    {lvl:7.2f}{n:9d}{imb:12.2f}   {sorted(dig)}")
        if (chosen is None and "thumb" in dig
                and len(dig - {"thumb"}) >= 2 and imb < 0.55):
            chosen = (lvl, sorted(dig), imb)

    if probe(0.0)[2] > 0:
        print("\n  ⚠️ NYITOTT kézzel is van kontaktus — a pont a kézben van, "
              "de a nyitott ujjak érintik. Kicsit nyitni kell a kézformán.")
    if chosen is None:
        print("\n  ❌ Nincs olyan zárási szint, ahol szembefogás jönne létre.")
        return 1

    lvl, dig, imb = chosen
    print(f"\n  ✅ ERŐZÁRÁS {lvl:.2f}-nél · {dig} · eredő/össz {imb:.2f}")

    if ap_write:
        plan["tweak_cm"] = [round(float(x), 2) for x in new_tweak]
        plan["close_amount"] = float(lvl)
        plan["_jaw"] = {
            "tool": "shelflife_jaw.py",
            "method": ("a fogási pont az állkapocs (hüvelyk ↔ ujjak legközelebbi "
                       "pontpárja) FELEZŐPONTJA — szerkesztve, nem keresve"),
            "opening_mm": round(opening * 1000, 1),
            "product_min_width_mm": round(need * 1000, 1),
            "clearance_mm": round((opening - need) * 1000, 1),
            "axis_palm": [round(float(x), 3) for x in (Rp.T @ axis)],
            "close_level": float(lvl), "digits": dig,
            "normal_imbalance": round(float(imb), 3),
        }
        _G.PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"  → beírva: {_G.PLAN_PATH.relative_to(_REPO)}")
    else:
        print("  (a beíráshoz: --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
