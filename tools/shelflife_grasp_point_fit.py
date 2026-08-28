"""
shelflife_grasp_point_fit.py — a FOGÁSI PONT újraillesztése (M2)

    python3 tools/shelflife_grasp_point_fit.py

────────────────────────────────────────────────────────────────────────────
MIÉRT KELL ÚJRA
────────────────────────────────────────────────────────────────────────────
A fogási pont eddig geometriai szerkesztésből jött (ujjbegy-centroid félig zárt
kéznél + korrekció). Az így kapott pont 3 mm-re pontosan a doboz közepén van —
csakhogy az a KÉZ NYÍLÁSÁNAK a közepe, nem a fogásé. Mérve, a zárás
indulásakor:

    hüvelyk  50 mm      mutató 70 mm     középső 73 mm
    gyűrűs   73 mm      kisujj 69 mm

A karton egy nála jóval nagyobb üreg közepén lebeg. Zárásnál minden ujj
5–7 cm-t tesz meg lendületet véve, a hüvelyk (a legközelebbi) ér oda elsőnek,
és eltolja a dobozt, mielőtt bármi szembefogná.

Ez magyarázza a korábbi tüneteket is:
  · a fáziskésleltetés nem segít eleget — a KONTAKTUS SZINTJÉBŐL számoltuk,
    nem a TÁVOLSÁGBÓL, és a kettő nem ugyanaz, ha a távolságok ennyire szórnak;
  · a kontaktusok merőlegesek egymásra, nem szemben (eredő 14.1 N > súly 10.1 N);
  · zárás közben a kar helyzetszabályzója a reakcióerőre válaszol, és a kéz
    „korrigálni" próbál — ez dönti fel a kartont.

────────────────────────────────────────────────────────────────────────────
A HELYES KRITÉRIUM
────────────────────────────────────────────────────────────────────────────
Ember úgy fog meg egy tárgyat, hogy az ujjak **egyszerre**, **ellentétes
oldalon** érnek hozzá. Ehhez nyitott kézben mindkét oldalnak KÖZEL és
KIEGYENLÍTETTEN kell állnia:

  1. ütközésmentes nyitott kézzel
  2. a hüvelyk és a négy ujj a doboz SZEMKÖZTI oldalán (oppozíciós szög > 120°)
  3. mindkét rés kicsi (5–25 mm) és közel egyenlő (eltérés < 10 mm)

A keresés a kart RÖGZÍTI a névleges pózban, és a TERMÉKET tolja végig egy
rácson — így a geometria egzakt, és nem kell pózonként IK-t futtatni. A
nyertes eltolásból adódik az új fogási pont; az elérhetőséget utána külön
ellenőrizzük.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                       # noqa: E402
import shelflife_grasp as _G                        # noqa: E402
from shelflife_api import Robot                     # noqa: E402

THUMB_BODIES = ["r_thumb_distal", "r_thumb_medial"]
FINGER_BODIES = [f"r_{f}_{p}" for f in ("index", "middle", "ring", "little")
                 for p in ("distal", "medial")]

GAP_MIN, GAP_MAX = 0.005, 0.025      # 5–25 mm
GAP_BALANCE = 0.010                  # a két oldal közti megengedett eltérés
OPPOSITION_DEG = 120.0               # ennél jobban szemben kell lenniük


def box_gap(p: np.ndarray, centre: np.ndarray, half: np.ndarray) -> float:
    """Egy pont előjeles távolsága egy tengelyparhuzamos doboz felületétől.

    Pozitív: kívül van. Negatív: belemetsz.
    """
    dv = np.abs(p - centre) - half
    if (dv > 0).any():
        return float(np.linalg.norm(np.maximum(dv, 0.0)))
    return float(dv.max())


def main() -> int:
    print("Shelf Life M2 — a fogási pont újraillesztése\n")
    r = Robot()
    g = r._r
    m, d = g.model, g.data

    # a kart a névleges fogási pózba visszük, kéz NYITVA
    r.approach_until(r.preset("pre_grasp"), until="goal")
    r.approach_until(r.preset("grasp"), until="goal")
    box, half = g.product_box()
    Rp = g.palm_R()
    raw = g._grasp_offset.copy()

    # a záráshoz szükséges kellékek: a kar pózát rögzítjük, a terméket toljuk
    qarm = d.qpos[np.array(g._arm_q)].copy()
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    adr = m.jnt_qposadr[jid]
    dorig = d.qpos[adr:adr + 3].copy() - box
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    s = g._scratch
    bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""

    def closes_on(centre: np.ndarray) -> tuple[float | None, set[str], float]:
        """Zárásnál létrejön-e SZEMBEFOGÁS ezen a helyen?

        Visszaadja: (a zárási szint, ahol létrejött; mely ujjak; a kontaktus-
        NORMÁLISOK eredőjének aránya az összerőhöz). Az utóbbi a valódi
        erőzárás mérőszáma: ha az eredő nagy, a kéz TOL, nem fog.
        """
        for lvl in (0.35, 0.45, 0.55, 0.65, 0.75):
            mujoco.mj_resetData(m, s)
            s.qpos[np.array(g._arm_q)] = qarm
            g.set_hand_qpos(s, lvl)
            s.qpos[adr:adr + 3] = centre + dorig
            s.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(m, s)
            digits, F, tot = set(), np.zeros(3), 0.0
            for k in range(s.ncon):
                con = s.contact[k]
                b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
                for x, y, sg in ((b1, b2, 1.0), (b2, b1, -1.0)):
                    if y != pb:
                        continue
                    nm = bn(x)
                    for dg in ("thumb", "index", "middle", "ring", "little"):
                        if dg in nm:
                            digits.add(dg)
                            n = con.frame[:3] * sg
                            F += n; tot += 1.0
            if "thumb" in digits and len(digits - {"thumb"}) >= 2:
                return lvl, digits, float(np.linalg.norm(F) / max(tot, 1))
        return None, set(), 1.0

    nid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    thumb_pts = np.array([d.xpos[nid(n)] for n in THUMB_BODIES])
    finger_pts = np.array([d.xpos[nid(n)] for n in FINGER_BODIES])

    print(f"  karton: közép {np.round(box, 3)} · félméret "
          f"{np.round(half * 100, 1)} cm")
    print(f"  jelenlegi rések — hüvelyk "
          f"{min(box_gap(p, box, half) for p in thumb_pts) * 1000:.0f} mm · "
          f"ujjak {min(box_gap(p, box, half) for p in finger_pts) * 1000:.0f} mm\n")

    # ── keresés: a TERMÉKET toljuk, a kéz áll ──────────────────────────────
    best = []
    # A rács SZÉLES: a fogási pont akár 10+ cm-rel is odébb lehet. Szűk
    # rácsnál a keresés a kiindulási póz környékére van bezárva, és ha az
    # rossz, semmit nem talál — bele is futottunk.
    rng = np.arange(-0.14, 0.141, 0.005)
    for du in rng:
        for dv in rng:
            for dw in rng:
                c = box + Rp @ np.array([du, dv, dw])
                gt = min(box_gap(p, c, half) for p in thumb_pts)
                gf = min(box_gap(p, c, half) for p in finger_pts)
                if not (GAP_MIN <= gt <= GAP_MAX and GAP_MIN <= gf <= GAP_MAX):
                    continue
                if abs(gt - gf) > GAP_BALANCE:
                    continue
                # oppozíció: a doboz közepéből nézve a két oldal ellentétes-e
                vt = thumb_pts.mean(0) - c
                vf = finger_pts.mean(0) - c
                vt /= np.linalg.norm(vt); vf /= np.linalg.norm(vf)
                ang = np.degrees(np.arccos(np.clip(vt @ vf, -1, 1)))
                if ang < OPPOSITION_DEG:
                    continue
                # Pontozás: elsősorban KIS rés (kevés lendületvétel a
                # kontaktusig), másodsorban jó oppozíció. A kiegyensúlyozottság
                # már szűrőfeltétel, nem cél — a tökéletesen egyenlő, de NAGY
                # rés rosszabb, mint a 0.3 mm-rel eltérő, de harmadakkora.
                # ⚠️ A NYITOTT kéz rése önmagában KEVÉS. Az első illesztés
                # csak ezt nézte, és olyan helyet választott, ahol a záródás
                # már NEM ér oda: nulla kontaktus, a kéz a levegőbe zárt.
                # Ezért itt le is zárjuk a kezet, és megköveteljük a
                # szembefogást — plusz mérjük a kontaktus-normálisok eredőjét,
                # mert az mondja meg, hogy FOG-e vagy TOL.
                lvl, digits, imbalance = closes_on(c)
                if lvl is None or imbalance > 0.55:
                    continue
                best.append((max(gt, gf) + 0.02 * imbalance, -ang,
                             gt, gf, ang, np.array([du, dv, dw]),
                             lvl, sorted(digits), imbalance))

    if not best:
        print("  ❌ NINCS olyan helyzet, ahol a hüvelyk és az ujjak SZEMBEN,")
        print("     kicsi és kiegyenlített réssel állnának.")
        print("     Ez azt jelentené, hogy ezzel a KÉZFORMÁVAL a karton nem")
        print("     fogható meg — a nyitott kéz alakját kell újratervezni,")
        print("     nem a pozíciót.")
        # diagnosztika: mi a legjobb elérhető oppozíció?
        bestang = (0.0, None)
        for du in rng[::2]:
            for dv in rng[::2]:
                for dw in rng[::2]:
                    c = box + Rp @ np.array([du, dv, dw])
                    vt = thumb_pts.mean(0) - c
                    vf = finger_pts.mean(0) - c
                    a = np.degrees(np.arccos(np.clip(
                        vt / np.linalg.norm(vt) @ (vf / np.linalg.norm(vf)),
                        -1, 1)))
                    if a > bestang[0]:
                        bestang = (a, np.array([du, dv, dw]))
        print(f"\n     A rácson elérhető legjobb oppozíciós szög: "
              f"{bestang[0]:.0f}° (küszöb {OPPOSITION_DEG:.0f}°)")
        return 1

    best.sort(key=lambda x: (x[0], x[1]))
    print(f"  {len(best)} megfelelő helyzet\n")
    print(f"{'rés hüvelyk':>12}{'rés ujjak':>11}{'oppozíció':>11}   "
          f"eltolás (u,v,w) cm   →  ÚJ tweak (cm)")
    # ⚠️ A TÉNYLEGESEN HATÁLYOS eltolás a TERVFÁJLBÓL jön, nem a modul
    # konstansából: a `_measure_grasp_offset()` a `grasp_plan.json`
    # `tweak_cm` mezőjét részesíti előnyben. Egy adat, két forrás — ugyanaz a
    # hibaosztály, mint amikor az `observe().holding` saját másolatot használt
    # a fogás-kritériumból. Ezért itt is a tervet olvassuk, és oda is írunk.
    cur = np.array(r._plan["tweak_cm"], float)
    #
    # ⚠️ ELŐJEL. A keresés a KEZET rögzíti és a TERMÉKET tolja `+dd`-vel.
    # Ugyanez a relatív helyzet úgy áll elő, hogy a termék marad és a TENYÉR
    # mozdul `−dd`-vel. A fogási pontot viszont a doboz közepére vezéreljük:
    #     fogáspont = tenyér + R·offset  =  doboz  (rögzített)
    # tehát ha a tenyér −dd-vel megy, az offsetnek **+dd**-vel kell nőnie.
    # Először kivontam — a rés 46 mm-ről 100 mm-re NŐTT.
    print(f"{'zárás':>7}{'ujjak':>34}{'eredő/össz':>12}")
    for _, _, gt, gf, ang, dd, lvl, dig, imb in best[:8]:
        print(f"{gt*1000:9.1f} mm{gf*1000:8.1f} mm{ang:10.0f}°   "
              f"{str(np.round(dd*100,1)):<20} → {np.round(cur + dd*100, 1)}")
        print(f"{lvl:>7.2f}{str(dig):>34}{imb:>12.2f}")

    _, _, gt, gf, ang, dd, lvl, dig, imb = best[0]
    tweak = cur + dd * 100
    print(f"\n  ✅ JAVASOLT tweak_cm = {np.round(tweak, 1).tolist()}")
    print(f"     rések {gt*1000:.1f} / {gf*1000:.1f} mm · oppozíció {ang:.0f}° · "
          f"zárás {lvl:.2f}-nél {dig} · eredő/össz {imb:.2f}")
    print(f"     (a hatályos, a tervfájlból: {cur.tolist()})")

    if "--write" in sys.argv:
        import json
        pth = _G.PLAN_PATH
        plan = json.loads(pth.read_text())
        plan["tweak_cm"] = [round(float(x), 1) for x in tweak]
        plan["grasp_offset_palm_cm"] = [
            round(float(x), 2) for x in (g._grasp_offset + dd) * 100]
        plan["_refit"] = {
            "tool": "shelflife_grasp_point_fit.py",
            "criterion": ("nyitott kézben a hüvelyk és a négy ujj a doboz "
                          "SZEMKÖZTI oldalán, kicsi és kiegyenlített réssel"),
            "gap_thumb_mm": round(gt * 1000, 1),
            "gap_fingers_mm": round(gf * 1000, 1),
            "opposition_deg": round(float(ang), 1),
            "close_level": float(lvl), "digits": dig,
            "normal_imbalance": round(float(imb), 3),
            "previous_gaps_mm": [46, 69],
        }
        pth.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"\n  → beírva: {pth.relative_to(_REPO)}")
    else:
        print("\n  (a beíráshoz: --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
