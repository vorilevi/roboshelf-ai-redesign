"""
shelflife_t1_show_date.py — A3/T1: ODA TUDJA-E VINNI A KAR A DÁTUMOT?

    python3 tools/shelflife_t1_show_date.py
    python3 tools/shelflife_t1_show_date.py --samples 120000 --render

────────────────────────────────────────────────────────────────────────────
MI EZ, ÉS MIÉRT MOST
────────────────────────────────────────────────────────────────────────────
A **2. fázis kapuja**. A demó íve — navigáció → fogás → megfordítás →
leolvasás → döntés — azon áll vagy bukik, hogy a kar be tudja-e forgatni a
megfogott doboz TALPÁT a fejkamera elé, olyan szögben és távolságban, ahol
a dátum ténylegesen olvasható.

Tisztán kinematika: nincs fogás, nincs dinamika. A termék MEREVEN a
csuklóhoz van kötve, egy feltételezett behelyezéssel.

────────────────────────────────────────────────────────────────────────────
AMI EZT MEGKÜLÖNBÖZTETI A GENE.01-ES A3-TÓL
────────────────────────────────────────────────────────────────────────────
Ott a kritérium **feltételezés** volt: „35°-nál ferdébb szögben a nyomtatás
összenyomódik", és 931 mm-es olvasható távolság egy rossz kamerafeltevésből.

Most **mérve van**. A 2026-08-16-i kontakt ívek a D455 valódi paramétereivel
(1280×800, 65°, bolti szórt fény), egész számú NEAREST nagyítással:

    ránézési szög   meddig olvasható
    ────────────────────────────────
         0°             ~700 mm
        30°             ~600 mm
        45°             ~504 mm
        60°          csak közelről
        75°             soha

A kritérium ezért nem egy szám, hanem egy BURKOLÓGÖRBE. A `readable_limit`
lineárisan interpolál a mért pontok között.

⚠️ EZ FELSŐ KORLÁT. A renderelés zajmentes, tömörítetlen, elmosódásmentes,
   és a képeket ember/VLM olvasta, nem a beépített lánc. A tényleges
   operációs pontot 10-ből 10 sikerrel kell igazolni.

────────────────────────────────────────────────────────────────────────────
A NÉGY KAPU
────────────────────────────────────────────────────────────────────────────
Egy kartartás akkor JÓ, ha mind a négy teljesül:

    1. RÁNÉZÉS   — a talp normálisa és a kamerairány közti szög ≤ 60°
    2. TÁVOLSÁG  — a mért burkológörbén belül, ÉS ≥ 150 mm
                   (a D455 FIX FÓKUSZÚ; a közeli határ alatt életlen)
    3. LÁTÓTÉR   — a talp középpontja a kamera látószögén belül,
                   5° tartalékkal (a D455: 90° vízszintes × 65° függőleges)
    4. ÜTKÖZÉS   — a kar ne érjen a polchoz, és ne ütközzön önmagával
                   jobban, mint alaphelyzetben
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env                       # noqa: E402,F401  (SORREND!)
import mujoco                                     # noqa: E402
from shelflife_t1_scene import (                  # noqa: E402
    build_model, apply_home, ARM, SOLE_Z, CAN_H, DATE_R, GRIP_OFFSET)

OUT = _REPO / "results/shelflife_t1_show_date"

# ── a MÉRT burkológörbe (2026-08-16, kontakt ívek) ─────────────────────────
ENVELOPE = [(0.0, 700.0), (30.0, 600.0), (45.0, 504.0), (60.0, 400.0),
            (75.0, 0.0)]
DIST_MIN_MM = 150.0            # a fix fókusz közeli határa
MAX_TILT_DEG = 60.0
FOV_H_DEG, FOV_V_DEG = 90.0, 65.0
FOV_MARGIN_DEG = 5.0

# ── a feltételezett behelyezések ───────────────────────────────────────────
# A fogó még nincs a csuklón, ezért három ésszerű változatot söprünk végig —
# ez egyben az A1 (fogófelszerelés) döntését is előkészíti.
MOUNTS = {
    "a) tengely a csukló X-e mentén": np.array([1.0, 0.0, 0.0]),
    "b) tengely a csukló Y-a mentén": np.array([0.0, 1.0, 0.0]),
    "c) tengely a csukló Z-je mentén": np.array([0.0, 0.0, 1.0]),
}
WRIST = "right_hand_link"


def readable_limit(tilt_deg: float) -> float:
    """Meddig olvasható ekkora ránézési szögben — a MÉRT pontok között."""
    xs = [t for t, _ in ENVELOPE]
    ys = [d for _, d in ENVELOPE]
    return float(np.interp(tilt_deg, xs, ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=80000)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    print("Shelf Life — A3/T1: MEGMUTATHATÓ-E A DÁTUM A KAMERÁNAK?\n")
    print("  A) konfigurációs réteg · a 2. fázis kapuja")
    print("  Tisztán kinematika. A termék mereven a csuklóhoz kötve.\n")

    m = build_model()
    d = mujoco.MjData(m)
    apply_home(m, d)

    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST)
    if cam < 0 or wrist < 0:
        print("  ❌ nincs fejkamera vagy csukló")
        return 1

    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
    jids = [j for j in jids if j >= 0]
    # ⚠️ A FEJ IS MOZOG. Az ember is odafordítja a fejét, ha valamit közelről
    #    meg akar nézni. A T1-nek 2 nyakízülete van — kihagyni őket ugyanaz a
    #    hiba lett volna, mint a GENE.01-nél a `torso_roll` kihagyása.
    for n in ("AAHead_yaw", "Head_pitch"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            jids.append(j)
    aq = np.array([m.jnt_qposadr[j] for j in jids])
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    print(f"  mozgatott ízületek ({len(jids)}):")
    for j in jids:
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"    {nm:<24}{np.degrees(m.jnt_range[j][0]):7.0f}° … "
              f"{np.degrees(m.jnt_range[j][1]):+.0f}°")

    # ütközés-alapállapot
    gname = lambda g: mujoco.mj_id2name(     # noqa: E731
        m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    bn = lambda b: mujoco.mj_id2name(        # noqa: E731
        m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    ARM_B = ("AR", "right_hand")
    arm_g = {g for g in range(m.ngeom) if bn(m.geom_bodyid[g]).startswith(ARM_B)}
    shelf_g = {g for g in range(m.ngeom) if gname(g).startswith("shelf")}
    # a polcon álló termék NE takarjon: a söprés alatt még ott van,
    # miközben matematikailag már a kézben van
    prod_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    mujoco.mj_forward(m, d)
    base_self = sum(1 for k in range(d.ncon)
                    if d.contact[k].geom1 in arm_g and d.contact[k].geom2 in arm_g)
    print(f"\n  önütközés alaphelyzetben: {base_self} (ez a viszonyítás)")
    print(f"  burkológörbe: 0°→{readable_limit(0):.0f}mm · "
          f"30°→{readable_limit(30):.0f} · 45°→{readable_limit(45):.0f} · "
          f"60°→{readable_limit(60):.0f}\n")

    rng = np.random.default_rng(20260816)
    half_h = np.radians(FOV_H_DEG / 2) - np.radians(FOV_MARGIN_DEG)
    half_v = np.radians(FOV_V_DEG / 2) - np.radians(FOV_MARGIN_DEG)
    results, best_all = {}, None

    for label, axis in MOUNTS.items():
        good, best = 0, None
        for _ in range(a.samples):
            q = rng.uniform(lo, hi)
            d.qpos[aq] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)

            W = d.xmat[wrist].reshape(3, 3)
            wp = d.xpos[wrist]
            ax_w = W @ axis                      # a doboz tengelye a világban
            # a fogáspont a csuklótól, a talp onnan fél dobozmagasságra
            grip = wp + ax_w * GRIP_OFFSET
            base_c = grip - ax_w * (CAN_H / 2)   # a TALP középpontja
            normal = -ax_w                        # a talp kifelé néző normálisa

            C = d.cam_xpos[cam]
            R = d.cam_xmat[cam].reshape(3, 3)
            v = base_c - C
            dist_mm = float(np.linalg.norm(v)) * 1000
            if dist_mm < DIST_MIN_MM:
                continue
            vhat = v / (dist_mm / 1000)

            # 1. ránézési szög: a talp normálisa mennyire néz a kamerába
            tilt = np.degrees(np.arccos(np.clip(float(np.dot(normal, -vhat)),
                                                -1, 1)))
            if tilt > MAX_TILT_DEG:
                continue
            # 2. a MÉRT burkológörbe
            if dist_mm > readable_limit(tilt):
                continue
            # 3. látótér — a kamera −z felé néz
            loc = R.T @ v
            if loc[2] >= 0:
                continue
            if (abs(np.arctan2(loc[0], -loc[2])) > half_h or
                    abs(np.arctan2(loc[1], -loc[2])) > half_v):
                continue
            # ── 5. TAKARÁS ────────────────────────────────────────────────
            # ⚠️ EZT A KAPUT ELŐSZÖR KIHAGYTAM, és a render leplezte le: a
            #    „legjobb" tartáson a KAR MAGA takarta el a dátumot. A 3.
            #    kapu csak azt nézi, hogy a talp középpontja beleesik-e a
            #    látószögbe — azt nem, hogy van-e SZABAD RÁLÁTÁS. Egy
            #    kamerának a látótér és a látás nem ugyanaz.
            # ⚠️ EGY SUGÁR KEVÉS. Először csak a talp KÖZEPÉRE lőttem, és az
            #    átfért két kartag közti résen — a render viszont mutatta,
            #    hogy a dátum jó része takarva van. Öt pont kell: a közép és
            #    a korong négy széle. A dátum a teljes korongon szétterül.
            mujoco.mj_forward(m, d)
            u = np.cross(normal, [0.0, 0.0, 1.0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(normal, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            w2 = np.cross(normal, u)
            probes = [base_c] + [base_c + s * DATE_R * 0.85
                                 for s in (u, -u, w2, -w2)]
            gid = np.zeros(1, dtype=np.int32)
            blocked = False
            for p in probes:
                vv = p - C
                L = float(np.linalg.norm(vv))
                hit = mujoco.mj_ray(m, d, C, vv / L, None, 1, prod_body, gid)
                if 0 <= hit < L - 0.005:
                    blocked = True
                    break
            if blocked:
                continue

            # 4. ütközés
            hit_shelf = any(
                (d.contact[k].geom1 in arm_g and d.contact[k].geom2 in shelf_g)
                or (d.contact[k].geom2 in arm_g and d.contact[k].geom1 in shelf_g)
                for k in range(d.ncon))
            if hit_shelf:
                continue
            nself = sum(1 for k in range(d.ncon)
                        if d.contact[k].geom1 in arm_g
                        and d.contact[k].geom2 in arm_g)
            if nself > base_self:
                continue

            good += 1
            margin = readable_limit(tilt) - dist_mm
            if best is None or margin > best["tartalek_mm"]:
                best = {"tilt_fok": round(tilt, 1),
                        "tavolsag_mm": round(dist_mm, 1),
                        "tartalek_mm": round(margin, 1),
                        "q": [round(float(x), 4) for x in q]}
        results[label] = {"jo_tartasok": good, "legjobb": best}
        print(f"  {label:<34}{good:>6} megfelelő kartartás"
              + (f"   · legjobb: {best['tilt_fok']}° · "
                 f"{best['tavolsag_mm']:.0f} mm · "
                 f"tartalék {best['tartalek_mm']:.0f} mm" if best else ""))
        if best and (best_all is None
                     or good > results[best_all]["jo_tartasok"]):
            best_all = label

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eredmeny.json").write_text(
        json.dumps({"minta": a.samples, "burkologorbe": ENVELOPE,
                    "eredmenyek": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print()
    if best_all is None:
        print("  ❌ EGYETLEN BEHELYEZÉSSEL SEM megy — a 2. fázis így nem áll össze.")
        print("     Lehetőségek: csuklókamera · más fogótájolás · a robot")
        print("     hátrébb áll · a terméket két kézzel forgatja.")
        return 1

    print(f"  ✅ A KAPU NYITVA — a legjobb behelyezés: {best_all}")
    b = results[best_all]["legjobb"]
    print(f"     {b['tilt_fok']}° ránézés · {b['tavolsag_mm']:.0f} mm · "
          f"{b['tartalek_mm']:.0f} mm tartalék az olvashatósági határig")
    print(f"\n  eredmény: {(OUT / 'eredmeny.json').relative_to(_REPO)}")

    if a.render:
        render_best(m, d, aq, results[best_all]["legjobb"]["q"], MOUNTS[best_all])
    return 0


def render_best(m, d, aq, q, axis) -> None:
    """A legjobb kartartás lefényképezve — a szám nem elég, LÁTNI kell.

    ⚠️ AZ ELSŐ VÁLTOZAT HASZNÁLHATATLAN KÉPET ADOTT: a dobozt csak
    MATEMATIKAILAG kötöttük a csuklóhoz, a jelenetben a polcon maradt. A
    render így a robotot mutatta üres kézzel — vagyis pont azt nem
    ellenőrizte, amiért készült. A terméket a számított helyre kell tenni.
    """
    import imageio.v3 as iio
    from shelflife_t1_scene import RENDER_W, RENDER_H
    shelflife_render_env.ensure(RENDER_W, RENDER_H)
    d.qpos[aq] = np.array(q)
    mujoco.mj_kinematics(m, d)

    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST)
    W = d.xmat[wrist].reshape(3, 3)
    ax_w = W @ axis
    base_c = d.xpos[wrist] + ax_w * GRIP_OFFSET - ax_w * (CAN_H / 2)

    # a doboz saját +z tengelye a talptól a teteje felé mutat → forgassuk ax_w-be
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, ax_w)
    c = float(np.dot(z, ax_w))
    if np.linalg.norm(v) < 1e-9:
        quat = np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    else:
        vn = v / np.linalg.norm(v)
        ang = np.arccos(np.clip(c, -1, 1))
        quat = np.concatenate([[np.cos(ang / 2)], vn * np.sin(ang / 2)])

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    if jid >= 0:
        adr = m.jnt_qposadr[jid]
        d.qpos[adr:adr + 3] = base_c
        d.qpos[adr + 3:adr + 7] = quat
    mujoco.mj_forward(m, d)
    OUT.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, RENDER_H, RENDER_W)
    for c in ("head_camera", "side", "overview"):
        try:
            r.update_scene(d, camera=c)
        except Exception:                          # noqa: BLE001
            continue
        img = r.render()
        iio.imwrite(OUT / f"legjobb_{c}.png", img)
        print(f"  kép: {(OUT / f'legjobb_{c}.png').relative_to(_REPO)}"
              f"  (szórás {img.std():.1f})")
    r.close()


if __name__ == "__main__":
    raise SystemExit(main())
