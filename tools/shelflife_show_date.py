"""
shelflife_show_date.py — A3 KÉPESSÉGVIZSGÁLAT: megmutatható-e a dátum?

    python3 tools/shelflife_show_date.py

────────────────────────────────────────────────────────────────────────────
MELYIK RÉTEGBE TARTOZIK
────────────────────────────────────────────────────────────────────────────
Ez az **A) KONFIGURÁCIÓ** réteg A3 pontja — egyszer kell lefuttatni a robot
élettartamára, nem minden dátumellenőrzésnél. A kérdés:

    Létezik-e olyan kartartás, amelyben a megfogott termék dátummezője
    a kamerába néz, olvasható távolságban?

Ha nincs, az nem a működési ciklus hibája, hanem **konfigurációs döntést**
kényszerít: más kamerahely, más végszerszám-tájolás, vagy a termékkategória
kizárása.

────────────────────────────────────────────────────────────────────────────
MIÉRT ÉPP EZ A KOCKÁZAT
────────────────────────────────────────────────────────────────────────────
A colásdobozon a dátum a TALPON van (`product_0_date`: Ø46,5 mm korong a
fenéken). Megmutatni tehát csak a termék MEGFORDÍTÁSÁVAL lehet.

A GENE.01 csuklójának gördülési tartománya MÉRVE:

    r_wrist_yaw    −1,484 … +1,484 rad   (−85° … +85°)
    r_wrist_pitch  −0,384 … +0,559 rad   (−22° … +32°)
    r_wrist_roll   −0,733 … +1,012 rad   (−42° … +58°)

A megfordításhoz legalább 90° kellene valamelyik tengely körül. **Egyik sem
elég önmagában** — a kérdés az, hogy a váll és a könyök kisegíti-e.

────────────────────────────────────────────────────────────────────────────
MIT CSINÁL, ÉS MIT NEM
────────────────────────────────────────────────────────────────────────────
✅ Tisztán KINEMATIKA. Nincs fogás, nincs dinamika, nincs fogó a modellen.
   A termék MEREVEN a csuklóhoz van kötve, egy feltételezett behelyezéssel.

❌ NEM válaszol arra, hogy a fogás túléli-e a forgatást (az a C réteg).
❌ NEM válaszol arra, hogy a kar oda tud-e menni dinamikusan (az A5).

⚠️ A BEHELYEZÉS FELTÉTELEZÉS, NEM MÉRÉS. A fogó még nincs a csuklón, ezért
   nem tudjuk pontosan, hogyan áll a doboz a csuklóhoz képest. Ezért NEM
   egyetlen behelyezést vizsgálunk, hanem hármat — így az eredmény azt is
   megmondja, hogy a VÉGSZERSZÁM TÁJOLÁSA (A1) számít-e. Ha csak az egyik
   behelyezésnél sikerül, akkor a fogót úgy kell felszerelni.

────────────────────────────────────────────────────────────────────────────
A KRITÉRIUM — ELŐRE KIMONDVA
────────────────────────────────────────────────────────────────────────────
Egy kartartás akkor JÓ, ha mind a négy teljesül valamelyik kamerára:

    1. RÁNÉZÉS   — a dátumkorong normálisa és a kamera iránya közti szög
                   ≤ 35°  (ferdébb szögben a nyomtatás összenyomódik)
    2. TÁVOLSÁG  — 80 … 931 mm
                   A felső határ a 7 képpontos olvashatóságból jön: MÉRT
                   5 mm-es karakter, Full HD (1080 px), 45°-os látószög.
                   224 px → 193 mm · 480 → 414 · 720 → 621 · 1080 → 931
                   ⚠️ Ez a korlát ezzel gyakorlatilag MEGSZŰNT: a kar teljes
                   elérése ~70 cm-en belül van. A számítás két korábbi
                   feltételezésen állt — 480 képpont és 2 mm-es karakter —,
                   és MINDKETTŐ rossz volt, ugyanabba az irányba.
    3. LÁTÓTÉR   — a korong középpontja a kamera 45°-os látószögén belül,
                   legalább 5°-os tartalékkal a szélétől
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

import mujoco                                    # noqa: E402
from shelflife_api import Robot                  # noqa: E402

OUT = _REPO / "results/shelflife_show_date"

# ── a termék, a MÉRT SKU-adatokból ──────────────────────────────────────────
CAN_R, CAN_H = 0.02905, 0.14540
DATE_R = 0.02324                 # a dátumkorong sugara — a doboz TALPÁN

# ── kritérium ───────────────────────────────────────────────────────────────
MAX_TILT_DEG = 35.0
# a felső határ a robot adatlapjából jön (Full HD, l. shelflife_robot_spec)
DIST_MIN_MM, DIST_MAX_MM = 80.0, 931.0
FOV_MARGIN_DEG = 5.0
SAMPLES = 40000

# ── a feltételezett behelyezések ────────────────────────────────────────────
# A doboz tengelye a CSUKLÓ keretében. A fogó még nincs felszerelve, ezért
# három ésszerű változatot nézünk — ez egyben az A1 döntést is előkészíti.
MOUNTS = {
    "a) tengely a csukló X-e mentén": np.array([1.0, 0.0, 0.0]),
    "b) tengely a csukló Y-a mentén": np.array([0.0, 1.0, 0.0]),
    "c) tengely a csukló Z-je mentén": np.array([0.0, 0.0, 1.0]),
}
GRIP_OFFSET_M = 0.16      # a doboz KÖZEPE ilyen messze a csukló origójától


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=SAMPLES)
    a = ap.parse_args()

    print("Shelf Life — A3: MEGMUTATHATÓ-E A DÁTUM?\n")
    print("  A) konfigurációs réteg · egyszer a robot élettartamára")
    print("  Tisztán kinematika. A termék mereven a csuklóhoz kötve.\n")

    r = Robot(); r.reset_home()
    g = r._r
    m, d = g.model, g.data

    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_3")
    cams = {}
    for c in range(m.ncam):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) or ""
        if "camera" in nm and "overview" not in nm and "side" not in nm:
            cams[nm] = c
    print(f"  kamerák: {', '.join(cams)}")
    print(f"  látószög: {np.degrees(2*np.arctan(np.tan(np.radians(45/2)))):.0f}°"
          f" (fovy=45)\n")

    # ⚠️ A TELJES LÁNCOT KELL MOZGATNI, NEM CSAK A KART.
    #
    # Az első változat a `g._arm_a` nyolc ízületét söpörte: torso_yaw, három
    # vállízület, könyök, három csuklóízület. A felkar és az alkar TESTEK,
    # nem ízületek — azok forgását a `r_shoulder_yaw` és a `r_elbow` adja,
    # tehát benne voltak. DE kimaradt a `torso_roll` (mellkas, −45°…+45°):
    # kilencven fok oldalra dőlés, amivel a robot a KAMERÁT is a kéz felé
    # fordítja. Egy dátumolvasásnál ez nem részletkérdés — az ember is
    # oldalra dönti a fejét, ha valamit közelről meg akar nézni.
    EXTRA = ["torso_roll"]
    jids = [int(m.actuator_trnid[act, 0]) for act in g._arm_a]
    for nm_ in EXTRA:
        jj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm_)
        if jj >= 0 and jj not in jids:
            jids.append(jj)
    aq = np.array([m.jnt_qposadr[j] for j in jids])
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    print("  a kar ízülethatárai:")
    for j, l, h in zip(jids, lo, hi):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"    {nm:<18}{np.degrees(l):7.0f}° … {np.degrees(h):+.0f}°")

    # ütközés-alapállapot
    bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    ARM = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
           "r_thumb", "r_index", "r_middle", "r_ring", "r_little")
    arm_g = {gg for gg in range(m.ngeom) if bn(m.geom_bodyid[gg]).startswith(ARM)}
    shelf_g = {gg for gg in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gg)
                   or "").startswith("shelf")}
    mujoco.mj_forward(m, d)
    base_self = sum(1 for k in range(d.ncon)
                    if d.contact[k].geom1 in arm_g and d.contact[k].geom2 in arm_g)
    print(f"\n  önütközés alaphelyzetben: {base_self} (ez a viszonyítás)\n")

    rng = np.random.default_rng(20260806)
    fovy_half = np.radians(45 / 2) - np.radians(FOV_MARGIN_DEG)

    results = {}
    for mount_name, axis_w in MOUNTS.items():
        best, hits = None, 0
        for _ in range(a.samples):
            q = lo + rng.random(len(lo)) * (hi - lo)
            d.qpos[aq] = q
            mujoco.mj_forward(m, d)

            # ütközés
            n_self = sum(1 for k in range(d.ncon)
                         if d.contact[k].geom1 in arm_g
                         and d.contact[k].geom2 in arm_g)
            n_shelf = sum(1 for k in range(d.ncon)
                          if (d.contact[k].geom1 in arm_g
                              and d.contact[k].geom2 in shelf_g)
                          or (d.contact[k].geom2 in arm_g
                              and d.contact[k].geom1 in shelf_g))
            if n_shelf > 0 or n_self > base_self + 3:
                continue

            W = d.xmat[wrist].reshape(3, 3)
            p = d.xpos[wrist]
            u = W @ axis_w                      # a doboz tengelye a világban
            centre = p + W @ (axis_w * GRIP_OFFSET_M)
            # a TALP középpontja és kifelé mutató normálisa
            date_c = centre - u * (CAN_H / 2)
            date_n = -u

            for cname, ci in cams.items():
                cp = d.cam_xpos[ci]
                cR = d.cam_xmat[ci].reshape(3, 3)
                v = cp - date_c
                dist = float(np.linalg.norm(v))
                if not (DIST_MIN_MM / 1000 <= dist <= DIST_MAX_MM / 1000):
                    continue
                vhat = v / dist
                tilt = np.degrees(np.arccos(np.clip(date_n @ vhat, -1, 1)))
                if tilt > MAX_TILT_DEG:
                    continue
                # a MuJoCo kamera −z felé néz
                fwd = -cR[:, 2]
                ang = np.arccos(np.clip(fwd @ (-vhat), -1, 1))
                if ang > fovy_half:
                    continue
                hits += 1
                score = tilt + abs(dist * 1000 - 120) / 10
                if best is None or score < best[0]:
                    best = (score, cname, tilt, dist * 1000, q.copy())

        results[mount_name] = (hits, best)
        print(f"  {mount_name}")
        if best is None:
            print(f"    ❌ {a.samples} mintából EGY SEM felel meg\n")
        else:
            _s, cname, tilt, dmm, _q = best
            print(f"    ✅ {hits} megfelelő tartás · legjobb: {cname} · "
                  f"rálátás {tilt:.0f}° · távolság {dmm:.0f} mm\n")

    ok = any(b is not None for _h, b in results.values())
    print("  " + "─" * 60)
    if ok:
        print("  ✅ A3 TELJESÜL — van olyan kartartás, ahol a dátum látszik.")
        print("     A végszerszám tájolását ehhez kell igazítani (A1).")
    else:
        print("  ❌ A3 NEM TELJESÜL egyik feltételezett behelyezéssel sem.")
        print("     Ez KONFIGURÁCIÓS döntést kényszerít: más kamerahely")
        print("     (pl. felfelé néző), vagy a termékkategória kizárása.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a3_eredmeny.json").write_text(json.dumps(
        {k: {"hits": h, "best": (None if b is None else
             {"camera": b[1], "tilt_deg": b[2], "dist_mm": b[3],
              "q": list(map(float, b[4]))})}
         for k, (h, b) in results.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  mentve: {OUT/'a3_eredmeny.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
