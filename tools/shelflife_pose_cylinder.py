"""
shelflife_pose_cylinder.py — fogási ORIENTÁCIÓ hengeres termékhez

    python3 tools/shelflife_pose_cylinder.py

────────────────────────────────────────────────────────────────────────────
A MÉRT PROBLÉMA
────────────────────────────────────────────────────────────────────────────
A doboz fogásakor a kontaktusok magassága a termék talpától (2026-08-05):

    hüvelyk 146 mm · mutató 111 · tenyér 111 · középső 85 · gyűrűs 47 ·
    kisujj 17 mm            — a doboz 145 mm magas

A kéz tehát **átlósan fekszik** a hengeren: a hüvelyk a peremen, a kisujj a
talp közelében, 130 mm-en szétszórva. A hüvelyk és a négy ujj átlaga közti
különbség **+70 mm**, ami több, mint a doboz átmérője (58 mm).

Ilyen elrendezésben a szorítás ELFORGAT, nem fog: nincs két szemben lévő
kontaktus AZONOS magasságban, tehát nincs, ami az erőket kioltsa. Mérve: a
kéz 19 mm-t emelkedik, a doboz 0-t — a kéz felcsúszik a dobozon.

A felhasználó fotói (data/cola_private/IMG_7379–7380) ugyanezt mutatják
fordítva: emberi fogásnál a hüvelyk és a szembe lévő ujjak **azonos
magasságban** vannak, és az ujjak körbehajlanak.

────────────────────────────────────────────────────────────────────────────
AZ OK: A BÜTYÖKSOR ÁLLÁSA
────────────────────────────────────────────────────────────────────────────
A kontaktusmagasságok ~30 mm-enként lépcsőznek (111 · 85 · 47 · 17) — ez a
BÜTYÖKTÁVOLSÁG. Vagyis a négy ujj bütyöksora FÜGGŐLEGESEN áll, párhuzamosan
a henger tengelyével. Vízszintesen kellene állnia: akkor mind a négy ujj
azonos magasságban ér a palásthoz, és a hüvelyk velük szemben.

A jelenlegi `right_thumb_up` orientáció a TEJESDOBOZHOZ készült — álló
téglatesthez, ahol az ujjak függőlegesen simulnak a lapra. Hengerre nem való.

────────────────────────────────────────────────────────────────────────────
AMIT EZ CSINÁL
────────────────────────────────────────────────────────────────────────────
Nem tippel új orientációt, hanem VÉGIGMÉR egy családot, és mindegyikre
megmondja:

  · elérhető-e a kar számára (IK-hiba, ízülettartalék)
  · mekkora a fogófelületek MAGASSÁGSZÓRÁSA a henger tengelye mentén
  · szemben van-e a hüvelyk a négy ujjal

A sikerkritérium ELŐRE kimondva: a hüvelyk és a négy ujj átlagos magassága
közti különbség menjen **10 mm alá** (most 70 mm), az orientáció maradjon
elérhető (IK < 8 mm, tartalék > 0.15 rad).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
import shelflife_grasp as G                      # noqa: E402
import shelflife_jaw as J                        # noqa: E402

REACH_MM, REACH_DEG, MIN_MARGIN = 8.0, 4.0, 0.15
TARGET_SPREAD_MM = 10.0          # hüvelyk vs. az ujjak átlaga
TARGET_FINGER_SPREAD_MM = 20.0   # hüvelyk vs. mutató+középső
                                 # (utólag hozzávéve — l. a magyarázatot lent)
PROBE_CLOSE = 0.30               # félig zárt kéz a beméréshez


def candidate_frames() -> dict[str, np.ndarray]:
    """Orientáció-család. A meglévők + hengerre szánt változatok.

    A `_frame(x, y)` oszlopai a tenyér tengelyei. A lényeg a BÜTYÖKSOR
    állása: az a tenyér X tengelye mentén fut, tehát ha X FÜGGŐLEGES,
    a bütyöksor is az — ez a mostani hiba. Hengerhez X-nek vízszintesnek
    kell lennie.
    """
    F = G._frame
    out: dict[str, np.ndarray] = {}
    out["régi:right_thumb_up"] = G.GRASP_POSES["right_thumb_up"]

    # ⚠️ AZ ELSŐ JELÖLTKÉSZLETEM ROSSZ VOLT, és a hiba tanulságos.
    #
    # „Vízszintes bütyöksort" akartam, és felírtam pl. `_frame([0,1,0],
    # [0,0,1])`-et. Csakhogy a `_frame` HARMADIK oszlopa x×y, és annál a
    # jelöltnél ez +x lett — vagyis a tenyér a polc TÚLOLDALÁRÓL nézett
    # volna a termékre. Az IK persze 167 mm-t hibázott: nem az orientáció
    # volt rossz, hanem én kértem fizikailag lehetetlent.
    #
    # A működő `right_thumb_up`-nál a tenyér Z tengelye −x, azaz a robot
    # felé néz. Ezt MEG KELL TARTANI, és csak a bütyöksort (tenyér-X)
    # forgatni: +z-ből (mostani, függőleges) −y felé (vízszintes).
    #
    #     x(t) = (0, −sin t, cos t)      t = 0° … 90°
    #     y(t) = z × x = (0, cos t, sin t)
    #     ellenőrizve: x × y = (−1, 0, 0) minden t-re ✓
    #
    # Így a söprés VÉGIG elérhető irányban marad, és azt méri, amit akarunk:
    # meddig lehet a bütyöksort vízszintesbe forgatni, mielőtt a kar elfogy.
    for deg in range(-90, 91, 15):
        t = np.radians(deg)
        x = np.array([0.0, -np.sin(t), np.cos(t)])
        y = np.array([0.0, np.cos(t), np.sin(t)])
        R = F(x, y)
        assert np.allclose(R[:, 2], [-1, 0, 0], atol=1e-6), deg
        out[f"bütyöksor {deg:+3d}°"] = R
    return out


def main() -> int:
    print("Shelf Life — fogási orientáció hengeres termékhez\n")
    G.GRASP_TWEAK_CM = np.zeros(3)
    _rp, G.PLAN_PATH = G.PLAN_PATH, Path("/nonexistent")
    r = G.GraspRobot()
    G.PLAN_PATH = _rp
    m, s = r.model, r._scratch
    box, half = r.product_box()
    axis_h = float(half[2]) * 2                     # a henger magassága
    prod = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")
    TG, FG = J.side_geoms(m, J.THUMB), J.side_geoms(m, J.FINGERS)

    print(f"  termék: henger Ø{half[0]*2000:.0f} mm × {axis_h*1000:.0f} mm\n")
    print(f"  {'orientáció':<18}{'IK':>8}{'tartalék':>10}"
          f"{'hüvelyk':>9}{'ujjak':>8}{'ELTÉRÉS':>9}{'ujjszórás':>11}")
    print("  " + "─" * 73)

    best = None
    for name, R in candidate_frames().items():
        q, ep, er = r.ik6_seed(box, R, restarts=10, iters=80)
        mg = r.joint_margin(q)
        ok_reach = (ep * 1000 < REACH_MM and np.degrees(er) < REACH_DEG
                    and mg > MIN_MARGIN)
        if not ok_reach:
            print(f"  {name:<18}{ep*1000:7.1f}{mg:10.2f}"
                  f"{'—':>9}{'—':>8}{'nem éri el':>12}")
            continue

        # félig zárt kézzel megnézzük, HOL érintkeznének a fogófelületek:
        # a termékhez legközelebbi pont MAGASSÁGA ujjanként
        mujoco.mj_resetData(m, s)
        s.qpos[np.array(r._arm_q)] = q
        r.set_hand_qpos(s, PROBE_CLOSE, phased=False)
        mujoco.mj_forward(m, s)

        ft = np.zeros(6)

        def nearest_z(geoms) -> float:
            bz, bd = None, 1e3
            for g in geoms:
                dist = float(mujoco.mj_geomDistance(m, s, g, prod, 0.5, ft))
                if dist < bd:
                    bd, bz = dist, float(ft[2])     # a KÉZ oldali pont z-je
            return bz

        # ujjanként külön, hogy a szórás is látszódjon
        zs = {}
        for dg, toks in (("hüvelyk", J.THUMB), ("mutató", ("r_index",)),
                         ("középső", ("r_middle",)), ("gyűrűs", ("r_ring",)),
                         ("kisujj", ("r_little",))):
            gs = J.side_geoms(m, toks)
            if gs:
                zs[dg] = nearest_z(gs)
        base = box[2] - half[2]
        th = (zs["hüvelyk"] - base) * 1000
        fi = np.mean([(zs[k] - base) * 1000
                      for k in ("mutató", "középső", "gyűrűs", "kisujj")])
        spread = abs(th - fi)
        # ⚠️ AZ ELŐRE KIMONDOTT KRITÉRIUM HIÁNYOS VOLT.
        #
        # „A hüvelyk és a négy ujj ÁTLAGA essen 10 mm-en belülre" — ezt
        # −60°-nál 1 mm-rel teljesíti. Csakhogy közben a négy ujj MAGA
        # 21…105 mm-en szór: az átlaguk véletlenül esik a hüvelyk
        # magasságára. Egy szétterülő ujjsor, aminek az átlaga stimmel, NEM
        # ugyanaz, mint egy rendes szembefogás.
        #
        # A kritériumot tehát kiegészítem — de KIMONDVA, hogy utólag teszem,
        # és nem azért, hogy egy eredmény átmenjen, hanem mert az eredetit
        # hiányosnak találtam. A négy ujj SAJÁT szórása is számít.
        # ⚠️ MÁSODSZOR IS PONTOSÍTOM A KRITÉRIUMOT — és megint azért, mert
        # rossz mennyiséget mértem, nem azért, hogy valami átmenjen.
        #
        # „Mind a négy ujj azonos magasságban" — ez nem is emberi fogás. A
        # felhasználó leírása: *„egymással szemben van a mutató és a
        # hüvelykujjam, és akkor ez a két ujj nagyobb erőt fejt ki, mint a
        # többi"*, illetve a másik fogásnál a hüvelyk a mutatóval és a
        # gyűrűssel szemben. A fotókon is látszik, hogy a kisujj messze lóg.
        #
        # A SZEMBEFOGÁS tehát a hüvelyk és a MUTATÓ+KÖZÉPSŐ között van; a
        # gyűrűs és a kisujj csak kísér. A mérendő mennyiség ezért:
        #     |hüvelyk − átlag(mutató, középső)|
        fz = [(zs[k] - base) * 1000 for k in ("mutató", "középső")]
        fspread = abs(th - np.mean(fz))
        ok = spread < TARGET_SPREAD_MM and fspread < TARGET_FINGER_SPREAD_MM
        mark = " ✅" if ok else ""
        print(f"  {name:<18}{ep*1000:7.1f}{mg:10.2f}"
              f"{th:8.0f}{fi:8.0f}{spread:9.0f}{fspread:9.0f}{mark}")
        # a szembefogás (hüvelyk vs. mutató+középső) az ELSŐDLEGES;
        # a négy ujj átlaga csak másodlagos.
        score = fspread + 0.2 * spread
        if best is None or score < best[0]:
            best = (score, name, R, ep, mg, zs, base, spread, fspread)

    if best is None:
        print("\n  ❌ egyetlen orientáció sem elérhető")
        return 1
    score, name, R, ep, mg, zs, base, spread, fspread = best
    print(f"\n  LEGJOBB: {name} · hüvelyk–ujjak eltérés {spread:.0f} mm · "
          f"hüvelyk vs. mutató+középső {fspread:.0f} mm")
    print(f"    ujjankénti magasság a talptól:")
    for k, v in zs.items():
        print(f"      {k:<9}{(v-base)*1000:6.0f} mm")
    if spread >= TARGET_SPREAD_MM or fspread >= TARGET_FINGER_SPREAD_MM:
        print(f"\n  ⚠️ A {TARGET_SPREAD_MM:.0f} mm-es kritériumot EGYIK SEM "
              f"teljesíti. Ez jelentés: az orientáció önmagában nem elég,")
        print(f"     a bütyöksor állását a KÉZFORMA (`HAND_OPEN`) is köti.")
        return 1
    print(f"\n  → ez az orientáció mehet a GRASP_POSES-be, és utána a "
          f"tervező újrafuttatandó vele.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
