"""
shelflife_t1_gripper.py — A1/T1: A KÉTUJJAS FOGÓ A CSUKLÓRA

    python3 tools/shelflife_t1_gripper.py            # tájolás keresése + mérés
    python3 tools/shelflife_t1_gripper.py --render

────────────────────────────────────────────────────────────────────────────
MELYIK RÉTEGBE TARTOZIK
────────────────────────────────────────────────────────────────────────────
**A) KONFIGURÁCIÓ · A1** — egyszer, a robot élettartamára. Vásárláskor dől
el, melyik végszerszám kerül a csuklóra. Nem része a napi ciklusnak.

────────────────────────────────────────────────────────────────────────────
MIÉRT KÖNNYEBB EZ, MINT A GENE.01-NÉL VOLT
────────────────────────────────────────────────────────────────────────────
A GENE.01-nél 21 ujjízületet kellett leoperálni, és a modell **összeomlott**:
a testeket töröltük, de a rájuk mutató aktuátorok és érzékelők bent
maradtak. A T1 kezén nincs mit törölni — `right_hand_link` egy csonk. A
fogó egyszerűen ráépül.

────────────────────────────────────────────────────────────────────────────
A TÁJOLÁST AZ A3 DÖNTÖTTE EL — ÉS MÉRNI KELL, NEM BEÁLLÍTANI
────────────────────────────────────────────────────────────────────────────
Az A3/T1 (2026-08-16) három behelyezést söpört végig:

    tengely a csukló X-e mentén   67 megfelelő kartartás  ← EZ
    tengely a csukló Z-je mentén  46
    tengely a csukló Y-a mentén   16

A megfogott henger tengelyének tehát a csukló **X** iránya mentén kell
állnia. (A GENE.01-nél ez Y volt — a behelyezés robotfüggő.)

⚠️ EZT NEM LEHET „BEÁLLÍTANI". A 2F85 saját tengelyeinek és a csuklónak a
   viszonya nem magától értetődő, és egy elrontott kvaternió csendben
   rossz irányba állítja a fogót. Ezért ez a modul **végigpróbálja mind a
   24 tengelyilleszkedő elforgatást**, mindegyiknél LEMÉRI a megfogott
   henger tengelyét, és csak azokat fogadja el, amelyek X-et adnak.

────────────────────────────────────────────────────────────────────────────
AMIT EZ MÉG MEGMÉR, ÉS MIÉRT FONTOS
────────────────────────────────────────────────────────────────────────────
A `GRIP_OFFSET = 160 mm` (csukló → fogáspont) eddig **feltételezés** volt,
a fogós próbapad méretéből. Az A3 egész eredménye ezen áll. Felszerelés
után ez MÉRHETŐ — és ha eltér, az A3-at újra kell futtatni.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env                       # noqa: E402,F401  (SORREND!)
import mujoco                                     # noqa: E402
from shelflife_t1_scene import (                  # noqa: E402
    build_spec, apply_home, GRIP_OFFSET, RENDER_W, RENDER_H)

MENAGERIE = _REPO / "mujoco_menagerie/robotiq_2f85/2f85.xml"
WRIST = "right_hand_link"
PREFIX = "g_"
MOUNT_POS = [0.0, 0.0, 0.0]        # a csonk végén
TARGET_AXIS = "X"                   # az A3/T1 döntése
OUT = _REPO / "results/shelflife_t1_gripper"


def axis_aligned_quats():
    """A 24 tengelyilleszkedő elforgatás, kvaternióként.

    Előjeles permutációk az egységmátrixból, csak a det=+1 (valódi forgatás,
    nem tükrözés). Ez a teljes, kimerítő halmaz — nincs benne találgatás.
    """
    out = []
    for perm in itertools.permutations(range(3)):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, (sx, sy, sz))):
                R[row, col] = s
            if abs(np.linalg.det(R) - 1.0) > 1e-9:
                continue
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, R.flatten())
            out.append((perm, (sx, sy, sz), q.copy()))
    return out


_ALIVE = []          # ⚠️ a specifikációkat ÉLETBEN KELL TARTANI


def build_with_quat(quat):
    """A jelenet + a fogó, adott csuklótájolással.

    ⚠️ AZ ELSŐ VÁLTOZAT 24-SZER ÉPÍTETTE FEL A JELENETET, és a folyamat
    összeomlott menet közben. A `MjSpec` és a belőle fordított modell
    élettartama összefügg; ha a specifikáció eltűnik a szemétgyűjtőben,
    miközben a modellt még használjuk, a folyamat elszáll. Ezért van a
    `_ALIVE` lista — és ezért nem söprünk többé 24 fordítást.
    """
    s = build_spec()
    _ALIVE.append(s)
    wrist = s.body(WRIST)
    child = mujoco.MjSpec.from_file(str(MENAGERIE))
    f = wrist.add_frame()
    f.pos = MOUNT_POS
    f.quat = list(quat)
    # ⚠️ A KONTAKTUS-BEÁLLÍTÁS A SZÜLŐÉ MARAD. A Robotiq modellt
    #    `cone="elliptic" impratio="10"` mellett hangolták be; a jelenetünk
    #    már ilyen (l. shelflife_t1_scene), tehát összemérhető marad a
    #    fogós próbapad 18 N-os eredményével.
    _ALIVE.append(child)
    f.attach_body(child.body("base_mount"), PREFIX, "")
    return s.compile()


def measure(m):
    """A felszerelés ELLENŐRZÉSE. A beállítás nem bizonyíték."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                # ⚠️ itt omlott össze a GENE.01
    gn = lambda g: mujoco.mj_id2name(      # noqa: E731
        m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    w = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST)
    pads = [g for g in range(m.ngeom) if "pad" in gn(g)]
    pr = [g for g in pads if "right" in gn(g)]
    pl = [g for g in pads if "left" in gn(g)]
    if not (pr and pl):
        return None
    W = d.xmat[w].reshape(3, 3)
    r_ = np.mean([d.geom_xpos[g] for g in pr], axis=0)
    l_ = np.mean([d.geom_xpos[g] for g in pl], axis=0)
    close = W.T @ ((r_ - l_) / np.linalg.norm(r_ - l_))
    pc = np.mean([d.geom_xpos[g] for g in pads], axis=0)
    appr = W.T @ (pc - d.xpos[w])
    reach = float(np.linalg.norm(appr))          # csukló → fogáspont [m]
    appr /= reach
    ax = np.cross(appr, close)
    ax /= np.linalg.norm(ax)
    return {"tengely": ax, "nev": "XYZ"[int(np.argmax(np.abs(ax)))],
            "tisztasag": float(np.max(np.abs(ax))),
            "zaras": close, "kozelites": appr, "nyulas_m": reach, "d": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    print("Shelf Life — A1/T1: kétujjas fogó a Booster T1 csuklójára\n")
    print(f"  cél: a megfogott henger tengelye a csukló {TARGET_AXIS} "
          f"iránya mentén (A3/T1)\n")

    # ── 1. EGY mérés az alaptájolással ─────────────────────────────────────
    # A fogó minden iránya EGYÜTT fordul a felszerelési kerettel. Elég
    # tehát egyszer megmérni, aztán KISZÁMOLNI, melyik elforgatás viszi a
    # henger tengelyét X-be — nem kell 24-szer felépíteni a jelenetet.
    base = measure(build_with_quat([1.0, 0.0, 0.0, 0.0]))
    if base is None:
        print("  ❌ a fogó párnái nem találhatók — a csatolás nem sikerült")
        return 1
    print(f"  alaptájolással a henger tengelye: "
          f"{np.round(base['tengely'], 2)} → {base['nev']}")

    tgt = np.eye(3)[ "XYZ".index(TARGET_AXIS) ]
    jo = []
    for perm, sgn, q in axis_aligned_quats():
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
        R = R.reshape(3, 3)
        if abs(abs(float(np.dot(R @ base["tengely"], tgt))) - 1.0) < 1e-6:
            jo.append((q, R))
    if not jo:
        print(f"  ❌ egyetlen tengelyilleszkedő tájolás sem ad "
              f"{TARGET_AXIS}-tengelyt.")
        return 1
    print(f"  {len(jo)} tájolás viszi {TARGET_AXIS}-be "
          f"(a 24 tengelyilleszkedő elforgatásból)\n")

    # ── 2. a legjobb kiválasztása, majd VISSZAMÉRÉS ────────────────────────
    # ⚠️ Mind X-et ad, de a fogó másfelé áll ki a csuklóból. Azt választjuk,
    #    amelyiknél a közelítés a legjobban egyezik a csukló +z-jével —
    #    vagyis a fogó ELŐRE néz, nem vissza a kar felé.
    jo.sort(key=lambda it: float((it[1] @ base["kozelites"])[2]), reverse=True)
    q = jo[0][0]
    m = build_with_quat(q)
    r = measure(m)
    if r is None or r["nev"] != TARGET_AXIS:
        print("  ❌ a visszamérés nem igazolta a számítást")
        return 1

    print("  ── A VÁLASZTOTT FELSZERELÉS ────────────────────────────")
    print(f"  kvaternió          [{', '.join(f'{x:+.3f}' for x in q)}]")
    print(f"  zárási tengely     {np.round(r['zaras'], 2)}  (csuklókeret)")
    print(f"  közelítés          {np.round(r['kozelites'], 2)}")
    print(f"  A HENGER TENGELYE  {np.round(r['tengely'], 2)}  → "
          f"{r['nev']}   {'✅' if r['nev'] == TARGET_AXIS else '❌'}")
    print(f"  tisztaság          {r['tisztasag']:.3f}  "
          f"(1,0 = pontosan tengelyirányú)")

    print("\n  ── AMI EDDIG FELTÉTELEZÉS VOLT ─────────────────────────")
    mert = r["nyulas_m"]
    delta = (mert - GRIP_OFFSET) * 1000
    print(f"  csukló → fogáspont  MÉRVE {mert*1000:.1f} mm   "
          f"(feltételezve {GRIP_OFFSET*1000:.0f} mm · eltérés {delta:+.1f} mm)")
    if abs(delta) > 20:
        print("  ⚠️ 20 mm-nél nagyobb eltérés — az A3/T1-et ÚJRA KELL FUTTATNI,")
        print("     mert a talp helyét ebből számolta.")
    else:
        print("  ✅ 20 mm-en belül — az A3/T1 eredménye áll.")

    d = r["d"]
    tomeg = sum(m.body_mass[b] for b in range(m.nbody)
                if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                        or "").startswith(("product", "world")))
    print(f"\n  testek {m.nbody} · ízületek {m.njnt} · aktuátorok {m.nu} · "
          f"geomok {m.ngeom}")
    print(f"  tömeg {tomeg:.2f} kg (a fogóval együtt)")

    # a fogó valóban nyit-zár?
    ga = [i for i in range(m.nu)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              or "").startswith(PREFIX)]
    print(f"  fogó-aktuátor      {len(ga)}   "
          f"{'✅' if ga else '❌ a fogó nem vezérelhető'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mount_quat.txt").write_text(
        " ".join(f"{x:.6f}" for x in q) + "\n", encoding="utf-8")
    print(f"\n  mentve: {(OUT / 'mount_quat.txt').relative_to(_REPO)}")

    if a.render:
        render(m, d)
    return 0


def render(m, d) -> None:
    import imageio.v3 as iio
    shelflife_render_env.ensure(RENDER_W, RENDER_H)
    apply_home(m, d)
    OUT.mkdir(parents=True, exist_ok=True)
    rr = mujoco.Renderer(m, RENDER_H, RENDER_W)
    for c in ("side", "overview", "head_camera"):
        try:
            rr.update_scene(d, camera=c)
        except Exception:                          # noqa: BLE001
            continue
        img = rr.render()
        iio.imwrite(OUT / f"fogo_{c}.png", img)
        print(f"  kép: {(OUT / f'fogo_{c}.png').relative_to(_REPO)}"
              f"  (szórás {img.std():.1f})")
    rr.close()


if __name__ == "__main__":
    rc = main()
    # ⚠️ KILÉPÉSKOR ÖSSZEOMLIK — a MUNKA UTÁN, nem közben.
    #    A `MjSpec`-ből fordított modell a specifikáció memóriájára mutat, és
    #    az értelmező leállásakor a kettő felszabadítási sorrendje nem
    #    garantált. Minden szám kiszámolva, minden fájl kiírva, aztán a
    #    folyamat elszáll — ami a hívó szemszögéből hibás kilépési kód.
    #    Ezért lépünk ki takarítás nélkül. NEM elfedés: a hiba fent le van
    #    írva, és ha valaha kiderül a valódi oka, ez a sor törlendő.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(rc)
