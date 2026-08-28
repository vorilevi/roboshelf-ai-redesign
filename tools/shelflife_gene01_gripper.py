"""
shelflife_gene01_gripper.py — A1: a KÉTUJJAS FOGÓ a GENE.01 csuklójára

    python3 tools/shelflife_gene01_gripper.py

────────────────────────────────────────────────────────────────────────────
MELYIK RÉTEGBE TARTOZIK
────────────────────────────────────────────────────────────────────────────
**A) KONFIGURÁCIÓ · A1** — egyszer, a robot élettartamára. Vásárláskor dől
el, melyik végszerszám kerül a csuklóra. Nem része a napi ciklusnak.

────────────────────────────────────────────────────────────────────────────
MIÉRT KÉTUJJAS FOGÓ AZ ÖTUJJAS KÉZ HELYETT
────────────────────────────────────────────────────────────────────────────
2026-08-06, mérve:

    ötujjas kéz          kétujjas fogó (Robotiq 2F85)
    ─────────────────────────────────────────────────
    feldönti a dobozt    a termék 0,0–2,0 mm-t mozdul a közelítés alatt
    2 ujj ér hozzá       12–18 kontaktus
    1587 N szabályozat-  18 N, a termék átmérőjéből számolva
      lanul, átzár
    nem emeli fel        100% követés (84,9 mm doboz / 84,8 mm fogó)
    21 ujjízület         1 aktuátor

Külső megerősítés: a Google Gemini Robotics 2 publikus mérésein a kétujjas
fogó JOBBAN teljesített, mint a többujjas kéz — a legjobb számaik (74,2%
felvétel-letevés) kétujjas fogóval születtek.

────────────────────────────────────────────────────────────────────────────
A TÁJOLÁS — EZT AZ A3 DÖNTÖTTE EL
────────────────────────────────────────────────────────────────────────────
A dátum a colásdoboz TALPÁN van, tehát a robotnak meg kell fordítania a
terméket. Az A3 képességvizsgálat három behelyezést söpört végig:

    tengely a csukló X-e mentén    2 megfelelő kartartás
    tengely a csukló Y-a mentén   49 megfelelő kartartás  ← EZ
    tengely a csukló Z-je mentén   4 megfelelő kartartás

A fogót tehát úgy kell felszerelni, hogy a megfogott henger tengelye a
csukló Y iránya mentén álljon.

⚠️ A TÁJOLÁST FELSZERELÉS UTÁN MEG KELL MÉRNI, nem elég beállítani. Ez a
   modul ki is írja a mért tengelyt — ha nem Y, a `MOUNT_QUAT`-ot kell
   javítani, nem az eredményt magyarázni.

────────────────────────────────────────────────────────────────────────────
AMI EZZEL ELVESZIK, ÉS AMI NEM
────────────────────────────────────────────────────────────────────────────
❌ elveszik: az ötujjas kéz és a 21 ujjízülete. Amit AZZAL lehetne csinálni —
   villanykörte kicsavarása, zacskókötés, cipzár — azt ez a robot nem tudja.
   Polcfeltöltéshez viszont egyik sem kell.
✅ megmarad: a teljes kar, a törzs, a kamerák, a taktilis bőr.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
from shelflife_robot_spec import apply_mass      # noqa: E402

SCENE = _REPO / "src/envs/assets/shelflife_scene_gene01_sku_v1.xml"
MENAGERIE = _REPO / "mujoco_menagerie/robotiq_2f85/2f85.xml"

WRIST = "r_wrist_3"
FINGERS = ("r_thumb", "r_index", "r_middle", "r_ring", "r_little")

# a fogó tövének helye és tájolása a csuklóban
MOUNT_POS = [0.0, 0.0, 0.02]
MOUNT_QUAT = [1.0, 0.0, 0.0, 0.0]        # felszerelés után MÉRNI kell


def build_model(verbose: bool = True):
    """GENE.01 kétujjas fogóval, egyszerű polccal, 80 kg-ra skálázva."""
    parent = mujoco.MjSpec.from_file(str(SCENE))

    # ── 1. az ötujjas kéz leválasztása ──────────────────────────────────────
    # ⚠️ A TÖRLÉS A SPEC METÓDUSA, NEM AZ ELEMÉ.
    #    Az első változat `b.delete()`-et hívott — a testnek nincs ilyen
    #    metódusa, a hívás csendben elszállt a `try` blokkban, és NULLA
    #    ujjat választott le. A modell mégis lefordult, 21 ujjtesttel.
    #    Ezért van az ellenőrzés: a beállítás nem bizonyíték.
    dropped = 0
    for b in list(parent.bodies):
        if any(b.name.startswith(f) for f in FINGERS):
            parent.delete(b)
            dropped += 1

    # ── 2. a felesleges polcelemek eltávolítása ─────────────────────────────
    # ⚠️ EGY POLCLAP, SEMMI TÖBB. A felső lap és a hátfal olyan korlátokat
    #    hozott be, aminek semmi köze a vizsgált kérdéshez.
    shelf_gone = 0
    for g in list(parent.geoms):
        if g.name in ("shelf_board_2", "shelf_back") or \
           g.name.startswith("shelf_side"):
            parent.delete(g)
            shelf_gone += 1

    # ── 3. a fogó felcsatolása ──────────────────────────────────────────────
    child = mujoco.MjSpec.from_file(str(MENAGERIE))
    wrist = parent.body(WRIST)
    frame = wrist.add_frame()
    frame.pos = MOUNT_POS
    frame.quat = MOUNT_QUAT
    frame.attach_body(child.body("base_mount"), "g_", "")

    # ⚠️ A KONTAKTUS-BEÁLLÍTÁS A SZÜLŐÉ MARAD, ÉS EZ SZÁMÍT.
    #    A Robotiq modell `cone="elliptic" impratio="10"` beállítással
    #    készült — ezen hangolták be a fogását. A humanoid jelenet
    #    alapértelmezett (piramis, impratio=1), és csatoláskor a SZÜLŐ nyer.
    #    A fogós próbapadon elliptikus kúppal mértük a 18 N-os fogást, tehát
    #    a hiteles összehasonlításhoz itt is az kell.
    parent.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    parent.option.impratio = 10.0

    model = parent.compile()
    apply_mass(model)                             # 152,4 kg → 80 kg

    if verbose:
        print(f"  leválasztott ujjtestek: {dropped}")
        print(f"  eltávolított polcelemek: {shelf_gone}")
        print(f"  testek {model.nbody} · ízületek {model.njnt} · "
              f"aktuátorok {model.nu} · geomok {model.ngeom}")
    return model


def measure(model) -> None:
    """A felszerelés ELLENŐRZÉSE — nem elég beállítani, mérni kell."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    bn = lambda b: mujoco.mj_id2name(          # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    gn = lambda g: mujoco.mj_id2name(          # noqa: E731
        model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    w = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST)
    pads = [g for g in range(model.ngeom) if "pad" in gn(g)]
    pr = [g for g in pads if "right" in gn(g)]
    pl = [g for g in pads if "left" in gn(g)]

    print("\n  ── ELLENŐRZÉS ──────────────────────────────────────")
    tomeg = sum(model.body_mass[b] for b in range(model.nbody)
                if not bn(b).startswith(("product", "world")))
    print(f"  tömeg                {tomeg:.1f} kg")

    finger_left = [b for b in range(model.nbody)
                   if any(bn(b).startswith(f) for f in FINGERS)]
    print(f"  megmaradt ujjtest    {len(finger_left)}"
          f"   {'✅' if not finger_left else '❌ nem sikerült leválasztani'}")

    ng = len([g for g in range(model.ngeom) if gn(g).startswith("shelf")])
    print(f"  polcgeomok           {ng}   {'✅ egy lap' if ng == 1 else '❌'}")

    if pr and pl:
        W = d.xmat[w].reshape(3, 3)
        r = np.mean([d.geom_xpos[g] for g in pr], axis=0)
        l = np.mean([d.geom_xpos[g] for g in pl], axis=0)
        close_w = (r - l) / np.linalg.norm(r - l)
        close_local = W.T @ close_w
        # a megfogott henger tengelye ⟂ a zárási tengelyre ÉS a közelítésre
        pc = np.mean([d.geom_xpos[g] for g in pads], axis=0)
        appr_local = W.T @ (pc - d.xpos[w])
        appr_local /= np.linalg.norm(appr_local)
        axis_local = np.cross(appr_local, close_local)
        axis_local /= np.linalg.norm(axis_local)
        nev = "XYZ"[int(np.argmax(np.abs(axis_local)))]
        print(f"  zárási tengely (csukló)  {np.round(close_local, 2)}")
        print(f"  közelítés     (csukló)  {np.round(appr_local, 2)}")
        print(f"  A HENGER TENGELYE       {np.round(axis_local, 2)}  → {nev}")
        print(f"  {'✅ Y — ahogy az A3 kérte' if nev == 'Y' else f'⚠️ {nev}, nem Y — a MOUNT_QUAT-ot kell javítani'}")
    else:
        print("  ❌ a fogó párnái nem találhatók — a csatolás nem sikerült")


def main() -> int:
    print("Shelf Life — A1: kétujjas fogó a GENE.01 csuklójára\n")
    m = build_model()
    measure(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
