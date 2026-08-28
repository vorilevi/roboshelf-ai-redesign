"""
shelflife_check_render.py — a renderelési lánc ellenőrzése (FELHASZNÁLÓ GÉPÉN)

A fejlesztői sandboxban nincs OpenGL, ezért ez a szkript ott nem fut le.
Ez az EGYETLEN lépés, amit a felhasználó gépén kell futtatni.

    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/shelflife_check_render.py
    # ha OpenGL-hibát ad (ritka macOS-en):
    mjpython tools/shelflife_check_render.py

────────────────────────────────────────────────────────────────────────────
MIT VÁLASZOL MEG
────────────────────────────────────────────────────────────────────────────
A fő kérdés: OLVASHATÓ-E a szavatossági dátum a robot saját kameranézetéből?
Eddig ezt csak szimulálva mértük (sík címke, enyhe életlenítés): 224 px →
3.6 px betűmagasság, olvashatatlan; 640 px → 10.3 px, tökéletes. A valódi
render ferde szögből, megvilágítással és textúraszűréssel rosszabb lehet.

AZ ELSŐ FUTÁS TANULSÁGA (2026-08-02): a képek üresek voltak, mert a POLCON
álló terméket egyik gyári kamera sem látja — 32°-kal kilóg a mellkasi kamera
22.5°-os fél-látószögéből, a fejkamerák pedig 44°-kal fölötte néznek el.
Ezért a szkript most GEOMETRIAILAG ellenőrzi a láthatóságot render ELŐTT.

A megoldás ugyanaz, mint embernél: a terméket a kamera elé kell emelni.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

OUT = _REPO / "results/shelflife_render_check"
ROBOT_CAMS = ("rgb_camera", "chest_camera_front", "head_camera_left")


def save(img: np.ndarray, name: str) -> None:
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(OUT / name)
    print(f"      → {name}  ({img.shape[1]}×{img.shape[0]})")


def main() -> int:
    import mujoco
    from shelflife_primitives import ShelfLifeRobot

    print("Shelf Life — renderelés-ellenőrzés\n")
    r = ShelfLifeRobot()
    p0 = r.product_pose().copy()

    # ── 1. stabilitás ───────────────────────────────────────────────────────
    print("[1/5] stabilitás")
    palm0 = r.palm_pose().copy()
    r.step(1200)
    drift = abs(r.palm_pose()[2] - palm0[2]) * 1000
    print(f"      kéz sodródás {drift:.2f} mm · termék "
          f"{np.linalg.norm(r.product_pose()-p0)*1000:.2f} mm "
          f"{'✅' if drift < 2 else '❌'}")
    r.reset()

    # ── 2. áttekintő képek (emberi ellenőrzéshez) ───────────────────────────
    print("\n[2/5] áttekintő nézetek")
    try:
        for cam in ("overview", "side_view"):
            save(r.render_view(cam, 960), f"00_{cam}.png")
    except RuntimeError as e:
        print(f"\n❌ A RENDERELÉS NEM MEGY:\n{e}")
        return 1
    except ValueError:
        print("      (nincs overview kamera — futtasd újra a build_scene-t)")

    # ── 3. látszik-e a termék a POLCON? (geometria, render nélkül) ─────────
    print("\n[3/5] látszik-e a termék a polcon — geometriai ellenőrzés")
    for cam in ROBOT_CAMS:
        print(f"      {r.in_view(r.product_pose(), cam).detail}")
    print("      ↑ ha mind KILÓG, az VÁRT: a terméket a kamera elé kell emelni")

    # ── 4. vizsgálati póz: a termék a kamera elé ────────────────────────────
    print("\n[4/5] vizsgálati póz")
    pos, fwd = r.camera_pose("rgb_camera")
    insp = pos + fwd * 0.18
    # TESZT-FIXTÚRA: a terméket ODATESSZÜK, mintha a kézben lenne.
    # Ez NEM képesség-állítás — a fogást az ügynöknek kell megoldania.
    # Itt csak azt mérjük, hogy OLVASHATÓ-E a dátum ilyen távolságból.
    jid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_JOINT, "product_0_free")
    adr = r.model.jnt_qposadr[jid]
    r.data.qpos[adr:adr+3] = insp
    r.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    r.data.qvel[:] = 0
    mujoco.mj_forward(r.model, r.data)
    v = r.in_view(r.product_pose(), "rgb_camera")
    print(f"      termék a kamera elé helyezve ({0.18:.2f} m)")
    print(f"      {v.detail}  {'✅' if v.ok else '❌'}")

    # ── 5. felbontás-sorozat — EZ A LÉNYEG ──────────────────────────────────
    print("\n[5/5] felbontás-sorozat (a dátum olvashatósága)")
    for res_px in (224, 448, 640, 960, 1280):
        try:
            save(r.render_view("rgb_camera", res_px), f"03_date_{res_px}px.png")
        except Exception as e:
            print(f"      {res_px}px: nem sikerült ({type(e).__name__})")

    gt = r._ground_truth(0)
    print("\n" + "─" * 64)
    print("GROUND TRUTH — ennek kellene olvashatónak lennie a 03_* képeken:")
    if gt:
        print(f"    {gt.get('product')}  ·  {gt.get('date_kind')} "
              f"{gt.get('date_printed')}  ·  helyes döntés: "
              f"{gt.get('ground_truth_decision')}")
    else:
        print("    nincs manifest — futtasd: python3 tools/shelflife_make_textures.py")
    print("─" * 64)
    print(f"\nKépek: {OUT}")
    print("KÜLDD VISSZA: 00_overview.png és a 03_date_*.png sorozatot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
