"""
shelflife_grasp_film.py — a fogási mozdulat FELVÉTELE és videóra tétele

    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/shelflife_grasp_film.py

Kimenet: results/shelflife_film/
    kozeli.mp4 / attekinto.mp4   (ha van imageio-ffmpeg)
    kozeli_0000.png ...          (mindenképp)
    kulcskepek/                  (érkezés · első kontaktus · fogás · emelés)

────────────────────────────────────────────────────────────────────────────
MIÉRT KÉT FÁZISBAN
────────────────────────────────────────────────────────────────────────────
1. FELVÉTEL — a szekvencia lefut, és közben `qpos`-t rögzítünk. Ehhez nem kell
   grafika, tehát fejlesztői gépen is megy.
2. RENDER — a rögzített állapotokat visszajátsszuk és képpé alakítjuk. Ehhez
   OpenGL kell (a sandboxban nincs).

A felvétel a `GraspRobot.step` ideiglenes kicserélésével történik. Ez TESZT-
ESZKÖZ, nem interfész-változás: a befagyasztott szótár (D1) egyetlen
szignatúrája sem módosul, és a felvétel nem befolyásolja a fizikát.

────────────────────────────────────────────────────────────────────────────
MIT KELL NÉZNI RAJTA
────────────────────────────────────────────────────────────────────────────
A mérés szerint a fogás azért nem tart, mert a kontaktusok NEM SZEMBEN vannak:

    mutató  a jobb lapon,  +y felé nyom
    középső a jobb lapon,  +y felé nyom
    hüvelyk az ELÜLSŐ lapon, +x felé nyom
    a túlsó lapon SEMMI

Eredő 14.1 N (a súly 10.1 N) és 1.03 Nm nyomaték. A videón ennek úgy kell
látszania, hogy a kéz **oldalról tolja és megbillenti** a kartont, nem
körbezárja.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env  # noqa: F401,E402  — Xvfb, ha kell
import mujoco                                       # noqa: E402
import shelflife_grasp as _G                        # noqa: E402
from shelflife_api import Robot                     # noqa: E402

OUT = _REPO / "results/shelflife_film"
EVERY = 150          # hány szimulációs lépésenként mentünk állapotot
RES = 720
FPS = 24


# ═══════════════════════════════════════════════════════════════════════════
# 1. fázis — felvétel
# ═══════════════════════════════════════════════════════════════════════════

def record() -> tuple[np.ndarray, list[tuple[int, str]]]:
    frames: list[np.ndarray] = []
    marks: list[tuple[int, str]] = []
    r = Robot()
    g = r._r
    orig_step = g.step
    counter = {"n": 0}

    def taped_step(n: int = 1) -> None:
        for _ in range(n):
            orig_step(1)
            counter["n"] += 1
            if counter["n"] % EVERY == 0:
                frames.append(g.data.qpos.copy())

    g.step = taped_step                       # type: ignore[method-assign]

    def mark(label: str) -> None:
        marks.append((len(frames), label))
        print(f"  [{len(frames):4d}] {label}")

    print("Shelf Life — a fogási mozdulat felvétele\n")
    mark("alaphelyzet")
    r.approach_until(r.preset("pre_grasp"), until="goal")
    mark("pre-grasp pont")
    r.approach_until(r.preset("grasp"), until="goal")
    mark("fogási pont (kéz nyitva, nulla kontaktus)")
    res = r.close_until(until="grip")
    mark(f"zárás vége — {res.reason}: {res.detail}")
    r.approach_until(r.preset("lift"), until="goal", guard_mm=1e9)
    orig_step(600)
    frames.append(g.data.qpos.copy())
    mark("emelés után")

    g.step = orig_step                        # type: ignore[method-assign]
    print(f"\n  {len(frames)} állapot rögzítve "
          f"({len(frames)/FPS:.1f} s @ {FPS} fps)")
    return np.array(frames), marks


# ═══════════════════════════════════════════════════════════════════════════
# 2. fázis — render
# ═══════════════════════════════════════════════════════════════════════════

def make_camera(model, data, distance: float, elevation: float,
                azimuth: float) -> "mujoco.MjvCamera":
    """Szabad kamera a TERMÉKRE nézve — a jelenet saját kamerái túl tágak."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.geom_xpos[gid]
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    return cam


def ghost_shelf(model, alpha: float = 0.22) -> None:
    """A polcot áttetszővé teszi — CSAK a képen, a fizikán nem változtat.

    A polclapok és az oldalfalak eltakarják a kezet: a termék a középső lapon
    ül (z=1.16), fölötte 1.35-nél a következő lap, oldalt ±0.45-nél a falak.
    Bármelyik hasznos nézetből belelóg valamelyik. Az áttetszőség megoldja,
    és nem kell a kamerát olyan helyre erőltetni, ahonnan nem látni semmit.
    """
    for i in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if n.startswith("shelf"):
            model.geom_rgba[i] = [0.62, 0.62, 0.60, alpha]
            model.geom_matid[i] = -1          # anyag nélkül a rgba érvényesül


def render(frames: np.ndarray, marks: list[tuple[int, str]],
           ghost: bool = True) -> None:
    from PIL import Image, ImageDraw

    r = Robot()
    m, d = r._r.model, r._r.data
    if ghost:
        ghost_shelf(m)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = OUT / "kulcskepek"
    keys.mkdir(exist_ok=True)
    label_at = {i: t for i, t in marks}

    # ⚠️ KAMERASZÖG — az első változat a POLC MÖGÜL nézett.
    #
    # A MuJoCo szabad kamerájánál a `forward` irány
    #     (cos az · cos el,  sin az · cos el,  sin el),
    # és a kamera a `lookat − távolság · forward` pontban ül. Ha tehát
    # cos(az) < 0, a kamera a +x oldalra kerül — a polc BELSEJÉBE, mert a
    # termék x=0.40-nál van, a polc eleje 0.34-nél. Az első próbánál az
    # azimut 145° volt: cos(145°) = −0.82, vagyis pontosan a polcon át
    # néztünk kifelé.
    #
    # A robot −x felől nyúl a polcra, tehát a nézőpontnak is ott a helye:
    # |az| < 90°. A −y (a robot jobbja) felőli oldalról látszanak az ujjak.
    views = {
        # szemből: a kéz és a karton, épp a polcnyílás előtt
        "elolrol": dict(distance=0.40, elevation=-8, azimuth=2),
        # jobb elölről: innen látszik, melyik lapra nyomnak az ujjak
        "jobb_elol": dict(distance=0.38, elevation=-12, azimuth=48),
        # felülről: a hüvelyk és az ujjak egymáshoz képesti helyzete
        "felulrol": dict(distance=0.45, elevation=-52, azimuth=25),
        # áttekintő: a robot és a polc
        "attekinto": dict(distance=1.5, elevation=-12, azimuth=25),
    }

    renderer = mujoco.Renderer(m, RES, RES)
    for vname, kw in views.items():
        print(f"\n  render: {vname} ({len(frames)} kép)")
        imgs = []
        for i, q in enumerate(frames):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            cam = make_camera(m, d, **kw)
            renderer.update_scene(d, camera=cam)
            im = Image.fromarray(renderer.render())
            if i in label_at:
                ImageDraw.Draw(im).text((14, 14), label_at[i], fill=(15, 15, 15))
                im.save(keys / f"{vname}_{i:04d}_{_slug(label_at[i])}.png")
            im.save(OUT / f"{vname}_{i:04d}.png")
            imgs.append(np.array(im))
        _write_video(OUT / f"{vname}.mp4", imgs)
    renderer.close()


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in s.lower())
    return keep[:40].strip("_")


def _write_video(path: Path, imgs: list[np.ndarray]) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(f"      (nincs imageio — csak PNG-k készültek; "
              f"telepítés: pip install imageio imageio-ffmpeg)")
        return
    try:
        imageio.mimsave(path, imgs, fps=FPS)
        print(f"      → {path.name}")
    except Exception as e:                            # noqa: BLE001
        gif = path.with_suffix(".gif")
        imageio.mimsave(gif, imgs[::2], fps=FPS // 2)
        print(f"      mp4 nem ment ({type(e).__name__}), GIF készült: {gif.name}")


def main() -> int:
    if OUT.exists():
        for p in OUT.glob("*.png"):
            try:
                p.unlink()
            except OSError:
                pass          # a csatolt mappában nem tudunk törölni — nem baj
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--solid-shelf", action="store_true",
                    help="ne tegye áttetszővé a polcot")
    a = ap.parse_args()
    frames, marks = record()
    np.save(_REPO / "results/shelflife_film_frames.npy", frames)
    try:
        render(frames, marks, ghost=not a.solid_shelf)
    except Exception as e:                            # noqa: BLE001
        print(f"\n  A RENDER NEM FUT: {type(e).__name__}: {e}")
        print("  A felvétel megvan (results/shelflife_film_frames.npy).")
        print("  Ha ez a fejlesztői sandbox, futtasd a saját gépeden — "
              "ott van OpenGL.")
        return 1
    print(f"\n  Kész: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
