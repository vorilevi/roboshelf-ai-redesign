"""
shelflife_film_aperture.py — a FOGÁSNYÍLÁS és a FOGÁSPONT felvétele

    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/shelflife_film_aperture.py

Kimenet: results/shelflife_film_aperture/
    tengely.mp4 / oldal.mp4 / szemben.mp4     a teljes közelítés + záródás
    kontakt_ív.png                            hat kulcsállapot egy lapon

────────────────────────────────────────────────────────────────────────────
MIÉRT KÉSZÜL EZ
────────────────────────────────────────────────────────────────────────────
2026-08-06-án a számok azt mondják, hogy a doboz NINCS a fogásnyílásban:

    nyitott kéz, minden ujj távolsága a doboztól:   20–25 mm
    hüvelyk↔mutató legszűkebb nyílás:               57,1 mm
    a doboz átmérője:                               58,0 mm

    záráskor 0,70-nél a hüvelyk ÉS a mutató EGYSZERRE hatol 4,6–4,7 mm-t
    a dobozba, miközben a köztük lévő nyílás 31 mm — egy 58 mm-es henger
    nem fér el két, egymástól 31 mm-re lévő ujj között

Ebből az következik, hogy a hüvelyk és az ujjak UGYANAZON AZ OLDALON vannak,
és a zárás nem rásimul a dobozra, hanem áthalad rajta — a megoldó pedig
kilövi (1587 N, 43 mm elmozdulás; erőkorláttal a padlóra esik).

⚠️ EZ EGYELŐRE KÖVETKEZTETÉS, NEM MEGFIGYELÉS. A projekt munkaszabálya
szerint mielőtt egy ilyen állításra átterveznénk a kézformát, MEG KELL NÉZNI.
Ez a felvétel azért készül, hogy a következtetés cáfolható legyen.

────────────────────────────────────────────────────────────────────────────
MIT KELL MEGNÉZNI — HÁROM ELDÖNTHETŐ KÉRDÉS
────────────────────────────────────────────────────────────────────────────
1. SZEMBEN VAN-E A HÜVELYK AZ UJJAKKAL?
   A `tengely` nézet a doboz tengelye mentén, FELÜLRŐL néz. Ebben látszik,
   hogy a hüvelyk a doboz túloldalán van-e, vagy mellette, ugyanazon az
   oldalon. Ha ugyanott: nincs oppozíció, és a fogás elvileg lehetetlen.

2. A DOBOZ A NYÍLÁSBAN VAN-E, VAGY MELLETTE?
   Ugyanez a nézet mutatja, hogy a doboz keresztmetszete a hüvelyk és az
   ujjak KÖZÉ esik-e. Ha nem, akkor nem a kézformával van baj, hanem a
   fogásponttal — és fordítva.

3. HOL ÉRI EL A KÉZ A DOBOZT A MAGASSÁG MENTÉN?
   Az `oldal` nézet mutatja, a doboz 145 mm-es palástján hol van a kéz:
   a peremen, a felső harmadon vagy középen.

A képekre rá van írva a zárási szint, a nyílás és ujjanként a távolság,
hogy a látvány és a szám EGYÜTT legyen ellenőrizhető.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env      # noqa: F401,E402  — Xvfb, ha kell
import mujoco                                        # noqa: E402
from shelflife_api import Robot                       # noqa: E402

OUT = _REPO / "results/shelflife_film_aperture"
RES = 640
FPS = 15
EVERY_APPROACH = 200       # a közelítés hosszú és érdektelen: ritka mintavétel
EVERY_CLOSE = 20           # a zárás a lényeg: sűrű
MAX_RENDER = 130           # ennyi kockánál többet nem renderelünk nézetenként
LEVELS = 20                # a zárás felbontása
SETTLE = 80                # lépés zárási szintenként
DIGITS = ("thumb", "index", "middle", "ring", "little")
HU = {"thumb": "hü", "index": "mut", "middle": "köz",
      "ring": "gyű", "little": "kis"}


class Probe:
    """A geometriai mérőrész — ugyanaz, ami a számokat adta."""

    def __init__(self, robot: Robot):
        self.g = robot._r
        self.m, self.d = self.g.model, self.g.data
        bn = lambda b: mujoco.mj_id2name(          # noqa: E731
            self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self.prod = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                      "product_0_col")
        self.dg = {dg: [g for g in range(self.m.ngeom)
                        if f"r_{dg}_" in bn(self.m.geom_bodyid[g])]
                   for dg in DIGITS}
        self._ft = np.zeros(6)

    def _dist(self, a: int, b: int) -> float:
        return float(mujoco.mj_geomDistance(self.m, self.d, a, b, 0.6,
                                            self._ft))

    def gaps(self) -> dict[str, float]:
        return {dg: min(self._dist(g, self.prod) for g in gs) * 1000
                for dg, gs in self.dg.items() if gs}

    def aperture(self) -> float:
        return min(self._dist(a, b) for a in self.dg["thumb"]
                   for b in self.dg["index"]) * 1000


def record() -> tuple[np.ndarray, list[str], int]:
    """qpos-sorozat + képfeliratok. Grafika nélkül."""
    r = Robot()
    r.reset_home()
    g = r._r
    p = Probe(r)

    frames: list[np.ndarray] = []
    labels: list[str] = []
    orig_step, n = g.step, {"i": 0}
    state = {"txt": "alaphelyzet"}

    every = {"v": EVERY_APPROACH}

    def taped(k: int = 1) -> None:
        for _ in range(k):
            orig_step(1)
            n["i"] += 1
            if n["i"] % every["v"] == 0:
                frames.append(g.data.qpos.copy())
                labels.append(state["txt"])

    g.step = taped                                   # type: ignore[method-assign]

    print("Shelf Life — a FOGÁSNYÍLÁS felvétele\n")
    state["txt"] = "közelítés a terv pályáján"
    fp = r.follow_plan(guard_mm=1e9)
    print(f"  pálya bejárva · a termék {fp.data['product_moved_mm']:.1f} mm-t "
          f"mozdult")

    gp = p.gaps()
    ap = p.aperture()
    print(f"\n  NYITOTT KÉZ:  nyílás {ap:.1f} mm · "
          + " · ".join(f"{HU[k]} {v:.0f}" for k, v in gp.items()))
    print("  a doboz átmérője: 58,0 mm\n")
    print(f"  {'szint':>6}{'nyílás':>9}"
          + "".join(f"{HU[k]:>7}" for k in DIGITS))
    print("  " + "─" * 50)

    every["v"] = EVERY_CLOSE
    close_start = len(frames)
    for i in range(1, LEVELS + 1):
        lvl = i / LEVELS
        gp, ap = p.gaps(), p.aperture()
        state["txt"] = (f"zárás {lvl:.2f} · nyílás {ap:.0f} mm · "
                        + " ".join(f"{HU[k]}{v:+.0f}" for k, v in gp.items()))
        g.close_fingers(lvl, settle=SETTLE)
        gp, ap = p.gaps(), p.aperture()
        print(f"  {lvl:6.2f}{ap:8.1f} "
              + "".join(f"{gp[k]:7.1f}" for k in DIGITS))

    g.step = orig_step                               # type: ignore[method-assign]
    print(f"\n  {len(frames)} állapot rögzítve "
          f"({close_start} közelítés + {len(frames)-close_start} zárás)")
    return np.array(frames), labels, close_start


VIEWS = {
    # a doboz TENGELYE mentén, felülről — ebben látszik az oppozíció
    "tengely": dict(distance=0.62, elevation=-88, azimuth=0),
    # oldalról — ebben látszik a fogás MAGASSÁGA a paláston
    "oldal": dict(distance=0.55, elevation=-8, azimuth=-70),
    # szemből — a nyílás nagysága
    "szemben": dict(distance=0.50, elevation=-14, azimuth=8),
}


def render(frames: np.ndarray, labels: list[str], close_start: int) -> None:
    from PIL import Image, ImageDraw

    r = Robot()
    m, d = r._r.model, r._r.data
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("shelf"):                   # áttetsző polc
            m.geom_rgba[i] = [0.62, 0.62, 0.60, 0.18]
            m.geom_matid[i] = -1
    OUT.mkdir(parents=True, exist_ok=True)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")

    # a felvétel sűrűbb, mint amennyit érdemes renderelni
    stride = max(1, len(frames) // MAX_RENDER)
    keep = list(range(0, len(frames), stride))
    close_r = len([i for i in keep if i < close_start])
    frames, labels = frames[keep], [labels[i] for i in keep]

    ren = mujoco.Renderer(m, RES, RES)
    shots: dict[str, list] = {}
    for name, kw in VIEWS.items():
        imgs = []
        for i, q in enumerate(frames):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = d.geom_xpos[gid]
            for k, v in kw.items():
                setattr(cam, k, v)
            ren.update_scene(d, camera=cam)
            im = Image.fromarray(ren.render())
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, RES, 22], fill=(255, 255, 255))
            dr.text((8, 6), labels[i][:96], fill=(10, 10, 10))
            im.save(OUT / f"{name}_{i:04d}.png")
            imgs.append(im)
        shots[name] = imgs
        print(f"  {name}: {len(imgs)} kép")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(OUT / f"{name}.mp4",
                            [np.array(x) for x in imgs], fps=FPS)
        except Exception as e:                        # noqa: BLE001
            print(f"    (videó nem készült: {type(e).__name__}: {e})")
    ren.close()

    # kontaktív: hat kulcsállapot a ZÁRÁS szakaszából (nem a közelítésből)
    lo, hi = close_r, len(frames) - 1
    idx = [min(hi, lo + int(k * (hi - lo))) for k in
           (0.0, 0.30, 0.55, 0.68, 0.78, 0.92)]
    for view in ("tengely", "oldal"):
        sheet = Image.new("RGB", (RES * 3, RES * 2), (255, 255, 255))
        for j, i in enumerate(idx):
            sheet.paste(shots[view][i], (RES * (j % 3), RES * (j // 3)))
        sheet.save(OUT / f"kontakt_iv_{view}.png")
    print("  kontakt_iv_tengely.png / _oldal.png: 6 kulcsállapot a zárásból")


def main() -> int:
    frames, labels, close_start = record()
    np.save(_REPO / "results/shelflife_film_aperture_frames.npy", frames)
    try:
        render(frames, labels, close_start)
    except Exception as e:                            # noqa: BLE001
        print(f"\n  A RENDER NEM FUT: {type(e).__name__}: {e}")
        return 1
    print(f"\n  Kész: {OUT}")
    print("  Nézd meg a `tengely` nézetet: a hüvelyk a doboz TÚLOLDALÁN van-e.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
