"""
shelflife_film_can.py — a doboz fogásának FELVÉTELE, sűrűn a kritikus pontokon

    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/shelflife_film_can.py

Kimenet: results/shelflife_film_can/
    zaras_*.png     a záródás, sűrű mintavétellel
    emeles_*.png    az első 2 cm emelés, MÉG sűrűbben
    *.mp4           ha van imageio-ffmpeg

────────────────────────────────────────────────────────────────────────────
MIÉRT KÉSZÜL EZ
────────────────────────────────────────────────────────────────────────────
A 2. SKU-n (Coca-Cola 330 sleek) először jött létre valódi ötujjas fogás:

    21/21 ízület megállt kontaktusra · 5 ujj · 62.3 N · observe().holding = True
    a termék záráskor 1.5 mm-t mozdult

…és az emelés első 2 centiméterén MINDEN kontaktus elveszik.

Számokból már két körön át keresem, és nem jutok tovább. A projekt saját
munkaszabálya erre az esetre: *„ha egy hibát két körnél tovább keresek
számokból, rendereljek egy képet."* A fejlesztői sandboxban nincs OpenGL,
ezért a felvétel ott készül, a RENDER viszont a felhasználó gépén.

────────────────────────────────────────────────────────────────────────────
MIT KELL MEGNÉZNI A KÉPEKEN — HÁROM KONKRÉT KÉRDÉS
────────────────────────────────────────────────────────────────────────────
1. RÁSIMULNAK-E AZ UJJAK, VAGY CSAK A BEGYÜK ÉR HOZZÁ?
   A kartonnál mérve MINDEN kontaktus a végpereceken volt — csipeszfogás,
   nem markolás. A dobozon 5 ujj ér hozzá, de nem tudom, mely perecekkel.
   Ha csak a begyek: a 25 mm-es rés-korlát a magyarázat, és a kézformát
   kell újratervezni.

2. HOGYAN VESZ EL A KONTAKTUS AZ ELSŐ 2 CM-EN?
   Három lehetőség, és egészen mást jelentenek:
     · a doboz LECSÚSZIK az ujjak közt (súrlódás kevés → csipeszfogás)
     · a doboz KILÖVŐDIK (a pozíciószabályzó túlnyom, mint a kartonnál 51 N-nál)
     · a KÉZ FORDUL EL a doboz alól (a kar korrigál, és elviszi a fogást)

3. HOL FOGJA MEG A DOBOZT?
   A terv „felülről" közelít. A 145 mm magas doboznál ez lehet a felső
   harmad vagy a perem — érdemes látni, hova kerül a kéz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env      # noqa: F401,E402  — Xvfb, ha kell
import mujoco                                        # noqa: E402
from shelflife_api import Robot, Pose                 # noqa: E402
from shelflife_conform import close_conforming        # noqa: E402

OUT = _REPO / "results/shelflife_film_can"
RES = 720
FPS = 20


def record():
    """qpos-sorozat + címkék. Grafika nélkül, tehát sandboxban is megy."""
    frames: list[np.ndarray] = []
    marks: list[tuple[int, str]] = []
    r = Robot()
    g = r._r
    orig_step = g.step
    n = {"i": 0}

    # A záródás LASSÚ folyamat, az emelés GYORS — más mintavétel kell.
    every = {"v": 60}

    def taped(k: int = 1) -> None:
        for _ in range(k):
            orig_step(1)
            n["i"] += 1
            if n["i"] % every["v"] == 0:
                frames.append(g.data.qpos.copy())

    g.step = taped                                   # type: ignore[method-assign]

    def mark(t: str) -> None:
        marks.append((len(frames), t))
        print(f"  [{len(frames):4d}] {t}")

    print("Shelf Life — a DOBOZ fogásának felvétele\n")
    mark("alaphelyzet")
    fp = r.follow_plan(guard_mm=1e9)
    mark(f"a terv pályája bejárva — termék {fp.data['product_moved_mm']:.1f} mm")

    every["v"] = 20                                  # sűrűbben: a záródás
    p0 = g.product_pose().copy()
    out = close_conforming(r)
    ob = r.observe()
    mark(f"zárás vége — {len(out['digits'])} ujj, {out['force_N']:.0f} N, "
         f"fogja={ob.holding}")

    # melyik PERECEK érnek hozzá — ez az 1. kérdés számszerű fele
    m, d = g.model, g.data
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    touch = set()
    for k in range(d.ncon):
        c = d.contact[k]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 == pb:
            touch.add(bn(b2))
        elif b2 == pb:
            touch.add(bn(b1))
    print(f"\n  ÉRINTKEZŐ PERECEK: {sorted(touch)}")
    lvl = {"distal": 0, "medial": 0, "prox": 0}
    for t in touch:
        for k in lvl:
            if k in t:
                lvl[k] += 1
    print(f"    végperec {lvl['distal']} · középperec {lvl['medial']} · "
          f"tőperec {lvl['prox']}")
    if lvl["medial"] == 0 and lvl["prox"] == 0:
        print("    → CSIPESZFOGÁS: csak ujjbegyek. Nincs körbezárás.")
    else:
        print("    → van rásimulás, nem csak begy-érintés.")

    every["v"] = 8                                   # MÉG sűrűbben: az emelés
    for i in range(4):                               # 4 × 0.5 cm
        tgt = Pose("lift", g.grasp_point() + np.array([0, 0, 0.005]), r._R_des)
        r.approach_until(tgt, until="goal", guard_mm=1e9)
        ob = r.observe()
        rise = (g.product_pose()[2] - p0[2]) * 1000
        mark(f"emelés +{(i+1)*5} mm — emelkedés {rise:+.0f} mm, "
             f"{len(ob.touching)} ujj")
        if not ob.touching:
            print("     (minden kontaktus elveszett — itt a hiba)")
            break

    g.step = orig_step                               # type: ignore[method-assign]
    print(f"\n  {len(frames)} állapot rögzítve")
    return np.array(frames), marks


def render(frames, marks) -> None:
    from PIL import Image, ImageDraw
    r = Robot()
    m, d = r._r.model, r._r.data
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("shelf"):                   # áttetsző polc
            m.geom_rgba[i] = [0.62, 0.62, 0.60, 0.20]
            m.geom_matid[i] = -1
    OUT.mkdir(parents=True, exist_ok=True)
    label = {i: t for i, t in marks}
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")

    # |azimut| < 90°, különben a kamera a polc BELSEJÉBE kerül (mértük).
    views = {
        "kozeli_oldal": dict(distance=0.26, elevation=-6, azimuth=-62),
        "kozeli_elol": dict(distance=0.26, elevation=-4, azimuth=4),
        "felulrol": dict(distance=0.32, elevation=-55, azimuth=20),
    }
    ren = mujoco.Renderer(m, RES, RES)
    for name, kw in views.items():
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
            if i in label:
                ImageDraw.Draw(im).text((12, 12), label[i], fill=(10, 10, 10))
            im.save(OUT / f"{name}_{i:04d}.png")
            imgs.append(np.array(im))
        print(f"  {name}: {len(imgs)} kép")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(OUT / f"{name}.mp4", imgs, fps=FPS)
        except Exception as e:                        # noqa: BLE001
            print(f"    (videó nem készült: {type(e).__name__})")
    ren.close()


def main() -> int:
    frames, marks = record()
    np.save(_REPO / "results/shelflife_film_can_frames.npy", frames)
    try:
        render(frames, marks)
    except Exception as e:                            # noqa: BLE001
        print(f"\n  A RENDER NEM FUT: {type(e).__name__}: {e}")
        print("  A felvétel megvan (results/shelflife_film_can_frames.npy).")
        print("  Futtasd a saját gépeden — ott van OpenGL.")
        return 1
    print(f"\n  Kész: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
