"""
shelflife_check_sku_render.py — a szkennelt SKU renderelése (FELHASZNÁLÓ GÉPÉN)

    cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
    python3 tools/shelflife_check_sku_render.py

Egyetlen kérdést válaszol meg: OLVASHATÓ-E a dátum a szkennelt Alpro-dobozon,
a robot vizsgálati távolságából, a valódi MuJoCo-renderen?

Eddig ezt csak számoltuk (4 mm-es betű, ~7 px olvashatósági küszöb) és külön
szimulálva mértük — a valódi renderen még senki nem látta.

A szkript render ELŐTT geometriailag ellenőrzi, hogy a dátum a képben lesz-e,
mert egyszer már kaptunk öt üres képet attól, hogy a tárgy kilógott a
látótérből.

────────────────────────────────────────────────────────────────────────────
EXPOZÍCIÓ: MÉRVE, NEM BECSÜLVE
────────────────────────────────────────────────────────────────────────────
Kétszer állítottuk a fényeket kézzel, és kétszer maradt kiégett a kép. Az ok
strukturális, nem szám-hangolási:

  A fotogrammetriás textúrában A MEGVILÁGÍTÁS MÁR BENNE VAN. A dobozt
  bevilágítva fényképeztük, tehát a textúra fehér részei eleve ~240/255-ön
  vannak. Ha erre a renderelő még rárak ambient+diffuse+specular járulékot,
  1.0 fölé megy és levágódik — a kupak és a zárófül tiszta fehér folttá válik.

Nyílt hurokban ezt nem lehet eltalálni: minden textúra máshol áll. Ezért itt
ZÁRT HURKOT csinálunk, ugyanazzal a mintával, mint a GR1T1 pozicionálásnál:
kis felbontáson végigmegyünk egy expozíciós létrán, SZEGMENTÁCIÓS maszkkal
kivágjuk a terméket a háttérből, megszámoljuk a levágott (kiégett) pixeleket,
és azon a szinten renderelünk élesben, ahol a levágás a küszöb alá esik.

Kimenet: results/shelflife_sku_render/*.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
SKU = "alpro_barista_coconut_1l"
D = _REPO / "models/shelflife_sku" / SKU
OUT = _REPO / "results/shelflife_sku_render"

# A robot vizsgálati távolsága: a terméket a mellkasi kamera elé emeli.
# 0.18 m-en a 8 cm-es termék a kép ~54%-át tölti ki (mérve a 0. kapuban).
INSPECT_DIST = 0.18

# Fényerő-alapértékek 1.0-s expozíciónál. A létra ezeket skálázza.
HL_AMBIENT, HL_DIFFUSE = 0.42, 0.22
KEY_DIFFUSE, FILL_DIFFUSE = 0.30, 0.16

EXPOSURE_LADDER = (1.00, 0.85, 0.72, 0.60, 0.50, 0.42, 0.35)
CLIP_BUDGET = 0.005      # a termék pixeleinek legfeljebb 0.5%-a éghet ki
CALIB_RES = 384          # a létrához elég kis felbontás


def build_xml(exposure: float) -> Path:
    """A jelenet XML-je adott expozícióval. Minden fényforrás egy szorzót kap."""
    e = exposure
    frag = (D / "product.xml").read_text()
    asset = frag.split("<asset>")[1].split("</asset>")[0].replace(
        f"../../../models/shelflife_sku/{SKU}/", "")
    body = "<body" + frag.split("<body", 1)[1]

    # Kamerák: a dátum köré, különböző rálátási szögekből.
    # A dátum a tetőn van (z≈0.204 m), normálisa ~(-0.2, 0, 0.98).
    cams = []
    for name, pos, target in [
        ("insp_top",   (0.023, -0.010, 0.204 + INSPECT_DIST), (0.023, -0.010, 0.204)),
        ("insp_45",    (0.023, -0.010 - INSPECT_DIST * 0.7, 0.204 + INSPECT_DIST * 0.7),
                       (0.023, -0.010, 0.204)),
        ("insp_60",    (0.023, -0.010 - INSPECT_DIST * 0.5, 0.204 + INSPECT_DIST * 0.87),
                       (0.023, -0.010, 0.204)),
        ("product",    (0.0, -0.42, 0.26), (0.0, 0.0, 0.10)),
    ]:
        p = np.array(pos, float); t = np.array(target, float)
        fwd = t - p; fwd /= np.linalg.norm(fwd)
        up0 = np.array([0, 0, 1.0]) if abs(fwd[2]) < 0.95 else np.array([0, 1.0, 0])
        right = np.cross(fwd, up0); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        cams.append(f'<camera name="{name}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
                    f'xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f}  '
                    f'{up[0]:.4f} {up[1]:.4f} {up[2]:.4f}" fovy="45"/>')

    def rgb(v: float) -> str:
        v = round(v * e, 4)
        return f"{v} {v} {v}"

    xml = f"""<mujoco model="sku_render">
  <compiler angle="radian" meshdir="{D}/" texturedir="{D}/"/>
  <option timestep="0.001"/>
  <visual>
    <global offwidth="1600" offheight="1600"/>
    <quality shadowsize="2048"/>
    <!-- a headlight specularja NULLA: a szkennelt textúrában a csillanás
         már benne van, a rárakott tükrös tag égette ki a kupakot -->
    <headlight ambient="{rgb(HL_AMBIENT)}" diffuse="{rgb(HL_DIFFUSE)}"
               specular="0 0 0"/>
  </visual>
  <asset>
    <!-- sötétebb ég + sötétebb padló: a korábbi majdnem fehér háttér
         mellett szemre nem lehetett megítélni, mi ég ki valójában -->
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.30 0.33 0.37" rgb2="0.52 0.55 0.58" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.58 0.58 0.55" rgb2="0.50 0.50 0.47"/>
    <material name="matfloor" texture="grid" texrepeat="6 6"
              specular="0" shininess="0" reflectance="0"/>
    {asset}
  </asset>
  <worldbody>
    <light name="key"  pos="0.3 -0.4 1.2" dir="-0.2 0.3 -1"
           diffuse="{rgb(KEY_DIFFUSE)}" specular="0 0 0"/>
    <light name="fill" pos="-0.4 0.3 0.9" dir="0.4 -0.3 -1"
           diffuse="{rgb(FILL_DIFFUSE)}" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 .05" material="matfloor"/>
    {" ".join(cams)}
    {body}
  </worldbody>
</mujoco>"""
    p = Path(f"/tmp/shelflife_sku_render_{int(e*100)}.xml")
    p.write_text(xml)
    return p


def settle(m, steps: int = 900):
    import mujoco
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{SKU}_free")
    adr = m.jnt_qposadr[jid]
    d.qpos[adr:adr + 3] = [0, 0, 0.005]
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return d


def product_mask(renderer, m, d, cam: str) -> np.ndarray:
    """Szegmentációs maszk: melyik pixel a TERMÉK (padló és háttér nélkül)."""
    import mujoco
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    oid, otype = seg[:, :, 0], seg[:, :, 1]
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)      # enum → int, hogy a numpy is értse
    return (otype == geom) & (oid >= 0) & (oid != int(floor))


def clipped_fraction(img: np.ndarray, mask: np.ndarray) -> float:
    """A termék hány százaléka égett ki (mindhárom csatorna ~telített)."""
    if mask.sum() == 0:
        return 1.0
    clip = (img.astype(np.int16).min(axis=2) >= 250)
    return float((clip & mask).sum() / mask.sum())


def calibrate_exposure(settled_qpos) -> tuple[float, float]:
    """Végigmegy az expozíciós létrán, és visszaadja az első elfogadhatót."""
    import mujoco
    print("\n[3/5] expozíció bemérése (kis felbontáson, szegmentációs maszkkal)")
    best = (EXPOSURE_LADDER[-1], 1.0)
    for e in EXPOSURE_LADDER:
        m = mujoco.MjModel.from_xml_path(str(build_xml(e)))
        d = mujoco.MjData(m)
        d.qpos[:] = settled_qpos
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, CALIB_RES, CALIB_RES)
        mask = product_mask(r, m, d, "insp_45")
        r.update_scene(d, camera="insp_45")
        frac = clipped_fraction(r.render(), mask)
        r.close()
        mark = "✅" if frac <= CLIP_BUDGET else "  "
        print(f"      expozíció {e:.2f} → kiégett {frac*100:5.2f} %  {mark}")
        if frac < best[1]:
            best = (e, frac)
        if frac <= CLIP_BUDGET:
            return e, frac
    print(f"      egyik szint sem fért a {CLIP_BUDGET*100:.1f}%-os keretbe — "
          f"a legjobbal megyünk tovább")
    return best


def main() -> int:
    import shelflife_render_env  # noqa: F401
    import mujoco
    from PIL import Image
    import json

    print("Shelf Life — a szkennelt SKU renderelése\n")

    print("[1/5] fizikai leülepedés")
    m0 = mujoco.MjModel.from_xml_path(str(build_xml(1.0)))
    d0 = settle(m0)

    gi = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, f"{SKU}_date")
    dec = d0.geom_xpos[gi].copy()
    nrm = d0.geom_xmat[gi].reshape(3, 3)[:, 2]
    print(f"      a dátum helye: z={dec[2]*100:.1f} cm · normális {np.round(nrm,2)}")

    # látszik-e — RENDER ELŐTT
    print("\n[2/5] geometriai ellenőrzés (render előtt)")
    ok_any = False
    for i in range(m0.ncam):
        nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_CAMERA, i)
        cp = d0.cam_xpos[i]; cm_ = d0.cam_xmat[i].reshape(3, 3)
        fwd = -cm_[:, 2]; right = cm_[:, 0]; up = cm_[:, 1]
        v = dec - cp; dist = np.linalg.norm(v); vn = v / dist
        half = m0.cam_fovy[i] / 2
        ah = np.degrees(np.arctan2(vn @ right, vn @ fwd))
        av = np.degrees(np.arctan2(vn @ up, vn @ fwd))
        face = np.degrees(np.arccos(np.clip(-vn @ nrm, -1, 1)))
        vis = abs(ah) < half and abs(av) < half and (vn @ fwd) > 0 and face < 80
        ok_any |= vis
        print(f"      {nm:<10} táv {dist*100:5.1f} cm · kép {ah:+5.1f}/{av:+5.1f}° · "
              f"rálátás {face:4.0f}°  {'✅' if vis else '❌'}")
    if not ok_any:
        print("\n❌ egyik kamera sem látja a dátumot — nincs értelme renderelni")
        return 1

    exp, frac = calibrate_exposure(d0.qpos.copy())

    print(f"\n[4/5] végleges jelenet — expozíció {exp:.2f} "
          f"(kiégett {frac*100:.2f} %)")
    m = mujoco.MjModel.from_xml_path(str(build_xml(exp)))
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    mujoco.mj_forward(m, d)

    print("\n[5/5] renderelés")
    OUT.mkdir(parents=True, exist_ok=True)
    r = None
    for cam in ("product", "insp_45", "insp_60", "insp_top"):
        for res in ((640, 1280) if cam.startswith("insp") else (960,)):
            try:
                if r is None or r.height != res:
                    if r is not None:
                        r.close()
                    r = mujoco.Renderer(m, res, res)
                mask = product_mask(r, m, d, cam)
                r.update_scene(d, camera=cam)
                img = r.render()
                fn = f"{cam}_{res}px.png"
                Image.fromarray(img).save(OUT / fn)
                print(f"      → {fn}   (termék {mask.mean()*100:4.1f} % of kép · "
                      f"kiégett {clipped_fraction(img, mask)*100:4.2f} %)")
            except Exception as e:
                print(f"      {cam}@{res}: HIBA {type(e).__name__}: {e}")
                return 1

    gt = json.loads((D / "ground_truth.json").read_text())
    print("\n" + "─" * 60)
    print("GROUND TRUTH — ennek kell olvashatónak lennie:")
    print(f"    dátum: {gt['date']}  ({gt['type']})  →  {gt['decision']}")
    print(f"    a képen: {gt['date'][8:10]}.{gt['date'][5:7]}.{gt['date'][2:4]}")
    print("─" * 60)
    print(f"\nKépek: {OUT}")
    print("Küldd vissza az insp_* képeket.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
