"""
shelflife_build_scene.py — Shelf Life scene-építő (Generative Bionics GENE.01)

Előállítja a `src/envs/assets/shelflife_scene_gene01_v1.xml` MuJoCo scene-t a
pip-telepített `gb-robot-models` URDF-ből, plusz a bolti környezetet
(polc, termékdobozok szavatossági dátummal, kamerák).

    pip install gb-robot-models
    python3 tools/shelflife_build_scene.py

Miért builder és nem kézzel írt XML?
  A GENE.01 modell a gyártótól jön, csomagkezelőn keresztül, és frissülhet.
  A meshek 23 MB-osak → nem kerülnek a gitbe (mint a mujoco_menagerie), hanem
  a builder másolja be őket a telepített csomagból.

────────────────────────────────────────────────────────────────────────────
A GR1T1-BŐL TANULT LECKÉK, AMIKET EZ A FÁJL AZ ELEJÉN KEZEL
────────────────────────────────────────────────────────────────────────────
1. PASSZÍV ÍZÜLETEK RUGÓERŐ NÉLKÜL ELROSKADNAK.
   A GR1T1-nél a `passive_joint` osztálynak volt csillapítása, de rugóereje
   nem — a csillapítás lassítja a roskadást, megállítani nem tudja. Az egész
   alsótest folyamatosan süllyedt (csípő 0.39 rad, boka 0.31 rad), a kar
   sosem került egyensúlyba, és emiatt a scripted expert egy TRANZIENST
   kapott el, nem egy állapotot. Ez tette a feladatot a VLA-nak
   tanulhatatlanná (20% eval SR).
   → Itt MINDEN nem működtetett ízület kap `stiffness`-t. Lásd PASSIVE_*.

2. A GEOMETRIÁT A MÉRT ELÉRÉSHEZ KELL KALIBRÁLNI, NEM FORDÍTVA.
   A GR1T1-nél az asztal x=0.45-re került, mert a MEGERESZKEDETT kar odáig
   ért. Merev testtel a valódi elérés 0.39 volt → a robot hozzá sem ért a
   termékhez.
   → Itt a polc a mért elérési burok ~68%-ára kerül, bőven margóval.
   Mért burok (ujjbegy, padlótól, 625 minta): z∈[0.95,1.15) m magasságban
   max előre-nyúlás x=0.658 m. SHELF_FRONT_X=0.40 → 61% kihasználtság.

3. A KAMERAFELBONTÁS NEM HANGOLÁSI KÉRDÉS.
   A 0. kapu mérése: 4 mm-es nyomtatott dátum 30 cm-ről 224 px-es kamerán
   3.6 pixel → olvashatatlan. 640 px-en 10.3 pixel → tökéletes.
   Gyakorlati küszöb ~7 px betűmagasság.
   → CAM_RES alapértéke 640, és NE menj 448 alá.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent

OUT_XML   = _REPO / "src/envs/assets/shelflife_scene_gene01_v1.xml"
# A meshek (23 MB, 219 fájl) a `models/` alá kerülnek, mert az a .gitignore-ban
# MÁR benne van — így nem kell meglévő fájlhoz nyúlni, és a gyártó geometriái
# nem hizlalják a publikus repót. Bármikor újraelőállíthatók:
#     pip install gb-robot-models && python3 tools/shelflife_build_scene.py
MESH_DIR  = _REPO / "models/gene01_meshes"
TEX_DIR   = _REPO / "src/envs/assets/shelflife_textures"

# ── Szimuláció ──────────────────────────────────────────────────────────────
TIMESTEP = 0.001

# ── Vezérlési paraméterek (a stabilitás-ellenőrzés ezeket méri, ne tippelj) ──
ARM_KP,    ARM_DAMP    = 3000.0, 10.0   # 7-DoF jobb kar (kp sweep: 300/1000/3000/
                                        # 10000/30000 → 3000 a legjobb kompromisszum
                                        # ízületi hiba és qacc-zaj között)
FINGER_KP, FINGER_DAMP =  20.0,   0.5   # 21-DoF jobb kéz
NECK_KP,   NECK_DAMP   = 100.0,   5.0   # 3-DoF nyak

# Nem működtetett ízületek: láb, törzs, bal kar, bal kéz.
# stiffness NÉLKÜL ezek elroskadnak — lásd az 1. leckét fent.
PASSIVE_STIFF, PASSIVE_DAMP = 100000.0, 500.0

# ── Bolti geometria (méterben) ──────────────────────────────────────────────
SHELF_FRONT_X   = 0.34    # a polc eleje — MÉRVE, nem tippelve (lásd 2. lecke)
SHELF_DEPTH     = 0.35
SHELF_WIDTH     = 0.90
SHELF_BOARD_Z   = [0.75, 1.05, 1.35]   # három polcszint
WORK_BOARD      = 1                     # ezen van a termék (z=1.05)

PRODUCT_W, PRODUCT_D, PRODUCT_H = 0.080, 0.060, 0.120   # 8×6×12 cm
PRODUCT_MASS = 0.35
PRODUCT_X    = SHELF_FRONT_X + 0.06     # kicsit beljebb a polc elején
# A termék y-eltolása a JOBB kar oldalára. MÉRVE (tenyér-burok a polc
# magasságában, 715 minta): y=0-nál csak 49 elérhető pont, max x=0.371 —
# testközépen átnyúlni szűk. y∈[-0.25,-0.15)-nél 85 pont, max x=0.488.
# A komfortzóna mediánja y=-0.33; -0.20 jó kompromisszum (bőven a polcon van).
PRODUCT_Y    = -0.20

# ── Kamera ──────────────────────────────────────────────────────────────────
CAM_RES = 640             # NE menj 448 alá (0. kapu mérés)
CAM_FOV = 45
# Az offscreen framebuffer a LEGNAGYOBB kérhető felbontás legyen, különben
# `Image width N > framebuffer width M` hibát kapunk renderelésnél. A
# felbontás-sorozat 1280-ig megy.
OFFSCREEN_MAX = 1280

# A gyártó URDF-jében definiált kamera-pózok (fix linkek, MuJoCo nem hozza át)
URDF_CAMERAS = {
    "rgb_camera":         ("chest", (0.154451, -0.0115,  -0.0174506)),
    "chest_camera_front": ("chest", (0.184418,  0.0,      0.230569)),
    "head_camera_left":   ("head",  (0.117516,  0.03,     0.24454)),
    "head_camera_right":  ("head",  (0.115878, -0.03,     0.24454)),
}

ARM_JOINTS = ["r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
              "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"]
NECK_JOINTS = ["neck_pitch", "neck_roll", "neck_yaw"]
FINGER_TOKENS = ("thumb", "index", "middle", "ring", "little")


# ═══════════════════════════════════════════════════════════════════════════
# 1. A telepített gb-robot-models megkeresése
# ═══════════════════════════════════════════════════════════════════════════

def find_gb_share() -> Path:
    import importlib.metadata as md
    try:
        dist = md.distribution("gb-robot-models")
    except md.PackageNotFoundError:
        sys.exit("HIÁNYZIK: pip install gb-robot-models")
    for f in dist.files:
        s = str(f)
        if "share/gb_robot_models/robots" in s and s.endswith("model.urdf"):
            return Path(dist.locate_file(f)).resolve().parent.parent.parent
    sys.exit("A gb_robot_models share könyvtára nem található.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. URDF → nyers MJCF
# ═══════════════════════════════════════════════════════════════════════════

def ensure_meshes(share: Path) -> int:
    """A meshek bemásolása a repóba, HIÁNYALAPÚ ellenőrzéssel.

    Korábban csak azt néztük, hogy a célkönyvtár LÉTEZIK-e — ha félkészen
    maradt (megszakadt másolás, kézzel törölt fájlok), a fordítás
    `Error opening file '...stl'` hibával szállt el, ami nem árulja el az okot.
    """
    src_dir = share / "meshes"
    src = sorted(p for p in src_dir.iterdir() if p.is_file())
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in MESH_DIR.iterdir() if p.is_file()}
    missing = [p for p in src if p.name not in have]
    if missing:
        print(f"  meshek másolása: {len(missing)} hiányzó → "
              f"{MESH_DIR.relative_to(_REPO)}")
        for p in missing:
            shutil.copy2(p, MESH_DIR / p.name)
    n = len([p for p in MESH_DIR.iterdir() if p.is_file()])
    if n < len(src):
        sys.exit(f"HIBA: {n}/{len(src)} mesh másolódott át ({MESH_DIR}). "
                 f"Ellenőrizd a lemezhelyet és a jogosultságokat.")
    print(f"  meshek: {n} db rendben")
    return n


def urdf_to_mjcf(share: Path, work: Path) -> Path:
    """URDF → MJCF.

    A MuJoCo URDF-betöltője alapértelmezésben LEVÁGJA az útvonalat a
    mesh-nevekről (`compiler/strippath` URDF-nél true), és csak a fájlnevet
    keresi a modellfájl mellett. Emiatt sem a relatív út + szimlink, sem az
    abszolút út nem működött megbízhatóan — mindkettő ugyanazt a
    `Error opening file 'sim_gene01_..._collision.stl'` hibát adta.

    A robusztus megoldás: az ideiglenes URDF-et A MESHEK MELLÉ írjuk, csupasz
    fájlnevekkel. Így mindegy, hogy a betöltő levágja-e az utat vagy sem.
    """
    import mujoco

    work.mkdir(parents=True, exist_ok=True)
    urdf_src = share / "robots/gene01_0/model.urdf"
    urdf_txt = urdf_src.read_text().replace(
        "package://gb_robot_models/meshes/", "")     # csupasz fájlnevek
    tmp_urdf = MESH_DIR / "_shelflife_build.urdf"    # a meshek MELLÉ
    tmp_urdf.write_text(urdf_txt)

    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
    except ValueError as e:
        sys.exit(f"HIBA az URDF fordításakor: {e}\n"
                 f"  mesh-könyvtár: {MESH_DIR}\n"
                 f"  létezik: {MESH_DIR.exists()}  "
                 f"fájlok: {len(list(MESH_DIR.glob('*'))) if MESH_DIR.exists() else 0}")
    out = work / "gene01_raw.mjcf"
    mujoco.mj_saveLastXML(str(out), model)
    return out


def measure_floor_offset(urdf_path: Path) -> float:
    """Mennyivel kell megemelni a robotot, hogy a talpa a padlón legyen.
    Mérve, nem tippelve — a modell gyökere a medencénél van."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(urdf_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lowest = float(d.geom_xpos[:, 2].min())
    return -lowest


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scene összeállítása
# ═══════════════════════════════════════════════════════════════════════════

def E(parent, tag, **kw):
    el = ET.SubElement(parent, tag)
    for k, v in kw.items():
        el.set(k.replace("_", "-") if k.startswith("rgba") is False and "_" in k and k in () else k, str(v))
    return el


def build(raw_mjcf: Path, floor_offset: float) -> ET.ElementTree:
    tree = ET.parse(raw_mjcf)
    root = tree.getroot()
    root.set("model", "shelflife_gene01_v1")

    # ── compiler / option ───────────────────────────────────────────────────
    comp = root.find("compiler")
    comp.set("angle", "radian")
    comp.set("meshdir", "../../../models/gene01_meshes/")
    # A nyers MJCF ABSZOLÚT mesh-utakat tartalmaz (az URDF-fordításból), ami
    # gépfüggő lenne. A scene-be csak a fájlnév kerül, az utat a meshdir adja —
    # így a kimenet hordozható.
    for mesh in root.find("asset").iter("mesh"):
        f = mesh.get("file", "")
        if f:
            mesh.set("file", Path(f).name)
    comp.set("texturedir", "shelflife_textures/")
    comp.set("autolimits", "true")

    opt = ET.Element("option")
    opt.set("timestep", str(TIMESTEP))
    opt.set("gravity", "0 0 -9.81")
    opt.set("integrator", "implicitfast")
    root.insert(list(root).index(comp) + 1, opt)

    vis = ET.Element("visual")
    ET.SubElement(vis, "global", {"offwidth": str(OFFSCREEN_MAX),
                                  "offheight": str(OFFSCREEN_MAX)})
    ET.SubElement(vis, "quality", {"shadowsize": "2048"})
    root.insert(list(root).index(opt) + 1, vis)

    # ── default osztályok ───────────────────────────────────────────────────
    dflt = ET.Element("default")
    for name, kw in [
        ("gene_arm",     {"damping": ARM_DAMP,    "armature": 0.01}),
        ("gene_finger",  {"damping": FINGER_DAMP, "armature": 0.001}),
        ("gene_neck",    {"damping": NECK_DAMP,   "armature": 0.005}),
        # ⚠️ stiffness KÖTELEZŐ — enélkül az alsótest elroskad (GR1T1-lecke)
        ("gene_passive", {"damping": PASSIVE_DAMP, "armature": 0.5,
                          "stiffness": PASSIVE_STIFF}),
    ]:
        c = ET.SubElement(dflt, "default", {"class": name})
        ET.SubElement(c, "joint", {k: str(v) for k, v in kw.items()})
    prod = ET.SubElement(dflt, "default", {"class": "product"})
    ET.SubElement(prod, "geom", {"friction": "0.9 0.02 0.002", "condim": "4",
                                 "solref": "0.02 1", "solimp": "0.9 0.95 0.001"})
    root.insert(list(root).index(vis) + 1, dflt)

    # ── ízület-osztályok kiosztása ──────────────────────────────────────────
    # FONTOS: csak a worldbody joint-jait, mert a root.iter() a <default>
    # blokkokban lévő joint-sablonokat is megtalálná, azokra viszont a
    # 'class' attribútum sémasértés.
    actuated = []
    for j in root.find("worldbody").iter("joint"):
        n = j.get("name") or ""
        if n in ARM_JOINTS:
            cls, act = "gene_arm", (n, "gene_arm", ARM_KP)
        elif n in NECK_JOINTS:
            cls, act = "gene_neck", (n, "gene_neck", NECK_KP)
        elif n.startswith("r_") and any(t in n for t in FINGER_TOKENS):
            cls, act = "gene_finger", (n, "gene_finger", FINGER_KP)
        else:
            cls, act = "gene_passive", None      # láb, törzs, bal kar+kéz
        j.set("class", cls)
        for attr in ("damping", "armature", "stiffness", "actuatorfrcrange"):
            j.attrib.pop(attr, None)
        if act:
            actuated.append(act)

    # ── textúrák + anyagok ──────────────────────────────────────────────────
    asset = root.find("asset")
    ET.SubElement(asset, "texture", {
        "name": "tex_floor", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512",
        "rgb1": "0.82 0.82 0.78", "rgb2": "0.78 0.78 0.74"})
    ET.SubElement(asset, "material", {"name": "mat_floor", "texture": "tex_floor",
                                      "texrepeat": "8 8", "reflectance": "0.05"})
    ET.SubElement(asset, "material", {"name": "mat_shelf", "rgba": "0.72 0.72 0.70 1"})
    ET.SubElement(asset, "material", {"name": "mat_board", "rgba": "0.88 0.86 0.82 1"})
    # a termék címkéje — a shelflife_make_textures.py állítja elő
    ET.SubElement(asset, "texture", {"name": "tex_product_0", "type": "2d",
                                     "file": "product_0.png"})
    ET.SubElement(asset, "material", {"name": "mat_product_0",
                                      "texture": "tex_product_0",
                                      "texuniform": "false", "specular": "0.1"})

    # ── worldbody: robot megemelése + környezet ─────────────────────────────
    wb = root.find("worldbody")
    robot_children = [c for c in list(wb)]
    for c in robot_children:
        wb.remove(c)

    ET.SubElement(wb, "light", {"name": "ceil", "pos": "0.3 0 3.0", "dir": "0 0 -1",
                                "diffuse": "0.9 0.9 0.9", "castshadow": "true"})
    ET.SubElement(wb, "light", {"name": "fill", "pos": "0.8 -0.8 2.2", "dir": "-0.3 0.3 -1",
                                "diffuse": "0.5 0.5 0.5", "castshadow": "false"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane",
                               "size": "4 4 0.01", "material": "mat_floor"})

    # Áttekintő kamerák — NEM a roboton vannak, csak hogy emberi szemmel
    # ellenőrizhető legyen a jelenet (a robot saját kamerái szűk látószögűek).
    ET.SubElement(wb, "camera", {
        "name": "overview", "pos": "1.2 -1.5 1.7",
        "xyaxes": "0.78 0.63 0  -0.28 0.35 0.89", "fovy": "50"})
    ET.SubElement(wb, "camera", {
        "name": "side_view", "pos": "0.3 -1.8 1.25",
        "xyaxes": "1 0 0  0 0 1", "fovy": "45"})

    # a robot egy emelő-bodyba kerül, hogy a talpa a padlón legyen
    base = ET.SubElement(wb, "body", {"name": "gene01_base",
                                      "pos": f"0 0 {floor_offset:.4f}"})
    for c in robot_children:
        base.append(c)

    # kamerák a gyártó által definiált pózokba
    body_by_name = {b.get("name"): b for b in root.iter("body")}
    for cam, (parent, xyz) in URDF_CAMERAS.items():
        b = body_by_name.get(parent)
        if b is None:
            continue
        ET.SubElement(b, "camera", {
            "name": cam, "pos": " ".join(f"{v:.5f}" for v in xyz),
            "fovy": str(CAM_FOV), "mode": "fixed",
            # +x-be néz (a robot előre), z felfelé
            "xyaxes": "0 -1 0  0 0 1"})

    # ── polc ────────────────────────────────────────────────────────────────
    sx = SHELF_FRONT_X + SHELF_DEPTH / 2
    shelf = ET.SubElement(wb, "body", {"name": "shelf", "pos": f"{sx} 0 0"})
    for i, z in enumerate(SHELF_BOARD_Z):
        ET.SubElement(shelf, "geom", {
            "name": f"shelf_board_{i}", "type": "box",
            "size": f"{SHELF_DEPTH/2} {SHELF_WIDTH/2} 0.012",
            "pos": f"0 0 {z}", "material": "mat_board"})
    ET.SubElement(shelf, "geom", {"name": "shelf_back", "type": "box",
                                  "size": f"0.01 {SHELF_WIDTH/2} 0.80",
                                  "pos": f"{SHELF_DEPTH/2} 0 0.80",
                                  "material": "mat_shelf"})
    for s in (-1, 1):
        ET.SubElement(shelf, "geom", {"name": f"shelf_side_{s}", "type": "box",
                                      "size": f"{SHELF_DEPTH/2} 0.01 0.80",
                                      "pos": f"0 {s*SHELF_WIDTH/2:.3f} 0.80",
                                      "material": "mat_shelf"})

    # ── termék ──────────────────────────────────────────────────────────────
    pz = SHELF_BOARD_Z[WORK_BOARD] + 0.012 + PRODUCT_H / 2
    p = ET.SubElement(wb, "body", {"name": "product_0",
                                   "pos": f"{PRODUCT_X} {PRODUCT_Y} {pz:.4f}"})
    ET.SubElement(p, "freejoint", {"name": "product_0_free"})
    ET.SubElement(p, "geom", {"name": "product_0_geom", "type": "box",
                              "size": f"{PRODUCT_D/2} {PRODUCT_W/2} {PRODUCT_H/2}",
                              "material": "mat_product_0", "mass": str(PRODUCT_MASS),
                              "class": "product"})
    ET.SubElement(p, "site", {"name": "product_0_site", "size": "0.005",
                              "rgba": "1 0 0 0.5"})

    # ── aktuátorok ──────────────────────────────────────────────────────────
    act = ET.SubElement(root, "actuator")
    for name, cls, kp in actuated:
        ET.SubElement(act, "position", {
            "name": f"act_{name}", "joint": name,
            "kp": str(kp), "inheritrange": "1"})

    # ── szenzorok ───────────────────────────────────────────────────────────
    sen = ET.SubElement(root, "sensor")
    ET.SubElement(sen, "framepos", {"name": "s_product", "objtype": "body",
                                    "objname": "product_0"})
    ET.SubElement(sen, "framepos", {"name": "s_palm", "objtype": "body",
                                    "objname": "r_wrist_3"})
    ET.SubElement(sen, "framequat", {"name": "s_product_quat", "objtype": "body",
                                     "objname": "product_0"})
    return tree


def main():
    print("Shelf Life — scene építése\n")
    share = find_gb_share()
    print(f"  gb_robot_models: {share}")
    ensure_meshes(share)
    work = Path(tempfile.mkdtemp(prefix="shelflife_build_"))
    raw = urdf_to_mjcf(share, work)
    off = measure_floor_offset(MESH_DIR / "_shelflife_build.urdf")
    print(f"  padló-offset (mérve): {off:.4f} m")

    tree = build(raw, off)
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUT_XML, encoding="unicode", xml_declaration=False)
    print(f"  scene: {OUT_XML.relative_to(_REPO)}")

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT_XML))
    print(f"\n✅ betöltve: nq={m.nq} nv={m.nv} nu={m.nu} "
          f"nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam}")
    print(f"   polc eleje x={SHELF_FRONT_X} · termék x={PRODUCT_X:.2f} "
          f"z={SHELF_BOARD_Z[WORK_BOARD]+0.012+PRODUCT_H/2:.3f}")
    print(f"   kamera: {CAM_RES}px, fovy={CAM_FOV}°")


if __name__ == "__main__":
    main()
