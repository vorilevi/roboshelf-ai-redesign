"""
shelflife_build_scene_sku.py — a polcjelenet VALÓDI, szkennelt termékkel

    python3 tools/shelflife_build_scene_sku.py
    python3 tools/shelflife_build_scene_sku.py --sku alpro_barista_coconut_1l

Kimenet: src/envs/assets/shelflife_scene_gene01_sku_v1.xml

────────────────────────────────────────────────────────────────────────────
MIÉRT KÜLÖN FÁJL, ÉS MIÉRT NEM A shelflife_build_scene.py BŐVÍTÉSE
────────────────────────────────────────────────────────────────────────────
A `shelflife_build_scene.py` a szintetikus demó dobozzal dolgozik, és az a
PUBLIKUS harness része (l. kísérleti terv 10b): enélkül más nem tudja
reprodukálni a jelenetet. Ha átírnánk a valódi SKU-ra, a publikus builder
függene a privát adatbázistól.

Ezért ez a fájl a meglévő buildert **változatlanul importálja**, és csak a
termék-testet cseréli ki. A publikus út érintetlen marad.

────────────────────────────────────────────────────────────────────────────
MIÉRT A VALÓDI TERMÉKKEL KELL A FOGÁST MÉRNI, NEM A DEMÓVAL
────────────────────────────────────────────────────────────────────────────
A két test nem hasonlít egymásra:

    szintetikus demó doboz :  8 × 6 × 12 cm,  0.35 kg
    Alpro Barista Coconut  :  7.9 × 8.0 × 20.4 cm,  1.03 kg

Háromszor nehezebb és majdnem kétszer magasabb. Egy magas, nehéz kartonnál a
fogási erő ELÉGTELENSÉGE nem csúszásban, hanem MEGDŐLÉSBEN jelentkezik — a
tömegközéppont magasan van a megfogási pont fölött. A demó dobozon mért
fogási siker semmit nem mondana a valódiról.

Ugyanaz a lecke, mint a GR1T1-nél: a geometriát a MÉRT valósághoz kell
igazítani, nem fordítva.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_build_scene as B          # noqa: E402  — a publikus builder

OUT_XML = _REPO / "src/envs/assets/shelflife_scene_gene01_sku_v1.xml"
SKU_ROOT = _REPO / "models/shelflife_sku"

# A termék elejét ide tesszük (a polc eleje 0.34). 2 cm-rel beljebb: elöl van,
# könnyen elérhető, de nem lóg le a polcról.
PRODUCT_FRONT_X = B.SHELF_FRONT_X + 0.02
CLEARANCE_M = 0.002        # ennyivel a polclap fölé, hogy ne induljon átfedésben


def load_fragment(sku: str) -> tuple[list[ET.Element], ET.Element]:
    """A product.xml töredék beolvasása: (asset-elemek, body-elem)."""
    p = SKU_ROOT / sku / "product.xml"
    if not p.exists():
        sys.exit(f"nincs SKU-eszköz: {p}\n"
                 f"Futtasd előbb: python3 tools/shelflife_sku_import.py --sku {sku}")
    root = ET.fromstring(f"<wrap>{p.read_text()}</wrap>")
    assets = list(root.find("asset"))
    body = root.find("body")
    return assets, body


def remap_paths(assets: list[ET.Element], sku: str) -> None:
    """A töredék útvonalai a scene compiler-könyvtáraihoz igazítva.

    A MuJoCóban a `meshdir` és a `texturedir` GLOBÁLIS — nem lehet
    eszközönként mást megadni. A scene meshdir-je a GENE.01 hálóira mutat,
    a texturedir a demó címkékre. Az SKU fájljai máshol vannak, ezért a
    hivatkozásokat ezekhez képest relatívvá tesszük.

        xml könyvtára   : src/envs/assets/
        meshdir         : ../../../models/gene01_meshes/   -> repo/models/gene01_meshes
        texturedir      : shelflife_textures/              -> src/envs/assets/shelflife_textures

    Innen az SKU könyvtára (repo/models/shelflife_sku/<sku>/):
        mesh    : ../shelflife_sku/<sku>/...
        texture : ../../../../models/shelflife_sku/<sku>/...
    """
    for el in assets:
        f = el.get("file")
        if not f:
            continue
        name = Path(f).name
        if el.tag == "mesh":
            el.set("file", f"../shelflife_sku/{sku}/{name}")
        elif el.tag == "texture":
            el.set("file", f"../../../../models/shelflife_sku/{sku}/{name}")


def rename_to_product_0(body: ET.Element, sku: str) -> None:
    """A testet `product_0`-ra nevezzük át.

    MIÉRT: a primitív-réteg és a szenzorok a `product_` prefixű testeket
    keresik (`shelflife_primitives.py`, `s_product` / `s_product_quat`).
    Ha az SKU-azonosítót használnánk testnévnek, minden hívót át kellene
    írni — és a scene csereszabatossága is elveszne: az eval ugyanazt a
    kódot futtatná bármelyik SKU-val.

    Az eszköznevek (mesh/textúra/anyag) MARADNAK SKU-prefixesek, hogy egy
    jelenetben több különböző termék is lehessen.
    """
    body.set("name", "product_0")
    for el in body.iter():
        n = el.get("name")
        if n and n.startswith(sku):
            el.set("name", "product_0" + n[len(sku):])


def geom_extent(body: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """Az ütközési doboz középpontja és félméretei a test frame-jében."""
    # ── TÖBBFÉLE ÜTKÖZŐ-ALAK ────────────────────────────────────────────────
    #
    # Az első változat CSAK `type="box"`-ot ismert, mert az 1. SKU (szkennelt
    # karton) doboz-proxyt kapott. A 2. SKU viszont HENGER — és nem proxyként,
    # hanem mert a valódi test is az. A MuJoCo natívan ütközteti, tehát az
    # alak egzakt.
    #
    # A jelenetépítőnek csak a befoglaló FÉLMÉRETEK kellenek (a polcra
    # állításhoz és a nyúlási pont számításához), azt pedig mindkét alakból
    # meg lehet adni:
    #     box       size = (hx, hy, hz)         → félméret ugyanez
    #     cylinder  size = (r, halfheight)      → félméret (r, r, halfheight)
    #
    # A doboz ága SZÓ SZERINT változatlan, hogy az 1. SKU jelenete bitre
    # ugyanaz maradjon.
    for g in body.iter("geom"):
        if not g.get("name", "").endswith("_col"):
            continue
        typ = g.get("type")
        s = [float(x) for x in g.get("size").split()]
        cen = np.array([float(x) for x in g.get("pos", "0 0 0").split()])
        if typ == "box":
            return cen, np.array(s)
        if typ == "cylinder":
            return cen, np.array([s[0], s[0], s[1]])
        if typ == "capsule":
            # a félgömb-sapkák a hosszra rájönnek
            return cen, np.array([s[0], s[0], s[1] + s[0]])
        sys.exit(f"ismeretlen ütköző-alak a '_col' geomon: {typ!r} "
                 f"(box | cylinder | capsule kezelt)")
    sys.exit("az SKU-töredékben nincs '_col' végű ütköző geom")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="alpro_barista_coconut_1l")
    a = ap.parse_args()

    print("Shelf Life — polcjelenet VALÓDI szkennelt termékkel\n")

    # ── 1. a robot és a bolt: a publikus builder változatlanul ──────────────
    share = B.find_gb_share()
    print(f"  gb_robot_models: {share}")
    B.ensure_meshes(share)
    work = Path(tempfile.mkdtemp(prefix="shelflife_sku_build_"))
    raw = B.urdf_to_mjcf(share, work)
    off = B.measure_floor_offset(B.MESH_DIR / "_shelflife_build.urdf")
    print(f"  padló-offset (mérve): {off:.4f} m")
    tree = B.build(raw, off)
    root = tree.getroot()
    root.set("model", f"shelflife_gene01_sku_{a.sku}")

    # ── 1b. A TÖRZS AKTIVÁLÁSA ──────────────────────────────────────────────
    #
    # MÉRVE: a fogási pózok VÉGIG az ízülethatárokon ülnek — a legjobb
    # tartalék 0.14 rad, és emiatt a zárt hurok nem tudja korrigálni a
    # gravitációs megereszkedést (33 mm-en beragad). A termék a jobb kar
    # munkaterének SZÉLÉN van.
    #
    # A robotnak viszont van 2-DoF törzse, amit a publikus builder passzívra
    # állít. Ember is elfordul, ha oldalt nyúl a polcra — ez nem trükk,
    # hanem meglévő, kihasználatlan képesség. A `torso_yaw` bevonása a
    # vezérlési láncba kinyitja a munkateret a kar határainak feszülése nélkül.
    # A törzs paraméterei MÉRVE (`tools/shelflife_torso_tune.py`, M0), nem
    # tippelve. Az első próbálkozás TORSO_KP=2000 volt — 50-szeresen alul:
    #
    #     kp        yaw-hiba 2000 lépésnél   tenyér-hiba   nyomaték
    #      2 000        0.387 rad              301 mm         —
    #     10 000        0.166 rad              252 mm         —
    #     30 000        0.023 rad                7.5 mm     257 Nm
    #  → 100 000        0.002 rad                1.6 mm     219 Nm
    #    300 000        0.002 rad                1.5 mm     460 Nm
    #
    # 100 000 a nyertes: teljesíti a kilépési feltételt (<0.010 rad, <5 mm),
    # és KEVESEBB nyomatékkal, mint a 300 000-es változat. A 219 Nm reális
    # nagyságrend egy humanoid derékízületére.
    #
    # A törzs NEM örökölheti a `gene_arm` osztályt: annak csillapítása (10) és
    # armatúrája (0.01) karra való, a törzs viszont a teljes felsőtestet hordja.
    TORSO = ["torso_yaw", "torso_roll"]
    TORSO_KP, TORSO_DAMP, TORSO_ARMATURE = 100000.0, 100.0, 0.5

    dflt = root.find("default")
    c = ET.SubElement(dflt, "default", {"class": "gene_torso"})
    ET.SubElement(c, "joint", {"damping": str(TORSO_DAMP),
                               "armature": str(TORSO_ARMATURE)})

    wb0 = root.find("worldbody")
    torso_found = []
    for j in wb0.iter("joint"):
        if (j.get("name") or "") in TORSO:
            j.set("class", "gene_torso")
            for attr in ("damping", "armature", "stiffness"):
                j.attrib.pop(attr, None)
            torso_found.append(j.get("name"))
    act0 = root.find("actuator")
    for n in torso_found:
        ET.SubElement(act0, "position", {"name": f"act_{n}", "joint": n,
                                         "kp": str(TORSO_KP), "inheritrange": "1"})
    print(f"  törzs aktiválva: {torso_found} (kp={TORSO_KP:.0f}, "
          f"csillapítás={TORSO_DAMP:.0f}, armatúra={TORSO_ARMATURE})")

    # ── 2. a szintetikus demó termék eltávolítása ───────────────────────────
    wb = root.find("worldbody")
    for b in list(wb.findall("body")):
        if b.get("name", "").startswith("product_"):
            wb.remove(b)
    asset = root.find("asset")
    for el in list(asset):
        if (el.get("name") or "").endswith("_product_0") or \
           (el.get("name") or "") in ("tex_product_0", "mat_product_0"):
            asset.remove(el)
    print("  szintetikus demó termék eltávolítva")

    # ── 3. a szkennelt SKU beillesztése ─────────────────────────────────────
    assets, body = load_fragment(a.sku)
    remap_paths(assets, a.sku)
    rename_to_product_0(body, a.sku)
    for el in assets:
        asset.append(el)

    cen, half = geom_extent(body)
    # A test z-jét úgy állítjuk be, hogy az ütközési doboz ALJA a polclapon
    # üljön. A szkennelt háló origója nem a talpán van, ezért ezt számolni
    # kell, nem tippelni.
    board_top = B.SHELF_BOARD_Z[B.WORK_BOARD] + 0.012
    pz = board_top - (cen[2] - half[2]) + CLEARANCE_M
    px = PRODUCT_FRONT_X - (cen[0] - half[0])
    py = B.PRODUCT_Y - cen[1]
    body.set("pos", f"{px:.4f} {py:.4f} {pz:.4f}")
    wb.append(body)

    print(f"  SKU: {a.sku}")
    print(f"    ütközési doboz : {np.round(half*200, 1)} cm (Sz×Mé×Ma)")
    print(f"    test-pozíció   : x={px:.3f} y={py:.3f} z={pz:.3f}")
    print(f"    a termék eleje : x={PRODUCT_FRONT_X:.3f} "
          f"(polc eleje {B.SHELF_FRONT_X})")

    # ── 4. kiírás és ellenőrző betöltés ─────────────────────────────────────
    ET.indent(tree, space="  ")
    OUT_XML.write_text(ET.tostring(root, encoding="unicode"))
    print(f"\n  scene: {OUT_XML.relative_to(_REPO)}")

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    print(f"\n✅ betöltve: nq={m.nq} nv={m.nv} nu={m.nu} "
          f"nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam}")
    print(f"   termék tömege: {m.body_mass[bid]:.3f} kg")

    # leülepedés — a termék nem eshet le és nem süllyedhet bele a polcba
    p0 = d.xpos[bid].copy()
    for _ in range(1500):
        mujoco.mj_step(m, d)
    p1 = d.xpos[bid].copy()
    drift = float(np.linalg.norm(p1 - p0))
    print(f"   1500 lépés után elmozdulás: {drift*1000:.2f} mm "
          f"({'✅ stabil' if drift < 0.005 else '⚠️ csúszik/süllyed'})")
    print(f"   végső pozíció: {np.round(p1, 4)}")

    (OUT_XML.parent / "shelflife_scene_gene01_sku_v1.meta.json").write_text(
        json.dumps({"sku_id": a.sku, "body": "product_0",
                    "mass_kg": float(m.body_mass[bid]),
                    "collision_half_m": half.tolist(),
                    "settled_pos": p1.tolist(),
                    "settle_drift_mm": drift * 1000}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
