"""
shelflife_sku_import.py — szkennelt SKU → MuJoCo eszközök

    python3 tools/shelflife_sku_import.py --sku alpro_barista_coconut_1l
    python3 tools/shelflife_sku_import.py --sku ... --date 2026-06-15

Bemenet : az SKU-bejegyzés (privát) + a szkennelt OBJ/JPG
Kimenet : models/shelflife_sku/<sku_id>/  — MuJoCo-kész eszközök
          + egy XML-töredék, amit a scene-builder beilleszt

────────────────────────────────────────────────────────────────────────────
HÁROM TERVEZÉSI DÖNTÉS, MINDEGYIK MÉRÉSBŐL
────────────────────────────────────────────────────────────────────────────
1. A DÁTUM KÜLÖN MATRICA, nem a textúrába festve.
   A szavatossági dátum tételenként változik — nem a SKU tulajdonsága. Így az
   eval minden epizódban más dátumot tehet ugyanoda, a 8K szkennelt textúrához
   hozzányúlás nélkül. (Amúgy sem lehetne: a szkennen a tető sima fehér, a
   fotogrammetria nem oldotta fel a nyomtatást.)

2. A MATRICA HÁLÓ, nem primitív.
   A MuJoCóban a `type="2d"` textúrák csak síkokra és magasságtérképekre
   működnek. Egy doboz-primitívre téve a geom SIMA SZÜRKE marad — ebbe már
   belefutottunk egyszer, öt üres renderelést kaptunk tőle. Ezért a matrica
   egy kétháromszöges OBJ, saját UV-kkel.

3. AZ ÜTKÖZÉS DOBOZ, nem a szkennelt háló.
   A MuJoCo csak konvex hálókkal ütköztet, a szkennelt felület pedig zajos és
   nem konvex. A karton amúgy is közel téglatest. Ugyanaz a minta, mint a
   GENE.01 saját modelljében: részletes .obj a látványhoz, egyszerű primitív
   a fizikához.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
SKU_DIR = _REPO / "src/envs/assets/shelflife_sku_private"
OUT_ROOT = _REPO / "models/shelflife_sku"        # models/ a .gitignore-ban van

TEX_MAX = 2048          # a 8K felesleges: a szkennelt textúra amúgy is elmosódott
DECAL_PX_PER_MM = 24    # a matrica textúra felbontása


def load_sku(sku_id: str) -> dict:
    p = SKU_DIR / f"{sku_id}.json"
    if not p.exists():
        raise SystemExit(f"nincs ilyen SKU-bejegyzés: {p}")
    return json.loads(p.read_text())


# ═══════════════════════════════════════════════════════════════════════════
# Vizuális háló + textúra
# ═══════════════════════════════════════════════════════════════════════════

def prepare_visual(sku: dict, out: Path) -> tuple[Path, Path]:
    src_obj = _REPO / sku["geometry"]["mesh"]
    src_tex = _REPO / sku["geometry"]["texture"]
    if not src_obj.exists():
        raise SystemExit(f"hiányzó háló: {src_obj}")

    # A forgatást (mesh y-fel → jelenet z-fel) MAGÁBA A HÁLÓBA sütjük.
    #
    # MIÉRT: ha a testnek freejoint-ja van, a qpos kvaternió FELÜLÍRJA a body
    # `quat` attribútumát — a statikus forgatás elveszik, amint a termék
    # szabadon mozoghat. Ebbe belefutottunk: a doboz fejjel lefelé állt, a
    # dátum -21 cm-en. A hálóba sütött forgatással a test frame-je eleve
    # helyes, és a freejoint természetesen működik.
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # Rx(+90°)
    lines_out = []
    for l in src_obj.read_text(errors="ignore").splitlines():
        p = l.split()
        if p and p[0] in ("v", "vn") and len(p) >= 4:
            x, y, z = (float(p[1]), float(p[2]), float(p[3]))
            nx, ny, nz = R @ np.array([x, y, z])
            lines_out.append(f"{p[0]} {nx:.6f} {ny:.6f} {nz:.6f}")
        elif p and p[0] in ("mtllib", "usemtl"):
            continue
        else:
            lines_out.append(l)
    dst_obj = out / "visual.obj"
    dst_obj.write_text("\n".join(lines_out))

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(src_tex).convert("RGB")
    if max(im.size) > TEX_MAX:
        im = im.resize((TEX_MAX, TEX_MAX), Image.LANCZOS)
    dst_tex = out / "visual.png"
    im.save(dst_tex)
    return dst_obj, dst_tex


# ═══════════════════════════════════════════════════════════════════════════
# Dátum-matrica
# ═══════════════════════════════════════════════════════════════════════════

def make_decal(sku: dict, when: _date, out: Path) -> tuple[Path, Path]:
    """Négyszög-háló + a rá nyomtatott dátum textúrája."""
    import sys
    sys.path.insert(0, str(_REPO / "tools"))
    from shelflife_date_render import render_date_block, date_lines

    df = sku["date_field"]
    pl = df["placement"]
    su, sv = pl["size_cm"]                   # u = a blokk "magassága", v = hossza

    lines = date_lines(when, sku["date_field"].get("observed_batch", "HA2225 12:14"),
                       fmt=df["format"])
    img = render_date_block(lines, px_per_mm=DECAL_PX_PER_MM)

    # A cube textúra NÉGYZETES képet követel ("PNG size must be integer
    # multiple of gridsize"). Ez nem gond: a négyzetes textúra a mező
    # arányára (4.7 x 1.8 cm) vetítve nyúlik meg, tehát ha a dátumblokkot
    # ELŐRE négyzetre nyújtjuk, a vetítés pontosan visszaállítja az alakját.
    S = 1024
    canvas = Image.new("RGB", (S, S), (247, 245, 240))
    inner = img.resize((int(S * 0.94), int(S * 0.90)), Image.LANCZOS)
    canvas.paste(inner, (int(S * 0.03), int(S * 0.05)))
    tex = out / "date_decal.png"
    canvas.save(tex)

    # NINCS matrica-HÁLÓ. Ok: a MuJoCo a hálókat a tehetetlenségi
    # főtengelyeikhez igazítja, és ezzel FELÜLÍRJA a megadott geom-quat-ot
    # (mértük: [0.704,-0.071,-0.071,0.704] helyett [-0.447,0.447,0.548,0.548]
    # került a modellbe, a matrica oldalra nézett a tető helyett).
    # Helyette vékony DOBOZ-primitív + CUBE textúra: primitívet a MuJoCo nem
    # orientál át, a cube textúra pedig — a 2d-vel ellentétben — működik rajta.
    return None, tex


# ═══════════════════════════════════════════════════════════════════════════
# Elhelyezés: a modell saját frame-je → a jelenet (z fel)
# ═══════════════════════════════════════════════════════════════════════════

def _quat_from_mat(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4)
        q[i + 1] = np.sqrt(max(0.0, 1 + 2 * R[i, i] - np.trace(R))) / 2
        return [float(x) for x in q]
    return [float(w),
            float((R[2, 1] - R[1, 2]) / (4 * w)),
            float((R[0, 2] - R[2, 0]) / (4 * w)),
            float((R[1, 0] - R[0, 1]) / (4 * w))]


def _obj_bbox(path: Path):
    V = [[float(x) for x in l.split()[1:4]]
         for l in path.read_text(errors="ignore").splitlines() if l.startswith("v ")]
    V = np.array(V)
    return V.min(0), V.max(0)


def placement(sku: dict, obj_path: Path) -> dict:
    """A modell y-fel-felfelé áll; a jelenetben z a felfelé.

    KÉT HIBA, AMIT ITT ELKÖVETTÜNK ÉS JAVÍTOTTUNK:
     1. Rossz forgásirány: Rx(-90°) fejjel lefelé fordítja a dobozt
        (a dátum z=-21 cm-re került). Helyesen Rx(+90°): mesh +y → jelenet +z.
     2. Kétszeres forgatás: a matrica pozícióját ELŐRE elforgattuk, holott a
        geom a TEST saját frame-jében van megadva, amit a body quat úgyis
        elforgat. Ezért a matrica adatai NYERS mesh-koordinátákban maradnak.
    """
    pl = sku["date_field"]["placement"]
    # a bbox-ot a MÁR ELFORGATOTT hálóból olvassuk (z a felfelé)
    mn, mx = _obj_bbox(obj_path)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # Rx(+90°)

    # az SKU-bejegyzés adatai az EREDETI mesh-frame-ben vannak → forgatni kell
    n = R @ np.array(pl["plane_normal"], float)
    u = R @ np.array(pl["axis_u_back"], float)
    v = R @ np.array(pl["axis_v_left"], float)
    c = R @ np.array(pl["center_xyz_m"], float)

    # a matrica lokális tengelyei: x = v (a szöveg iránya), y = u, z = normális
    M = np.column_stack([v, u, n])
    if np.linalg.det(M) < 0:
        M[:, 1] *= -1
    return {
        "body_quat": [1.0, 0.0, 0.0, 0.0],      # a forgatás a hálóban van
        "decal_pos": (c + 0.0008 * n).tolist(),  # 0.8 mm-rel a felület fölé
                                                 # (1.5 mm-en ferde szögből
                                                 #  elvált a kartontól)
        "decal_quat": _quat_from_mat(M),
        "collision_center": ((mn + mx) / 2).tolist(),
        "collision_half": ((mx - mn) / 2).tolist(),
        "decal_half": [pl["size_cm"][1] / 200.0, pl["size_cm"][0] / 200.0],
        "height_m": float(mx[2] - mn[2]),        # most z a magasság
        "bottom_z": float(mn[2]),
    }


def field(sku: dict, path: str, fallback):
    """Egy `{value, source, confidence}` mező kiolvasása az SKU-bejegyzésből.

    ────────────────────────────────────────────────────────────────────────
    MIÉRT NEM ELÉG AZ ÉRTÉKET KIOLVASNI
    ────────────────────────────────────────────────────────────────────────
    2026-08-05-ig a tömeg (1.03 kg) és a súrlódás (0.9) ITT, a szkriptben volt
    beírva — az SKU-bejegyzésben nem is szerepelt. Két következménye lett:

      · a tömeg a GENERÁLT jelenetfájlban élt, tehát az „adatbázisunk" nem
        tartalmazta a legalapvetőbb fizikai jellemzőt;
      · a súrlódásról senki nem tudta, hogy tipp — és a fogás pont a
        csúszáson bukott.

    Ezért a jelenet mostantól az SKU-bejegyzésből generálódik, és minden
    beolvasott mezőnél KIÍRJUK, hogy mért adat-e vagy alapértelmezés.
    Ha `default` vagy `estimated`, az figyelmeztetéssel jár: a belőle
    származó eredmény nem publikálható mérésként.
    """
    node = sku
    for part in path.split("."):
        node = (node or {}).get(part) if isinstance(node, dict) else None
    if not isinstance(node, dict) or "value" not in node or node["value"] is None:
        print(f"    ⚠️  {path}: nincs érték az SKU-bejegyzésben → "
              f"tartalék {fallback!r}")
        return fallback
    src, conf = node.get("source", "?"), node.get("confidence", "?")
    mark = "⚠️ " if src in ("default", "estimated") else "   "
    print(f"    {mark}{path:<28} {node['value']!r:<22} "
          f"[{src} / {conf}]")
    return node["value"]


def emit_xml(sku_id: str, p: dict, rel: str, sku: dict) -> str:
    print("  fizikai paraméterek az SKU-bejegyzésből:")
    mass = float(field(sku, "physics.mass_kg", 1.03))
    fric = field(sku, "contact.mujoco_friction", [0.9, 0.02, 0.002])
    condim = int(field(sku, "contact.condim", 4))
    fric_s = " ".join(str(x) for x in fric)
    q = " ".join(f"{x:.6f}" for x in p["body_quat"])
    dq = " ".join(f"{x:.6f}" for x in p["decal_quat"])
    dp = " ".join(f"{x:.5f}" for x in p["decal_pos"])
    hx, hy, hz = p["collision_half"]
    cx, cy, cz = p["collision_center"]
    dsx, dsy = p["decal_half"]
    return f"""<!-- SKU: {sku_id} — szkennelt vizuális háló + doboz-ütközés + dátum-matrica -->
<asset>
  <mesh name="{sku_id}_visual" file="{rel}/visual.obj"/>
  <texture name="{sku_id}_tex"   type="2d" file="{rel}/visual.png"/>
  <texture name="{sku_id}_dtex"  type="cube" file="{rel}/date_decal.png"/>
  <!-- FONTOS a fényerő miatt: a MuJoCo material-alapértelmezése specular=0.5,
       shininess=0.5. Egy FOTOGRAMMETRIÁS textúrában viszont a csillanás MÁR
       BENNE VAN (a szkennelés bevilágítva készült), tehát a renderelő tükrös
       tagja RÁADÓDIK — ettől égett ki a kupak és a fehér zárófül fehér
       foltokká. A szkennelt textúra közel diffúzként kezelendő. -->
  <material name="{sku_id}_mat"  texture="{sku_id}_tex"
            specular="0.08" shininess="0.15" reflectance="0"/>
  <!-- a matrica matt nyomtatás papíron: nulla csillanás -->
  <material name="{sku_id}_dmat" texture="{sku_id}_dtex"
            specular="0.0" shininess="0.0" reflectance="0"/>
</asset>
<body name="{sku_id}" pos="0 0 0" quat="{q}">
  <freejoint name="{sku_id}_free"/>
  <!-- látvány: a szkennelt háló, ütközés nélkül -->
  <geom type="mesh" mesh="{sku_id}_visual" material="{sku_id}_mat"
        contype="0" conaffinity="0" group="2" mass="0"/>
  <!-- fizika: egyszerű téglatest (a MuJoCo csak konvex hálót ütköztet) -->
  <geom name="{sku_id}_col" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}"
        pos="{cx:.4f} {cy:.4f} {cz:.4f}" group="3" rgba="0 0 0 0"
        mass="{mass}" friction="{fric_s}" condim="{condim}"/>
  <!-- a dátum: külön matrica, epizódonként cserélhető -->
  <geom name="{sku_id}_date" type="box" size="{dsx:.4f} {dsy:.4f} 0.0004"
        material="{sku_id}_dmat" pos="{dp}" quat="{dq}"
        contype="0" conaffinity="0" group="2" mass="0"/>
</body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="alpro_barista_coconut_1l")
    ap.add_argument("--date", default=None, help="ÉÉÉÉ-HH-NN (alap: az eredeti)")
    a = ap.parse_args()

    sku = load_sku(a.sku)
    out = OUT_ROOT / a.sku
    out.mkdir(parents=True, exist_ok=True)

    when = (_date.fromisoformat(a.date) if a.date
            else _date.fromisoformat(sku["date_field"]["observed_parsed_iso"]))

    print(f"SKU: {sku['brand']} {sku['product_name']}")
    obj, tex = prepare_visual(sku, out)
    print(f"  vizuális háló : {obj.relative_to(_REPO)}")
    print(f"  textúra       : {tex.relative_to(_REPO)}  ({TEX_MAX}px)")
    _, d_tex = make_decal(sku, when, out)
    print(f"  dátum-matrica : {d_tex.name} (cube textúra)   dátum={when.isoformat()}")

    p = placement(sku, obj)
    rel = f"../../../models/shelflife_sku/{a.sku}"
    xml = emit_xml(a.sku, p, rel, sku)
    (out / "product.xml").write_text(xml)
    print(f"  XML-töredék   : {(out/'product.xml').relative_to(_REPO)}")
    print(f"\n  magasság {p['height_m']*100:.1f} cm · ütközési doboz "
          f"{[round(x*200,1) for x in p['collision_half']]} cm")
    print(f"  a talp z={p['bottom_z']:.4f}-nél van")

    gt = {"date": when.isoformat(), "type": sku["date_field"]["type"],
          "expired": when < _date.today(),
          "decision": sku["decision_rules"][
              "expired" if when < _date.today() else "not_expired"]}
    (out / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2))
    print(f"  ground truth  : {gt}")


if __name__ == "__main__":
    main()
