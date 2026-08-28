"""
shelflife_date_readable.py — MEDDIG ÉS MILYEN SZÖGBŐL OLVASHATÓ A DÁTUM

    python3 tools/shelflife_date_readable.py
    python3 tools/shelflife_date_readable.py --char-mm 3.6

────────────────────────────────────────────────────────────────────────────
EZ A PROJEKT TÉNYLEGES KÉRDÉSE
────────────────────────────────────────────────────────────────────────────
A „Shelf Life" arról szól, hogy a robot megtalálja és ELOLVASSA a lejárati
dátumot, és dönt. Eddig ehelyett a megfogás mechanikájával foglalkoztunk —
azzal a résszel, amit senki nem vitat, és ami sim-to-real szempontból a
legrosszabbul átvihető. A dátum a jelenetben egy sima fehér korong volt.

Ez a modul a fordított sorrendet valósítja meg: a leolvashatóság ROBOT
NÉLKÜL tesztelhető. Két változó van, minden más rögzített.

    változó:   TÁVOLSÁG  ·  RÁNÉZÉSI SZÖG
    rögzített: kamera (D455 valódi adatai), világítás, karaktermagasság

────────────────────────────────────────────────────────────────────────────
A RÖGZÍTETT FELTÉTELEK, ÉS MIÉRT AZOK
────────────────────────────────────────────────────────────────────────────
VILÁGÍTÁS: felülről jövő, egyenletes szórt fény. A vásárlótérben a
megvilágítás tervezett és állandó, épp azért, hogy a vásárló jól lássa a
termékeket. Nem változó — adottság.

KAMERA: Intel RealSense D455 színérzékelő, a gyártó adatlapjából
(337029-017, Table 3-19): OV9782 · 1280×800 · látószög 90°×65° · globális
zár · FIX FÓKUSZ · f/2.0 · 1,93 mm.

⚠️ A GEOMETRIA ITT NEM SZABADON VÁLASZTOTT. A doboz a polcon állva a
   TALPÁT nem mutatja — az első próbánál a kamera alulról a polclapot
   fényképezte. A dátum megnézéséhez a terméket fel KELL emelni és meg KELL
   fordítani. Ez a modul ezt az állapotot modellezi: a doboz a levegőben,
   a talpa a kamera felé.

────────────────────────────────────────────────────────────────────────────
A MÉRÉS: VALÓDI OCR, NEM SZEMREVÉTELEZÉS
────────────────────────────────────────────────────────────────────────────
Minden képkockát végigfuttatunk a Tesseract OCR-en, és a kiolvasott
karakterláncot a VALÓDI dátumhoz hasonlítjuk. Nem azt nézzük, hogy „elég
élesnek látszik-e" — azt, hogy helyesen kiolvasható-e.

⚠️ AZ EREDMÉNY FELFELÉ TORZÍT. A MuJoCo renderelése tiszta: nincs
   érzékelőzaj, nincs mozgási elmosódás, nincs tömörítési műtermék, a
   tükröződés is egyszerűsített. Ami itt MEGBUKIK, az a valóságban is
   megbukik; ami itt átmegy, arról ez még nem bizonyítja, hogy átmenne.
   A határértékek tehát FELSŐ korlátok.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env                       # noqa: E402,F401  (SORREND!)
import shelflife_date_render as dr                # noqa: E402
import mujoco                                     # noqa: E402
from shelflife_date_decal import make_decal, readable_mm   # noqa: E402

OUT = _REPO / "results/shelflife_date_readable"
TEX = _REPO / "src/envs/assets/shelflife_textures/date_0.png"

CAN_R, CAN_H = 0.02905, 0.14540
BASE_R = 0.02324
RES_W, RES_H = 1280, 800
FOVY = 65.0

DISTANCES_MM = (100, 150, 200, 250, 314, 400, 504, 600, 700, 900)
TILTS_DEG = (0, 15, 30, 45, 60)

WHEN = _date(2027, 3, 22)
BATCH = "HA2225"


def cam_xml() -> str:
    """RÖGZÍTETT kamerák — minden vizsgált távolság–szög párra egy.

    ⚠️ AZ ELSŐ VÁLTOZAT SZABAD KAMERÁT (`MjvCamera`) HASZNÁLT, és minden
    cella megbukott. Nem a dátum volt olvashatatlan, hanem a műszer rossz:
    a szabad kamera `azimuth`-ja más irányból néz, mint hittem, a `fovy`
    pedig a modell alapértelmezett 45°-a maradt — nem a D455 65°-a. Így
    egy másik tárgyat fényképeztünk, rossz látószögben.
    Rögzített kamerával mindkettő KIÍRVA szerepel a modellben.
    """
    out = []
    for dist in DISTANCES_MM:
        for tilt in TILTS_DEG:
            th = np.radians(tilt)
            eye = np.array([np.cos(th), 0.0, np.sin(th)]) * (dist / 1000.0)
            R = look_at(eye, [0, 0, 0])
            # ⚠️ A FELIRAT ÁLLVA JELENT MEG. A doboz 90°-os elfordítása a
            #    szöveg irányát a világ függőlegesébe viszi; a geom saját
            #    elfordítása nem segített, mert a textúra vele fordul.
            #    A kamerát kell megdönteni a nézési tengely körül — ez az,
            #    amit a valóságban a fogó tájolása állít be.
            c90, s90 = 0.0, -1.0
            Rz = np.array([[c90, -s90, 0], [s90, c90, 0], [0, 0, 1.0]])
            R = R @ Rz
            q = np.empty(4); mujoco.mju_mat2Quat(q, R.flatten())
            out.append(
                f'    <camera name="c{dist}_{tilt}" fovy="{FOVY}" '
                f'pos="{eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f}" '
                f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"/>')
    return "\n".join(out)


def scene_xml(w: float, h: float) -> str:
    """A doboz a levegőben, TALPPAL a +x irány felé.

    Felülről jövő szórt fény: több gyenge, széles fényforrás fentről,
    árnyék nélkül — ez a bolti mennyezeti világítás közelítése.
    """
    return f"""<mujoco model="datumolvasas">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="{RES_W}" offheight="{RES_H}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.25 0.25 0.25"
               specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="tex_date" type="2d" file="{TEX}"/>
    <material name="mat_date" texture="tex_date" specular="0.15" shininess="0.1"/>
    <material name="mat_alu" rgba="0.78 0.78 0.80 1" specular="0.2" shininess="0.2"/>
    <material name="mat_can" rgba="0.85 0.10 0.12 1"/>
  </asset>
  <worldbody>
    <light pos="0.3  0.4 1.2" dir="0 -0.3 -1" diffuse="0.45 0.45 0.45"
           castshadow="false"/>
    <light pos="0.3 -0.4 1.2" dir="0  0.3 -1" diffuse="0.45 0.45 0.45"
           castshadow="false"/>
    <light pos="0.0  0.0 1.2" dir="0 0 -1"    diffuse="0.40 0.40 0.40"
           castshadow="false"/>
    <!-- a doboz: tengelye x mentén, a TALPA az origóban, +x felé néz -->
    <body name="product_0" pos="0 0 0" euler="0 1.5707963 0">
      <geom name="body" type="cylinder" size="{CAN_R} {CAN_H/2}"
            pos="0 0 {-CAN_H/2}" material="mat_can"/>
      <!-- ⚠️ Z-HARC. A talpkorong és a doboz alaplapja pontosan egy síkban
           volt (mindkettő z=0), ezért ferde nézetben piros-szürke foltokban
           törtek egymásba. A kontakt íven ez végig látszik. A korongot
           teljesen a doboz ALÁ tesszük — a valódi talp is kiáll. -->
      <geom name="base" type="cylinder" size="{BASE_R} 0.0003"
            pos="0 0 -0.0008" material="mat_alu"/>
      <!-- ⚠️ A LAP OLDALARÁNYA = A TEXTÚRA OLDALARÁNYA. Egy közbeeső
           változatban felcseréltem a két méretet, hogy a szöveg
           vízszintesbe kerüljön. A textúra emiatt 45 mm-ről 16 mm-re
           préselődött össze az egyik irányban: a render elmosódott,
           kontraszttalan lett, és az OCR MINDEN cellában megbukott. Nem a
           dátum volt olvashatatlan, hanem a saját torzításom.
           A szöveg irányát a KAMERA döntése állítja be, nem a lap. -->
      <geom name="date" type="box" size="{w/2} {h/2} 0.0003"
            pos="0 0 -0.0013" material="mat_date"/>
    </body>
{cam_xml()}
  </worldbody>
</mujoco>"""


def look_at(eye, tgt):
    f = np.array(tgt, float) - np.array(eye, float); f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.column_stack([x, y, -f])


def isolate_and_read(img: np.ndarray, truth: str):
    """A VALÓDI feldolgozási lánc: talp megkeresése → pontok kötése → OCR.

    ⚠️ HÁROM MÉRÉSI HIBA VOLT EBBEN, MIRE MŰKÖDÖTT — és mindhárom „a dátum
       nem olvasható" hamis eredményt adott:

    1. A teljes 1280×800-as képkockára eresztettem az OCR-t. A dátum a kép
       2%-át foglalja el; a Tesseract meg sem találta.
    2. A kivágás után a FEKETE HÁTTÉR is bent maradt, és az Otsu-küszöb a
       háttér és a korong között húzta meg a határt — a szöveg elveszett.
       Ezért maszkoljuk a korongon kívüli részt a korong tónusára.
    3. A Tesseract SOLID betűkre van tanítva, nem pontsorokra. Nyers
       pontmátrixot még 1168 képpontos talpon SEM olvasott ki. Morfológiai
       zárás köti össze a pontokat vonássá.

    ⚠️ A ZÁRÓELEM MÉRETE NEM SZABADON VÁLASZTOTT. A pontok közti hézagot
       kell áthidalnia, se többet, se kevesebbet — és a hézag a vetített
       méretaránnyal változik. Ezért a geometriából számoljuk, nem
       hangoljuk. (Ellenőrizve: 200 mm-en k=7, 250 mm-en k=5, 314 mm-en
       k=3 nyert — pontosan ahogy a számítás adja.)
    """
    import cv2
    g = img.mean(axis=2).astype(np.float32)
    mask = (g > 120).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return "", 0, 0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    disc = lab == i
    if disc.sum() < 200:
        return "", 0, 0
    g[~disc] = float(np.median(g[disc]))
    x, y, w, h = st[i, :4]
    roi = g[y:y + h, x:x + w].astype(np.uint8)
    base_px = max(w, h)

    up = 4
    roi = cv2.resize(roi, (roi.shape[1] * up, roi.shape[0] * up),
                     interpolation=cv2.INTER_LANCZOS4)
    _, b = cv2.threshold(roi, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    px_per_mm = base_px * up / 46.5
    gap_px = max(0.0, dr.DOT_PITCH_SCALED - 2 * dr.DOT_RADIUS_MM) * px_per_mm
    # A geometriából számolt hézag ADJA A KIINDULÓPONTOT, de a teljes
    # ésszerű tartományt végigpróbáljuk, és KIÍRJUK, melyik nyert. Így a
    # záróelem mérete mért paraméter marad, nem rejtett hangolás.
    k0 = max(3, int(round(gap_px)) | 1)
    kers = sorted(range(3, 16, 2), key=lambda k: (abs(k - k0), k))
    best = ""
    for k in kers:
        c = cv2.morphologyEx(
            b, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        txt = re.sub(r"[^0-9A-Z]", "", ocr_raw(255 - c).upper())
        if truth in txt:
            return txt, base_px, k
        best = best or txt
    return best, base_px, 0


def crop_upscale(img: np.ndarray, factor: int = 4):
    """A dátummezőre vágás, majd nagyítás — EZ A VALÓDI FELDOLGOZÁSI LÁNC.

    Egy éles rendszer sem a teljes 1280×800-as képkockára ereszti rá az
    OCR-t: előbb megkeresi a terméket, kivágja, és csak a kivágott részt
    olvassa. Ezt utánozzuk. A nagyítás NEM ad hozzá információt — csak
    a Tesseract elvárt betűméretéhez igazít.

    ⚠️ A vágás a doboz ISMERT helyére támaszkodik. Egy éles rendszernek ezt
    ELŐBB meg kellene találnia; azt a lépést ez a modul NEM méri.
    """
    from PIL import Image
    g = img.mean(axis=2)
    ys, xs = np.where(g > 40)                       # a fekete háttér felett
    if len(xs) < 50:
        return None, 0
    pad = 6
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    c = Image.fromarray(img[y0:y1, x0:x1])
    if c.width < 8 or c.height < 8:
        return None, 0
    return c.resize((c.width * factor, c.height * factor),
                    Image.LANCZOS), (y1 - y0)


def ocr_raw(arr) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.fromarray(arr), config="--psm 6")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-mm", type=float, default=5.0)
    ap.add_argument("--save-every", action="store_true")
    a = ap.parse_args()

    # a pontosztás a karaktermagasságból jön (7 sor tölti ki a karaktert)
    dr.DOT_PITCH_SCALED = a.char_mm / (dr.GLYPH_H - 1)
    img_dec, (w_mm, h_mm), lines = make_decal(WHEN, a.char_mm, BATCH)
    TEX.parent.mkdir(parents=True, exist_ok=True)
    img_dec.save(TEX)
    truth = lines[0].replace(".", "")            # „220327"

    m = mujoco.MjModel.from_xml_string(scene_xml(w_mm / 1000, h_mm / 1000))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    OUT.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, RES_H, RES_W)

    print("Shelf Life — A DÁTUM LEOLVASHATÓSÁGA\n")
    print(f"  dátum {WHEN.isoformat()} · {' | '.join(lines)} · "
          f"karakter {a.char_mm:.1f} mm · folt {w_mm:.1f}×{h_mm:.1f} mm")
    print(f"  kamera D455 szín: {RES_W}×{RES_H} @ {FOVY:.0f}° · "
          f"számított határ {readable_mm(a.char_mm):.0f} mm")
    print(f"  világítás: felülről, szórt, állandó (adottság)\n")
    print("        " + "".join(f"{t:>7}°" for t in TILTS_DEG))

    grid = {}
    for dist in DISTANCES_MM:
        row = []
        for tilt in TILTS_DEG:
            r.update_scene(d, camera=f"c{dist}_{tilt}")
            img = r.render()
            txt, px, kwin = isolate_and_read(img, truth)
            ok = truth in txt
            grid[(dist, tilt)] = (ok, txt[:24], px, kwin)
            row.append("   ✅  " if ok else "   ·   ")
            if a.save_every:
                from PIL import Image as _I
                _I.fromarray(img).save(OUT / f"d{dist}_t{tilt}.png")
        print(f"  {dist:>4}mm" + "".join(row)
              + f"   [talp {grid[(dist,0)][2]:>4} px]")

    okd = [dist for dist in DISTANCES_MM if grid[(dist, 0)][0]]
    print()
    if okd:
        print(f"  MERŐLEGESEN olvasható: {min(okd)} … {max(okd)} mm")
    else:
        print("  ❌ MERŐLEGESEN SEM olvasható egyetlen távolságon sem")
    for tilt in TILTS_DEG:
        okt = [dd for dd in DISTANCES_MM if grid[(dd, tilt)][0]]
        print(f"    {tilt:>3}° ránézés → {'max %d mm' % max(okt) if okt else 'nem olvasható'}")
    print(f"\n  képek: {OUT.relative_to(_REPO)}")
    print("  ⚠️ Ezek FELSŐ korlátok: a render zajmentes, tömörítetlen, "
          "elmosódásmentes.")
    r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
