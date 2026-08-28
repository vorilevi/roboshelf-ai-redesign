"""
shelflife_date_render.py — szavatossági dátum renderelése MÁTRIXNYOMTATÁS stílusban

    python3 tools/shelflife_date_render.py --demo
    python3 tools/shelflife_date_render.py --date 2027-03-22 --batch "HA2225 12:14"

MIÉRT KÜLÖN MODUL — a dátum NEM a termék tulajdonsága
─────────────────────────────────────────────────────────────────────────────
A szavatossági dátum gyártási tételenként változik: minden doboz más. Ezért
akkor sem szabadna beleégetni a szkennelt SKU-textúrába, ha a fotogrammetria
feloldotta volna (nem oldotta fel — az Alpro-szkennen a tető sima fehér).

A helyes felbontás:
  SKU-eszköz (állandó) : geometria + alaptextúra + A DÁTUMMEZŐ HELYE
  egyed (változó)      : maga a dátum, a mezőbe komponálva

Ez egyben megoldja az eval randomizálását is: tetszőleges lejárt/érvényes
dátum generálható ugyanabba a mezőbe.

A STÍLUS a valódi nyomtatásból jön
─────────────────────────────────────────────────────────────────────────────
Referencia: `data/doboz/IMG_7347.jpeg` — Alpro Barista Coconut tetőgerinc,
tintasugaras mátrixnyomtatás. Jellemzők, amiket utánzunk:
  - jól látható, kerek különálló pontok (nem folytonos vonal)
  - enyhe dőlés és pontonkénti szabálytalanság (a fej mozgásából)
  - sötét kékesfekete, tompa szélek
  - három sor: dátum · tételkód + idő · egy számjegy

Vektorszöveget renderelni HIBA lenne: a VLM-nek éles, tökéletes betűket adna,
miközben a valóságban halvány, szaggatott pontsorokat lát. A kiértékelés
ettől lenne hamisan optimista.
"""

from __future__ import annotations

import argparse
from datetime import date as _date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent

# A valódi nyomtatás mért/becsült paraméterei (IMG_7347 alapján)
DOT_PITCH_MM   = 0.52     # pontok közepének távolsága
DOT_RADIUS_MM  = 0.23     # egy pont sugara — a valódin a pontok KÜLÖNÁLLÓAK,
                          # nem érnek össze (pitch 0.52 mellett 0.26 összeolvadna)
CHAR_H_MM      = 3.6      # betűmagasság (a 0. kapu ~4 mm-es feltételezésével egyezik)
LINE_GAP_MM    = 1.5
SLANT          = 0.10     # enyhe dőlés
JITTER_MM      = 0.035    # pontonkénti szabálytalanság
INK            = (28, 26, 38)     # sötét kékesfekete

_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]


def _font(px: int):
    for p in _FONTS:
        try:
            return ImageFont.truetype(p, max(6, int(px)))
        except Exception:
            continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# 5×7-es MÁTRIXFONT
#
# Miért nem TrueType-ot raszterezünk: a valódi nyomtatófej 5×7-es rácson
# dolgozik, EGY pont széles vonalakkal. Egy TrueType betű raszterezve vagy
# 2-3 pont széles vonalakat ad (a betűk vastag foltokká olvadnak), vagy ha
# kicsire vesszük, a glifa szétesik. Mértük mindkettőt — egyik sem hasonlított.
# Ez a tábla adja a valódi szerkezetet.
# ─────────────────────────────────────────────────────────────────────────────
_M57 = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110",
    "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 10001 11001 10101 10011 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001",
    "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100",
    "Z": "11111 00001 00010 00100 01000 10000 11111",
    ".": "00000 00000 00000 00000 00000 01100 01100",
    ":": "00000 01100 01100 00000 01100 01100 00000",
    "-": "00000 00000 00000 11111 00000 00000 00000",
    "/": "00001 00010 00010 00100 01000 01000 10000",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}
GLYPH_W, GLYPH_H, CHAR_GAP = 5, 7, 1     # rácspont-egységben


def _glyph_dots(ch: str):
    """A karakter világító pontjai (oszlop, sor) párokként."""
    pat = _M57.get(ch.upper())
    if pat is None:
        pat = _M57[" "]
    rows = pat.split()
    return [(c, r) for r, row in enumerate(rows)
            for c, v in enumerate(row) if v == "1"]


def render_date_block(lines, px_per_mm: float = 40.0,
                      bg=(247, 245, 240), rng: np.random.Generator | None = None
                      ) -> Image.Image:
    """Szöveg → mátrixnyomtatás-szerű kép, fizikailag méretezve.

    A trükk: a szöveget először normálisan kirajzoljuk egy maszkba, majd a
    maszkot PONTRÁCSON mintavételezzük, és ahol fedett, oda kerek pontot
    rajzolunk. Így bármilyen karakter működik, nem kell 5×7-es fonttábla.
    """
    rng = rng or np.random.default_rng(0)
    # A pontosztás a betűmagasságból adódik: 7 sor tölti ki a karaktert.
    pitch = (CHAR_H_MM / (GLYPH_H - 1)) * px_per_mm
    rad = DOT_RADIUS_MM * px_per_mm
    line_h = (GLYPH_H - 1) * pitch + LINE_GAP_MM * px_per_mm

    ncols = max(len(s) for s in lines) * (GLYPH_W + CHAR_GAP)
    W = int(ncols * pitch + 6 * pitch)
    H = int(len(lines) * line_h + 4 * pitch)

    out = Image.new("RGB", (W, H), bg)
    od = ImageDraw.Draw(out)
    for li, s in enumerate(lines):
        y0 = 2 * pitch + li * line_h
        for ci, chx in enumerate(s):
            x0 = 2 * pitch + ci * (GLYPH_W + CHAR_GAP) * pitch
            for cx, cy in _glyph_dots(chx):
                x = x0 + cx * pitch
                y = y0 + cy * pitch
                x += SLANT * ((GLYPH_H - 1) * pitch - cy * pitch)   # dőlés
                jx, jy = rng.normal(0, JITTER_MM * px_per_mm, 2)
                r = rad * rng.uniform(0.85, 1.15)   # a pontméret ingadozik
                od.ellipse([x + jx - r, y + jy - r, x + jx + r, y + jy + r],
                           fill=INK)
    return out


def date_lines(when: _date, batch: str = "HA2225 12:14",
               fmt: str = "%d.%m.%y", tail: str = "1"):
    """A valódi Alpro-elrendezés: dátum / tételkód+idő / egy számjegy."""
    return [when.strftime(fmt), batch, tail]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2027-03-22", help="ÉÉÉÉ-HH-NN")
    ap.add_argument("--batch", default="HA2225 12:14")
    ap.add_argument("--out", default=None)
    ap.add_argument("--px-per-mm", type=float, default=40.0)
    ap.add_argument("--demo", action="store_true",
                    help="összehasonlító lap több dátummal")
    a = ap.parse_args()

    outdir = _REPO / "results/shelflife_date_render"
    outdir.mkdir(parents=True, exist_ok=True)

    if a.demo:
        rows = []
        for i, (d, note) in enumerate([
                (_date(2027, 3, 22), "eredeti (nem járt le)"),
                (_date(2026, 6, 15), "lejárt"),
                (_date(2026, 12, 1), "érvényes"),
        ]):
            img = render_date_block(date_lines(d), a.px_per_mm,
                                    rng=np.random.default_rng(i))
            rows.append((img, note))
        W = max(i.width for i, _ in rows) + 20
        H = sum(i.height for i, _ in rows) + 20 * len(rows)
        sheet = Image.new("RGB", (W, H), (247, 245, 240))
        y = 10
        for img, _ in rows:
            sheet.paste(img, (10, y)); y += img.height + 20
        p = outdir / "datum_demo.png"
        sheet.save(p)
        print(f"demó: {p}  ({sheet.size[0]}×{sheet.size[1]})")
        for _, n in rows:
            print(f"  · {n}")
        return

    d = _date.fromisoformat(a.date)
    img = render_date_block(date_lines(d, a.batch), a.px_per_mm)
    p = Path(a.out) if a.out else outdir / f"datum_{a.date}.png"
    img.save(p)
    print(f"{p}  ({img.size[0]}×{img.size[1]} px, "
          f"{img.size[0]/a.px_per_mm:.1f}×{img.size[1]/a.px_per_mm:.1f} mm)")


if __name__ == "__main__":
    main()
