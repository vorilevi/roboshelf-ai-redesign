"""
shelflife_date_decal.py — a dátum RÁKERÜL a doboz talpára

    python3 tools/shelflife_date_decal.py
    python3 tools/shelflife_date_decal.py --date 2026-06-15 --char-mm 5.0

────────────────────────────────────────────────────────────────────────────
MIÉRT EZ HIÁNYZOTT
────────────────────────────────────────────────────────────────────────────
A `shelflife_date_render` modul régóta megvan, és mátrixnyomtatás-szerű,
hiteles dátumképet állít elő. De SOHA NEM KERÜLT RÁ a jelenetre: a doboz
talpán egy sima fehér korong van, felirat nélkül.

Emiatt a „megmutatható-e a dátum" kérdést eddig csak GEOMETRIAILAG tudtuk
vizsgálni — látószög, távolság, takarás. Azt, hogy a kép ELOLVASHATÓ-e,
nem. Pedig a projekt tétje pontosan ez.

────────────────────────────────────────────────────────────────────────────
MI EZ A MODUL
────────────────────────────────────────────────────────────────────────────
Előállít egy textúrát a dátummal, és megmondja, mekkora FIZIKAI foltot kell
neki a talpon. A jelenet ebből egy vékony lapot tesz a doboz aljára.

⚠️ NEM A TELJES TALP. A valódi dobozon a nyomtatás egy kis folt a fenéken,
   nem borítja be az egészet. Ha az egész talpat lefednénk a dátumképpel,
   a karakterek arányosan óriásiak lennének, és a teszt hamisan optimista.

────────────────────────────────────────────────────────────────────────────
⚠️ AZ EREDETI ELRENDEZÉS NEM FÉR RÁ A DOBOZ TALPÁRA
────────────────────────────────────────────────────────────────────────────
Az első futás megbukott a saját ellenőrzésén, és jól tette:

    3 sor, „HA2225 12:14" tételkód, 5 mm-es karakter → 65,0 mm széles
    a colásdoboz talpa                                → 46,5 mm átmérő

A referencia (`data/doboz/IMG_7347.jpeg`) egy **Alpro italoskarton
tetőgerince**, ami jóval szélesebb, mint egy konzervdoboz feneke. Az
elrendezést onnan örököltük, a méretet pedig innen — a kettő együtt
fizikailag lehetetlen.

Ebből adódik a korlát: 46,5 mm-es talpon, 5 mm-es karakterrel **legfeljebb
8 karakter fér el egy sorban**. Ezért lett az alapértelmezés rövidebb
tételkód, és ezért van benne a szélességi kapu. Egy valódi dobozon ezt
le kell fényképezni és megmérni — addig `decided`, nem `measured`.

────────────────────────────────────────────────────────────────────────────
⚠️ ELLENTMONDÁS A KARAKTERMAGASSÁGBAN — FELOLDATLAN
────────────────────────────────────────────────────────────────────────────
    shelflife_date_render.CHAR_H_MM = 3.6 mm   („a 0. kapu ~4 mm-es
                                                 feltételezésével egyezik")
    a projektben rögzített érték     = 5,0 mm   (measured)

Ez 1,4 mm különbség, és az olvashatóságban közel 40%. A 65°-os látószögű,
800 képpont magas D455-tel, 7 képpontos küszöbbel:

    3,6 mm karakter  →  363 mm-ig olvasható
    5,0 mm karakter  →  504 mm-ig olvasható

A jelenetben a fejkamera 314 mm-re van a doboztól, tehát MINDKETTŐ belefér —
de a tartalék az egyik esetben 15%, a másikban 60%. Ezért lehet a
`--char-mm` kapcsolóval állítani, és ezért íratja ki a modul, melyikkel
dolgozott. A kérdést egy valódi dobozon MEG KELL MÉRNI.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_date_render as dr                # noqa: E402

OUT_DIR = _REPO / "src/envs/assets/shelflife_textures"
PX_PER_MM = 40.0
BASE_D_MM = 46.5                 # a colásdoboz talpának átmérője (mérve)
MAX_CHARS = 8                    # ennyi fér el 5 mm-es karakterrel a talpon

# a talp fém alapszíne — csupasz alumínium, nem fehér papír
ALU = (198, 199, 203)


def make_decal(when: _date, char_mm: float, batch: str = "HA2225",
               px_per_mm: float = PX_PER_MM):
    """Visszaadja a képet és a hozzá tartozó FIZIKAI méretet (mm).

    ⚠️ A `date_render.CHAR_H_MM` MODULSZINTŰ állandó, a `render_date_block`
    nem vesz át betűmagasságot. Ideiglenesen átírjuk, majd VISSZAÁLLÍTJUK —
    különben a `shelflife_sku_import` is ezt a értéket kapná meg, csendben.
    """
    lines = [when.strftime("%d.%m.%y"), batch]
    old = dr.CHAR_H_MM
    dr.CHAR_H_MM = char_mm
    try:
        img = dr.render_date_block(lines, px_per_mm=px_per_mm, bg=ALU)
    finally:
        dr.CHAR_H_MM = old
    return img, (img.width / px_per_mm, img.height / px_per_mm), lines


def readable_mm(char_mm: float, px: int = 800, fov_deg: float = 65.0,
                need_px: int = 7) -> float:
    """Meddig olvasható — a KAMERA VALÓDI adataival.

    D455 színérzékelő (Intel D400 adatlap, 337029-017, Table 3-19):
    OmniVision OV9782 · 1280×800 aktív képpont · látószög 90°×65° ·
    globális zár · FIX FÓKUSZ · f/2.0 · 1,93 mm gyújtótávolság.
    """
    return char_mm / (2 * np.tan(np.radians(need_px / 2 * fov_deg / px)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2027-03-22")
    ap.add_argument("--char-mm", type=float, default=5.0,
                    help="karaktermagasság; 3.6 a date_render alapértéke")
    ap.add_argument("--batch", default="HA2225")
    a = ap.parse_args()

    when = _date.fromisoformat(a.date)
    img, (w_mm, h_mm), lines = make_decal(when, a.char_mm, a.batch)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "date_0.png"
    img.save(p)
    fits = w_mm <= BASE_D_MM

    print("Shelf Life — DÁTUMMATRICA A DOBOZ TALPÁRA\n")
    print(f"  dátum              {when.isoformat()}")
    print(f"  sorok              {' | '.join(lines)}  "
          f"({max(len(s) for s in lines)} karakter)")
    print(f"  karaktermagasság   {a.char_mm:.1f} mm   [decided — l. a fejlécet]")
    print(f"  a folt mérete      {w_mm:.1f} × {h_mm:.1f} mm")
    print(f"  a talp átmérője    {BASE_D_MM:.1f} mm  "
          f"{'✅ elfér' if fits else '❌ SZÉLESEBB, MINT A TALP'}")
    if not fits:
        print(f"     → {BASE_D_MM/w_mm*a.char_mm:.1f} mm-es karakterrel férne "
              f"el, vagy {int(BASE_D_MM/(w_mm/max(len(s) for s in lines)))} "
              f"karakterrel soronként")
    print(f"  textúra            {img.width}×{img.height} px "
          f"({PX_PER_MM:.0f} px/mm)")
    print(f"  mentve             {p.relative_to(_REPO)}")

    print("\n  OLVASHATÓ TÁVOLSÁG (D455 szín: 800 px @ 65°, 7 képpont):")
    for c in sorted({3.6, a.char_mm, 5.0}):
        print(f"    {c:.1f} mm karakter → {readable_mm(c):.0f} mm")
    print("\n  ⚠️ A jelenetben a fejkamera 314 mm-re van a doboztól —")
    print("     de a D455 FIX FÓKUSZÚ, és ez alatt már élességi kérdés is.")
    return 0 if fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
