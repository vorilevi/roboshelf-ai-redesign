"""
shelflife_make_textures.py — termékcímke-textúrák szavatossági dátummal

    python3 tools/shelflife_make_textures.py
    python3 tools/shelflife_make_textures.py --n 8 --seed 42

A 0. kapu mérése alapján (2026-08-02):
  A dátum betűmagassága a kamerán ~7 pixel alatt olvashatatlan.
  4 mm-es nyomtatott dátum · 8 cm-es termékoldal · 45° FOV:
     224 px kamera, 30 cm →  3.6 px  ❌
     448 px kamera, 30 cm →  7.2 px  ⚠️
     640 px kamera, 30 cm → 10.3 px  ✅
  Ezért a textúra felbontása bőven a szükséges fölött van, és a dátum
  fizikailag helyes méretben (DATE_CAP_MM) kerül rá.

EU-jelölés — a feladat üzleti magja:
  „use by"     = fogyaszthatósági idő — BIZTONSÁGI kategória, lejárat után
                 kötelező kivonni
  „best before"= minőségmegőrzési idő — MINŐSÉGI kategória, lejárat után
                 sok termék még legálisan árusítható, jelöléssel
  A robotnak nem elég a dátumot elolvasnia: a TÍPUST is fel kell ismernie,
  mert más a teendő. A textúrák ezért mindkét feliratot használják.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent
TEX_DIR = _REPO / "src/envs/assets/shelflife_textures"

# fizikai méretek (a scene-builderrel egyezzenek)
PRODUCT_W_MM = 80.0
PRODUCT_H_MM = 120.0
DATE_CAP_MM  = 4.0          # valós nyomtatott dátum betűmagassága
TEX_PX_W     = 1024         # textúra felbontás (bőven a kamera fölött)

PRODUCTS = [
    ("TEJFÖL",   "20% zsírtartalom",  (178, 32, 42)),
    ("JOGHURT",  "natúr, 150 g",      (40, 90, 170)),
    ("VAJ",      "82%, 250 g",        (225, 180, 40)),
    ("SAJT",     "trappista, 200 g",  (200, 120, 30)),
    ("TEJSZÍN",  "30%, 200 ml",       (60, 140, 90)),
]
DATE_KINDS = ["USE BY", "BEST BEFORE"]

_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size_px: int):
    for p in _FONTS:
        try:
            return ImageFont.truetype(p, max(6, int(size_px)))
        except Exception:
            continue
    return ImageFont.load_default()


def make_label(name: str, sub: str, colour, kind: str, datestr: str) -> Image.Image:
    w = TEX_PX_W
    h = int(w * PRODUCT_H_MM / PRODUCT_W_MM)
    px_per_mm = w / PRODUCT_W_MM

    img = Image.new("RGB", (w, h), (242, 240, 232))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(120, 120, 110), width=max(1, w // 220))
    d.rectangle([w * 0.08, h * 0.10, w * 0.92, h * 0.34], fill=colour)
    d.text((w * 0.12, h * 0.15), name, font=_font(9 * px_per_mm), fill=(255, 255, 255))
    d.text((w * 0.12, h * 0.42), sub, font=_font(3.5 * px_per_mm), fill=(60, 60, 60))

    # A DÁTUM — ez az, amit a robotnak el kell olvasnia
    f = _font(DATE_CAP_MM * px_per_mm)
    y = h * 0.72
    d.text((w * 0.10, y), kind, font=f, fill=(20, 20, 20))
    d.text((w * 0.10, y + DATE_CAP_MM * px_per_mm * 1.15), datestr, font=f, fill=(20, 20, 20))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="hány termékváltozat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--today", type=str, default=None,
                    help="referencia-dátum ÉÉÉÉ-HH-NN (alap: ma)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    today = date.fromisoformat(a.today) if a.today else date.today()
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(a.n):
        name, sub, colour = PRODUCTS[i % len(PRODUCTS)]
        kind = rng.choice(DATE_KINDS)
        # fele lejárt, fele érvényes — hogy a döntés ne legyen triviális
        offset = rng.randint(-40, -1) if rng.random() < 0.5 else rng.randint(1, 90)
        dt = today + timedelta(days=offset)
        datestr = dt.strftime("%d.%m.%Y")

        img = make_label(name, sub, colour, kind, datestr)
        fn = f"product_{i}.png"
        img.save(TEX_DIR / fn)

        expired = offset < 0
        if not expired:
            decision = "MARADHAT"
        elif kind == "USE BY":
            decision = "KIVONNI"          # biztonsági kategória
        else:
            decision = "JELÖLNI"          # minőségi kategória
        manifest.append({
            "file": fn, "product": name, "date_kind": kind,
            "date": dt.isoformat(), "date_printed": datestr,
            "days_from_today": offset, "expired": expired,
            "ground_truth_decision": decision,
        })
        print(f"  {fn:<14} {name:<9} {kind:<12} {datestr}  "
              f"({offset:+d} nap) → {decision}")

    (TEX_DIR / "manifest.json").write_text(
        json.dumps({"reference_date": today.isoformat(), "items": manifest},
                   ensure_ascii=False, indent=2))
    print(f"\n{a.n} textúra + manifest.json → {TEX_DIR.relative_to(_REPO)}")
    print("A manifest a ground truth az eval-hoz (döntési pontosság méréséhez).")


if __name__ == "__main__":
    main()
