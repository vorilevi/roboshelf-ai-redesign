"""
shelflife_sku_audit.py — mi az, amit az SKU-bejegyzésben NEM MÉRTÜNK MEG

    python3 tools/shelflife_sku_audit.py
    python3 tools/shelflife_sku_audit.py --strict     # kilépési kód 1, ha van hiány

────────────────────────────────────────────────────────────────────────────
MIÉRT KÓD ÉS NEM JEGYZET
────────────────────────────────────────────────────────────────────────────
A hiányzó mérések listája eddig jegyzetben élt. A jegyzet viszont nem állít
meg egy futást: 2026-08-05-ig a szimuláció vígan dolgozott egy beírt
súrlódással, amit ráadásul a kéz alapértelmezése felül is írt — és semmi nem
szólt.

Ez az eszköz **a szimuláció része**, nem dokumentáció. Kiírja, mely mezők
`default` vagy `estimated` forrásúak, és `--strict` módban meg is bukik.
Így egy eredmény nem hivatkozhat mérésre, ha alatta feltételezés van.

────────────────────────────────────────────────────────────────────────────
A SZABÁLY
────────────────────────────────────────────────────────────────────────────
    measured / manufacturer / derived   →  az eredmény mérésként idézhető
    estimated                           →  az eredmény FELTÉTELES
    default                             →  az eredmény NEM idézhető mérésként

A `LOAD_BEARING` halmaz azokat a mezőket sorolja, amelyeken a JELENLEGI
kísérlet közvetlenül múlik. Ha ezek közül bármelyik `default`, a fogási
eredményekre nem szabad számként hivatkozni.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SKU_DIR = _REPO / "src/envs/assets/shelflife_sku_private"

# Amin a fogási kísérlet KÖZVETLENÜL múlik.
LOAD_BEARING = {
    "physics.mass_kg": "a szükséges szorítóerő ebből jön",
    "physics.com_m": "a kifordulási nyomaték ebből jön",
    "contact.friction_static": "F_min = m·g/(n·μ) — enélkül nem számolható",
    "contact.mujoco_friction": "a szimulált csúszás ezt használja",
    "packaging.grip_force_min_N": "a fogás alsó küszöbe",
    "packaging.grip_force_max_N": "a fogás felső küszöbe (deformáció)",
}

OK_SOURCES = {"measured", "manufacturer", "derived"}

# ⚠️ „NEM TUDJUK" ≠ „NEM KORLÁTOZ".
#
# A 2. SKU-nál (bontatlan szénsavas fémdoboz) a `grip_force_max_N` azért üres,
# mert a doboz nagyságrendekkel többet bír, mint amekkora erőt a kéz kifejt —
# nem azért, mert nem mértük meg. A súlypont ugyanígy: egy zárt, homogén,
# forgásszimmetrikus hengeré a geometriai közép, ez következmény, nem becslés.
#
# Ha ezt nem különböztetjük meg, az audit minden futásnál riaszt olyanra, ami
# rendben van — és pont a hitelét veszti el. Ezért a bejegyzés kimondhatja egy
# mezőről, hogy NEM TEHERHORDÓ ennél a terméknél, DE csak indoklással
# (`note`), és a riportban akkor is látszik.
NOT_BINDING_KEY = "not_binding"


def walk(node, prefix=""):
    """Minden `{value, source, confidence}` mező bejárása."""
    if isinstance(node, dict):
        if "source" in node and "confidence" in node:
            yield prefix, node
            return
        for k, v in node.items():
            if k.startswith("_"):
                continue
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{prefix}[{i}]")


def audit(sku_id: str, verbose: bool = True) -> dict:
    p = SKU_DIR / f"{sku_id}.json"
    if not p.exists():
        raise SystemExit(f"nincs ilyen SKU-bejegyzés: {sku_id}")
    d = json.loads(p.read_text(encoding="utf-8"))

    rows = list(walk(d))
    by_src: dict[str, list[str]] = {}
    for path, node in rows:
        by_src.setdefault(node.get("source", "?"), []).append(path)

    missing = []
    for path, why in LOAD_BEARING.items():
        node = next((n for pth, n in rows if pth == path), None)
        if node is None:
            missing.append((path, "nincs ilyen mező", why))
        elif node.get(NOT_BINDING_KEY) is True:
            pass                       # kimondottan nem korlátoz — l. `note`
        elif node.get("value") is None:
            missing.append((path, f"üres ({node.get('source')})", why))
        elif node.get("source") not in OK_SOURCES:
            missing.append((path, node.get("source", "?"), why))

    if verbose:
        print(f"SKU-audit — {sku_id}\n")
        print(f"  {len(rows)} forrásjelölt mező")
        for src in ("measured", "manufacturer", "derived", "estimated", "default"):
            n = len(by_src.get(src, []))
            if n:
                mark = "  " if src in OK_SOURCES else "⚠️"
                print(f"    {mark} {src:<14} {n:3d}")
        unmarked = [k for k in d
                    if not k.startswith("_")
                    and not any(pth.startswith(k) for pth, _ in rows)]
        if unmarked:
            print(f"\n  forrásjelölés NÉLKÜLI blokkok: {', '.join(unmarked)}")

        print(f"\n  A KÍSÉRLET SZEMPONTJÁBÓL TEHERHORDÓ MEZŐK")
        for path, why in LOAD_BEARING.items():
            node = next((n for pth, n in rows if pth == path), None)
            if node is None:
                print(f"    ❌ {path:<32} nincs ilyen mező")
                continue
            src = node.get("source", "?")
            nb = node.get(NOT_BINDING_KEY) is True
            ok = nb or (node.get("value") is not None and src in OK_SOURCES)
            val = node.get("value")
            sval = ("NEM KORLÁTOZ" if nb else
                    ("üres" if val is None else str(val)))
            print(f"    {'✅' if ok else '❌'} {path:<32} {sval:<22} [{src}]")

        if missing:
            print(f"\n  ⚠️  {len(missing)} TEHERHORDÓ MEZŐ NINCS MEGMÉRVE:\n")
            for path, state, why in missing:
                print(f"      · {path}  ({state})")
                print(f"        {why}")
            print("\n  KÖVETKEZMÉNY: a fogási eredmények erre az SKU-ra")
            print("  SZIMULÁCIÓ-BELSŐ értékek. Nem hivatkozhatók a valódi")
            print("  termék fizikai viselkedéseként, és nem publikálhatók")
            print("  mérésként. Ez nem formaság — a szimulált súrlódás")
            print("  jelenleg a MuJoCo alapértelmezése (1.0), nem a kartoné.")
        else:
            print("\n  ✅ minden teherhordó mező mért vagy gyártói adat")

    return {"fields": len(rows), "by_source": {k: len(v) for k, v in by_src.items()},
            "missing_load_bearing": [m[0] for m in missing]}


def one_line(sku_id: str) -> str:
    """Egysoros figyelmeztetés futásidejű használatra."""
    try:
        r = audit(sku_id, verbose=False)
    except SystemExit:
        return ""
    n = len(r["missing_load_bearing"])
    if not n:
        return ""
    return (f"⚠️  {sku_id}: {n} teherhordó fizikai mező nincs megmérve "
            f"(shelflife_sku_audit.py) — az eredmények szimuláció-belsők")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="alpro_barista_coconut_1l")
    ap.add_argument("--strict", action="store_true",
                    help="kilépési kód 1, ha teherhordó mező hiányzik")
    a = ap.parse_args()
    r = audit(a.sku)
    return 1 if (a.strict and r["missing_load_bearing"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
