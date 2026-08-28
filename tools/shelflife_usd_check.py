"""
shelflife_usd_check.py — a generált .usda ELLENŐRZÉSE (a te gépeden fut)

    pip install usd-core
    python3 tools/shelflife_usd_check.py --sku coca_cola_zero_330_sleek

────────────────────────────────────────────────────────────────────────────
MIÉRT KÜLÖN FÁJL ÉS MIÉRT NEM FUTOTT LE NÁLAM
────────────────────────────────────────────────────────────────────────────
A fejlesztői sandboxban nincs `pxr` (OpenUSD), és a `usd-core` sem
telepíthető onnan. Az `.usda`-t ezért meg tudtam ÍRNI, de nem tudtam
PARSE-OLNI — a fájl fejlécében ezért áll, hogy ELLENŐRIZETLEN.

Ez a szkript zárja le a kört. Nem GUI: nem kell hozzá usdview, PySide vagy
OpenGL, csak a core könyvtár. Amit ellenőriz:

  1. megnyílik-e a stage egyáltalán (ez a fő kérdés — szintaktikai hiba)
  2. a `defaultPrim` létezik-e, és Xform-e
  3. a fizikai attribútumok olvashatók-e, és EGYEZNEK-e az SKU-bejegyzéssel
  4. a `variantSet` váltható-e, és minden variáns értelmes adatot ad-e
  5. a henger geometriája egyezik-e a bejegyzéssel

A 3. és 5. pont a fontos: nem elég, hogy a fájl SZINTAKTIKAILAG jó — azt is
tudni akarjuk, hogy ugyanazt a terméket írja le, mint a forrás. Ez ugyanaz az
elv, mint az MJCF-nél: a generált kimenetet a forráshoz mérjük, nem magához.

Kilépési kód 0, ha minden rendben; 1, ha bármi eltér.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SKU_DIR = _REPO / "src/envs/assets/shelflife_sku_private"
OUT_ROOT = _REPO / "models/shelflife_sku"

TOL = 1e-6


def val(node, default=None):
    if isinstance(node, dict) and "value" in node:
        return node["value"] if node["value"] is not None else default
    return node if node is not None else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="coca_cola_zero_330_sleek")
    a = ap.parse_args()

    try:
        from pxr import Usd, UsdGeom, UsdPhysics       # noqa: F401
    except ImportError:
        print("Nincs OpenUSD. Telepítés:\n\n    pip install usd-core\n")
        print("(Ez csak a core könyvtár — usdview NINCS benne, de az")
        print(" ellenőrzéshez nem is kell.)")
        return 2

    usda = OUT_ROOT / a.sku / "product.usda"
    rec = json.loads((SKU_DIR / f"{a.sku}.json").read_text(encoding="utf-8"))
    print(f"Ellenőrzés — {usda.relative_to(_REPO)}\n")

    bad: list[str] = []

    def check(name: str, got, want, unit: str = "") -> None:
        ok = (abs(float(got) - float(want)) < TOL
              if isinstance(want, (int, float)) else got == want)
        print(f"  {'✅' if ok else '❌'} {name:<26} {got}{unit}"
              + ("" if ok else f"   ← a bejegyzésben: {want}{unit}"))
        if not ok:
            bad.append(name)

    # ── 1. megnyílik-e ──────────────────────────────────────────────────
    stage = Usd.Stage.Open(str(usda))
    if stage is None:
        print("  ❌ A STAGE NEM NYÍLT MEG — szintaktikai hiba az .usda-ban.")
        return 1
    print("  ✅ a stage megnyílt")

    # ── 2. defaultPrim ──────────────────────────────────────────────────
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        print("  ❌ nincs érvényes defaultPrim")
        return 1
    print(f"  ✅ defaultPrim: <{prim.GetPath()}>  ({prim.GetTypeName()})")
    print(f"     apiSchemas: {list(prim.GetAppliedSchemas())}")

    # ── 3. fizika ───────────────────────────────────────────────────────
    print("\n  FIZIKA — a bejegyzéshez mérve")
    mass_attr = prim.GetAttribute("physics:mass")
    if not mass_attr:
        print("  ❌ nincs physics:mass")
        bad.append("physics:mass")
    else:
        check("physics:mass", mass_attr.Get(),
              val(rec["physics"]["mass_kg"]), " kg")

    body = stage.GetPrimAtPath(prim.GetPath().AppendChild("Body"))
    if not body or not body.IsValid():
        print("  ❌ nincs /Product/Body")
        return 1
    cyl = UsdGeom.Cylinder(body)
    check("henger sugár", cyl.GetRadiusAttr().Get(),
          val(rec["geometry"]["diameter_m"]) / 2, " m")
    check("henger magasság", cyl.GetHeightAttr().Get(),
          val(rec["geometry"]["height_m"]), " m")

    mat = stage.GetPrimAtPath(
        prim.GetPath().AppendChild("Looks").AppendChild("CanSurface"))
    if mat and mat.IsValid():
        mu = val(rec["contact"]["mujoco_friction"])[0]
        check("statikus súrlódás",
              mat.GetAttribute("physics:staticFriction").Get(), mu)
    else:
        print("  ❌ nincs fizikai anyag (/Product/Looks/CanSurface)")
        bad.append("CanSurface")

    # ── 4. variánsok ────────────────────────────────────────────────────
    print("\n  DÁTUM-VARIÁNSOK")
    vsets = prim.GetVariantSets()
    if "expiryDate" not in vsets.GetNames():
        print("  ❌ nincs `expiryDate` variantSet")
        bad.append("expiryDate")
    else:
        vs = vsets.GetVariantSet("expiryDate")
        names = vs.GetVariantNames()
        print(f"     {len(names)} variáns: {names}")
        original = vs.GetVariantSelection()
        for n in names:
            vs.SetVariantSelection(n)
            decal = stage.GetPrimAtPath(
                prim.GetPath().AppendChild("DateDecal"))
            iso = decal.GetAttribute("roboshelf:date:iso").Get()
            dec = decal.GetAttribute(
                "roboshelf:date:expectedDecision").Get()
            ok = iso == n and dec
            print(f"     {'✅' if ok else '❌'} {n:<14} iso={iso!r} "
                  f"várt döntés={dec!r}")
            if not ok:
                bad.append(f"variáns:{n}")
        vs.SetVariantSelection(original)

    # ── 5. összegzés ────────────────────────────────────────────────────
    print()
    if bad:
        print(f"  ❌ {len(bad)} eltérés: {', '.join(bad)}")
        print("     A generátort kell javítani (tools/shelflife_usd_export.py),")
        print("     NEM az .usda-t kézzel — az újragenerálásnál elveszne.")
        return 1
    print("  ✅ MINDEN RENDBEN — az .usda a bejegyzéssel egyező terméket ír le.")
    print("     Ezután az export fejlécéből törölhető az ELLENŐRIZETLEN jelzés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
