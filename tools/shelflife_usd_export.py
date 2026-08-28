"""
shelflife_usd_export.py — SKU-bejegyzés → OpenUSD (.usda) + MuJoCo (.xml)

    python3 tools/shelflife_usd_export.py --sku coca_cola_zero_330_sleek
    python3 tools/shelflife_usd_export.py --sku ... --dates 2026-12-19,2025-01-05

Kimenet: models/shelflife_sku/<sku>/
    product.usda      OpenUSD — geometria + fizika + dátum-VariantSet
    product.xml       MuJoCo XML-töredék (a jelenetépítő ezt illeszti be)

────────────────────────────────────────────────────────────────────────────
EGY FORRÁS, KÉT KIMENET
────────────────────────────────────────────────────────────────────────────
Az SKU-bejegyzés a forrás; az USD és az MJCF egyaránt GENERÁLT. Ez szándékos.

Kézenfekvő lenne az USD-t tenni forrásnak — csakhogy:

  · a MuJoCo 3.11 NEM olvas USD-t (nincs `mujoco.usd` almodul),
  · tehát ha USD-ben szerkesztenénk, az MJCF-hez kellene egy konverter,
    és a fejlesztői környezetben `pxr` sincs telepítve.

Ha viszont az USD-t szerkesztenénk ÉS az MJCF-et külön írnánk, két forrásunk
lenne ugyanarra az adatra — pontosan az a duplikáció, amit 2026-08-05-én
felszámoltunk (a tömeg a jelenetépítőben élt, az SKU-bejegyzésben nem is
szerepelt).

A felosztás ezért ELVI, nem kényszer:

    SZERKESZTETT geometria (szkennelt háló)   →  USD-ben él, a bejegyzés
                                                 hivatkozza
    PARAMETRIKUS geometria (henger: 2 szám)   →  a bejegyzésben
    mért/gyártói/levezetett adat              →  a bejegyzésben, forrásjelölve
    USD és MJCF                               →  mindkettő generált

────────────────────────────────────────────────────────────────────────────
A DÁTUM MINT VARIANTSET — EZ A VALÓDI NYERESÉG
────────────────────────────────────────────────────────────────────────────
A szavatossági dátum tételenként változik, tehát NEM a SKU tulajdonsága. Az
1. SKU-nál ezt külön geommal oldottuk meg (matrica), és a textúrát epizódonként
újragyártottuk. USD-ben erre van szabványos eszköz: a `variantSet`.

Az eval epizódonként variánst vált — beleértve a SZÁNDÉKOSAN OLVASHATATLAN
esetet is, amire az M4-nél szükség lesz —, és a geometriához nem nyúl.

────────────────────────────────────────────────────────────────────────────
⚠️ ELLENŐRZÉSI KORLÁT
────────────────────────────────────────────────────────────────────────────
A fejlesztői sandboxban NINCS `pxr` (OpenUSD), és a `usd-core` sem telepíthető.
Az `.usda` szöveges formátum, tehát előállítható — de **itt nem tudjuk
parse-olni**. A generált fájl fejlécében ezért ott áll, hogy ELLENŐRIZETLEN,
amíg valaki meg nem nyitja `usdview`-ban vagy Isaac Simben.

Ez ugyanaz az elv, mint az SKU-mezők `source` jelölése: nem adunk ki
érvényesként olyat, amit nem ellenőriztünk.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SKU_DIR = _REPO / "src/envs/assets/shelflife_sku_private"
OUT_ROOT = _REPO / "models/shelflife_sku"


# A dátummatrica magassága a talp fölött. EGY HELYEN, mert mindkét kimenet
# használja — az első változatban az USD 1.0 mm-t, az MJCF 1.2 mm-t írt.
# Pontosan az a duplikáció, amit ma reggel a tömegnél felszámoltunk.
DECAL_Z = 0.0012


def val(node, default=None):
    """Egy `{value, source, confidence}` mező értéke."""
    if isinstance(node, dict) and "value" in node:
        return node["value"] if node["value"] is not None else default
    return node if node is not None else default


def src(node) -> str:
    return node.get("source", "?") if isinstance(node, dict) else "?"


# ═══════════════════════════════════════════════════════════════════════════
# OpenUSD
# ═══════════════════════════════════════════════════════════════════════════

def emit_usda(sku: dict, dates: list[str]) -> str:
    ident, geo = sku["identity"], sku["geometry"]
    phys, cont, pack = sku["physics"], sku["contact"], sku["packaging"]
    df = sku["date_field"]

    r = float(val(geo["diameter_m"])) / 2.0
    h = float(val(geo["height_m"]))
    mass = float(val(phys["mass_kg"]))
    mu = val(cont["mujoco_friction"], [0.7, 0, 0])[0]
    sku_id = sku["sku_id"]

    # ⚠️ ORIGÓ-KONVENCIÓ — ITT VOLT EGY CSENDES ELTÉRÉS.
    #
    # A USD `Cylinder` a prim origójára KÖZÉPPONTOS; az MJCF-ben viszont a
    # test origóját a TALPRA tettük (a polcra állításhoz az kényelmes).
    # Az első generálás emiatt a dátummezőt az USD-ben z=−71.7 mm-re, az
    # MJCF-ben z=+1.2 mm-re tette — ugyanaz a mező, két helyen.
    #
    # Egységesítve: MINDKETTŐ TALP-ORIGÓ. Az USD-ben ezért a hengert
    # eltoljuk +h/2-vel, és a mező z≈1 mm-re kerül, mint az MJCF-ben.
    decal_z = DECAL_Z
    body_z = h / 2.0

    def variant_block(d: str) -> str:
        iso = d
        try:
            human = date.fromisoformat(d).strftime("%d/%m/%Y")
        except ValueError:
            human = d                       # pl. "unreadable"
        expired = (iso < date.today().isoformat()) if iso[:4].isdigit() else None
        dec = ("MARADHAT" if expired is False else
               "JELÖLNI" if expired is True else "NEM_OLVASHATO")
        return f'''        "{d}" {{
            over "DateDecal" {{
                custom string roboshelf:date:iso = "{iso}"
                custom string roboshelf:date:printed = "{human}"
                custom string roboshelf:date:expectedDecision = "{dec}"
            }}
        }}
'''

    variants = "".join(variant_block(d) for d in dates)

    return f'''#usda 1.0
(
    """
    {ident['product_name']['value']} — Roboshelf SKU
    ═══════════════════════════════════════════════════════════════════════
    GENERÁLT FÁJL. Forrás: az SKU-bejegyzés (privát). Kézzel ne szerkeszd —
    a következő export felülírja. Generátor: tools/shelflife_usd_export.py

    ⚠️ ELLENŐRIZETLEN: a fejlesztői környezetben nincs OpenUSD (`pxr`), ezért
    ez a fájl NEM lett parse-olva. Nyisd meg `usdview`-ban vagy Isaac Simben,
    mielőtt érvényesnek tekintenéd.

    Mértékegység: méter. Felfelé: Z (a MuJoCo-jelenettel egyezően).
    """
    defaultPrim = "Product"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string roboshelf:skuId = "{sku_id}"
        string roboshelf:ean = "{val(ident['ean_gtin'])}"
        string roboshelf:schema = "docs/roboshelf_sku_sema.md"
        string roboshelf:generator = "tools/shelflife_usd_export.py"
        string roboshelf:note = "A szemantikus réteg (döntési szabályok, dátum-típus) SZÁNDÉKOSAN nem itt van: üzleti szabály, nem jelenetadat."
    }}
)

def Xform "Product" (
    kind = "component"
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    prepend variantSets = "expiryDate"
    variants = {{
        string expiryDate = "{dates[0]}"
    }}
)
{{
    # ── FIZIKA ──────────────────────────────────────────────────────────
    # A tömeg MÉRT adat (konyhamérleg). A súlypont NEM becslés: zárt,
    # homogén, forgásszimmetrikus hengeré a geometriai közép.
    float physics:mass = {mass}
    point3f physics:centerOfMass = (0, 0, 0)

    custom string roboshelf:mass:source = "{src(phys['mass_kg'])}"
    custom string roboshelf:friction:source = "{src(cont['mujoco_friction'])}"
    custom string roboshelf:friction:warning = "A súrlódás NINCS MEGMÉRVE. Az ebből származó eredmény szimuláció-belső."

    def Cylinder "Body" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        uniform token axis = "Z"
        double radius = {r:.5f}
        double height = {h:.5f}
        # TALP-ORIGÓ: a USD Cylinder magától középpontos, ezért toljuk.
        # Ugyanaz a konvenció, mint az MJCF-ben — l. a generátor megjegyzését.
        double3 xformOp:translate = (0, 0, {body_z:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(0.78, 0.06, 0.13)]

        # Az ütközési alak EGZAKT — a valódi test is henger, tehát nincs
        # közelítés. (`physics:approximation` szándékosan NINCS itt: az a
        # PhysicsMeshCollisionAPI attribútuma, analitikus gprimre nem
        # értelmezett. Az 1. SKU-nál volt doboz-proxy, ami a gable-top
        # tetejét levágta — itt nincs mit levágni.)
        rel material:binding:physics = </Product/Looks/CanSurface>
    }}

    # A dátummező helye: a doboz ALJÁN, a homorú fenék síkján.
    # Ezt egyetlen robot sem találná ki magától — az adatbázis mondja meg.
    def Xform "DateDecal"
    {{
        double3 xformOp:translate = (0, 0, {decal_z:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        custom string roboshelf:date:location = "{val(df['location_human'])}"
        custom string roboshelf:date:format = "{val(df['format'])}"
        custom string roboshelf:date:ocrDifficulty = "{val(df['ocr_difficulty'])}"
    }}

    def Scope "Looks"
    {{
        def Material "CanSurface" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            # ⚠️ BECSÜLT ÉRTÉKEK. Lakkozott alumínium, simább mint a karton.
            float physics:staticFriction = {mu}
            float physics:dynamicFriction = {mu}
            float physics:restitution = 0.1
        }}
    }}

    # ── A DÁTUM MINT VARIÁNS ────────────────────────────────────────────
    # A szavatossági dátum tételenként változik, tehát nem SKU-tulajdonság.
    # Az eval epizódonként variánst vált, a geometriához nem nyúlva.
    variantSet "expiryDate" = {{
{variants}    }}
}}
'''


# ═══════════════════════════════════════════════════════════════════════════
# MuJoCo
# ═══════════════════════════════════════════════════════════════════════════

def emit_mjcf(sku: dict) -> str:
    geo, phys, cont = sku["geometry"], sku["physics"], sku["contact"]
    sku_id = sku["sku_id"]
    r = float(val(geo["diameter_m"])) / 2.0
    h = float(val(geo["height_m"]))
    mass = float(val(phys["mass_kg"]))
    fr = " ".join(str(x) for x in val(cont["mujoco_friction"]))
    condim = int(val(cont["condim"], 4))

    return f'''<!-- SKU: {sku_id} — GENERÁLT (tools/shelflife_usd_export.py). Kézzel ne szerkeszd.

     A test ANALITIKUS henger, nem szkennelt háló. Ez nem egyszerűsítés,
     hanem PONTOSABB: a doboz szabványos henger, és a MuJoCo natívan tud
     hengert ütköztetni — tehát az ütközési alak egzakt, nem proxy.
     Az 1. SKU-nál a fotogrammetria a formát jól adta, a nyomtatást viszont
     elmosta; itt nincs mit elmosni.  -->
<asset>
  <material name="{sku_id}_mat" rgba="0.78 0.06 0.13 1"
            specular="0.25" shininess="0.55" reflectance="0.08"/>
  <material name="{sku_id}_metal" rgba="0.82 0.82 0.84 1"
            specular="0.35" shininess="0.7" reflectance="0.12"/>
</asset>
<body name="{sku_id}" pos="0 0 0">
  <freejoint name="{sku_id}_free"/>
  <!-- fizika ÉS látvány egyben: a henger mindkettőre jó -->
  <geom name="{sku_id}_col" type="cylinder" size="{r:.5f} {h/2:.5f}"
        pos="0 0 {h/2:.5f}" material="{sku_id}_mat"
        mass="{mass}" friction="{fr}" condim="{condim}"/>
  <!-- a perem és a talp: csak látvány -->
  <geom name="{sku_id}_top" type="cylinder" size="{r*0.93:.5f} 0.0015"
        pos="0 0 {h - 0.0015:.5f}" material="{sku_id}_metal"
        contype="0" conaffinity="0" group="2" mass="0"/>
  <!-- a DÁTUM helye: a doboz ALJÁN. Külön geom, hogy epizódonként
       cserélhető legyen — USD-ben ez a `expiryDate` variantSet. -->
  <geom name="{sku_id}_date" type="cylinder" size="{r*0.80:.5f} 0.0004"
        pos="0 0 {DECAL_Z}" material="{sku_id}_metal"
        contype="0" conaffinity="0" group="2" mass="0"/>
</body>
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="coca_cola_zero_330_sleek")
    ap.add_argument("--dates", default=None,
                    help="vesszős lista; alap: a megfigyelt + 2 próbaeset")
    a = ap.parse_args()

    p = SKU_DIR / f"{a.sku}.json"
    if not p.exists():
        raise SystemExit(f"nincs ilyen SKU-bejegyzés: {a.sku}")
    sku = json.loads(p.read_text(encoding="utf-8"))

    if a.dates:
        dates = [d.strip() for d in a.dates.split(",") if d.strip()]
    else:
        obs = val(sku["date_field"]["observed_parsed_iso"])
        # A harmadik variáns SZÁNDÉKOSAN olvashatatlan — az M4-nél a
        # `NEM_OLVASHATO` kategóriát is mérni kell, nem csak a helyeset.
        dates = [obs, "2025-01-05", "unreadable"]

    out = OUT_ROOT / a.sku
    out.mkdir(parents=True, exist_ok=True)
    (out / "product.usda").write_text(emit_usda(sku, dates), encoding="utf-8")
    (out / "product.xml").write_text(emit_mjcf(sku), encoding="utf-8")

    geo = sku["geometry"]
    print(f"SKU → USD + MJCF  ·  {a.sku}\n")
    print(f"  henger  Ø{float(val(geo['diameter_m']))*1000:.1f} mm × "
          f"{float(val(geo['height_m']))*1000:.1f} mm  "
          f"[{src(geo['diameter_m'])}]")
    print(f"  tömeg   {float(val(sku['physics']['mass_kg']))*1000:.0f} g   "
          f"[{src(sku['physics']['mass_kg'])}]")
    print(f"  dátum-variánsok: {', '.join(dates)}")
    print(f"\n  → {(out / 'product.usda').relative_to(_REPO)}   ⚠️ ELLENŐRIZETLEN")
    print(f"  → {(out / 'product.xml').relative_to(_REPO)}")
    print("\n  Az USD ellenőrzése a saját gépeden:")
    print("      usdview models/shelflife_sku/%s/product.usda" % a.sku)
    print("      (vagy: python -c \"from pxr import Usd; "
          "Usd.Stage.Open('.../product.usda')\")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
