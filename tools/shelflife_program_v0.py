"""
shelflife_program_v0.py — a TELJES feladat, kézzel írva, csak a szótárból

    python3 tools/shelflife_program_v0.py

Két szerepe van:

**1. Lefedettségi teszt (M1/D1 előtt).** Ha a feladat nem írható meg a
szótárból, akkor nem a feladat nehéz, hanem a szótár hiányos. Ez a fájl
végigmegy a teljes láncon — fogás, leemelés, forgatás, olvasás, döntés,
visszatétel/félrerakás —, és ahol nincs rá ige, ott elhasal. Épp ezért íródott
a fogáson TÚL is: a szótárt a leggyengébb pontja minősíti.

**2. Baseline (M5/D4).** Ez az EMBER által írt program. Az ügynök változatát
ehhez mérjük — nem abban, hogy pontosabb-e, hanem hogy mennyivel kevesebb
emberi ráfordítással jut el ugyaneddig.

────────────────────────────────────────────────────────────────────────────
AMI A SZÓTÁRON KÍVÜL VAN
────────────────────────────────────────────────────────────────────────────
A dátum LEOLVASÁSA (kép → szöveg) VLM-hívás, nem robotprimitív. A sandboxban
nincs OpenGL, tehát a render nem fut — ilyenkor a program a `NEM_OLVASHATO`
ágon megy tovább, és ez nem hiba: a kísérleti terv 8. szakasza szerint ez
LEGITIM kimenet. Egy boltban a „nem tudom megállapítani, nézze meg valaki"
végtelenül jobb, mint a magabiztos tévedés.
"""

from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

from shelflife_api import Robot                      # noqa: E402

# Milyen szögekben próbáljuk megmutatni a dátumot a kamerának.
# Az ügynöknek ezt kell kitalálnia; a baseline egyszerű pásztázást használ.
TURN_SEARCH = ((0, "right"), (-40, "right"), (-70, "right"),
               (-70, "up"), (40, "right"))


def read_date(image):
    """VLM-híd — M4-ben készül el. Itt csak a helye van meg."""
    return None


def decide(parsed_iso, date_type: str, rules: dict, today: _date) -> str:
    """A döntés — EZ a feladat üzleti magja, nem az OCR.

    Az EU két különböző dátumot ismer, és nem felcserélhetők:
      · fogyaszthatósági idő (use by)  → BIZTONSÁGI  → kötelező kivonni
      · minőségét megőrzi (best before) → MINŐSÉGI    → jelöléssel árusítható
    """
    if parsed_iso is None:
        return rules.get("unreadable", "NEM_OLVASHATO")
    expired = _date.fromisoformat(parsed_iso) < today
    return rules["expired"] if expired else rules["not_expired"]


def run(verbose: bool = True, continue_on_fail: bool = False) -> dict:
    """`continue_on_fail=True`: a fogás bukása után is végigmegy.

    MIÉRT KELL: a lefedettségi teszt azt kérdezi, hogy a feladat MEGÍRHATÓ-E
    a szótárból. Ha a program a fogásnál megáll (vezérlési probléma, D2),
    akkor a lánc többi igéjéről — forgatás, letétel, félrerakás — semmit nem
    tudunk meg. Ez a kapcsoló szétválasztja a két kérdést.
    """
    r = Robot()
    log: list[str] = []

    def step(label: str, res) -> bool:
        log.append(f"{label}: {res.reason} — {res.detail}")
        if verbose:
            print(f"  {label:<28} {res.reason:<10} {res.detail}")
        return bool(res)

    if verbose:
        print("Shelf Life — teljes program, csak a szótárból\n")
        print(f"  SKU: {r.sku_info()['sku']} · dátum típusa: "
              f"{r.sku_info()['date_type']} · helye: {r.sku_info()['date_location']}\n")

    # ── 1. FOGÁS ────────────────────────────────────────────────────────────
    step("reset_home", r.reset_home())
    step("→ pre_grasp", r.approach_until(r.preset("pre_grasp"), until="goal"))
    step("→ grasp", r.approach_until(r.preset("grasp"), until="goal"))
    grip = r.close_until(until="grip")
    step("close_until(grip)", grip)
    failed_at = None
    if not r.observe().holding:
        failed_at = "fogás"
        log.append("A FOGÁS NEM ZÁRT BE (D2). "
                   + ("folytatás lefedettség-ellenőrzéshez"
                      if continue_on_fail else "leállás"))
        if verbose:
            print(f"  {'⚠ fogás':<28} nem zárt be — "
                  f"{'folytatjuk (lefedettség)' if continue_on_fail else 'leállunk'}")
        if not continue_on_fail:
            return {"decision": None, "failed_at": failed_at, "log": log,
                    "obs": repr(r.observe())}

    # ── 2. LEEMELÉS ÉS KIHÚZÁS ──────────────────────────────────────────────
    step("→ lift", r.approach_until(r.preset("lift"), until="goal"))
    if r.observe().supported_by:
        log.append("figyelem: a termék még alátámasztott az emelés után")
    step("→ shelf_out", r.approach_until(r.preset("shelf_out"), until="goal"))

    # ── 3. A KAMERA ELÉ ─────────────────────────────────────────────────────
    step("→ inspect", r.approach_until(r.preset("inspect"), until="goal"))

    # ── 4. FORGATÁS, AMÍG A DÁTUM LÁTSZIK ───────────────────────────────────
    # Ez a szemantikus leállási feltétel: nem lépésszám dönt, hanem hogy
    # LÁTSZIK-E. A `can_see_date()` geometriai, tehát olcsó — nem kell hozzá
    # render és VLM-hívás.
    seen = False
    for deg, about in TURN_SEARCH:
        if r.can_see_date().ok:
            seen = True
            break
        pose = r.preset("inspect").turned(deg, about)
        if not r.reachable(pose):
            log.append(f"forgatás {deg:+}° {about}: elérhetetlen, kihagyva")
            continue
        step(f"forgatás {deg:+}° {about}", r.approach_until(pose, until="goal"))
    seen = seen or r.can_see_date().ok
    log.append(f"a dátum {'LÁTSZIK' if seen else 'NEM látszik'} a forgatás után")
    if verbose:
        print(f"  {'dátum látszik?':<28} {'IGEN' if seen else 'NEM'}")

    # ── 5. OLVASÁS (VLM — a szótáron kívül) ─────────────────────────────────
    parsed = None
    if seen:
        try:
            parsed = read_date(r.view(res=640))
        except Exception as e:                        # noqa: BLE001
            log.append(f"render nem fut: {type(e).__name__}")

    # ── 6. DÖNTÉS ───────────────────────────────────────────────────────────
    info = r.sku_info()
    decision = decide(parsed, info["date_type"], info["decision_rules"],
                      _date.today())
    if verbose:
        print(f"  {'DÖNTÉS':<28} {decision}")

    # ── 7. VISSZATÉTEL VAGY FÉLRERAKÁS ──────────────────────────────────────
    where = "grasp" if decision == "MARADHAT" else "aside"
    target = r.preset(where)
    step(f"→ {where} fölé", r.approach_until(target.offset(dz=0.05), until="goal"))
    step("place_until(support)", r.place_until(target))
    step("open_hand", r.open_hand())
    step("→ shelf_out", r.approach_until(r.preset("shelf_out"), until="goal"))

    obs = r.observe()
    return {"decision": decision, "date_seen": seen, "parsed": parsed,
            "failed_at": failed_at, "released": not obs.holding,
            "supported_by": obs.supported_by, "log": log, "obs": repr(obs)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true",
                    help="a fogás bukása után is fusson végig (lefedettség)")
    a = ap.parse_args()
    out = run(continue_on_fail=a.coverage)
    print("\n" + "─" * 72)
    if out["failed_at"] and "decision" not in out:
        print(f"  ELAKADT: {out['failed_at']}")
    else:
        if out["failed_at"]:
            print(f"  ⚠ a fogás nem zárt be (D2) — a lánc többi igéje "
                  f"lefedettségre futott")
        print(f"  döntés          {out['decision']}")
        print(f"  dátum látszott  {'igen' if out['date_seen'] else 'nem'}")
        print(f"  elengedte       {'igen' if out['released'] else 'NEM'}")
        print(f"  alátámasztás    {out['supported_by'] or '—'}")
    print(f"  végállapot      {out['obs']}")
    print("─" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
