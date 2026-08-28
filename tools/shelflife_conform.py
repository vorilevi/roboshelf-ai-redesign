"""
shelflife_conform.py — PERECENKÉNT MEGÁLLÓ ZÁRÁS (M2)

    python3 tools/shelflife_conform.py            # összevetés a jelenlegivel
    python3 tools/shelflife_conform.py --verbose

────────────────────────────────────────────────────────────────────────────
MIÉRT ÚJ FÁJL ÉS NEM A SZÓTÁR ÁTÍRÁSA
────────────────────────────────────────────────────────────────────────────
Ez a `close_until` viselkedésének alternatívája, tehát D1-változás lenne.
A szabály az, hogy előbb BIZONYÍTJUK, aztán javasoljuk. Ez a fájl a bizonyítás;
ha jobb, akkor kerül be a szótárba, külön döntéssel.

────────────────────────────────────────────────────────────────────────────
A HIBA, AMIT JAVÍT
────────────────────────────────────────────────────────────────────────────
A GENE.01 keze **21 ízület, 21 aktuátor** (hüvelyk 5, a négy ujj 4–4), a
csuklóval együtt 24 DoF — pontosan annyi, amennyit a gyártó megad.

Mi ezt a 21 aktuátort **egyetlen 0–1 skalárral** vezéreltük:

    close_fingers(amount):
        minden aktuátor  ←  nyitott + (zárt − nyitott) · fázis(amount)

Az ujjankénti `DIGIT_DELAY` csak IDŐZÍTÉST tol, alakot nem. Következmény:
egy ujj három perece **rögzített arányban** hajlik, tehát az ujj egyetlen
görbe mentén tud alakot venni. Ez nem ujj, hanem hajlított merev pofa.

Egy valódi ujj nem így fog meg semmit: amelyik perec hozzáér, az **megáll**,
a többi tovább zár, és így SIMUL RÁ a tárgyra. Ebből lesz a KÖRBEZÁRÁS
(formazárás), ami sem a súrlódástól, sem a pontos előpozicionálástól nem függ.

Ez magyarázza a korábbi méréseket is:

  · a „rés legyen kicsi és kiegyenlített" kritérium a MEREV POFA következménye
    — ha az ujjak nem simulnak rá, tényleg pontosan kell pozicionálni;
  · a 25,3 mm-es fal (a legkisebb kiegyenlített rés, ami mellett a tenyér még
    kimarad a dobozból) ugyanennek a geometriája;
  · a doboz azért borult fel, mert az odaérő ujj nem állt meg, hanem TOLTA
    tovább — a vezérlés nem kontaktust figyelt, hanem egy skalárt követett.

────────────────────────────────────────────────────────────────────────────
AMIT EZ CSINÁL
────────────────────────────────────────────────────────────────────────────
Aktuátoronként külön zárási szint. Minden lépésben:

  1. megnézzük, melyik PERECET érinti a termék;
  2. az azt a perecet mozgató ízületet BEFAGYASZTJUK (tovább csak nyomná);
  3. a többi ízület zár tovább.

Egy ízület a `jnt_bodyid` szerinti testet mozgatja — ha az a test már ér a
termékhez, akkor annak az ízületnek a további zárása tolás, nem fogás.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
import shelflife_grasp as _G                     # noqa: E402
from shelflife_api import Robot                  # noqa: E402

STEP = 0.02          # egy menetben ennyit zár egy még szabad ízület (0–1)
SETTLE = 25          # szimulációs lépés menetenként
MAX_ROUNDS = 90
FORCE_CAP_N = 45.0   # ennél nagyobb összerőnél megállunk (a karton kilövése ellen)


def close_conforming(r: Robot, step: float = STEP, settle: int = SETTLE,
                     max_rounds: int = MAX_ROUNDS,
                     force_cap: float = FORCE_CAP_N,
                     verbose: bool = False) -> dict:
    g = r._r
    m, d = g.model, g.data
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""

    # aktuátor → az általa mozgatott TEST, és a test kinematikai MÉLYSÉGE
    #
    # ⚠️ AZ ELSŐ VÁLTOZAT ITT HIBÁZOTT. Csak azt az ízületet fagyasztottam be,
    # amelyik az érintkező perecet mozgatja. Csakhogy egy perec TÉRBEN a nála
    # PROXIMÁLISABB ízületektől is mozog: ha az ujjbegy hozzáér, de a tő
    # tovább fordul, a begy TOLJA a tárgyat. Mérve pontosan ez történt —
    # a hüvelyk, a középső, a mutató és a kisujj BEGYE is hozzáért
    # (9., 13., 15. kör), a doboz mégis kicsúszott.
    #
    # A helyes alulaktuált szabály: ha egy perec érintkezik, befagy az őt
    # mozgató ízület ÉS MINDEN NÁLA PROXIMÁLISABB az adott ujjban. A
    # disztálisabbak zárnak tovább — így simul rá az ujj.
    body_of, depth_of, digit_of = {}, {}, {}
    DIGITS = ("thumb", "index", "middle", "ring", "little")

    def _depth(b: int) -> int:
        n = 0
        while b > 0:
            b = int(m.body_parentid[b]); n += 1
        return n

    for act in g._pose:
        jid = int(m.actuator_trnid[act, 0])
        b = int(m.jnt_bodyid[jid])
        body_of[act] = b
        depth_of[act] = _depth(b)
        nm = bn(b)
        digit_of[act] = next((dg for dg in DIGITS if f"_{dg}_" in nm
                              or nm.endswith(f"_{dg}")), "")

    level = {act: 0.0 for act in g._pose}
    frozen: set[int] = set()

    def touching_bodies() -> set[int]:
        out = set()
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == pb:
                out.add(int(b2))
            elif b2 == pb:
                out.add(int(b1))
        return out

    def total_force() -> float:
        f = np.zeros(6)
        tot = 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if pb in (b1, b2):
                mujoco.mj_contactForce(m, d, k, f)
                tot += float(abs(f[0]))
        return tot

    for rnd in range(max_rounds):
        hit = touching_bodies()
        # az érintkező perecek mélysége ujjanként
        touch_depth: dict[str, int] = {}
        for a in g._pose:
            if body_of[a] in hit and digit_of[a]:
                dg = digit_of[a]
                touch_depth[dg] = max(touch_depth.get(dg, -1), depth_of[a])
        # befagy: az érintkező perec ízülete ÉS minden proximálisabb
        newly = {a for a in g._pose
                 if a not in frozen and digit_of[a] in touch_depth
                 and depth_of[a] <= touch_depth[digit_of[a]]}
        if newly and verbose:
            names = sorted(bn(body_of[a]).replace("r_", "") for a in newly)
            print(f"    [{rnd:2d}] megállt: {', '.join(names)}")
        frozen |= newly

        free = [a for a in g._pose if a not in frozen and level[a] < 1.0]
        if not free:
            reason = "minden perec megállt" if frozen else "nincs mit zárni"
            break
        if total_force() > force_cap:
            reason = f"erőkorlát {force_cap:.0f} N"
            break

        for act in free:
            level[act] = min(1.0, level[act] + step)
            o, c = g._pose[act]
            d.ctrl[act] = o + (c - o) * level[act]
        g.step(settle)
    else:
        reason = "körlimit"

    hit = touching_bodies()
    if verbose:
        print(f"    érintkező testek: "
              f"{sorted(bn(b) for b in hit) if hit else '— nincs'}")
    digits = set()
    for b in hit:
        nm = bn(b)
        for dg in ("thumb", "index", "middle", "ring", "little"):
            if f"_{dg}_" in nm or nm.endswith(f"_{dg}"):
                digits.add(dg)
    return {
        "reason": reason,
        "frozen": len(frozen),
        "of": len(g._pose),
        "digits": sorted(digits),
        "contacts": len(hit),
        "force_N": total_force(),
        "mean_level": float(np.mean(list(level.values()))),
    }


def lift_test(r: Robot, label: str) -> dict:
    """A VALÓDI mérce: megtartja-e emelés közben."""
    p0 = r._r.product_pose().copy()
    res = r.approach_until(r.preset("lift"), until="goal", guard_mm=1e9)
    o = r.observe()
    moved = float(np.linalg.norm(r._r.product_pose() - p0)) * 1000
    rise = float(r._r.product_pose()[2] - p0[2]) * 1000
    print(f"    emelés: {res.reason} · termék {moved:.0f} mm "
          f"(ebből emelkedés {rise:.0f} mm) · "
          f"{'✅ MEGTARTOTTA' if o.holding else '❌ elengedte'}")
    return {"held": bool(o.holding), "rise_mm": rise, "moved_mm": moved}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("Shelf Life M2 — perecenként megálló zárás\n")
    print("A kéz 21 ízülete, 21 aktuátorral. Eddig EGY skalár vezérelte mind.\n")

    # ── A) a jelenlegi, skaláris zárás ──────────────────────────────────────
    print("  A) JELENLEGI — egy skalár, fázistolással")
    r = Robot()
    r.reset_home()
    # ⚠️ UGYANAZ a megközelítés, mint a baseline-ban (shelflife_program_v0.py).
    # Az első változat `follow_plan()`-nel indult — az más pózba visz, és így
    # nem a ZÁRÁST hasonlítottam volna össze, hanem a megközelítést is.
    r.approach_until(r.preset("pre_grasp"), until="goal")
    r.approach_until(r.preset("grasp"), until="goal")
    res = r.close_until(until="grip")
    print(f"    close_until → {res.reason}: {res.detail}")
    a_lift = lift_test(r, "skaláris")

    # ── B) perecenként megálló zárás ────────────────────────────────────────
    print("\n  B) ÚJ — perecenként megálló")
    r2 = Robot()
    r2.reset_home()
    r2.approach_until(r2.preset("pre_grasp"), until="goal")
    r2.approach_until(r2.preset("grasp"), until="goal")
    out = close_conforming(r2, verbose=a.verbose)
    print(f"    {out['frozen']}/{out['of']} ízület állt meg kontaktusra · "
          f"{out['contacts']} érintkező perec · {len(out['digits'])} ujj "
          f"{out['digits']}")
    print(f"    összerő {out['force_N']:.1f} N · átlagos zárás "
          f"{out['mean_level']:.2f} · vége: {out['reason']}")
    b_lift = lift_test(r2, "perecenkénti")

    # ── összevetés ──────────────────────────────────────────────────────────
    print("\n" + "─" * 66)
    print(f"  {'':22}{'skaláris':>16}{'perecenkénti':>18}")
    print(f"  {'ujj kontaktusban':22}{len(res.data.get('parts', [])):>16}"
          f"{len(out['digits']):>18}")
    print(f"  {'emelkedés (mm)':22}{a_lift['rise_mm']:>16.0f}"
          f"{b_lift['rise_mm']:>18.0f}")
    print(f"  {'megtartotta':22}{('IGEN' if a_lift['held'] else 'nem'):>16}"
          f"{('IGEN' if b_lift['held'] else 'nem'):>18}")
    print("─" * 66)
    if b_lift["held"] and not a_lift["held"]:
        print("\n  ✅ A perecenkénti zárás megtartja, a skaláris nem.")
        print("     → javaslat a szótár bővítésére (D1-döntés kell hozzá).")
    elif b_lift["held"] and a_lift["held"]:
        print("\n  Mindkettő megtartja — az összevetés a stabilitásról szól.")
    else:
        print("\n  ❌ A perecenkénti zárás sem tartja meg. A záródás modellje")
        print("     tehát nem az EGYETLEN ok — jelenteni kell, nem tovább")
        print("     hangolni.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
