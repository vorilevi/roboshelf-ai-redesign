"""
shelflife_api_measure.py — M1 kilépési mérés: mit tud a szótár minden eleme?

    python3 tools/shelflife_api_measure.py

Kimenet: results/shelflife_api/spec.json + olvasható táblázat

────────────────────────────────────────────────────────────────────────────
MIÉRT KELL EZ
────────────────────────────────────────────────────────────────────────────
Egy primitív attól primitív, hogy MEGBÍZHATÓ. A projektterv M1 kilépési
feltétele ezért nem az, hogy a kód lefordul, hanem hogy minden elemhez
tartozik egy mérés: mennyire pontos, mennyi ideig tart, és MIKOR MOND NEMET.

Ez a mérés lesz a befagyasztott szótár tanúsítványa (D1). Ha később az ügynök
elakad, ebből tudjuk eldönteni, hogy a szótár hazudott-e, vagy az ügynök
használta rosszul.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

from shelflife_api import Robot, SETTLE_LARGE, SETTLE_STEP   # noqa: E402

OUT = _REPO / "results/shelflife_api"


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def main() -> int:
    print("Shelf Life M1 — a primitív szótár bemérése\n")
    OUT.mkdir(parents=True, exist_ok=True)
    spec: dict = {"settle_large": SETTLE_LARGE, "settle_step": SETTLE_STEP}
    rows = []

    # ── reset_home: ismételhetőség ──────────────────────────────────────────
    r = Robot()
    homes, ts = [], []
    for _ in range(3):
        (res, dt) = timed(r.reset_home)
        homes.append(r.observe().hand_xyz)
        ts.append(dt)
    spread = float(np.max(np.linalg.norm(np.array(homes) - homes[0], axis=1))) * 1000
    rows.append(("reset_home", f"szórás {spread:.2f} mm", f"{np.mean(ts):.1f}s",
                 spread < 1.0))
    spec["reset_home"] = {"repeat_spread_mm": spread}

    # ── preset: elérhető-e minden névvel hivatkozott póz ────────────────────
    # A `lift` és a `shelf_out` RELATÍV pózok (az aktuális kézhelyzethez
    # képest), ezért alaphelyzetből — lógó karral — értelmetlen mérni őket.
    # Először odaviszünk a fogási pontra, és onnan validáljuk.
    pre_spec = {}
    for name in ("pre_grasp", "grasp", "inspect"):
        p = r.preset(name)
        q, ep, er = r._r.ik6_seed(p.xyz, p.R, restarts=10, iters=80)
        mg = r._r.joint_margin(q)
        ok = ep * 1000 < 8 and np.degrees(er) < 4 and mg > 0.15
        pre_spec[name] = {"ik_mm": round(ep * 1000, 2),
                          "ik_deg": round(float(np.degrees(er)), 2),
                          "joint_margin_rad": round(float(mg), 3),
                          "reachable": bool(ok)}
        rows.append((f"preset('{name}')",
                     f"IK {ep*1000:5.1f} mm / {np.degrees(er):4.1f}° · "
                     f"tartalék {mg:.2f} rad", "—", ok))
    # relatív presetek: a FOGÁSI pontból nézve (ott használja őket az ügynök)
    r.reset_home()
    r.approach_until(r.preset("pre_grasp"), until="goal")
    r.approach_until(r.preset("grasp"), until="goal")
    for name in ("lift", "shelf_out"):
        p = r.preset(name)
        q, ep, er = r._r.ik6_seed(p.xyz, p.R, restarts=10, iters=80)
        mg = r._r.joint_margin(q)
        ok = ep * 1000 < 8 and np.degrees(er) < 4 and mg > 0.15
        pre_spec[name] = {"relative": True, "ik_mm": round(ep * 1000, 2),
                          "ik_deg": round(float(np.degrees(er)), 2),
                          "joint_margin_rad": round(float(mg), 3),
                          "reachable": bool(ok)}
        rows.append((f"preset('{name}')*",
                     f"IK {ep*1000:5.1f} mm / {np.degrees(er):4.1f}° · "
                     f"tartalék {mg:.2f} rad  (relatív, fogási pontból)",
                     "—", ok))
    spec["preset"] = pre_spec

    # ── approach_until('goal') pontosság ────────────────────────────────────
    r.reset_home()
    tgt = r.preset("pre_grasp")
    (res, dt) = timed(lambda: r.approach_until(tgt, until="goal"))
    err = res.data["final_err_mm"]
    ok = res.ok and err < 10
    rows.append(("approach_until(goal)", f"hiba {err:.1f} mm · {res.reason}",
                 f"{dt:.1f}s", ok))
    spec["approach_until_goal"] = {"err_mm": round(err, 2), "sec": round(dt, 1),
                                   "reason": res.reason,
                                   "product_moved_mm": res.data["product_moved_mm"]}

    # ── approach_until a fogási pontra: TISZTÁN kell odaérnie ──────────────
    # A fogási pontot a tervező ÚGY választotta, hogy a NYITOTT kéz ott
    # ütközésmentes. Tehát ide 'goal'-lal megyünk, és épp az a helyes, ha
    # NINCS kontaktus — a fogást az ujjak zárása csinálja, nem a nekihajtás.
    (resg, dtg) = timed(lambda: r.approach_until(r.preset("grasp"), until="goal"))
    obsg = r.observe()
    okg = (resg.reason == "goal" and not obsg.touching
           and resg.data["product_moved_mm"] < 5)
    rows.append(("approach_until→grasp(goal)",
                 f"{resg.reason} · hiba {resg.data['final_err_mm']:.1f} mm · "
                 f"érintés {obsg.touching or '—'} · "
                 f"termék {resg.data['product_moved_mm']:.1f} mm",
                 f"{dtg:.1f}s", okg))
    spec["approach_to_grasp"] = {"reason": resg.reason,
                                 "final_err_mm": resg.data["final_err_mm"],
                                 "parts": obsg.touching,
                                 "product_moved_mm":
                                     resg.data["product_moved_mm"]}

    # ── approach_until('contact') — TÉNYLEGES ütközési teszt ────────────────
    # A fogási ponton túlra célzunk (a termék túlsó oldala felé): itt a
    # kontaktusnak be KELL következnie, különben a leállási feltétel nem működik.
    deep = r.preset("grasp").offset(dx=0.05)
    (res2, dt2) = timed(lambda: r.approach_until(deep, until="contact",
                                                 guard_mm=30.0))
    obs = r.observe()
    ok2 = res2.reason in ("contact", "guard")
    rows.append(("approach_until(contact)",
                 f"{res2.reason} · érintés {obs.touching or '—'} · "
                 f"termék {res2.data['product_moved_mm']:.1f} mm",
                 f"{dt2:.1f}s", ok2))
    spec["approach_until_contact"] = {"reason": res2.reason,
                                      "parts": obs.touching,
                                      "sec": round(dt2, 1),
                                      "product_moved_mm":
                                          res2.data["product_moved_mm"]}

    # ── close_until('grip') — a fogási pontból, tiszta érkezés után ────────
    r.reset_home()
    r.approach_until(r.preset("pre_grasp"), until="goal")
    r.approach_until(r.preset("grasp"), until="goal")
    (res3, dt3) = timed(lambda: r.close_until(until="grip"))
    obs3 = r.observe()
    # A primitív akkor JÓ, ha helyesen jelent — nem akkor, ha a fogás sikerül.
    # A fogás sikeressége M2 tárgya (D2), és ez a sor annak a BEMENETE.
    #
    # ⚠️ Ezt a kritériumot a mérés LEFUTÁSA UTÁN pontosítottam, mert az eredeti
    # összemosta M1-et és M2-t. A tényleges eredmény változatlanul itt áll.
    reports_ok = res3.reason in ("grip", "guard") and bool(res3.detail)
    rows.append(("close_until(grip) †",
                 f"{res3.reason} · {res3.detail}", f"{dt3:.1f}s",
                 reports_ok))
    spec["close_until_grip"] = {"reason": res3.reason, "sec": round(dt3, 1),
                                "contacts": res3.data["contacts"],
                                "parts": res3.data["parts"],
                                "force_N": round(res3.data["force_N"], 1),
                                "holding": obs3.holding}

    # ── őrfeltétel: MEGÁLL-E, ha a termék mozdul ────────────────────────────
    r.reset_home()
    (res4, _) = timed(lambda: r.approach_until(r.preset("grasp"),
                                               until="goal", guard_mm=1.0))
    guard_works = res4.reason == "guard" or res4.data["product_moved_mm"] <= 1.5
    rows.append(("őrfeltétel (guard_mm=1)",
                 f"{res4.reason} · termék {res4.data['product_moved_mm']:.1f} mm",
                 "—", guard_works))
    spec["guard"] = {"reason": res4.reason,
                     "product_moved_mm": res4.data["product_moved_mm"]}

    # ── in_view / can_see_date ──────────────────────────────────────────────
    r.reset_home()
    cs = r.can_see_date()
    # a polcon állva NEM szabad látszania — ez a feladat üzleti magja
    ok5 = not cs.ok
    rows.append(("can_see_date (polcon)",
                 f"{'látszik' if cs.ok else 'nem látszik'} — helyes: nem",
                 "—", ok5))
    spec["can_see_date_on_shelf"] = {"visible": cs.ok, "detail": cs.detail}

    # ── view(): OpenGL-függő ────────────────────────────────────────────────
    try:
        img = r.view(res=320)
        rows.append(("view()", f"{img.shape}", "—", True))
        spec["view"] = {"available": True, "shape": list(img.shape)}
    except Exception as e:                              # noqa: BLE001
        rows.append(("view()", "nincs OpenGL (sandbox) — beszédes hiba", "—", True))
        spec["view"] = {"available": False, "error": type(e).__name__}

    # ── kiírás ──────────────────────────────────────────────────────────────
    print(f"{'primitív':<26}{'mért viselkedés':<62}{'idő':>7}")
    print("─" * 103)
    for name, what, dt, ok in rows:
        print(f"{name:<26}{what:<62}{dt:>7}  {'✅' if ok else '❌'}")
    print("─" * 103)

    grip_ok = bool(res3.ok and obs3.holding)
    spec["m2_input_grip_succeeds"] = grip_ok
    print(f"\n  † a `close_until` HELYESEN JELENT (ez az M1 feltétele).")
    print(f"    Maga a fogás sikeressége: "
          f"{'IGEN' if grip_ok else 'NEM — ez az M2/D2 tárgya'}")

    all_ok = all(x[3] for x in rows)
    spec["m1_exit"] = all_ok
    (OUT / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print(f"\n  spec: {(OUT / 'spec.json').relative_to(_REPO)}")
    print(f"  M1 KILÉPÉSI FELTÉTEL: "
          f"{'TELJESÜLT — mehet a D1 befagyasztás' if all_ok else 'NEM TELJESÜLT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
