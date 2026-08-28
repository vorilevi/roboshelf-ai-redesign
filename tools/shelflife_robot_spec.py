"""
shelflife_robot_spec.py — a ROBOT adatlapja, forrásmegjelöléssel

    from shelflife_robot_spec import SPEC, apply_mass
    apply_mass(model)          # a modell tömegét a döntött értékre skálázza

────────────────────────────────────────────────────────────────────────────
MIÉRT LÉTEZIK EZ
────────────────────────────────────────────────────────────────────────────
A TERMÉKEKNÉL szigorú rendet tartunk: minden fizikai mező hordozza, hogy
`measured`, `manufacturer`, `derived` vagy `estimated` (l.
`docs/roboshelf_sku_sema.md`). A ROBOTNÁL ezt eddig NEM tettük meg — pedig
ugyanúgy mérések épülnek rá.

2026-08-06-án derült ki, mennyire kellett volna: a szimulációs modell
152,4 kg-ot mutat, a nyilvános katalógus 60–80 kg-ot ír. Kétszeres eltérés,
és MINDEN erő- és nyomatékszámunk ebből jön.

────────────────────────────────────────────────────────────────────────────
FORRÁSJELÖLÉS
────────────────────────────────────────────────────────────────────────────
    measured     — mi mértük meg a modellen vagy a valóságban
    manufacturer — a gyártó közölte
    catalogue    — nem hitelesített katalógusadat (humanoid.guide: „Not Verified")
    derived      — másik adatból számolt
    inferred     — MÁS eszközből következtetett (nem ezé a robopté!)
    decided      — mi döntöttük el, mert nincs megbízható adat
    unknown      — nincs adat
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# A ROBOT — Generative Bionics GENE.01
# ═══════════════════════════════════════════════════════════════════════════

SPEC: dict[str, dict] = {

    # ── alapadatok ──────────────────────────────────────────────────────────
    "gyarto": {"ertek": "Generative Bionics (IIT leányvállalat)",
               "forras": "manufacturer"},
    "statusz": {"ertek": "prototípus · sorozatgyártás várhatóan 2026 Q4",
                "forras": "catalogue",
                "megj": "CES 2026 bemutató, majd AMD Advancing AI 2026-07"},
    "licenc": {"ertek": "CC-BY-NC-4.0 (gb-robot-models)",
               "forras": "manufacturer",
               "megj": "NEM kereskedelmi. Termékbe építéshez külön megállapodás."},

    # ── tömeg — ITT VOLT AZ ELLENTMONDÁS ────────────────────────────────────
    "tomeg_kg": {
        "ertek": 80.0,
        "forras": "decided",
        "megj": (
            "⚠️ HÁROM FORRÁS, HÁROM ÉRTÉK.\n"
            "  · a szimulációs modell összege: 152,4 kg  (measured)\n"
            "  · humanoid.guide katalógus:      60–80 kg  (catalogue, „Not Verified\")\n"
            "  · DÖNTÉS 2026-08-06:              80 kg\n"
            "A 152 kg irreálisan sok egy 162–170 cm-es humanoidhoz; a mezőny "
            "hasonló robotjai 55–80 kg között vannak (Fourier GR-1: 55 kg). "
            "A modell tömegeit ezért a 80 kg-ra SKÁLÁZZUK. Minden korábbi "
            "erő- és nyomatékszám a 152 kg-os modellen készült, tehát "
            "ÚJRA KELL MÉRNI."),
    },
    "magassag_cm": {"ertek": 162.0, "forras": "measured",
                    "megj": "a modellen mérve; a katalógus ~170-et ír (catalogue)"},
    "teherbiras_kg": {"ertek": 14.0, "forras": "catalogue"},

    # ── kinematika — EZEK MÉRVE VANNAK ──────────────────────────────────────
    "izuletek_test": {"ertek": 31, "forras": "measured"},
    "izuletek_kez": {"ertek": 42, "forras": "measured", "megj": "21 kezenként"},
    "aktuatorok": {"ertek": 33, "forras": "measured"},
    "kar_lanc": {
        "ertek": ["torso_yaw", "torso_roll", "r_shoulder_pitch",
                  "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
                  "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"],
        "forras": "measured",
        "megj": "kilenc ízület a talptól a végszerszámig; a felkar és az "
                "alkar TESTEK, nem ízületek",
    },
    "csuklo_tartomany_fok": {
        "ertek": {"yaw": (-85, 85), "pitch": (-22, 32), "roll": (-42, 58)},
        "forras": "measured",
        "megj": "a gördülés és a bólintás szűk — a termék megfordítását a "
                "váll és a könyök végzi",
    },

    # ── kamera — ITT A LEGNAGYOBB BIZONYTALANSÁG ────────────────────────────
    "kamera_darab": {
        "ertek": None, "forras": "unknown",
        "megj": ("A gyártó NEM publikálta. A „Humanoid robotok kameraszenzorai\" "
                 "tanulmány több független forrással erősíti meg. A "
                 "„Kamerarendszerek elemzése\" 2–4 darabot BECSÜL az elődmodell "
                 "(ergoCub, IIT) alapján — az `inferred`, nem adat.\n"
                 "A mi jelenetünkben 4 kamera van: fej ×2, mellkas, rgb."),
    },
    "kamera_felbontas": {
        "ertek": None, "forras": "unknown",
        "megj": ("Katalógus: „Camera resolution: N/A\".\n"
                 "⚠️ MI 480 KÉPPONTTAL SZÁMOLTUNK a dátumolvasásnál — ez "
                 "FELTÉTELEZÉS volt, nem adat. Az ergoCub-ból következtetett "
                 "érték 640×480 @ 30 fps (`inferred`)."),
    },
    "kamera_latoszog_fok": {"ertek": 45.0, "forras": "measured",
                            "megj": "a MI jelenetünk `fovy=45` értéke — nem gyártói adat"},
    "halozati_bemenet_px": {
        "ertek": 224, "forras": "inferred",
        "megj": ("A látórendszer a képet 224×224-re skálázza a neurális "
                 "feldolgozáshoz (YOLOv8 / VLA — a Figure-nél is 224 vagy 336).\n"
                 "⚠️ A DÁTUMOT NEM EBBŐL kell olvasni, hanem a TELJES "
                 "felbontású képkockából. Ez architektúra-döntés, és eddig "
                 "hallgatólagos volt."),
    },
    "fo_erzekelo": {
        "ertek": "teljes testet borító taktilis bőr (érintés, erő, hő, "
                 "ToF közelség)",
        "forras": "manufacturer",
        "megj": ("A GENE.01-nél a kamera KIEGÉSZÍTŐ. A precíziós utolsó "
                 "centimétereket a bőr ToF-érzékelői vezérlik — a tanulmány "
                 "szerint épp ezért „N/A\" a kamerafelbontás.\n"
                 "Ez a MI fogási munkánkat is érinti: a valódi roboton lenne "
                 "közelségérzékelés a kontaktus ELŐTT."),
    },
}

# a szimulációs modell nyers összege, amiből skálázunk
MODEL_MASS_KG = 152.4          # measured, 2026-08-06
TARGET_MASS_KG = SPEC["tomeg_kg"]["ertek"]


def apply_mass(model, target_kg: float = TARGET_MASS_KG,
               exclude=("product", "world")) -> float:
    """A robot testeinek tömegét a döntött összegre skálázza.

    A tehetetlenségi nyomatékokat is ugyanazzal a szorzóval — a geometria
    változatlan, csak a sűrűség. Visszaadja az alkalmazott szorzót.

    ⚠️ Ez NEM javítja a modellt, csak konzisztenssé teszi a döntött
    össztömeggel. Ha valaha megkapjuk a gyártói tömegeloszlást, azt kell
    használni helyette.
    """
    import mujoco
    bn = lambda b: mujoco.mj_id2name(          # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    ids = [b for b in range(model.nbody)
           if not bn(b).startswith(exclude) and model.body_mass[b] > 0]
    cur = float(sum(model.body_mass[b] for b in ids))
    if cur <= 0:
        return 1.0
    k = target_kg / cur
    for b in ids:
        model.body_mass[b] *= k
        model.body_inertia[b] *= k
    return k


def report() -> None:
    print("GENE.01 — a robot adatlapja\n")
    for k, v in SPEC.items():
        e = v["ertek"]
        if isinstance(e, (list, dict)):
            e = f"({len(e)} tétel)"
        print(f"  {k:<24}{str(e):<28}[{v['forras']}]")
    print(f"\n  ⚠️ `unknown` és `decided` mezők külön figyelmet igényelnek.")
    print(f"  A tömeg {MODEL_MASS_KG} kg-ról {TARGET_MASS_KG} kg-ra skálázva "
          f"(×{TARGET_MASS_KG/MODEL_MASS_KG:.3f}).")


if __name__ == "__main__":
    report()
