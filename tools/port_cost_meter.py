"""
port_cost_meter.py — A PORTOLÁSI KÖLTSÉG MÉRÉSE a git történetből

    python3 tools/port_cost_meter.py
    python3 tools/port_cost_meter.py --json

────────────────────────────────────────────────────────────────────────────
MIT MÉR, ÉS MIÉRT
────────────────────────────────────────────────────────────────────────────
A projekt két állítást tesz. A **hordozhatóság** mérve van (sikerarány
robotonként, 50 epizód). A **fejlesztési sebesség** nincs — arra csak
anekdota volt: „napok", „5 döntés", „25 perc".

Ez az eszköz azt gyűjti ki, amit a git objektíven meg tud mondani.
A definíció: `docs/roboshelf_portolasi_koltseg_metrika.md` — a mérés ELŐTT
rögzítve.

────────────────────────────────────────────────────────────────────────────
⚠️ A FŐ VESZÉLY: AZ ÖSSZEHASONLÍTÁS ÖNMAGÁBAN FÉLREVEZET
────────────────────────────────────────────────────────────────────────────
A robotok nem egyformán nehezek. A T1 portolása azért volt gyors, mert
könnyebb volt — nem volt benne rejtett fizikai hiba. A GR1T1-ben volt.

    ❌ TILOS:  „a T1 két nap alatt ment, a GR1T1 nyolc — ezért a módszer"
    ✅ SZABAD: ugyanazon a roboton belül, ugyanarra a hibára:
               GR1T1 · P2 · ember-vezérelt   vs   GR1T1 · P2 · ügynök

Ez a modul ezért a P2 szakaszokat állítja szembe, nem a robotokat.

────────────────────────────────────────────────────────────────────────────
AMIT A GIT NEM TUD MEGMONDANI
────────────────────────────────────────────────────────────────────────────
Az emberi hibakeresési idő és az iterációszám a kézi ágra csak BECSÜLHETŐ.
A modul ezeket `estimated`-ként jelöli, és külön kiírja. Becsült mezőre
külső kommunikációban nem szabad állítást építeni.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results/port_cost"


# ── a vizsgált szakaszok — a metrika 3. pontja szerint ─────────────────────
@dataclass
class Phase:
    nev: str
    robot: str
    modszer: str
    tol: str                      # ISO dátum, bezárólag
    ig: str
    minta: str                    # git pathspec
    megj: str = ""
    # ⚠️ ESTIMATED mezők — a jegyzetekből, nem a gitből
    emberi_dontes: int | None = None
    hipotezis_elvetve: int | None = None
    iteracio: int | None = None
    iteracio_median_s: float | None = None
    gpu_perc_betanitas: int | None = None
    gpu_perc_diagnozis: int | None = None
    eredmeny_sr: float | None = None
    forras: dict = field(default_factory=dict)


PHASES = [
    Phase(
        nev="GR1T1 · P0+P1 · felépítés",
        robot="Fourier GR1T1", modszer="ember-vezérelt",
        tol="2026-07-26", ig="2026-07-26", minta="*gr1*",
        megj="jelenet, env, diagnosztika, szakértő, Kaggle pipeline",
    ),
    Phase(
        nev="GR1T1 · P2 · ember-vezérelt hibakeresés",
        robot="Fourier GR1T1", modszer="ember-vezérelt",
        tol="2026-07-27", ig="2026-07-28", minta="*gr1*",
        megj="öt javítási commit, mind a tünetet kezelte",
        emberi_dontes=None, hipotezis_elvetve=3, iteracio=3,
        iteracio_median_s=9480.0,        # ~158 perc: demó+export+GPU+eval
        gpu_perc_betanitas=474,          # 3 teljes kör × ~158 perc
        gpu_perc_diagnozis=0,
        eredmeny_sr=20.0,
        forras={"hipotezis_elvetve": "measured (git+jegyzet)",
                "iteracio": "measured (git)",
                "iteracio_median_s": "derived (158 perc/kör)",
                "gpu_perc_betanitas": "derived",
                "emberi_dontes": "unknown",
                "eredmeny_sr": "measured (eval)"},
    ),
    Phase(
        nev="GR1T1 · P2 · ügynök-asszisztált hibakeresés",
        robot="Fourier GR1T1", modszer="ügynök-asszisztált",
        tol="2026-08-01", ig="2026-08-02", minta="*gr1*",
        megj="merev-test gyökérok megtalálva, v2 ág",
        emberi_dontes=5, hipotezis_elvetve=4, iteracio=25,
        iteracio_median_s=30.0,
        gpu_perc_betanitas=148,
        gpu_perc_diagnozis=0,
        eredmeny_sr=94.0,
        forras={"emberi_dontes": "measured (jegyzet 5.10)",
                "hipotezis_elvetve": "measured (jegyzet 5.10 táblázat)",
                "iteracio": "measured (jegyzet: ~25, mind <45 s)",
                "iteracio_median_s": "estimated (<45 s felső korlátból)",
                "gpu_perc_betanitas": "measured (148,4 perc)",
                "gpu_perc_diagnozis": "measured (0)",
                "eredmeny_sr": "measured (eval 47/50)"},
    ),
    Phase(
        nev="Booster T1 · teljes portolás  [KONTEXTUS, nem összemérhető]",
        robot="Booster T1", modszer="ember-vezérelt",
        tol="2026-07-24", ig="2026-07-25", minta="*t1*",
        megj="⚠️ nem volt benne rejtett fizikai hiba — könnyebb eset",
        gpu_perc_betanitas=158, eredmeny_sr=86.0,
        forras={"gpu_perc_betanitas": "measured", "eredmeny_sr": "measured"},
    ),
]


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args],
                          capture_output=True, text=True).stdout


def measure(p: Phase) -> dict:
    """A GIT-BŐL kinyerhető mezők. Ezek mind `measured`."""
    rng = [f"--since={p.tol} 00:00", f"--until={p.ig} 23:59"]
    log = git("log", "--all", *rng, "--format=%H|%at|%s", "--", p.minta)
    rows = [l.split("|", 2) for l in log.strip().splitlines() if l]
    if not rows:
        return {"commit": 0}
    ts = sorted(int(r[1]) for r in rows)
    # ⚠️ EGYETLEN COMMITBÓL NEM SZÁMÍTHATÓ ÁTFUTÁS. Az ügynök-ág egyetlen
    #    commitba tette a munkát, ezért a git 0,0 órát mutatott — ami nem
    #    „nulla idő", hanem NINCS ADAT. A commitszám-aszimmetria maga is
    #    lelet (5 javítási commit vs 1), de átfutásnak nem használható.
    span_h = round((ts[-1] - ts[0]) / 3600.0, 1) if len(ts) > 1 else None

    # ⚠️ A `--format=C` MIATT VOLT MINDEN SORSZÁM NULLA. A `git log` a
    #    numstat blokkot csak akkor írja ki, ha a formátum után üres sor
    #    következik; a „C" fejléc a numstat sorokat is elnyelte. Üres
    #    formátum + `--no-renames` a helyes.
    stat = git("log", "--all", *rng, "--numstat", "--no-renames",
               "--format=", "--", p.minta)
    add = dele = 0
    files: set[str] = set()
    for line in stat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit():
            add += int(parts[0])
            dele += int(parts[1])
            files.add(parts[2])
    # új vs módosított: ha a fájl első előfordulása ebben az ablakban van
    uj = 0
    for f in files:
        first = git("log", "--all", "--diff-filter=A", "--format=%at", "--", f)
        first = first.strip().splitlines()
        if first and p.tol <= datetime.utcfromtimestamp(
                int(first[-1])).strftime("%Y-%m-%d") <= p.ig:
            uj += 1
    return {"commit": len(rows), "atfutas_ora": span_h,
            "uj_sor": add, "torolt_sor": dele,
            "fajl": len(files), "uj_fajl": uj, "modositott_fajl": len(files) - uj}


def fmt(v, unit=""):
    return "—" if v is None else f"{v}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    print("Roboshelf — PORTOLÁSI KÖLTSÉG\n")
    print("  definíció: docs/roboshelf_portolasi_koltseg_metrika.md")
    print("  (a metrika a mérés ELŐTT lett rögzítve)\n")

    out = []
    for p in PHASES:
        g = measure(p)
        rec = {**asdict(p), **g}
        out.append(rec)
        print(f"  ── {p.nev}")
        print(f"     {p.robot} · {p.modszer} · {p.tol} → {p.ig}")
        if p.megj:
            print(f"     {p.megj}")
        print(f"     GIT      commit {g['commit']} · "
              f"átfutás {fmt(g.get('atfutas_ora'), ' óra')} · "
              f"+{g.get('uj_sor', 0)} −{g.get('torolt_sor', 0)} sor · "
              f"{g.get('uj_fajl', 0)} új / {g.get('modositott_fajl', 0)} mód. fájl")
        print(f"     ITERÁCIÓ {fmt(p.iteracio)} db · "
              f"medián {fmt(p.iteracio_median_s, ' s')} · "
              f"elvetett hipotézis {fmt(p.hipotezis_elvetve)}")
        print(f"     GPU      betanítás {fmt(p.gpu_perc_betanitas, ' perc')} · "
              f"diagnózis {fmt(p.gpu_perc_diagnozis, ' perc')}")
        print(f"     EMBER    döntés {fmt(p.emberi_dontes)}")
        print(f"     EREDMÉNY {fmt(p.eredmeny_sr, '%')}\n")

    # ── A TISZTA ÖSSZEHASONLÍTÁS ──────────────────────────────────────────
    h = next(r for r in out if "ember-vezérelt hibakeresés" in r["nev"])
    u = next(r for r in out if "ügynök-asszisztált" in r["nev"])
    print("  ══ A TISZTA ÖSSZEHASONLÍTÁS ═══════════════════════════════════")
    print("  Ugyanaz a robot, ugyanaz a hiba, ugyanaz a feladat.\n")
    print(f"  {'':<26}{'ember-vezérelt':>16}{'ügynök':>14}")
    print("  " + "─" * 56)
    for cim, kh, ku, unit in [
            ("naptári átfutás", h.get("atfutas_ora"), u.get("atfutas_ora"), " ó"),
            ("iterációk", h["iteracio"], u["iteracio"], ""),
            ("iteráció medián", h["iteracio_median_s"], u["iteracio_median_s"], " s"),
            ("elvetett hipotézis", h["hipotezis_elvetve"], u["hipotezis_elvetve"], ""),
            ("GPU-perc betanítás", h["gpu_perc_betanitas"], u["gpu_perc_betanitas"], ""),
            ("GPU-perc diagnózis", h["gpu_perc_diagnozis"], u["gpu_perc_diagnozis"], ""),
            ("eredmény", h["eredmeny_sr"], u["eredmeny_sr"], "%")]:
        print(f"  {cim:<26}{fmt(kh, unit):>16}{fmt(ku, unit):>14}")

    if h["iteracio_median_s"] and u["iteracio_median_s"]:
        arany = h["iteracio_median_s"] / u["iteracio_median_s"]
        print(f"\n  → egy iteráció {arany:.0f}× olcsóbb az ügynök ágon")
        print(f"  → ezért {u['iteracio']/h['iteracio']:.1f}× több MÉRÉS fért bele")
        print(f"  → és {u['hipotezis_elvetve']/h['hipotezis_elvetve']:.1f}× "
              f"több hipotézist lehetett elvetni a sikeres előtt")
        print("\n  ⚠️ A commitszám fordítva viselkedik, mint várnánk:")
        print(f"     ember-vezérelt {h['commit']} commit (mind javítási kísérlet)")
        print(f"     ügynök          {u['commit']} commit (a diagnózis nem hagy nyomot)")
        print("     A git a KÍSÉRLETEKET látja, a MÉRÉSEKET nem.")

    print("\n  ⚠️ BECSÜLT MEZŐK — nem szabad rájuk állítást építeni:")
    for r in out:
        for k, v in (r.get("forras") or {}).items():
            if v.startswith(("estimated", "unknown")):
                print(f"     {r['robot']:<16}{k:<22}{v}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "portolasi_koltseg.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  mentve: {(OUT / 'portolasi_koltseg.json').relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
