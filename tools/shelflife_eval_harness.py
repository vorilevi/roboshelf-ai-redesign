"""
shelflife_eval_harness.py — A SIKERARÁNY MÉRÉSE. Egy mérce, hat cellára.

    python3 tools/shelflife_eval_harness.py --selftest
    python3 tools/shelflife_eval_harness.py --policy nulla --episodes 50

────────────────────────────────────────────────────────────────────────────
MIÉRT EZ AZ ELSŐ LÉPÉS
────────────────────────────────────────────────────────────────────────────
A befektetői demó terve **3 robot × 2 módszer** — hat cella, mindegyikben
egy szám: hány százalékban sikerül. 2026-08-16-án kiderült, hogy ez a szám
**sehol nem létezik a projektben**:

    12 betanítási futás naplója közül amelyikben van szám, az mind 0%
    a ténycsomag saját szavaival: „Nincs eval-harness. Nincs 50 epizódos
    futás, nincs szórás, nincs sikerarány. Minden szám EGYETLEN futásból."

Sőt: a fejemben élő „86%-os eredmény a Booster T1-en" **nem létezik** — a
86 az milliméter, egy pozicionálási hiba. Ez a fajta tévedés pontosan az,
amit egy közös mérce megelőz.

Amíg nincs ilyen, a mátrix hat üres cella marad, akárhány videót veszünk.

────────────────────────────────────────────────────────────────────────────
A FELADAT ÉS A SIKER — ELŐRE KIMONDVA
────────────────────────────────────────────────────────────────────────────
FELADAT: a robot vegye le a dobozt a polcról és emelje meg.

SIKER akkor, és csak akkor, ha MINDHÁROM teljesül:

    1. EMELÉS   — a doboz alja ≥ 50 mm-rel a polclap teteje fölött
    2. TARTÁS   — és ott is marad 100 lépésen át (nem esik vissza)
    3. ÉPSÉG    — a doboz dőlése ≤ 30° (nem borult fel, nem csúszott ki)

⚠️ A KRITÉRIUMOT A FUTÁS ELŐTT KELL KIMONDANI, nem utólag hozzáigazítani.
   2026-08-06-án hat rossz kritériumot írtunk egyetlen nap alatt, mert
   mindig az eredmény után fogalmaztuk meg őket.

────────────────────────────────────────────────────────────────────────────
A VÉLETLENÍTÉS NEM DÍSZ
────────────────────────────────────────────────────────────────────────────
Ötven epizód ugyanabból a kezdőállapotból nem ötven mérés, hanem EGY mérés
ötvenszer. Ezért minden epizódban változik:

    a doboz helye     ±25 mm előre-hátra, ±35 mm oldalra
    a doboz fordulása ±180°  (a dátum bárhol lehet)
    a doboz tömege    ±8%    (gyártási szórás)

────────────────────────────────────────────────────────────────────────────
AMIT A SZÁM MELLÉ MINDIG KIÍRUNK
────────────────────────────────────────────────────────────────────────────
Egy sikerarány önmagában félrevezető: 3/5 és 30/50 ugyanaz a 60%, de nem
ugyanaz a tudás. Ezért **Wilson-féle konfidenciaintervallum** jár hozzá,
ami kis mintánál is becsületes. 10 epizódnál a 60% valójában 31–83% — ezt
egy befektetői anyagban látni kell.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env                       # noqa: E402,F401  (SORREND!)
import mujoco                                     # noqa: E402

OUT = _REPO / "results/shelflife_eval"

# ── a siker kritériuma — ELŐRE KIMONDVA ────────────────────────────────────
LIFT_MM = 50.0
HOLD_STEPS = 100
MAX_TILT_DEG = 30.0
MAX_STEPS = 1500

# ── a véletlenítés tartományai ─────────────────────────────────────────────
JITTER_X_MM = 25.0
JITTER_Y_MM = 35.0
MASS_JITTER = 0.08


@dataclass
class EpisodeResult:
    index: int
    siker: bool
    ok: str                       # miért nem sikerült
    max_emeles_mm: float
    max_doles_fok: float
    lepesek: int
    kezdet: dict = field(default_factory=dict)


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson-intervallum — kis mintánál is becsületes, a naiv k/n nem az."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


class Task:
    """A feladat: EGY jelenet, EGY termék, EGY polc. Robotfüggetlen.

    A robot és a módszer kívülről jön — a `policy` egy hívható objektum,
    ami minden lépésben megkapja a modellt és az adatot, és beállítja a
    vezérlést. Így ugyanaz a mérce fut mind a hat cellára.
    """

    def __init__(self, build_model, seed: int = 0):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.pg = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                    "product_0_col")
        self.pb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                    "product_0")
        self.pj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                    "product_0_free")
        sb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                               "shelf_board_1")
        mujoco.mj_forward(self.m, self.d)
        self.shelf_top = float(self.d.geom_xpos[sb][2]
                               + self.m.geom_size[sb][2])
        self.base_pos = self.d.xpos[self.pb].copy()
        self.base_mass = float(self.m.body_mass[self.pb])
        if min(self.pg, self.pb, self.pj) < 0:
            raise RuntimeError("a jelenet nem tartalmazza a terméket")

    def reset(self) -> dict:
        mujoco.mj_resetData(self.m, self.d)
        dx = self.rng.uniform(-JITTER_X_MM, JITTER_X_MM) / 1000
        dy = self.rng.uniform(-JITTER_Y_MM, JITTER_Y_MM) / 1000
        yaw = self.rng.uniform(-np.pi, np.pi)
        mscale = 1.0 + self.rng.uniform(-MASS_JITTER, MASS_JITTER)
        adr = self.m.jnt_qposadr[self.pj]
        self.d.qpos[adr:adr + 3] = self.base_pos + np.array([dx, dy, 0.0])
        self.d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.m.body_mass[self.pb] = self.base_mass * mscale
        mujoco.mj_forward(self.m, self.d)
        return {"dx_mm": round(dx * 1000, 1), "dy_mm": round(dy * 1000, 1),
                "fordulas_fok": round(float(np.degrees(yaw)), 1),
                "tomeg_g": round(self.base_mass * mscale * 1000, 1)}

    # ── a siker MÉRÉSE ────────────────────────────────────────────────────
    def lift_mm(self) -> float:
        """A doboz ALJA a polclap teteje fölött [mm].

        ⚠️ NEM a test origója és NEM a geom közepe. A henger közepe fél
        dobozmagassággal feljebb van, mint az alja — ha azt mérnénk, a
        polcon ÁLLÓ doboz is 72 mm „emelést" mutatna, és minden epizód
        sikeres lenne. Ez pontosan az a hiba, amit a projekt már
        elkövetett a fogáspontnál.
        """
        half = float(self.m.geom_size[self.pg][1])
        bottom = float(self.d.geom_xpos[self.pg][2]) - half
        return (bottom - self.shelf_top) * 1000

    def tilt_deg(self) -> float:
        R = self.d.geom_xmat[self.pg].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def run(self, policy, idx: int) -> EpisodeResult:
        start = self.reset()
        if hasattr(policy, "reset"):
            policy.reset(self.m, self.d)
        held, max_lift, max_tilt, ok = 0, -1e9, 0.0, "időtúllépés"
        for step in range(MAX_STEPS):
            policy(self.m, self.d, step)
            mujoco.mj_step(self.m, self.d)
            lift, tilt = self.lift_mm(), self.tilt_deg()
            max_lift, max_tilt = max(max_lift, lift), max(max_tilt, tilt)
            if tilt > MAX_TILT_DEG and lift < LIFT_MM:
                ok = "feldőlt"
                break
            if lift >= LIFT_MM and tilt <= MAX_TILT_DEG:
                held += 1
                if held >= HOLD_STEPS:
                    return EpisodeResult(idx, True, "siker", round(max_lift, 1),
                                         round(max_tilt, 1), step, start)
            else:
                if held > 0:
                    ok = "visszaesett"
                held = 0
        return EpisodeResult(idx, False, ok, round(max_lift, 1),
                             round(max_tilt, 1), MAX_STEPS, start)


class NullPolicy:
    """Nem csinál semmit. A mérce ALSÓ referenciája: ennek 0%-ot kell adnia."""

    def __call__(self, m, d, step):
        d.ctrl[:] = 0.0


class TeleportPolicy:
    """CSAK ÖNTESZTHEZ: felemeli a dobozt „csalással".

    ⚠️ Ez nem szabályzó, hanem a MÉRŐESZKÖZ ellenőrzése. Ha a
    sikerdetektor erre sem jelez sikert, akkor a detektor romlott el —
    és akkor a 0%-os eredmények semmit nem jelentenek. Egy nap alatt
    nyolcszor fordult elő, hogy a műszer volt hibás, nem a jelenség.
    """

    def __init__(self, task: Task, height_mm: float = 120.0):
        self.t, self.h = task, height_mm / 1000

    def __call__(self, m, d, step):
        adr = m.jnt_qposadr[self.t.pj]
        d.qpos[adr + 2] = self.t.shelf_top + self.h
        d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        d.qvel[m.jnt_dofadr[self.t.pj]:m.jnt_dofadr[self.t.pj] + 6] = 0
        mujoco.mj_forward(m, d)


def evaluate(build_model, policy_factory, episodes: int, seed: int = 0,
             nev: str = "névtelen") -> dict:
    task = Task(build_model, seed=seed)
    pol = policy_factory(task)
    res = [task.run(pol, i) for i in range(episodes)]
    k = sum(r.siker for r in res)
    p, lo, hi = wilson(k, episodes)
    okok: dict[str, int] = {}
    for r in res:
        if not r.siker:
            okok[r.ok] = okok.get(r.ok, 0) + 1
    return {"nev": nev, "epizod": episodes, "siker": k,
            "sikerarany": round(p * 100, 1),
            "konfidencia_95": [round(lo * 100, 1), round(hi * 100, 1)],
            "bukas_okai": okok,
            "kriterium": {"emeles_mm": LIFT_MM, "tartas_lepes": HOLD_STEPS,
                          "max_doles_fok": MAX_TILT_DEG},
            "epizodok": [asdict(r) for r in res]}


def report(r: dict) -> None:
    print(f"\n  ── {r['nev']} ──────────────────────────────────")
    print(f"  siker            {r['siker']}/{r['epizod']}  =  "
          f"{r['sikerarany']:.1f}%")
    print(f"  95% konfidencia  {r['konfidencia_95'][0]:.1f}% … "
          f"{r['konfidencia_95'][1]:.1f}%")
    if r["bukas_okai"]:
        print("  bukások:")
        for ok, n in sorted(r["bukas_okai"].items(), key=lambda x: -x[1]):
            print(f"    {ok:<16}{n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    from shelflife_t1_scene import build_model

    print("Shelf Life — KIÉRTÉKELŐ: egy mérce, hat cellára\n")
    print(f"  feladat:  vedd le a dobozt a polcról és emeld meg")
    print(f"  siker:    ≥{LIFT_MM:.0f} mm emelés · {HOLD_STEPS} lépésen át "
          f"megtartva · dőlés ≤{MAX_TILT_DEG:.0f}°")
    print(f"  véletlen: hely ±{JITTER_X_MM:.0f}/±{JITTER_Y_MM:.0f} mm · "
          f"fordulás ±180° · tömeg ±{MASS_JITTER*100:.0f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    if a.selftest:
        # ⚠️ A MŰSZERT ELŐBB HITELESÍTJÜK. Alsó és felső referencia.
        print("\n  ÖNTESZT — a mérce hitelesítése két ismert végponton\n")
        n = min(10, a.episodes)
        r0 = evaluate(build_model, lambda t: NullPolicy(), n,
                      nev="alsó referencia: semmit sem csinál")
        report(r0)
        r1 = evaluate(build_model, lambda t: TeleportPolicy(t), n,
                      nev="felső referencia: a doboz felemelve (csalás)")
        report(r1)
        jo = r0["siker"] == 0 and r1["siker"] == n
        print(f"\n  {'✅ A MÉRCE HITELES' if jo else '❌ A MÉRCE ROMLOTT'}"
              f" — alul {r0['sikerarany']:.0f}%, felül {r1['sikerarany']:.0f}%")
        if not jo:
            print("     Amíg ez nem 0% és 100%, egyetlen eredmény sem jelent "
                  "semmit.")
        (OUT / "onteszt.json").write_text(
            json.dumps({"also": r0, "felso": r1}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  mentve: {(OUT / 'onteszt.json').relative_to(_REPO)}")
        return 0 if jo else 1

    r = evaluate(build_model, lambda t: NullPolicy(), a.episodes,
                 nev="nulla-szabályzó")
    report(r)
    (OUT / "nulla.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)          # l. shelflife_t1_gripper: MjSpec kilépési összeomlás
