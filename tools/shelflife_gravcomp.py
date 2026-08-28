"""
shelflife_gravcomp.py — GRAVITÁCIÓKOMPENZÁCIÓ a robot ízületeire

    python3 tools/shelflife_gravcomp.py --selftest   # a kompenzáció hitelesítése
    python3 tools/shelflife_gravcomp.py              # mennyit javít a pontosságon

    # használat kódból:
    from shelflife_gravcomp import enable
    r = Robot(); enable(r)          # innentől a kar oda megy, ahová küldjük

────────────────────────────────────────────────────────────────────────────
A MÉRT PROBLÉMA
────────────────────────────────────────────────────────────────────────────
2026-08-06, a fogás újratervezése közben mérve:

    az inverz kinematika hibája                         1,1 mm
    a fogáspont a céltól, SZÁMOLVA (mj_forward)         2,2 mm
    a fogáspont a céltól, a kar TÉNYLEGES mozgása után  86  mm

És nem a beállási idő hiányzik:

    ramp_to(n=14, settle= 60)   ízülethiba 0,114 rad   eltérés 92,9 mm
    ramp_to(n=20, settle=140)   ízülethiba 0,104 rad   eltérés 86,9 mm
    ramp_to(n=40, settle=400)   ízülethiba 0,105 rad   eltérés 86,0 mm
    ramp_to(n=80, settle=800)   ízülethiba 0,105 rad   eltérés 86,3 mm

Hatszor hosszabb idő, ugyanaz a hiba: ez ÁLLANDÓSULT ÁLLAPOT, nem tranziens.

────────────────────────────────────────────────────────────────────────────
AZ OK
────────────────────────────────────────────────────────────────────────────
A kar aktuátorai arányos-differenciáló (PD) pozíciószabályzók, integráló tag
és gravitációkompenzáció nélkül. Egy ilyen szabályzó állandó terhelést csak
állandó hibával tud tartani:

    τ_szükséges = kp · Δq      →      Δq = τ_nehézségi / kp

Vagyis a kar SOHA nem áll be a parancsolt szöghelyzetbe, amíg a gravitáció
nyomatékot fejt ki rá. 0,105 rad ≈ 6°, és egy ilyen hosszú karon ez 86 mm.

86 mm a colásdoboz átmérőjének MÁSFÉLSZERESE. Semmilyen fogásgeometria nem
éli túl — ezért adott a mai kinematikai söprés minden „nyertese" nulla ujjat
a valódi futásban.

────────────────────────────────────────────────────────────────────────────
A MEGOLDÁS
────────────────────────────────────────────────────────────────────────────
A `data.qfrc_bias` MuJoCo-ban pontosan a nehézségi + Coriolis + centrifugális
nyomatékok vektora. Ha ezt előrecsatoljuk (`qfrc_applied`), a szabályzónak
már nem kell terhelést tartania, csak a hibát javítani — és a hiba nullához
tart. Ez a robotikában szabványos „computed torque" előrecsatolás.

⚠️ CSAK A ROBOT ÍZÜLETEIRE SZABAD ALKALMAZNI. A termék szabad ízülete
   (`product_0_free`) is szerepel a `qfrc_bias`-ban; ha azt is kompenzálnánk,
   a doboz SÚLYTALANNÁ válna, és minden fogásmérés hazudna. Ezért a modul
   kizárja a szabad ízületeket, és a `--selftest` ELLENŐRZI, hogy az elengedett
   doboz továbbra is leesik.

⚠️ Ez a modul EGYETLEN MEGLÉVŐ FÁJLT SEM ÍR ÁT. Futásidőben cseréli ki a
   `step()` metódust az adott robotpéldányon, tehát a régi viselkedés
   érintetlen marad, és a kompenzáció bármikor visszavonható (`disable`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
import shelflife_grasp as G                      # noqa: E402
from shelflife_api import Robot                  # noqa: E402


def robot_dofs(m) -> np.ndarray:
    """A robot ízületeinek DOF-indexei — a SZABAD ízületek nélkül.

    A szabad ízület (a termék) kihagyása nem részletkérdés: ha azt is
    kompenzálnánk, a doboz lebegne, és minden fogásmérés értelmetlen lenne.
    """
    out = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            continue
        n = int({mujoco.mjtJoint.mjJNT_BALL: 3}.get(int(m.jnt_type[j]), 1))
        adr = int(m.jnt_dofadr[j])
        out.extend(range(adr, adr + n))
    return np.array(sorted(set(out)), dtype=int)


def enable(robot: Robot, ratio: float = 1.0):
    """Bekapcsolja a kompenzációt EZEN a robotpéldányon. Visszaad egy kikapcsolót.

    `ratio` = 1.0 teljes kompenzáció; kisebb érték részleges (méréshez hasznos,
    mert így látszik, hogy a hiba tényleg a nehézségi nyomatékkal arányos).
    """
    g = robot._r
    m, d = g.model, g.data
    dofs = robot_dofs(m)
    orig = g.step

    def step(k: int = 1) -> None:
        for _ in range(int(k)):
            d.qfrc_applied[dofs] = ratio * d.qfrc_bias[dofs]
            orig(1)

    g.step = step                                # type: ignore[method-assign]

    def disable() -> None:
        g.step = orig                            # type: ignore[method-assign]
        d.qfrc_applied[dofs] = 0.0

    return disable


# ═══════════════════════════════════════════════════════════════════════════
# MÉRÉS
# ═══════════════════════════════════════════════════════════════════════════

def _target_pose():
    """A 2026-08-06-i söprés nyertese — ezen mértük a 86 mm-t."""
    from shelflife_grasp_redesign import hand_pose, frames
    th, pf, dz, kn = (1.40, 0.20, 0.40), 0.25, -40, "+60°"
    G.HAND_OPEN.update(hand_pose(th, pf))
    G.GRASP_TWEAK_CM = np.zeros(3)
    return frames()[kn], dz


def accuracy(ratio: float | None) -> tuple[float, float]:
    """(ízülethiba rad, a fogáspont eltérése a kinematikaitól mm)."""
    R, dz = _target_pose()
    r = Robot(); r.reset_home()
    g = r._r
    m, d = g.model, g.data
    box, _ = g.product_box()
    q, _ep, _er = g.ik6_seed(box + np.array([0, 0, dz / 1000.0]), R,
                             restarts=16, iters=110)
    aq = np.array(g._arm_q)
    d.qpos[aq] = q
    mujoco.mj_forward(m, d)
    kin = g.grasp_point().copy()

    r.reset_home()
    if ratio is not None:
        enable(r, ratio)
    g.ramp_to(q, n=20, settle=140)
    err = float(np.abs(d.qpos[aq] - q).max())
    dist = float(np.linalg.norm(g.grasp_point() - kin)) * 1000
    return err, dist


def selftest() -> int:
    """A kompenzáció HITELESÍTÉSE. Két dolgot kell egyszerre teljesítenie."""
    print("Gravitációkompenzáció — HITELESÍTÉS\n")
    ok = True

    # ── 1. ISMERT ROSSZ: a TERMÉKNEK továbbra is le kell esnie
    print("  [1] a termék NEM válhat súlytalanná")
    r = Robot(); r.reset_home()
    enable(r)
    g = r._r
    gid = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_GEOM, "product_0_col")
    jid = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_JOINT,
                            "product_0_free")
    adr = g.model.jnt_qposadr[jid]
    g.data.qpos[adr + 2] += 0.30                 # 30 cm-rel a polc fölé
    mujoco.mj_forward(g.model, g.data)
    z0 = float(g.data.geom_xpos[gid][2])
    g.step(600)
    drop = (z0 - float(g.data.geom_xpos[gid][2])) * 1000
    good = drop > 100.0
    print(f"      elengedve {drop:.0f} mm-t esett  "
          f"{'✅' if good else '❌ (kellene > 100 mm — a doboz LEBEG)'}")
    ok &= good

    # ── 2. ISMERT JÓ: a KAR pontossága javuljon
    print("  [2] a kar pontossága javul")
    e0, d0 = accuracy(None)
    e1, d1 = accuracy(1.0)
    good = d1 < d0 / 4
    print(f"      kompenzáció nélkül  ízülethiba {e0:.3f} rad · "
          f"eltérés {d0:5.1f} mm")
    print(f"      kompenzációval      ízülethiba {e1:.3f} rad · "
          f"eltérés {d1:5.1f} mm  "
          f"{'✅' if good else '❌ (a negyedére kellene csökkennie)'}")
    ok &= good

    print(f"\n  {'✅ A KOMPENZÁCIÓ HITELES' if ok else '❌ NEM MEGBÍZHATÓ'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    print("Gravitációkompenzáció — mennyit javít a kar pontosságán\n")
    print(f"  {'kompenzáció':>13}{'ízülethiba':>13}{'eltérés a célponttól':>23}")
    print("  " + "─" * 49)
    for ratio in (None, 0.0, 0.5, 0.9, 1.0):
        e, dist = accuracy(ratio)
        nm = "nincs" if ratio is None else f"{ratio*100:.0f}%"
        print(f"  {nm:>13}{e:10.3f} rad{dist:18.1f} mm")
    print("\n  A 0% sor és a „nincs" ' sor ugyanannak kell lennie — ez a mérés\n'
          "  belső ellenőrzése: ha eltérnek, maga a bekötés változtat valamit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
