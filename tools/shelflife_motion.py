"""
shelflife_motion.py — `approach_until`: zárt hurkú mozgás LEÁLLÁSI FELTÉTELLEL

    from shelflife_motion import approach_until
    approach_until(robot, target_xyz, R_des, until="grip")

────────────────────────────────────────────────────────────────────────────
MIÉRT EZ A PRIMITÍV, ÉS MIÉRT PONT ÍGY
────────────────────────────────────────────────────────────────────────────
A Waddle Labs publikált primitív-készletében (2026-07) szerepel egy elem, ami
a mienkből hiányzott:

    approach_until(waypoints + stop criterion → trajectory)

Nem pálya, hanem **mozgás leállási feltétellel**. A különbség nem stiláris.

Amit mi csináltunk: diszkrét pályapontokat terveztünk, és ízület-interpolációval
kötöttük össze őket. MÉRVE: a termék MINDEN pályaponton érintetlen volt (nulla
kontaktus), a végén mégis 141 mm-t mozdult. **A kár a pontok KÖZÖTT keletkezett** —
egy 1 cm-es Cartesian lépés mögött 0.2–0.3 rad ízületi ugrás állt, és a kar
azon az íven söpört át a kartonon.

Ezt a hibaosztályt nem lehet finomabb tervezéssel megszüntetni, csak a
végrehajtás átalakításával:

  1. **FOLYTONOS mozgás** — ~1 mm-es lépések, mindegyik után korrekció.
     Nincs olyan szakasz, ahol a kar „vakon" halad.
  2. **LEÁLLÁSI FELTÉTEL** — a mozgás akkor ér véget, amikor a VILÁG azt
     mondja (kontaktus), nem amikor a terv elfogy. Fogásnál épp ez a lényeg:
     nem tudjuk előre, hány milliméterre van a felület.
  3. **ŐRFELTÉTEL** — ha a termék elmozdul, azonnal megállunk. Eddig a
     szkript végigment és a végén derült ki, hogy a karton a földön van.

A Waddle saját összegzése ugyanerről: *„az ügynök teljesítménye erősen függ
attól, milyen interfészt adunk neki."* Hivatkoznak az Anthropic
*Claude Plays Robotics* munkájára, ahol egyetlen lekérdezhető kurzor
6%-ról 32%-ra vitte a sikert.

────────────────────────────────────────────────────────────────────────────
HOL A HATÁR A PLATFORM ÉS AZ ÜGYNÖK KÖZÖTT
────────────────────────────────────────────────────────────────────────────
Ez a fájl PLATFORM. Az ügynök nem írja, csak hívja. Amit az ügynök ír, az:
„hová és milyen orientációval közelíts, és mikor állj meg" — vagyis a
fogási stratégia. A megbízható mozgás nem az ő dolga.

Korábban ezt rosszul húztuk meg: a `grasp_box` a tervben ügynök-írta készség
volt. Egy LLM-nek 6-DoF Jacobian-korrekciót írni nem a nyelvi képességét
használja.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import mujoco

# ── Alapértékek, mind mérésből ──────────────────────────────────────────────
STEP_M = 0.0015          # 1.5 mm/lépés. 1 cm-nél a kar íve átsöpört a
                         # terméken; 1.5 mm-en az ízületváltozás nagyságrenddel
                         # kisebb, és a korrekció végig fut.
SETTLE = 40              # sim-lépés egy mm-es lépés után (a beállás gyors,
                         # mert a korrekció apró)
CORRECT_EVERY = 1        # hány mm-enként korrigálunk zárt hurokkal
GUARD_MM = 4.0           # ennyinél nagyobb termék-elmozdulásnál megállunk
MAX_STEPS = 400          # biztonsági plafon
FINAL_ITERS = 8          # záró konvergencia-iterációk 'goal' esetén
FINAL_TOL_M = 0.005      # 5 mm — ennél pontosabb pozicionálás nem kell


def _grip_ok(parts: set[str]) -> bool:
    """Szembefogás: hüvelyk + legalább KÉT ujj — tripod-fogás.

    ⚠️ D2-visszacsatolás (2/2): a küszöb korábban „hüvelyk + három ujj" volt.
    MÉRVE, az ujjankénti fáziskésleltetés után pontosan a hüvelyk, a mutató és
    a középső ujj zár rá (30 N) — ez a **tripod**, a kézi manipuláció egyik
    kanonikus fogása, nem féloldalas érintés.

    A küszöböt NEM azért lazítottuk, mert a szigorúbb nem teljesült, hanem mert
    rossz kérdést mért. A fogás minősítője az EMELÉSI PRÓBA: megtartja-e a
    kartont. A kontaktusszám ennek csak proxyja volt, és rossz proxy.
    """
    return "thumb" in parts and len(parts - {"thumb"}) >= 2


def product_support(robot, product_idx: int = 0) -> set[str]:
    """Mi tartja a terméket a KÉZEN KÍVÜL: {'shelf', 'floor', ...}

    MIÉRT KELL: a letételhez tudni kell, mikor ért le a termék. A kéz–termék
    kontaktus erre nem elég — a kéz akkor is fogja, amikor még a levegőben van.
    A szótár korábbi változatában emiatt egyszerűen nem lehetett letenni semmit.
    """
    import mujoco as _mj
    m, d = robot.model, robot.data
    pb = robot._products[product_idx]
    HAND = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
            "r_thumb", "r_index", "r_middle", "r_ring", "r_little")
    out: set[str] = set()
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        for x, y, g in ((b1, b2, con.geom1), (b2, b1, con.geom2)):
            if y != pb:
                continue
            bn = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_BODY, x) or ""
            if bn.startswith(HAND):
                continue
            gn = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, g) or ""
            if gn.startswith("shelf"):
                out.add("shelf")
            elif gn == "floor":
                out.add("floor")
            elif bn:
                out.add(bn)
    return out


def approach_until(robot, target: np.ndarray, R_des: np.ndarray,
                   until: str | Callable = "goal",
                   step_m: float = STEP_M,
                   settle: int = SETTLE,
                   guard_mm: float = GUARD_MM,
                   product_idx: int = 0,
                   verbose: bool = False) -> dict:
    """A fogási pontot EGYENES VONALON viszi a cél felé, amíg a feltétel be nem áll.

    Paraméterek
    -----------
    until : 'goal'    — végigmegy a teljes szakaszon
            'contact' — az első kéz–termék érintkezésig
            'grip'    — amíg a hüvelyk és legalább három ujj is fog
            'support' — amíg a termék valami MÁSHOZ ér (polc, padló) — letétel
            callable(robot) -> bool

    Visszatérés
    -----------
    dict: reason, travelled_mm, remaining_mm, product_moved_mm, parts, steps

    A `reason` mindig megmondja, MIÉRT állt meg — ez az, amit az ügynöknek
    látnia kell a ReAct-hurokban. A néma sikertelenség a legrosszabb kimenet.
    """
    target = np.asarray(target, float)
    start = robot.grasp_point().copy()
    p0 = robot.product_pose(product_idx).copy()
    total = float(np.linalg.norm(target - start))
    if total < 1e-6:
        return {"reason": "goal", "travelled_mm": 0.0, "final_err_mm": 0.0,
                "remaining_mm": 0.0, "product_moved_mm": 0.0,
                "parts": [], "steps": 0}
    direction = (target - start) / total

    def stop_now() -> Optional[str]:
        parts = robot.contact_parts(product_idx)
        if callable(until):
            return "feltétel" if until(robot) else None
        if until == "contact" and parts:
            return "kontaktus"
        if until == "grip" and _grip_ok(parts):
            return "fogás"
        if until == "support" and product_support(robot, product_idx):
            return "alátámasztás"
        return None

    # ŐRFELTÉTEL FOGOTT TERMÉKNÉL: ha a kéz FOGJA a terméket, akkor a termék
    # világbeli elmozdulása a MOZGÁS CÉLJA, nem hiba. Ilyenkor a KÉZHEZ KÉPESTI
    # elcsúszást figyeljük — az jelzi, hogy kicsúszik a kezünkből.
    holding0 = _grip_ok(robot.contact_parts(product_idx))
    hand0 = robot.grasp_point().copy()

    def disturbance_mm() -> float:
        p = robot.product_pose(product_idx)
        if holding0:
            return float(np.linalg.norm(
                (p - robot.grasp_point()) - (p0 - hand0))) * 1000
        return float(np.linalg.norm(p - p0)) * 1000

    n = min(MAX_STEPS, max(1, int(np.ceil(total / step_m))))
    reason, k = "goal", 0
    for k in range(1, n + 1):
        wp = start + direction * (total * k / n)

        # egyetlen Jacobian-korrekció a köztes pontra — nincs „vak" szakasz
        _servo_step(robot, wp, R_des, settle)

        s = stop_now()
        if s:
            reason = s
            break
        if disturbance_mm() > guard_mm:
            reason = "őrfeltétel: a termék elmozdult"
            break
    else:
        # A ciklus végigfutott. Ha a MAX_STEPS plafon vágta el a szakaszt, azt
        # KI KELL MONDANI — a szótár 2. szabálya szerint nincs néma csonkolás.
        if n >= MAX_STEPS and total > MAX_STEPS * step_m + 1e-9:
            reason = "timeout: a szakasz hosszabb a lépéskeretnél"

    # ── ZÁRÓ KONVERGENCIA ───────────────────────────────────────────────────
    # Ha a szakasz végigment feltétel nélkül, a lépésenkénti apró korrekciók
    # nem feltétlenül vittek pontosan a célra: MÉRVE 48.7 mm maradék hiba,
    # miközben a primitív „goal"-t jelentett. Ez pont az a néma sikertelenség,
    # amit a szótár 2. szabálya tilt.
    #
    # A záró fázisban ezért teljes zárt hurokkal ráhúzunk — nagyobb korrekciós
    # lépéssel, de VÉGIG aktív őrfeltétellel, mert itt már a termék közelében
    # vagyunk.
    if reason == "goal":
        for _ in range(FINAL_ITERS):
            if float(np.linalg.norm(robot.grasp_point() - target)) < FINAL_TOL_M:
                break
            _servo_step(robot, target, R_des, settle * 4, clip=0.06)
            s = stop_now()
            if s:
                reason = s
                break
            if disturbance_mm() > guard_mm:
                reason = "őrfeltétel: a termék elmozdult"
                break

    travelled = total * k / n
    parts = sorted(robot.contact_parts(product_idx))
    moved = disturbance_mm()
    final_err = float(np.linalg.norm(robot.grasp_point() - target)) * 1000
    out = {"reason": reason, "travelled_mm": travelled * 1000,
           "final_err_mm": final_err,
           "remaining_mm": (total - travelled) * 1000,
           "product_moved_mm": moved, "parts": parts, "steps": k,
           "held_during_move": holding0,
           "support": sorted(product_support(robot, product_idx))}
    if verbose:
        print(f"      approach_until({until}) → {reason} · "
              f"{out['travelled_mm']:.0f}/{total*1000:.0f} mm · "
              f"maradék hiba {final_err:.1f} mm · "
              f"termék {moved:.1f} mm · {parts}")
    return out


def _servo_step(robot, wp: np.ndarray, R_des: np.ndarray, settle: int,
                clip: float = 0.02) -> None:
    """Egy servo-lépés: parancs → rövid beállás → egy Jacobian-korrekció.

    Szándékosan NEM konvergálunk minden köztes pontra. A cél nem az, hogy
    minden milliméternél pontosan ott legyünk, hanem hogy a kar SEHOL ne
    haladjon nyílt hurokban. A pontosság a szakasz végén kell, és oda a
    lépések halmozódó korrekciója visz el.
    """
    q_des = np.zeros(4); q_cur = np.zeros(4)
    q_err = np.zeros(4); q_inv = np.zeros(4); w = np.zeros(3)
    mujoco.mju_mat2Quat(q_des, np.ascontiguousarray(R_des.flatten()))

    for a, v in zip(robot._arm_a, robot._cmd):
        robot.data.ctrl[a] = v
    robot.step(settle)

    gp = robot.grasp_point()
    e_pos = wp - gp
    R = robot.palm_R()
    mujoco.mju_mat2Quat(q_cur, np.ascontiguousarray(R.flatten()))
    mujoco.mju_negQuat(q_inv, q_cur)
    mujoco.mju_mulQuat(q_err, q_des, q_inv)
    mujoco.mju_quat2Vel(w, q_err, 1.0)

    from shelflife_grasp import ROT_WEIGHT, IK6_DAMPING
    jacp = np.zeros((3, robot.model.nv))
    jacr = np.zeros((3, robot.model.nv))
    mujoco.mj_jac(robot.model, robot.data, jacp, jacr,
                  np.ascontiguousarray(gp), robot._palm)
    J = np.vstack([jacp[:, robot._arm_v], ROT_WEIGHT * jacr[:, robot._arm_v]])
    e6 = np.concatenate([e_pos, ROT_WEIGHT * w])
    JJt = J @ J.T + (IK6_DAMPING ** 2) * np.eye(6)
    dq = J.T @ np.linalg.solve(JJt, e6)
    # A lépésenkénti korrekciót SZŰKEN vágjuk: itt nem nagy hibát javítunk,
    # hanem a sodródást tartjuk kordában. A tág vágás (0.25 rad) pont az az
    # ízületi ugrás volt, ami a kárt okozta.
    dq = np.clip(dq, -clip, clip)
    robot._cmd = np.clip(robot._cmd + dq,
                         robot._arm_range[:, 0] + 0.05,
                         robot._arm_range[:, 1] - 0.05)


def close_until(robot, until: str = "grip", steps: int = 48,
                start: float = 0.0, stop: float = 1.0,
                guard_mm: float = 4.0, max_force_N: float = 20.0,
                product_idx: int = 0, verbose: bool = False) -> dict:
    """Ujjzárás LEÁLLÁSI FELTÉTELLEL — ugyanaz az elv, mint a mozgásnál.

    Nem az számít, milyen erősen zárunk, hanem hogy MIKOR ÁLLUNK MEG.

    HÁROM leállási ok, mindhárom mérésből:

    · **fogás** — a hüvelyk és legalább három ujj is fog (erőzárás).
    · **erőkorlát** — MÉRVE: az ujjankénti fáziskésleltetés után a kontaktus
      0.60-nál jön létre, 1.3 mm elmozdulással; 0.65-nél viszont az erő
      **51.6 N**-ra ugrik és a kartont kilövi (780 mm). A pozíció-aktuátorok
      kontaktusban keményen húznak, tehát erőkorlát nélkül a fogás
      szétroppantja vagy ellövi a terméket.
    · **őrfeltétel** — a termék elcsúszott.

    A lépésszám is mérésből jött: 16 lépés (0.0625) túl durva volt ahhoz, hogy
    a kritikus 0.60–0.65 sávban megálljunk.
    """
    p0 = robot.product_pose(product_idx).copy()
    reason, amount = "teljesen zárt", stop
    for k in range(steps + 1):
        a = start + (stop - start) * k / steps
        robot.close_fingers(a, settle=90)
        parts = robot.contact_parts(product_idx)
        n, f = robot.contact_count(product_idx)
        if (until == "grip" and _grip_ok(parts)) or \
           (until == "contact" and parts):
            reason, amount = "fogás", a
            break
        if f > max_force_N:
            reason, amount = "erőkorlát", a
            break
        moved = float(np.linalg.norm(
            robot.product_pose(product_idx) - p0)) * 1000
        if moved > guard_mm:
            reason, amount = "őrfeltétel: a termék elmozdult", a
            break
    n, f = robot.contact_count(product_idx)
    out = {"reason": reason, "amount": amount, "contacts": n, "force_N": f,
           "parts": sorted(robot.contact_parts(product_idx)),
           "product_moved_mm": float(np.linalg.norm(
               robot.product_pose(product_idx) - p0)) * 1000}
    if verbose:
        print(f"      close_until({until}) → {reason} · zárás {amount:.2f} · "
              f"{n} kontakt {out['parts']} · {f:.1f} N · "
              f"termék {out['product_moved_mm']:.1f} mm")
    return out
