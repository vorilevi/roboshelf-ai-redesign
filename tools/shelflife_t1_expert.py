"""
shelflife_t1_expert.py — SCRIPTELT SZAKÉRTŐ: vedd le a dobozt és emeld meg

    python3 tools/shelflife_t1_expert.py --episodes 20
    python3 tools/shelflife_t1_expert.py --episodes 5 --render

────────────────────────────────────────────────────────────────────────────
MIÉRT EZ KELL, ÉS MIRE
────────────────────────────────────────────────────────────────────────────
A CÉL: **3 robot × 2 betanítási módszer** (l. `shelflife_CEL`). Az utánzásos
ág úgy tanul, hogy megnézi, hogyan kell — tehát kell valaki, aki megmutatja.
Ez a modul az a valaki: egy kézzel megírt szabályzó, ami leveszi a dobozt.

⚠️ EZ NEM AZ EGYIK MÓDSZER. Ez a TANÍTÓ. A mátrix egyik oszlopa az, amit
   ebből a bemutatóból tanul egy modell; a másik oszlop az LLM által írt
   program. A scriptelt szakértő maga egyik cellába sem kerül be.

A meglévő `scripted_expert_t1.py` egy TOLÁSI feladatra készült, más
jelenetben, fogó nélkül. Erre nem használható.

────────────────────────────────────────────────────────────────────────────
A FOGÁSI TERV — A KORÁBBI MÉRÉSEKBŐL
────────────────────────────────────────────────────────────────────────────
A fogós próbapadon (2026-08-11) mérve:

    felülről fogni NEM MEGY — a fogó alaplapja ráül a doboz fedelére
                              (mérve: 103 N lefelé, feldönti)
    OLDALRÓL fogni MEGY     — 12–18 kontaktus, 100% követés emeléskor

Ezért oldalsó közelítés, a doboz fél magasságában.

A fogó tájolása az A1/T1-ből (mérve, nem feltételezve):

    zárási tengely   a csukló Y-a
    közelítés        a csukló Z-je
    henger tengelye  a csukló −X-e     ← ennek FÜGGŐLEGESNEK kell lennie

Ebből a csukló céltájolása egyértelmű, és a modul ki is számolja — nem
kézzel beírt kvaternió.

────────────────────────────────────────────────────────────────────────────
NÉGY SZAKASZ
────────────────────────────────────────────────────────────────────────────
    1. ELŐKÉSZÍTÉS  a doboz elé, 130 mm-rel hátrébb, nyitott fogóval
    2. KÖZELÍTÉS    egyenesen be, a fogáspontig
    3. ZÁRÁS        a fogó összezár
    4. EMELÉS       a csukló célpontja 120 mm-rel feljebb

⚠️ A SZAKASZHATÁR NEM IDŐ, HANEM ÁLLAPOT. Ha lépésszámra váltanánk, a
   lassabb epizódok félúton zárnának. Minden szakasz akkor ér véget, ha a
   célját elérte (vagy kifutott az idejéből).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env                       # noqa: E402,F401  (SORREND!)
import mujoco                                     # noqa: E402
from shelflife_t1_scene import build_spec, ARM, CAN_R, CAN_H   # noqa: E402

MENAGERIE = _REPO / "mujoco_menagerie/robotiq_2f85/2f85.xml"
WRIST = "right_hand_link"
QUAT_FILE = _REPO / "results/shelflife_t1_gripper/mount_quat.txt"
OUT = _REPO / "results/shelflife_t1_expert"

REACH = 0.1415              # csukló → fogáspont, MÉRVE az A1/T1-ben
PRE_BACK = 0.130            # ennyivel hátrébb indul a közelítés
LIFT_M = 0.120
GRIP_OPEN, GRIP_CLOSE = 0.0, 255.0     # a 2F85 vezérlési tartománya

_ALIVE = []


def load_mount_quat():
    """Az A1/T1-ben MÉRT tájolás. Ha nincs meg, a modul megáll."""
    if not QUAT_FILE.exists():
        raise FileNotFoundError(
            f"nincs meg a felszerelési tájolás: {QUAT_FILE}\n"
            "Előbb futtasd: python3 tools/shelflife_t1_gripper.py")
    return [float(x) for x in QUAT_FILE.read_text().split()]


def build_model():
    """A T1 jelenet a felszerelt fogóval."""
    s = build_spec()
    _ALIVE.append(s)
    child = mujoco.MjSpec.from_file(str(MENAGERIE))
    _ALIVE.append(child)
    f = s.body(WRIST).add_frame()
    f.pos = [0.0, 0.0, 0.0]
    f.quat = load_mount_quat()
    f.attach_body(child.body("base_mount"), "g_", "")
    return s.compile()


class Expert:
    """Négyszakaszos fogás, differenciális inverz kinematikával."""

    def __init__(self, m, d):
        self.m = m
        self.wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST)
        self.pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "product_0")
        self.pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                    "product_0_col")
        # a kar ízületei és a hozzájuk tartozó aktuátorok
        self.jid, self.aid = [], []
        for nm in ARM:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{nm}")
            if j >= 0 and a >= 0:
                self.jid.append(j)
                self.aid.append(a)
        self.dof = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.lo = np.array([m.jnt_range[j][0] for j in self.jid])
        self.hi = np.array([m.jnt_range[j][1] for j in self.jid])
        # a fogó aktuátora
        self.ga = next((i for i in range(m.nu)
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            or "").startswith("g_")), None)
        gn = lambda g: mujoco.mj_id2name(   # noqa: E731
            m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self._pads = [g for g in range(m.ngeom) if "pad" in gn(g)]
        self.phase = 0
        self.tick = 0

    # ── a célpóz a doboz aktuális helyéből ────────────────────────────────
    def targets(self, d):
        """A fogáspont és a csukló céltájolása. MINDEN LÉPÉSBEN újraszámolva.

        ⚠️ A doboz helye epizódonként változik (a kiértékelő véletlenít), és
        a közelítés közben el is mozdulhat. Egy egyszer kiszámolt célpont
        rossz epizódot ad — a fogós próbapadon pont ez okozta, hogy a fogó
        a doboz MELLETT zárt.
        """
        can = d.xpos[self.pb].copy()
        grasp = can + np.array([0.0, 0.0, CAN_H / 2])     # fél magasságban
        # ⚠️ A KÖZELÍTÉSI IRÁNY NEM A CSUKLÓTÓL FÜGG. Az első változat a
        #    `grasp − csukló` vektorból számolta, minden lépésben újra — így
        #    a célpont a saját mozgásunktól függött, és ahogy a kéz közeledett,
        #    az irány elfordult. Mérve: záráskor a fogó közepe 26 mm-re volt a
        #    doboz tengelyétől, miközben a hézag oldalanként 9,5 mm. Ezért
        #    döntötte fel minden epizódban.
        #    A doboz HELYÉT követni kell (mozdulhat), az IRÁNYT nem.
        base = np.array([0.0, 0.0, grasp[2]])       # a törzs függőlegese
        appr = grasp - base
        appr[2] = 0.0
        n = np.linalg.norm(appr)
        appr = appr / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        # a csukló céltájolása: −X függőlegesen fel, Z a közelítés mentén
        Xw = np.array([0.0, 0.0, -1.0])
        Zw = appr
        Yw = np.cross(Zw, Xw)
        Yw /= np.linalg.norm(Yw)
        R = np.column_stack([Xw, Yw, Zw])
        return grasp, appr, R

    def wrist_goal(self, d, back: float, lift: float):
        grasp, appr, R = self.targets(d)
        pos = grasp - appr * REACH - appr * back + np.array([0, 0, lift])
        return pos, R

    # ── differenciális inverz kinematika ──────────────────────────────────
    def ik_step(self, d, pos_goal, R_goal, gain=1.0):
        m = self.m
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self.wrist)
        J = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])

        e_pos = pos_goal - d.xpos[self.wrist]
        Rc = d.xmat[self.wrist].reshape(3, 3)
        Rerr = R_goal @ Rc.T
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, Rerr.flatten())
        ang = 2 * np.arccos(np.clip(q[0], -1, 1))
        v = q[1:]
        nv = np.linalg.norm(v)
        e_rot = (v / nv * ang) if nv > 1e-9 else np.zeros(3)

        e = np.concatenate([e_pos * 2.0, e_rot])
        # ⚠️ CSILLAPÍTOTT legkisebb négyzetek. A sima pszeudoinverz a
        #    szinguláris tartásoknál (kinyújtott kar) elszáll — a
        #    csillapítás ára némi pontatlanság, cserébe nem robban.
        lam = 0.08
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
        qcur = d.qpos[self.qadr]
        return np.clip(qcur + gain * dq, self.lo, self.hi)

    # ── a szakaszgép ──────────────────────────────────────────────────────
    def reset(self, m, d):
        self.phase = 0
        self.tick = 0

    def __call__(self, m, d, step):
        self.tick += 1
        if self.phase == 0:                      # ELŐKÉSZÍTÉS
            pos, R = self.wrist_goal(d, PRE_BACK, 0.0)
            close = False
            if self._reached(d, pos, R, 0.02) or self.tick > 500:
                self.phase, self.tick = 1, 0
        elif self.phase == 1:                    # KÖZELÍTÉS
            pos, R = self.wrist_goal(d, 0.0, 0.0)
            close = False
            # ⚠️ NEM A CSUKLÓ POZÍCIÓJA A FELTÉTEL, HANEM A FOGÓ PÁRNÁIÉ.
            #    A csukló lehet 12 mm-en belül, miközben a párnák közepe
            #    26 mm-re van a doboz tengelyétől — a fogásnál ez utóbbi
            #    számít, és ennyinél már ütközik.
            if self._pads_aligned(d, 0.008) or self.tick > 700:
                self.phase, self.tick = 2, 0
        elif self.phase == 2:                    # ZÁRÁS
            pos, R = self.wrist_goal(d, 0.0, 0.0)
            close = True
            if self.tick > 150:
                self.phase, self.tick = 3, 0
        else:                                    # EMELÉS
            pos, R = self.wrist_goal(d, 0.0, LIFT_M)
            close = True

        d.ctrl[self.aid] = self.ik_step(d, pos, R)
        if self.ga is not None:
            d.ctrl[self.ga] = GRIP_CLOSE if close else GRIP_OPEN

    def _pads_aligned(self, d, tol) -> bool:
        """A fogópárnák közepe a doboz TENGELYÉN van-e (vízszintesen)."""
        if not self._pads:
            return False
        pc = np.mean([d.geom_xpos[g] for g in self._pads], axis=0)
        can = d.xpos[self.pb]
        return bool(np.linalg.norm((pc - can)[:2]) < tol)

    def _reached(self, d, pos, R, tol) -> bool:
        if np.linalg.norm(pos - d.xpos[self.wrist]) > tol:
            return False
        Rc = d.xmat[self.wrist].reshape(3, 3)
        return float(np.trace(R @ Rc.T)) > 2.7      # ~25°-on belül


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    import shelflife_eval_harness as H

    print("Shelf Life — SCRIPTELT SZAKÉRTŐ a T1-re (fogás)\n")
    print("  cél: 3 robot × 2 módszer · ez a TANÍTÓ, nem az egyik módszer\n")

    r = H.evaluate(build_model, lambda t: Expert(t.m, t.d), a.episodes,
                   nev="scriptelt szakértő · Booster T1")
    H.report(r)
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    (OUT / "eredmeny.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  mentve: {(OUT / 'eredmeny.json').relative_to(_REPO)}")

    if r["siker"] == 0:
        print("\n  ⚠️ NULLA SIKER. Mielőtt a szabályzót hangolnánk: a mérce")
        print("     hitelesítve van (0%/100% önteszt), tehát a hiba a")
        print("     szabályzóban vagy a fogás geometriájában van.")
        print("     Futtasd `--render`-rel és NÉZD MEG.")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
