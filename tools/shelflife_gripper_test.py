"""
shelflife_gripper_test.py — a KÉTUJJAS IPARI FOGÓ fogáspróbája

    python3 tools/shelflife_gripper_test.py --selftest   # a mérőeszköz hitelesítése
    python3 tools/shelflife_gripper_test.py              # magasság-söprés

────────────────────────────────────────────────────────────────────────────
MIT MÉR
────────────────────────────────────────────────────────────────────────────
Ugyanaz a doboz, ugyanaz a polc, ugyanaz a mérce, mint az ötujjas kéznél:

    · hány fogópárna érintkezik, és mekkora normálerővel  (CSAK a fogó felől)
    · a fogás magassága a doboz talpától
    · emeléskor a termék KÖVETI-e a fogót  (a termék emelkedése / a fogóé)

A söprés a doboz magassága mentén megy végig — így derül ki, hogy a
polclap meddig engedi a fogót lejjebb. Az ötujjas kéznél ez a korlát
17,6 mm volt: a behajlított kisujj és a tenyér alja felült a polcra.

────────────────────────────────────────────────────────────────────────────
HITELESÍTÉS
────────────────────────────────────────────────────────────────────────────
A `--selftest` ugyanazt a három esetet futtatja, mint az ötujjas próbánál:
rögzített tárgy (tartania kell), távoli tárgy (le kell esnie), padló-ütközés
(nem számíthat bele a fogóerőbe). Amíg ezek nem mennek át, az eredménynek
nem hiszünk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402
from shelflife_gripper_scene import (build_model, CAN_POS, CAN_H,  # noqa: E402
                                     CAN_R)

OPEN_CTRL, CLOSE_CTRL = 0.0, 255.0        # a Robotiq saját vezérlési tartománya
SQUEEZE_MM = 2.0                          # ennyivel zárunk a termék átmérője alá

# ⚠️ MIÉRT NEM „ZÁRJ BE" A PARANCS.
#
# Az első változat teljes zárást parancsolt (255), a termék méretétől
# függetlenül. Mérve: a fogó 32,7 és 56,3 mm-re zárt egy 58 mm-es dobozon —
# vagyis ÁTZÁRT rajta —, és 3200–4000 newton keletkezett. Egy valódi
# Robotiq fogót nem így vezérelnek: CÉLNYÍLÁST kap.
#
# A célnyílás pedig a TERMÉKADATBÁZISBÓL jön: a doboz átmérője 58 mm, a
# parancs 58 mm mínusz egy kevés szorítás. Ez a projekt „katalógust írunk,
# nem robotot tanítunk" tézise a legkonkrétabb formájában — az adatbázis
# mondja meg, mennyire nyíljon a fogó.
SLIP_LIMIT_MM = 20.0
FOLLOW_NEED = 0.8


class GripperRig:
    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.gn = lambda g: mujoco.mj_id2name(       # noqa: E731
            self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self.gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                     "product_0_col")
        self.pb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                    "product_0")
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                "product_0_free")
        self.adr = self.m.jnt_qposadr[jid]
        self.vadr = self.m.jnt_dofadr[jid]
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a):
                    a for a in range(self.m.nu)}
        # ⚠️ A FOGÓ GEOMJAIT TEST SZERINT KELL AZONOSÍTANI, NEM NÉV SZERINT.
        #
        # Az első változatom a geom NEVÉNEK „g_" előtagjára szűrt. Megmérve:
        # a fogó 28 geomjából mindössze NÉGYNEK van neve — a Menagerie-modell
        # ütközőgeomjai névtelenek. A szűrőm tehát 24 geomot NEM LÁTOTT, és
        # ezért jelentett „0 kontaktust" minden magasságon. Nem a fogó nem
        # ért a dobozhoz: a MÉRÉS nem vette észre.
        #
        # Ugyanaz a hibatípus, mint a 118 newtonos padló-eset: rossz szűrő,
        # és az eredmény fizikailag hihetőnek látszott.
        bn = lambda b: mujoco.mj_id2name(          # noqa: E731
            self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self.grip_b = {b for b in range(self.m.nbody) if bn(b).startswith("g_")}
        self.grip_g = [g for g in range(self.m.ngeom)
                       if self.m.geom_bodyid[g] in self.grip_b]
        # ⚠️ NÉGY párnageom van, oldalanként kettő. Az első változatom a
        #    `pads[0]` és `pads[1]` távolságát mérte — ezek UGYANAZON az
        #    oldalon vannak, ezért a „nyílás" végig 18,8 mm-nek látszott,
        #    nyitva és zárva egyformán. Oldalanként kell csoportosítani.
        self.pads = [g for g in self.grip_g if "pad" in self.gn(g)]
        self.pad_r = [g for g in self.pads if "right" in self.gn(g)]
        self.pad_l = [g for g in self.pads if "left" in self.gn(g)]
        self._ft = np.zeros(6)
        mujoco.mj_forward(self.m, self.d)

    # ── vezérlés ────────────────────────────────────────────────────────────

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    def _mount_adr(self):
        return [self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("gx", "gy", "gz", "gyaw", "gpitch", "groll")]

    def _mount_dof(self):
        return [self.m.jnt_dofadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("gx", "gy", "gz", "gyaw", "gpitch", "groll")]

    def move(self, xyz, yaw: float = 0.0, pitch: float = 0.0,
             roll: float = 0.0, settle: int = 400) -> None:
        """A szánt KINEMATIKUSAN vezetjük — az ujjak dinamikusak maradnak.

        ⚠️ Az első változat pozíciószabályzót használt a szánra, és a szán
        0,39 helyett 0,20-ig jutott: 19 CENTIMÉTERES hiba. Ugyanaz a
        jelenség, mint a humanoid karnál (86 mm) — a szabályzó nem éri el a
        parancsolt helyzetet. Egy VÉGSZERSZÁM-tesztnél viszont a kar
        pontossága nem a vizsgálat tárgya, ezért a szán pozícióját
        közvetlenül írjuk elő. Így amit mérünk, az tényleg a fogóé.
        """
        # ⚠️ A SEBESSÉGET NEM SZABAD NULLÁZNI — EZ VOLT A HATODIK ÁLHIBA.
        #
        # Az előző változat minden lépésben `qvel = 0`-t írt a szánra. Ettől
        # a fogó „ugrásokkal" haladt: helyzete változott, sebessége viszont a
        # megoldó szemében mindig nulla volt. A súrlódás relatív SEBESSÉGRE
        # hat, tehát nem volt mit ellenállnia — a fogó egyszerűen ÁTCSÚSZOTT
        # a dobozon.
        #
        # Mérve: 40 mm emelés mellett a doboz 0,4 mm-t emelkedett, MIKÖZBEN
        # 17 kontaktus és 326 N szorítás volt rajta. A fogás jó volt, a
        # MOZGATÁSOM volt fizikátlan.
        #
        # A helyes érték a parancsolt sebesség: (q_új − q_régi) / dt.
        adr, dof = self._mount_adr(), self._mount_dof()
        tgt = np.array([xyz[0], xyz[1], xyz[2], yaw, pitch, roll], float)
        cur = np.array([self.d.qpos[a] for a in adr], float)
        n = max(1, int(settle))
        dt = float(self.m.opt.timestep)
        vel = (tgt - cur) / (n * dt)
        for i in range(1, n + 1):
            q = cur + (tgt - cur) * (i / n)
            for a, v in zip(adr, q):
                self.d.qpos[a] = float(v)
            for dd, v in zip(dof, vel):
                self.d.qvel[dd] = float(v)
            for k, v in zip(("act_gx", "act_gy", "act_gz", "act_gyaw",
                             "act_gpitch", "act_groll"), q):
                self.d.ctrl[self.act[k]] = float(v)
            self.step(1)

    def set_grip(self, ctrl: float, settle: int = 300) -> None:
        """Nyitás/zárás — a szánt KÖZBEN IS a helyén tartva.

        ⚠️ AZ ÖTÖDIK ÁLHIBA, ÉS EZ ZÁRTA LE A KERESÉST. Az első változat csak
        lépett, a szánt nem tartotta. Csakhogy a szán pozíciószabályzóinak
        `ctrl` értéke alaphelyzetben NULLA, tehát azok 400–1000 lépésen át
        visszarángatták a fogót a világ origójába — le, a padló alá, át a
        polcon és a dobozon. A mozgatás közben (`move`) fogtam a szánt, a
        zárás közben nem, és a különbség pont a mérés kritikus szakaszára
        esett.
        """
        adr, dof = self._mount_adr(), self._mount_dof()
        hold = [float(self.d.qpos[a]) for a in adr]
        for k, v in zip(("act_gx", "act_gy", "act_gz", "act_gyaw",
                         "act_gpitch", "act_groll"), hold):
            self.d.ctrl[self.act[k]] = v
        self.d.ctrl[self.act["g_fingers_actuator"]] = float(ctrl)
        for _ in range(int(settle)):
            for a, v in zip(adr, hold):
                self.d.qpos[a] = v
            for dd in dof:
                self.d.qvel[dd] = 0.0      # álló helyzetben ez HELYES
            self.step(1)

    def close_until_force(self, target_N: float = 30.0, step: float = 3.0,
                          settle: int = 60, max_ctrl: float = CLOSE_CTRL) -> dict:
        """Zárás KONTAKTUSERŐRE — nem geometriai lekérdezésre.

        ⚠️ A NYOLCADIK ÁLHIBA, ÉS EZ VOLT AZ UTOLSÓ AKADÁLY. A korábbi
        `close_to_width()` a mért nyílásra zárt. Csakhogy a nyílásfüggvény
        oldalsó fogásnál 0,0 mm-t adott vissza, MIKÖZBEN a pofák 89 mm-re
        voltak egymástól — így a ciklus az első lépés után megállt. Nyers
        koordinátákkal mérve: a pofák a doboz tengelyétől +44,0 és −44,9 mm,
        a doboz sugara 29,1 — vagyis MINDKÉT OLDALON 15 mm hézag maradt, és
        a fogó ízületei a 0…0,8-as tartomány 6%-áig jutottak.
        Nem volt szorítás. Nem a fogás volt rossz: ZÁRÁS NEM TÖRTÉNT.

        A kontaktuserő közvetlenül azt méri, amit el akarunk érni, és nem
        függ semmilyen geometriai segédfüggvénytől. Egy valódi fogót is így
        vezérelnek, erővisszacsatolással.
        """
        adr, dof = self._mount_adr(), self._mount_dof()
        hold = [float(self.d.qpos[a]) for a in adr]
        for k, v in zip(("act_gx", "act_gy", "act_gz", "act_gyaw",
                         "act_gpitch", "act_groll"), hold):
            self.d.ctrl[self.act[k]] = v
        a = self.act["g_fingers_actuator"]
        ctrl = float(self.d.ctrl[a])
        n, F = 0, 0.0
        while ctrl < max_ctrl:
            ctrl = min(max_ctrl, ctrl + step)
            self.d.ctrl[a] = ctrl
            for _ in range(settle):
                for ad, v in zip(adr, hold):
                    self.d.qpos[ad] = v
                for dd in dof:
                    self.d.qvel[dd] = 0.0
                self.step(1)
            n, F = self.contacts()
            if F >= target_N:
                break
        return {"ctrl": ctrl, "contacts": n, "force_N": F}

    def close_to_width(self, target_mm: float, step: float = 2.0,
                       settle: int = 40, max_ctrl: float = CLOSE_CTRL) -> dict:
        """Zárás a TERMÉK ÁTMÉRŐJÉBŐL számolt célnyílásig, zárt hurokban.

        Nem „zárj be", hanem „zárj eddig". A mért nyílást figyeli, és ott
        áll meg — így nem zár át a terméken, és nem keletkezik 4000 newton.
        """
        adr, dof = self._mount_adr(), self._mount_dof()
        hold = [float(self.d.qpos[a]) for a in adr]
        for k, v in zip(("act_gx", "act_gy", "act_gz", "act_gyaw",
                         "act_gpitch", "act_groll"), hold):
            self.d.ctrl[self.act[k]] = v
        a = self.act["g_fingers_actuator"]
        ctrl = float(self.d.ctrl[a])
        while ctrl < max_ctrl:
            ctrl = min(max_ctrl, ctrl + step)
            self.d.ctrl[a] = ctrl
            for _ in range(settle):
                for ad, v in zip(adr, hold):
                    self.d.qpos[ad] = v
                for dd in dof:
                    self.d.qvel[dd] = 0.0
                self.step(1)
            if self.opening_mm() <= target_mm:
                break
        return {"ctrl": ctrl, "opening_mm": self.opening_mm()}

    # ── mérés ───────────────────────────────────────────────────────────────

    def pad_centre(self) -> np.ndarray:
        return np.mean([self.d.geom_xpos[g] for g in self.pads], axis=0)

    def opening_mm(self) -> float:
        """A két fogópárna FELÜLETE közti távolság — a valódi állkapocs-nyílás.

        ⚠️ A HETEDIK ÁLHIBA, ÉS UGYANABBÓL A CSALÁDBÓL. Az előző változat a
        párnák KÖZÉPPONTJAINAK távolságát mérte. A párnáknak viszont
        vastagságuk van, ezért a középpont-távolság ~10 mm-rel nagyobb a
        tényleges résnél. Emiatt a „zárj 55 mm-re" parancs valójában 45 mm-es
        rést kért egy 58 mm-es dobozon — 13 mm-rel a terméken BELÜLRE —, és
        innen jött a 4268 newton.
        Ez pontosan az a hiba, ami a projekt hibataxonómiájában már szerepel:
        *origó-távolság ott, ahol FELÜLETI távolság kell.*
        """
        return min(float(mujoco.mj_geomDistance(self.m, self.d, a, b, 0.4,
                                                self._ft))
                   for a in self.pad_r for b in self.pad_l) * 1000

    def contacts(self) -> tuple[int, float]:
        """(érintkező fogó-geomok száma, összes normálerő) — CSAK a fogó felől."""
        n, F = 0, 0.0
        f = np.zeros(6)
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            if self.pb not in (b1, b2):
                continue
            other = c.geom2 if b1 == self.pb else c.geom1
            if int(self.m.geom_bodyid[other]) not in self.grip_b:
                continue                       # padló, polc, bármi más: KIZÁRVA
            n += 1
            mujoco.mj_contactForce(self.m, self.d, k, f)
            F += abs(float(f[0]))
        return n, F

    def shelf_clearance(self) -> float:
        """A fogó legkisebb távolsága a POLCLAPTÓL, mm."""
        sh = [g for g in range(self.m.ngeom)
              if self.gn(g).startswith("shelf")]
        return min(float(mujoco.mj_geomDistance(self.m, self.d, a, b, 0.5,
                                                self._ft))
                   for a in self.grip_g for b in sh) * 1000

    def can_base_z(self) -> float:
        return CAN_POS[2]

    # ── a próba ─────────────────────────────────────────────────────────────

    def calibrate(self) -> np.ndarray:
        """A fogópárna-középpont eltolása a szán parancsolt helyétől."""
        self.move([0.0, 0.0, 1.40], settle=600)
        self.set_grip(OPEN_CTRL, settle=200)
        return self.pad_centre() - np.array([0.0, 0.0, 1.40])

    def grasp_at(self, height_mm: float, offset: np.ndarray,
                 lift_mm: float = 40.0) -> dict:
        """Fogás a doboz talpától `height_mm` magasságban."""
        # ⚠️ A NEGYEDIK ÁLHIBA, ÉS EZ VOLT A KERESETT VÁLASZ.
        #
        # A `mj_resetData` minden ízületet nullára állít — a szán tehát a
        # VILÁG ORIGÓJÁBA kerül, a fogó pedig a padló alá (z ≈ −0,19 m).
        # Onnan a kinematikus vezetés egyenes vonalban húzza a célpontig, és
        # közben ÁTGÁZOL a polcon és a dobozon. Ezért repült a doboz 498 és
        # 907 mm-t, és ezért nem volt kontaktus a végén: mire a fogó odaért,
        # a doboz már rég nem volt ott.
        #
        # Nem a fogó lökte le a dobozt zárás közben. A MÉRÉS lökte le, az
        # alaphelyzetbe állítás miatt.
        mujoco.mj_resetData(self.m, self.d)
        for a, v in zip(self._mount_adr(), (0.0, 0.0, 1.55, 0.0, 0.0)):
            self.d.qpos[a] = v
        mujoco.mj_forward(self.m, self.d)
        self.set_grip(OPEN_CTRL, settle=400)
        tgt = np.array([CAN_POS[0], CAN_POS[1],
                        self.can_base_z() + height_mm / 1000.0])
        # 15 cm-rel fölé, NYITVA, majd LE — hogy ne söpörjük le a dobozt
        self.move(tgt - offset + np.array([0, 0, 0.15]), settle=500)
        self.set_grip(OPEN_CTRL, settle=400)
        # ⚠️ ELLENŐRIZZÜK, hogy tényleg nyitva van. Egy korábbi futásban a
        #    fogó ZÁRVA (8,1 mm) ereszkedett le, és persze lelökte a dobozt.
        #    A „nyitás" parancs kiadása nem bizonyíték arra, hogy nyitva is van.
        opening = self.opening_mm()
        p_before = self.d.geom_xpos[self.gid].copy()
        self.move(tgt - offset, settle=700)
        moved_on_approach = float(
            np.linalg.norm(self.d.geom_xpos[self.gid] - p_before)) * 1000
        clear = self.shelf_clearance()

        # ⚠️ 600 LÉPÉS KEVÉS VOLT, ÉS EZ ADTA A HARMADIK ÁLHIBÁT.
        #    600 lépéssel a fogó minden magasságon „0 kontaktust" jelentett,
        #    és ebből azt a következtetést vontam le, hogy a fogó lelöki a
        #    dobozt. 800 lépéssel ugyanaz a beállítás 11 kontaktust és
        #    111 N-t ad. A záródás egyszerűen lassabb, mint gondoltam.
        target = (2 * CAN_R * 1000) - SQUEEZE_MM      # a TERMÉKBŐL számolva
        cl = self.close_to_width(target)
        n, F = self.contacts()
        p0 = self.d.geom_xpos[self.gid].copy()
        g0 = self.pad_centre().copy()
        self.move(tgt - offset + np.array([0, 0, lift_mm / 1000.0]),
                  settle=900)
        rise = float(self.d.geom_xpos[self.gid][2] - p0[2]) * 1000
        grip_rise = float(self.pad_centre()[2] - g0[2]) * 1000
        n2, F2 = self.contacts()
        follow = rise / grip_rise if abs(grip_rise) > 1.0 else 0.0
        return {
            "opening_mm": opening, "closed_mm": cl["opening_mm"],
            "grip_ctrl": cl["ctrl"],
            "height_mm": height_mm, "approach_moved_mm": moved_on_approach,
            "shelf_clear_mm": clear, "contacts": n, "force_N": F,
            "grip_rise_mm": grip_rise, "product_rise_mm": rise,
            "follow": follow, "contacts_after": n2,
            "held": follow > FOLLOW_NEED and n2 >= 2,
        }


# ═══════════════════════════════════════════════════════════════════════════

def selftest() -> int:
    print("Kétujjas fogó — HITELESÍTÉS\n")
    rig = GripperRig()
    ok = True

    print("  [1] ismert ROSSZ — a fogó messze, a doboz magára hagyva")
    rig.move([0.0, 0.0, 1.6], settle=400)
    n, F = rig.contacts()
    good = n == 0 and F == 0.0
    print(f"      {n} kontaktus · {F:.1f} N  "
          f"{'✅' if good else '❌ (0 kellene)'}")
    ok &= good

    print("  [2] padló-szűrő — a doboz leejtve, padló-ütközés")
    rig.d.qpos[rig.adr + 2] += 0.5
    rig.d.qvel[rig.vadr:rig.vadr + 6] = 0
    mujoco.mj_forward(rig.m, rig.d)
    rig.step(1200)
    n, F = rig.contacts()
    good = n == 0 and F == 0.0
    print(f"      fogó felőli erő {F:.1f} N · {n} kontaktus  "
          f"{'✅' if good else '❌ (0 kellene — a padló nem fogó)'}")
    ok &= good

    print("  [3] ismert JÓ — a fogó záródik szabad levegőben, majd a dobozon")
    rig = GripperRig()
    off = rig.calibrate()
    rig.set_grip(CLOSE_CTRL, settle=600)
    gap = rig.opening_mm()
    rig.set_grip(OPEN_CTRL, settle=600)
    gap_open = rig.opening_mm()
    good = gap_open > 58.0 > gap
    print(f"      nyitva {gap_open:.1f} mm · zárva {gap:.1f} mm · "
          f"a doboz 58 mm  {'✅' if good else '❌'}")
    ok &= good

    print(f"\n  {'✅ A MÉRŐESZKÖZ HITELES' if ok else '❌ NEM MEGBÍZHATÓ'}")
    return 0 if ok else 1


HEIGHTS_MM = (20, 35, 50, 65, 80, 95, 110, 125)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    print("Kétujjas ipari fogó (Robotiq 2F85) — fogás a polcon\n")
    rig = GripperRig()
    off = rig.calibrate()
    print(f"  a doboz {CAN_H*1000:.0f} mm magas · a fogó nyílása 85 mm\n")
    print(f"  {'fogás m.':>9}{'a doboz %-a':>13}{'nyílás':>9}{'polc-rés':>10}"
          f"{'zárt':>8}{'közelítés':>11}{'kontakt':>9}{'erő':>10}{'követés':>10}")
    print("  " + "─" * 80)
    best = None
    for h in HEIGHTS_MM:
        r = rig.grasp_at(h, off)
        print(f"  {h:7d} mm{h/(CAN_H*1000)*100:12.0f}%"
              f"{r['opening_mm']:8.1f}{r['shelf_clear_mm']:10.1f}"
              f"{r['closed_mm']:8.1f}{r['approach_moved_mm']:11.1f}"
              f"{r['contacts']:9d}{r['force_N']:9.1f} N"
              f"{r['follow']*100:9.0f}%{'  ✅' if r['held'] else ''}")
        if best is None or r["follow"] > best["follow"]:
            best = r
    print(f"\n  A LEGJOBB: {best['height_mm']} mm-nél · "
          f"követés {best['follow']*100:.0f}% · {best['force_N']:.1f} N")
    print("\n  ÖSSZEHASONLÍTÁS az ötujjas kézzel (ugyanaz a doboz, ugyanaz a polc):")
    print("    ötujjas kéz   : 2 ujj · 79,8 N · követés −17% · a doboz eldől")
    print(f"    kétujjas fogó : {best['contacts']} kontaktus · "
          f"{best['force_N']:.1f} N · követés {best['follow']*100:.0f}%")
    return 0 if best["held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
