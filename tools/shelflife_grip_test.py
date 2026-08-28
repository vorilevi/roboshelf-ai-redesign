"""
shelflife_grip_test.py — HITELESÍTETT fogáspróba

    python3 tools/shelflife_grip_test.py --selftest    # a mérőeszköz hitelesítése
    python3 tools/shelflife_grip_test.py               # a jelenlegi kéz mérése

────────────────────────────────────────────────────────────────────────────
MIÉRT EZ A FÁJL LÉTEZIK
────────────────────────────────────────────────────────────────────────────
2026-08-05 délutánján öt egymást követő körben mértem hibásan, és mind az öt
hibát utólag találtam meg — nem a mérés jelezte:

    1. „118 N-nal lövi ki az első kontaktus"  →  a PADLÓVAL való ütközés volt,
       miután a doboz már leesett. A `total_force()` a padlót is beszámolta.
    2. Az erre épített erőkorlát-söprés emiatt értelmetlen lett (kisebb
       korlátnál „nagyobb erő" — fizikailag lehetetlen, és ez árulta el).
    3. „A `PALM_CLEAR` a 25 mm-es fal oka"  →  megmérve 1,8 mm-t hoz a 26-ból.
    4. A kézforma-söprés 15 mm-es proxyja a ROSSZ kézformát is átengedte.
    5. Az előhajlított kéznél a behelyezés volt rossz, nem a kézforma.

A közös ok: minden körben ÚJ, egyszeri szkriptet írtam, és egyiket sem
hitelesítettem semmin. Ezért ez a modul mást csinál: van benne `--selftest`,
ami ISMERT JÓ és ISMERT ROSSZ esetre is lefut, és ha azokat nem adja vissza
helyesen, akkor a mérésnek nem szabad hinni.

────────────────────────────────────────────────────────────────────────────
A PRÓBA
────────────────────────────────────────────────────────────────────────────
1. a tárgyat a kézbe helyezzük, ÁTHATOLÁS NÉLKÜL (1-D keresés a tenyér
   normálisa mentén — nem az „állkapocs közepére", mert az behajlított
   kéznél a hajlaton belülre esik);
2. RÖGZÍTJÜK, és zárunk — így a gravitáció nem ejti ki, mielőtt az ujjak
   odaérnének (ez volt az 5. hiba);
3. elengedjük, és 1000 lépésen át nézzük, mennyit csúszik.

Amit mérünk, mind KÉZ-oldali kontaktusokra szűrve (soha nem padló, nem polc):
    · hány ujj érintkezik, és melyik perecével
    · a normálerők összege
    · a tárgy elmozdulása elengedés után

A KRITÉRIUM (előre kimondva, 2026-08-05):
    csúszás < 20 mm · erő < 50 N · ≥ 4 ujj · ebből ≥ 2 középpereccel
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

DIGITS = ("thumb", "index", "middle", "ring", "little")
SLIP_LIMIT_MM = 20.0
FORCE_LIMIT_N = 50.0
NEED_DIGITS = 4
NEED_MEDIAL = 2
RELEASE_STEPS = 1000


class GripRig:
    """Egy fogáspróba környezete. Minden mérés KÉZ-oldali kontaktusra szűrt."""

    def __init__(self, robot: Robot):
        self.r = robot
        self.g = robot._r
        self.m, self.d = self.g.model, self.g.data
        self.pb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                    "product_0")
        self.gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                     "product_0_col")
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                "product_0_free")
        self.adr = self.m.jnt_qposadr[jid]
        self.vadr = self.m.jnt_dofadr[jid]
        self._bn = lambda b: mujoco.mj_id2name(
            self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        ARM = ("r_shoulder", "r_upper_arm", "r_forearm", "r_wrist",
               "r_thumb", "r_index", "r_middle", "r_ring", "r_little")
        self.hand_g = [gg for gg in range(self.m.ngeom)
                       if self._bn(self.m.geom_bodyid[gg]).startswith(ARM)]

    # ── mérés ───────────────────────────────────────────────────────────────

    def contacts(self) -> tuple[set[str], set[str], float]:
        """(ujjak, érintkező perecek, összes normálerő) — CSAK a kézre szűrve.

        ⚠️ Ez a szűrés a 1. hiba javítása. A korábbi változat minden
        `product_0`-t érintő kontaktust összegzett, tehát a PADLÓ-ütközést
        is — így lett 118 N egy 343 g-os dobozból.
        """
        digits, parts, F = set(), set(), 0.0
        f = np.zeros(6)
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            if self.pb not in (b1, b2):
                continue
            other = self._bn(b2 if b1 == self.pb else b1)
            if not other.startswith("r_"):
                continue                     # padló, polc, bármi más: KIZÁRVA
            parts.add(other)
            for dg in DIGITS:
                if f"_{dg}_" in other or other.endswith(f"_{dg}"):
                    digits.add(dg)
            mujoco.mj_contactForce(self.m, self.d, k, f)
            F += abs(float(f[0]))
        return digits, parts, F

    def hand_gap(self) -> float:
        """A kéz és a tárgy legkisebb FELÜLETI távolsága (negatív = áthatolás)."""
        ft = np.zeros(6)
        return min(float(mujoco.mj_geomDistance(self.m, self.d, gg,
                                                self.gid, 0.5, ft))
                   for gg in self.hand_g)

    # ── behelyezés ──────────────────────────────────────────────────────────

    def place(self, pos: np.ndarray, quat=(1, 0, 0, 0)) -> None:
        self.d.qpos[self.adr + 3:self.adr + 7] = quat
        mujoco.mj_forward(self.m, self.d)
        off = self.d.geom_xpos[self.gid] - self.d.qpos[self.adr:self.adr + 3]
        self.d.qpos[self.adr:self.adr + 3] = pos - off
        self.d.qvel[self.vadr:self.vadr + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def seat_in_hand(self) -> np.ndarray | None:
        """A tárgy legmélyebb ÁTHATOLÁSMENTES helye a kézben.

        ⚠️ NEM az „állkapocs közepe". Behajlított ujjaknál az a hajlaton
        BELÜLRE esik, és a tárgy már a zárás első lépésénél a kézben van —
        ez volt az 5. hiba. Itt a tenyér normálisa mentén kívülről befelé
        keresünk, és az utolsó olyan helyet vesszük, ahol még nincs áthatolás.
        """
        # ⚠️ HARMADIK JAVÍTÁS — a hitelesített eszköz ezt is megmutatta.
        #
        # (a) Az első változat a tenyér normálisa mentén keresett; az
        #     alaphelyzetben a FELKAR felé mutat, tehát sehol nem volt hely.
        # (b) A második egyenes mentén keresett, és a kéz KÜLSEJÉRE tette a
        #     tárgyat: teljes záráskor NULLA ujj ért hozzá, mert az ujjak
        #     befelé hajlottak, el a tárgytól.
        #
        # A helyes kérdés nem egy irány, hanem: hol a LEGMÉLYEBB pont a
        # tenyér közelében, ahol a tárgy még nem hatol a kézbe? Ez 3-D
        # keresés, de kicsi: ±6 cm a tenyér körül, 1 cm-es lépéssel, majd
        # finomítás. A cél a tenyérközépponthoz legközelebbi ÁTHATOLÁSMENTES
        # hely — ott van a markolat.
        palm = self.d.xpos[self.g._palm].copy()
        Rp = self.d.xmat[self.g._palm].reshape(3, 3)

        def scan(centre, reach, stepm):
            best, bestd = None, 1e9
            rng = np.arange(-reach, reach + 1e-9, stepm)
            for du in rng:
                for dv in rng:
                    for dw in rng:
                        c = centre + Rp @ np.array([du, dv, dw])
                        self.place(c)
                        if self.hand_gap() < 0.0005:
                            continue
                        dist = float(np.linalg.norm(c - palm))
                        if dist < bestd:
                            bestd, best = dist, c
            return best

        # ⚠️ NEGYEDIK JAVÍTÁS. A „legmélyebb áthatolásmentes hely" sem jó:
        # a hitelesített próba szerint teljes záráskor NULLA ujj ért hozzá,
        # mert az a pont a tenyérhez van közel, de NEM ott, ahová az ujjak
        # záródnak. A markolat helyét nem a NYITOTT kéz geometriája adja,
        # hanem az, hogy hová ér az ujjak PÁLYÁJA.
        #
        # Ezért a behelyezés kritériuma: az a hely, ahol FÉLIG ZÁRT kéznél a
        # legtöbb ujj érintkezne, és nyitott kézzel még nincs áthatolás.
        # Ez keresési heurisztika — a VERDIKTET nem ez mondja ki, hanem a
        # hitelesített próba.
        def contacts_at(c, level):
            self.place(c)
            if self.hand_gap() < 0.0:
                return -1, 0
            self.g.set_hand_qpos(self.d, level, phased=False)
            mujoco.mj_forward(self.m, self.d)
            dg, parts, _ = self.contacts()
            return len(dg), len([x for x in parts if "medial" in x])

        open_q = self.d.qpos.copy()
        best, bestk = None, (-1, -1, 1e9)
        for du in np.arange(-0.10, 0.101, 0.02):
            for dv in np.arange(-0.10, 0.101, 0.02):
                for dw in np.arange(-0.10, 0.101, 0.02):
                    c = palm + Rp @ np.array([du, dv, dw])
                    self.d.qpos[:] = open_q
                    nd, nm_ = contacts_at(c, 0.45)
                    if nd < 0:
                        continue
                    k = (nd, nm_, -float(np.linalg.norm(c - palm)))
                    if k > bestk:
                        bestk, best = k, c
        self.d.qpos[:] = open_q
        if best is None:
            return None
        self.place(best)
        self.g.close_fingers(0.0, settle=1)
        mujoco.mj_forward(self.m, self.d)
        return best

    # ── a próba ─────────────────────────────────────────────────────────────

    def run(self, target_N: float = 15.0, step: float = 0.05,
            settle: int = 25, verbose: bool = False) -> dict:
        pos = self.seat_in_hand()
        if pos is None:
            return {"ok": False, "why": "a tárgy nem fér a kézbe áthatolás nélkül"}
        quat = self.d.qpos[self.adr + 3:self.adr + 7].copy()
        qpos = self.d.qpos[self.adr:self.adr + 3].copy()

        lvl = 0.0
        while lvl < 1.0:
            lvl += step
            self.g.close_fingers(float(lvl), settle=1)
            for _ in range(settle):
                self.d.qpos[self.adr:self.adr + 3] = qpos
                self.d.qpos[self.adr + 3:self.adr + 7] = quat
                self.d.qvel[self.vadr:self.vadr + 6] = 0
                self.g.step(1)
            dg, parts, F = self.contacts()
            if F >= target_N and len(dg) >= 2:
                break

        dg, parts, F = self.contacts()
        p0 = self.d.geom_xpos[self.gid].copy()
        self.g.step(RELEASE_STEPS)
        slip = float(np.linalg.norm(self.d.geom_xpos[self.gid] - p0)) * 1000
        dg2, parts2, F2 = self.contacts()
        med = len([p for p in parts if "medial" in p])
        return {
            "ok": True, "close": lvl, "digits": sorted(dg), "medial": med,
            "force_N": F, "slip_mm": slip,
            "digits_after": sorted(dg2), "force_after_N": F2,
            "held": slip < SLIP_LIMIT_MM and len(dg2) >= 2,
            "pass": (slip < SLIP_LIMIT_MM and F < FORCE_LIMIT_N
                     and len(dg) >= NEED_DIGITS and med >= NEED_MEDIAL),
        }


    # ═══════════════════════════════════════════════════════════════════════
    # MEGKÖZELÍTÉSES PROTOKOLL — ez modellezi a VALÓDI fogást
    # ═══════════════════════════════════════════════════════════════════════

    def settle_to_steady(self, tol_N: float = 0.5, tol_mm: float = 0.05,
                         window: int = 25, max_steps: int = 4000) -> dict:
        """Lép, amíg a kontaktuserő ÉS a tárgy helyzete meg nem nyugszik.

        Nem fix lépésszám: a kilépés feltétele, hogy három egymást követő
        ablakban se az erő (< `tol_N`), se a tárgy (< `tol_mm`) ne változzon.
        Ha `max_steps`-ig ez nem áll be, a visszaadott `converged` HAMIS, és
        akkor a mérésre nem szabad építeni.
        """
        prevF, prevP, stable = None, self.d.geom_xpos[self.gid].copy(), 0
        done = 0
        while done < max_steps:
            self.g.step(window)
            done += window
            _, _, F = self.contacts()
            p = self.d.geom_xpos[self.gid].copy()
            dmm = float(np.linalg.norm(p - prevP)) * 1000
            if prevF is not None and abs(F - prevF) < tol_N and dmm < tol_mm:
                stable += 1
                if stable >= 3:
                    return {"converged": True, "steps": done, "force_N": F}
            else:
                stable = 0
            prevF, prevP = F, p
        return {"converged": False, "steps": done,
                "force_N": float(prevF or 0.0)}

    def run_approach(self, close_level: float = 1.0, ramp: int = 20,
                     ramp_settle: int = 10, lift_mm: float = 20.0) -> dict:
        """A kéz MEGY a tárgyhoz, nem a tárgyat teleportáljuk.

        ⚠️ ÖTÖDIK JAVÍTÁS — 2026-08-06. Az első változat ERŐKÜSZÖBRE állt meg
        („zárj, amíg 15 N-t el nem érsz"). Ezt megmértük, és kiderült, hogy a
        beállási idő önmagában 17,5 N és 102,3 N között, a követést −106% és
        +32% között mozgatja. Az ok: hosszabb beállásnál minden zárási szinten
        több erő épül fel, tehát a ciklus MÁS SZINTEN áll meg — a `settle` nem
        a mérés pontosságát változtatta, hanem AZT, HOGY MELYIK FOGÁST MÉRJÜK.
        Emiatt a 08-05-i összes egyfutásos paraméter-összehasonlítás
        értelmezhetetlen volt.

        A javítás: a zárás RÖGZÍTETT végállapotig megy (`close_level`), utána
        `settle_to_steady()` vár a tényleges egyensúlyig, és csak onnan mérünk.
        Így a felfutás időzítése már csak a pályát befolyásolja, a mért
        állapotot nem — ezt a `--repeat` hitelesítési eset ellenőrzi.

        Mellékhaszon: teljesen zárt kéznél a szorítóerőt a szervó ERŐKORLÁTJA
        határozza meg, nem egy önkényes küszöb. Épp ezt akarjuk mérni.

        ⚠️ MIÉRT KELLETT EZ. A teleportálásos próba azt kérdezi: „van-e olyan
        hely, ahová a tárgyat betéve a kéz rázár?" — és arra azt felelte, hogy
        nincs. Csakhogy egy valódi fogás nem így történik: ott a KÉZ megy a
        tárgyhoz, és a tárgy az ujjbegyek közti nyíláson át kerül a markolatba.
        A polcon pontosan ez történt (5 ujj, 62 N), tehát a teleportálásos
        „nem fér be" válasz a protokoll műterméke volt, nem a kézé.

        A mérőrész (kéz-oldali erő, ujjak, perecek) VÁLTOZATLAN — az a rész
        hitelesített. Csak a behelyezést cseréljük megközelítésre.

        A megtartás mércéje itt NEM az esés (a polc alátámasztja), hanem az
        EMELÉS: felmegy-e a termék a kézzel.
        """
        from shelflife_api import Pose
        fp = self.r.follow_plan(guard_mm=1e9)
        if not fp.ok:
            return {"ok": False, "why": f"a pálya nem járható: {fp.detail}"}
        moved_on_path = fp.data.get("product_moved_mm", float("nan"))

        # RÖGZÍTETT végállapotig zárunk — nincs erőküszöbös megállás.
        for i in range(1, ramp + 1):
            self.g.close_fingers(close_level * i / ramp, settle=ramp_settle)
        st = self.settle_to_steady()

        dg, parts, F = self.contacts()
        med = len([x for x in parts if "medial" in x])

        p0 = self.d.geom_xpos[self.gid].copy()
        h0 = self.g.grasp_point().copy()
        for _ in range(4):
            tgt = Pose("lift", self.g.grasp_point() + np.array([0, 0, lift_mm / 4000.0]),
                       self.r._R_des)
            self.r.approach_until(tgt, until="goal", guard_mm=1e9)
        rise = float(self.d.geom_xpos[self.gid][2] - p0[2]) * 1000
        hand_rise = float(self.g.grasp_point()[2] - h0[2]) * 1000
        dg2, parts2, F2 = self.contacts()
        follow = rise / hand_rise if abs(hand_rise) > 1.0 else 0.0
        return {
            "ok": True, "close": close_level, "digits": sorted(dg),
            "medial": med, "force_N": F, "path_moved_mm": moved_on_path,
            "converged": st["converged"], "settle_steps": st["steps"],
            "hand_rise_mm": hand_rise, "product_rise_mm": rise,
            "follow": follow, "digits_after": sorted(dg2),
            "pass": (st["converged"] and follow > 0.8 and F < FORCE_LIMIT_N
                     and len(dg) >= NEED_DIGITS and med >= NEED_MEDIAL),
        }


def _fresh() -> GripRig:
    """Friss robot, a karral a FOGÁSI PÓZBAN.

    ⚠️ A HITELESÍTÉS MÁSODIK FUTÁSA EZT FOGTA MEG. Az első változat
    alaphelyzetből indult — ott viszont a kéz a TEST MELLETT LÓG, és egy
    58 × 145 mm-es hengernek egyszerűen nincs szabad helye a tenyér
    környékén, egyik irányban sem. A behelyezés ezért mindig meghiúsult.
    Nem a kéz volt szűk: rossz pózban mértem.

    A próbát ott kell futtatni, ahol a fogás történik — a kart tehát a
    névleges fogási pózba visszük, mielőtt bármit mérnénk.
    """
    r = Robot()
    r.reset_home()
    g = r._r
    box, _ = g.product_box()
    R = G.GRASP_POSES[(g.plan or {}).get("pose", "right_thumb_up")]
    q, ep, er = g.ik6_seed(box, R, restarts=16, iters=110)
    g.ramp_to(q, n=18, settle=120)
    return GripRig(r)


# ═══════════════════════════════════════════════════════════════════════════
# HITELESÍTÉS
# ═══════════════════════════════════════════════════════════════════════════

def selftest() -> int:
    """Ismert JÓ és ismert ROSSZ eset. Ha ezek rosszul jönnek ki, a mérés
    nem használható — akkor a hiba a mérőeszközben van, nem a kézben."""
    print("Fogáspróba — HITELESÍTÉS\n")
    ok = True

    # ── 1. ISMERT JÓ: a tárgyat VÉGIG rögzítve tartjuk. Kell, hogy „tartsa".
    print("  [1] ismert JÓ — a tárgy végig rögzítve (nem eshet le)")
    rig = _fresh()
    pos = rig.seat_in_hand()
    if pos is None:
        print("      ❌ a behelyezés meghiúsult"); return 1
    q = rig.d.qpos[rig.adr:rig.adr + 3].copy()
    qq = rig.d.qpos[rig.adr + 3:rig.adr + 7].copy()
    p0 = rig.d.geom_xpos[rig.gid].copy()
    for _ in range(RELEASE_STEPS):
        rig.d.qpos[rig.adr:rig.adr + 3] = q
        rig.d.qpos[rig.adr + 3:rig.adr + 7] = qq
        rig.d.qvel[rig.vadr:rig.vadr + 6] = 0
        rig.g.step(1)
    slip = float(np.linalg.norm(rig.d.geom_xpos[rig.gid] - p0)) * 1000
    good = slip < 1.0
    print(f"      csúszás {slip:.2f} mm  {'✅' if good else '❌ (kellene < 1 mm)'}")
    ok &= good

    # ── 2. ISMERT ROSSZ: a tárgy a kéztől MESSZE, elengedve. Le kell esnie.
    print("  [2] ismert ROSSZ — a tárgy a kéztől 30 cm-re, elengedve")
    rig = _fresh()
    palm = rig.d.xpos[rig.g._palm].copy()
    rig.place(palm + np.array([0.0, 0.0, 0.30]))
    p0 = rig.d.geom_xpos[rig.gid].copy()
    rig.g.step(RELEASE_STEPS)
    slip = float(np.linalg.norm(rig.d.geom_xpos[rig.gid] - p0)) * 1000
    dg, parts, F = rig.contacts()
    bad = slip > 100.0 and len(dg) == 0
    print(f"      csúszás {slip:.0f} mm · {len(dg)} ujj · {F:.1f} N  "
          f"{'✅' if bad else '❌ (kellene: nagy csúszás, 0 ujj)'}")
    ok &= bad

    # ── 3. A PADLÓ-SZŰRŐ: leesett tárgynál az ERŐ legyen 0 (ez volt az 1. hiba)
    print("  [3] padló-szűrő — a leesett tárgy padló-ütközése NEM számít bele")
    rig.g.step(500)
    dg, parts, F = rig.contacts()
    filt = F == 0.0 and len(parts) == 0
    print(f"      kéz-oldali erő {F:.1f} N · {len(parts)} perec  "
          f"{'✅' if filt else '❌ (kellene 0 N — a padló nem kéz)'}")
    ok &= filt

    print(f"\n  {'✅ A MÉRŐESZKÖZ HITELES' if ok else '❌ A MÉRŐESZKÖZ NEM MEGBÍZHATÓ'}")
    if not ok:
        print("     Amíg ez nem megy át, a fogási eredményeknek nem hiszünk.")
        return 1
    print("\n  Az ISMÉTELHETŐSÉGET külön kell futtatni:  --repeat")
    return 0


# ── ismételhetőség ──────────────────────────────────────────────────────────

REPEAT_RAMP_SETTLE = (5, 10, 20, 40)
SPREAD_FORCE_N = 10.0        # megengedett szórás a kontaktuserőben
SPREAD_FOLLOW = 0.15         # megengedett szórás a követésben


def repeat_test() -> int:
    """Ugyanaz a beállítás, KÜLÖNBÖZŐ felfutási időzítéssel.

    ⚠️ EZ A TESZT AZÉRT LÉTEZIK, mert 2026-08-05-én a mérés a beállási időtől
    17,5–102,3 N-t és −106%…+32% követést adott — UGYANARRA a paraméterre.
    Egy mérőeszköz, ami így viselkedik, nem használható rangsorolásra, még ha
    a három korábbi hitelesítési esetet át is engedi.

    A kritérium ELŐRE kimondva: a felfutási időzítés végigsöprésén az erő
    szórása maradjon 10 N alatt, a követésé 15 százalékpont alatt, és MINDEN
    futás konvergáljon.
    """
    print("Fogáspróba — ISMÉTELHETŐSÉG\n")
    print("  Ugyanaz a konfiguráció, csak a felfutás időzítése változik.")
    print("  Ha az eredmény ettől függ, a mérés nem használható.\n")
    print(f"  {'ramp_settle':>12}{'konv.':>8}{'lépés':>8}{'ujj':>5}"
          f"{'közép':>7}{'erő':>9}{'követés':>10}")
    print("  " + "─" * 59)

    Fs, Ls, all_conv = [], [], True
    for rs in REPEAT_RAMP_SETTLE:
        r = Robot(); r.reset_home()
        res = GripRig(r).run_approach(ramp_settle=rs)
        if not res["ok"]:
            print(f"  {rs:12d}  {res['why']}")
            return 1
        all_conv &= bool(res["converged"])
        Fs.append(res["force_N"]); Ls.append(res["follow"])
        print(f"  {rs:12d}{'✅' if res['converged'] else '❌':>7}"
              f"{res['settle_steps']:8d}{len(res['digits']):5d}"
              f"{res['medial']:7d}{res['force_N']:8.1f} N"
              f"{res['follow']*100:9.0f}%")

    dF = max(Fs) - min(Fs)
    dL = max(Ls) - min(Ls)
    good = all_conv and dF < SPREAD_FORCE_N and dL < SPREAD_FOLLOW
    print(f"\n  erőszórás {dF:.1f} N (< {SPREAD_FORCE_N:.0f} kell) · "
          f"követésszórás {dL*100:.0f} pp (< {SPREAD_FOLLOW*100:.0f} kell) · "
          f"konvergencia {'mind' if all_conv else 'NEM mind'}")
    if good:
        print("\n  ✅ ISMÉTELHETŐ — a változatok összehasonlíthatók.")
        return 0
    print("\n  ❌ NEM ISMÉTELHETŐ. A paraméter-változatok rangsorolása "
          "ÉRTELMETLEN,\n     amíg ez nem megy át. Ez jelentés, nem hiba.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--approach", action="store_true",
                    help="megközelítéses protokoll (a valódi fogás modellje)")
    ap.add_argument("--repeat", action="store_true",
                    help="ismételhetőség: ugyanaz a beállítás több időzítéssel")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.repeat:
        return repeat_test()
    if a.approach:
        print("Fogáspróba — MEGKÖZELÍTÉSES protokoll\n")
        r = Robot(); r.reset_home()
        res = GripRig(r).run_approach()
        if not res["ok"]:
            print(f"  ❌ {res['why']}"); return 1
        print(f"  pálya alatt a termék {res['path_moved_mm']:.1f} mm-t mozdult")
        print(f"  beállás: {res['settle_steps']} lépés · "
              f"{'✅ konvergált' if res['converged'] else '❌ NEM konvergált'}")
        print(f"  zárás {res['close']:.2f} · {len(res['digits'])} ujj "
              f"{res['digits']} · {res['medial']} középperec · {res['force_N']:.1f} N")
        print(f"  EMELÉS: kéz {res['hand_rise_mm']:+.1f} mm · "
              f"termék {res['product_rise_mm']:+.1f} mm · "
              f"követés {res['follow']*100:.0f}%")
        print(f"\n  kritérium (követés>80%% · erő<{FORCE_LIMIT_N:.0f} N · "
              f"≥{NEED_DIGITS} ujj · ≥{NEED_MEDIAL} középperec): "
              f"{'✅ ÁTMENT' if res['pass'] else '❌ nem'}")
        return 0

    print("Fogáspróba — a JELENLEGI kéz (alapállapot)\n")
    res = _fresh().run(verbose=True)
    if not res["ok"]:
        print(f"  ❌ {res['why']}")
        return 1
    print(f"  zárás {res['close']:.2f} · {len(res['digits'])} ujj {res['digits']}"
          f" · {res['medial']} középperec")
    print(f"  erő {res['force_N']:.1f} N · csúszás {res['slip_mm']:.1f} mm"
          f" · utána {len(res['digits_after'])} ujj")
    print(f"\n  kritérium (csúszás<{SLIP_LIMIT_MM:.0f} mm · erő<{FORCE_LIMIT_N:.0f} N"
          f" · ≥{NEED_DIGITS} ujj · ≥{NEED_MEDIAL} középperec): "
          f"{'✅ ÁTMENT' if res['pass'] else '❌ nem'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
