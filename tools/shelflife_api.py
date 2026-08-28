"""
shelflife_api.py — A PRIMITÍV SZÓTÁR (M1). Ezt hívja az ügynök, és semmi mást.

    from shelflife_api import Robot
    r = Robot(sku="alpro_barista_coconut_1l")
    r.approach_until(r.preset("pre_grasp"), until="goal")
    r.approach_until(r.preset("grasp"),     until="contact")
    r.close_until(until="grip")
    r.approach_until(r.preset("lift", height=0.10), until="goal")

════════════════════════════════════════════════════════════════════════════
A SZÓTÁR HÁROM SZABÁLYA
════════════════════════════════════════════════════════════════════════════

**1. Minden mozgásnak van leállási feltétele.**
   A „menj a (x,y,z) pontba" típusú primitív hazudik: fogásnál nem tudjuk
   előre, hány milliméterre van a felület. A Waddle szótárának központi eleme
   ezért az `approach_until(waypoints, until=...)`, és ez a hiányzó darab volt
   nálunk is.

**2. Minden hívás megmondja, MIÉRT állt meg.**
   A néma sikertelenség a legrosszabb kimenet: az ügynök nem tud belőle
   újratervezni. Minden `Result` tartalmaz `reason`-t, akkor is, ha sikerült.

**3. Az ügynök nem lát lineáris algebrát.**
   Se forgatásmátrix, se Jacobian, se ízületvektor. A célpózokat a `preset()`
   adja, névvel. Ez nem kényelmi kérdés: az Anthropic *Claude Plays Robotics*
   mérése szerint az interfész megválasztása 6%-ról 32%-ra vitte a sikert,
   és a mi legdrágább hibánk is interfész-hiba volt (az `r_wrist_3` origója
   15,3 cm-rel a bütykök előtt van — bármi, ami azt tenyérnek nézi, 17 cm-t
   téved).

════════════════════════════════════════════════════════════════════════════
AMIT AZ ÜGYNÖK NEM LÁT
════════════════════════════════════════════════════════════════════════════
A `grasp_plan.json` (fogási pont, orientáció, közelítési irány, zárási szint)
és az SKU dátummező-adatai az SKU-ADATBÁZIS részei. Az ügynök annyit lát,
hogy „ehhez a termékhez van bejegyzett fogástechnika", és hivatkozni tud rá
névvel. Ez szándékos: ez a védőárok, és az eval nem mérheti újra epizódonként.

Ground truth (a valódi dátum, a helyes döntés) egyáltalán NEM elérhető innen.

════════════════════════════════════════════════════════════════════════════
BEFAGYASZTVA — D1, 2026-08-04
════════════════════════════════════════════════════════════════════════════
Ettől a ponttól a szótár menet közben nem módosul. Ha az ügynöknek hiányzik
valami, azt JELENTÉSKÉNT rögzítjük (mit akart, mit nem tudott), és külön
döntéssel nyúlunk hozzá. Minden ilyen eset önmagában eredmény: pont ez a
„milyen interfészt szeretnek az ügynökök" kérdés.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import shelflife_render_env  # noqa: F401,E402  — Xvfb, ha kell
import mujoco                                      # noqa: E402
import shelflife_grasp as _G                       # noqa: E402
from shelflife_motion import approach_until as _approach_until  # noqa: E402
from shelflife_motion import close_until as _close_until        # noqa: E402
from shelflife_motion import product_support as _product_support  # noqa: E402
from shelflife_motion import _grip_ok as _is_grip                 # noqa: E402

# ── MÉRT időállandók (M0) ───────────────────────────────────────────────────
# A törzs bevonása után a beállás lassabb, mint a puszta karnál:
#     500 lépés → 166 mm hiba · 1000 → 6.6 mm · 2000 → 1.8 mm
# A primitív rétegben örökölt SETTLE_STEPS=250 a KARRA volt kalibrálva.
SETTLE_LARGE = 2000        # nagy ízület-ugrás után
SETTLE_STEP = 60           # egy servo-lépés (~1.5 mm) után

STOP_REASONS = ("goal", "contact", "grip", "support", "force",
                "guard", "limit", "timeout")


# ═══════════════════════════════════════════════════════════════════════════
# Visszatérési típusok
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    """Minden primitív ezt adja vissza. A `reason` MINDIG ki van töltve."""
    ok: bool
    reason: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        mark = "ok" if self.ok else "NEM"
        return f"<{mark}: {self.reason}{(' — ' + self.detail) if self.detail else ''}>"


@dataclass
class Pose:
    """Célpóz. Az ügynök NEM állítja elő kézzel — a `preset()` adja."""
    name: str
    xyz: np.ndarray
    R: np.ndarray

    def __repr__(self) -> str:
        return f"<Pose {self.name} @ {np.round(self.xyz, 3).tolist()}>"

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Pose":
        """Eltolt változat — ennyi lineáris algebrát az ügynök kaphat."""
        return Pose(f"{self.name}+off", self.xyz + np.array([dx, dy, dz]), self.R)

    def turned(self, degrees: float, about: str = "up") -> "Pose":
        """Elforgatott változat — VILÁGTENGELY körül, fokban.

        about: 'up'      — függőleges tengely (a termék megpörgetése)
               'forward' — előre-hátra tengely (megdöntés oldalra)
               'right'   — bal-jobb tengely (előre-hátra billentés)

        MIÉRT KELL: a dátum a doboz TETEJÉN van, elölről nem látszik. Enélkül
        az ügynök nem tudja megnézni — és a feladat épp ez lenne. A szótár
        első változatából ez hiányzott: `approach_until` csak POZÍCIÓT mozgat,
        rögzített orientációval.
        """
        ax = {"up": np.array([0.0, 0.0, 1.0]),
              "forward": np.array([1.0, 0.0, 0.0]),
              "right": np.array([0.0, 1.0, 0.0])}.get(about)
        if ax is None:
            raise ValueError(f"about: 'up' | 'forward' | 'right', nem {about!r}")
        th = np.radians(degrees)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rw = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        return Pose(f"{self.name}+{degrees:+.0f}°{about}", self.xyz.copy(), Rw @ self.R)


@dataclass
class Observation:
    """Egyetlen struktúra az érzékelt állapotról."""
    hand_xyz: np.ndarray
    product_xyz: np.ndarray
    product_size_cm: np.ndarray
    touching: list[str]
    holding: bool
    supported_by: list[str]

    def __repr__(self) -> str:
        return (f"<Obs kéz={np.round(self.hand_xyz, 3).tolist()} "
                f"termék={np.round(self.product_xyz, 3).tolist()} "
                f"érintés={self.touching or '—'} "
                f"alátámasztás={self.supported_by or '—'} "
                f"{'FOGJA' if self.holding else 'nem fogja'}>")


# ═══════════════════════════════════════════════════════════════════════════
# A robot
# ═══════════════════════════════════════════════════════════════════════════

class Robot:
    """A GENE.01 az ügynök szemszögéből.

    Minden metódus `Result`-ot vagy egyszerű adatot ad vissza. MuJoCo-objektum,
    ízületvektor, forgatásmátrix nem szivárog ki.
    """

    def __init__(self, sku: str = "alpro_barista_coconut_1l", seed: int = 0):
        self._r = _G.GraspRobot(seed=seed)
        self._sku = sku
        self._plan = self._r.plan
        if not self._plan:
            raise RuntimeError(
                "nincs fogási terv ehhez az SKU-hoz — a terméktudásbázis "
                "hiányos. Futtasd: python3 tools/shelflife_grasp_plan.py")
        self._R_des = _G.GRASP_POSES[self._plan["pose"]]
        ax, sg = self._plan["approach_palm_axis"]
        self._approach_dir = sg * self._R_des[:, ax]

        # ── FIGYELMEZTETÉS A HIÁNYZÓ MÉRÉSEKRŐL ─────────────────────────────
        # Ez nem naplózás, hanem szándékos zaj. A hiányzó fizikai mérések
        # listája eddig JEGYZETBEN élt, a jegyzet viszont nem állít meg egy
        # futást: a szimuláció vígan dolgozott egy beírt súrlódással, amit a
        # kéz alapértelmezése felül is írt, és semmi nem szólt.
        # Amíg a teherhordó mezők nincsenek megmérve, minden futás kiírja.
        try:
            from shelflife_sku_audit import one_line
            msg = one_line((self._plan or {}).get("sku", ""))
            if msg:
                print(msg)
        except Exception:                                  # noqa: BLE001
            pass                     # az audit hiánya SOHA ne törjön el futást

        self.reset_home()

    # ── életciklus ──────────────────────────────────────────────────────────

    def reset_home(self) -> Result:
        """Jelenet és robot alaphelyzetbe, kéz nyitva."""
        self._r.reset()
        self._r.close_fingers(0.0, settle=200)
        return Result(True, "goal", "alaphelyzet",
                      {"hand": self._r.grasp_point().tolist()})

    # ── észlelés ────────────────────────────────────────────────────────────

    def observe(self) -> Observation:
        """Az aktuális állapot egyetlen struktúrában."""
        box, half = self._r.product_box()
        parts = sorted(self._r.contact_parts())
        return Observation(
            hand_xyz=self._r.grasp_point(),
            product_xyz=box,
            product_size_cm=np.round(half * 200, 1),
            touching=parts,
            # EGY forrásból: ugyanaz a kritérium, amit a `close_until` használ.
            # Korábban itt saját másolat volt („hüvelyk + három ujj"), és a
            # `close_until` már `grip`-et jelentett, miközben az `observe()`
            # még azt mondta, hogy nem fogja. Két definíció ugyanarra a
            # fogalomra — pont az a fajta csendes ellentmondás, amit a szótár
            # 2. szabálya tilt.
            holding=_is_grip(set(parts)),
            supported_by=sorted(_product_support(self._r)),
        )

    def view(self, camera: str = "rgb_camera", res: int = 640):
        """Kamerakép a VLM-nek.

        FIGYELEM: OpenGL kell hozzá. A fejlesztői sandboxban NINCS — ilyenkor
        beszédes hibát dobunk, mert a néma fekete kép sokkal rosszabb lenne.
        """
        return self._r.render_view(camera=camera, res=res)

    def in_view(self, point: Sequence[float], camera: str = "rgb_camera") -> Result:
        """Látszik-e a pont a kamerából — GEOMETRIAILAG, render nélkül.

        Olcsó kapu a ReAct-hurokban: mielőtt az ügynök képet kérne és VLM-et
        hívna, megnézheti, hogy egyáltalán benne van-e a látótérben.
        """
        r = self._r.in_view(point, camera)
        return Result(bool(r), "goal" if r else "limit", r.detail, r.data)

    def can_see_date(self) -> Result:
        """Látszik-e MOST a dátummező a mellkasi kamerából.

        A dátummező helye az SKU-adatbázisból jön (2. pillér), nem képi
        kereséssel. Ha nem látszik, a terméket forgatni/emelni kell — és pont
        ez a szemantikus mag feladata.
        """
        p = self._date_point_world()
        if p is None:
            return Result(False, "limit", "ehhez az SKU-hoz nincs dátummező-adat")
        return self.in_view(p)

    # ── célpózok ────────────────────────────────────────────────────────────

    def preset(self, name: str, height: float = 0.10) -> Pose:
        """Névvel hivatkozott célpóz. Az ügynök NEM számol koordinátát.

        name:
          'pre_grasp'  — fogás előtti pont, a bejegyzett közelítési irány mentén
          'grasp'      — a bejegyzett fogási pont a terméken
          'lift'       — az aktuális kézpozíció + `height` felfelé
          'inspect'    — a termék a mellkasi kamera elé emelve
          'shelf_out'  — a polc síkja elé, szabad térbe
          'aside'      — félrerakási hely a polc előtt (JELÖLNI / KIVONNI)
        """
        box, _ = self._r.product_box()
        if name == "grasp":
            return Pose(name, box.copy(), self._R_des)
        if name == "pre_grasp":
            so = self._plan["waypoints"][0]["standoff_cm"] / 100.0
            return Pose(name, box - self._approach_dir * so, self._R_des)
        if name == "lift":
            # A preset GARANTÁLJA az elérhetőséget: ha a kért emelés az
            # ízülethatárra vinne, a legnagyobb még biztonságos magasságot
            # adja vissza. MÉRVE: 10 cm-es emelés a fogási pontból 0.05 rad
            # tartalékot hagy — onnan a zárt hurok nem tud korrigálni.
            g = self._r.grasp_point()
            for h in (height, 0.08, 0.06, 0.04):
                p = Pose(name, g + np.array([0.0, 0.0, h]), self._R_des)
                if self._reachable(p):
                    return p
            return Pose(name, g + np.array([0.0, 0.0, 0.04]), self._R_des)
        if name == "aside":
            # Félrerakási hely: UGYANAZON A POLCON, oldalra tolva. Boltban is
            # így megy: a kivont/jelölt árut a polc egy elkülönített részére
            # teszik. NEM az aktuális kézmagasságból számoljuk — a `lift`-nél
            # már belefutottunk abba, hogy lógó karral értelmetlen pózt ad.
            for dy in (-0.16, -0.20, -0.12, -0.24):
                p = Pose(name, box + np.array([0.0, dy, 0.0]), self._R_des)
                if self._reachable(p):
                    return p
            return Pose(name, box + np.array([0.0, -0.16, 0.0]), self._R_des)
        if name == "inspect":
            pos, fwd = self._r.camera_pose("rgb_camera")
            return Pose(name, pos + fwd * 0.18, self._R_des)
        if name == "shelf_out":
            g = self._r.grasp_point()
            for x in (0.26, 0.24, 0.22, 0.20):
                p = Pose(name, np.array([x, g[1], g[2]]), self._R_des)
                if self._reachable(p):
                    return p
            return Pose(name, np.array([0.24, g[1], g[2]]), self._R_des)
        raise ValueError(f"nincs ilyen preset: {name} (pre_grasp | grasp | "
                         f"lift | inspect | shelf_out | aside)")

    # ── mozgás ──────────────────────────────────────────────────────────────

    def place_until(self, target: Pose, guard_mm: float = 1e9,
                    verbose: bool = False) -> Result:
        """Letétel: ereszkedés, amíg a termék valami MÁSHOZ nem ér.

        A kéz–termék kontaktus erre nem elég: a kéz akkor is fogja, amikor még
        a levegőben van. A leállási feltétel ezért a termék ALÁTÁMASZTÁSA.
        Utána az `open_hand()` engedi el — a kettő szándékosan külön van, hogy
        az ügynök ellenőrizhessen közben.
        """
        return self.approach_until(target, until="support",
                                   guard_mm=guard_mm, verbose=verbose)

    def approach_until(self, target: Pose,
                       until: str | Callable = "goal",
                       guard_mm: float = 4.0,
                       verbose: bool = False) -> Result:
        """Egyenes vonalú, FOLYTONOS mozgás a célpóz felé, feltételig.

        until : 'goal'    — végigmegy a szakaszon
                'contact' — az első kéz–termék érintkezésig
                'grip'    — amíg a hüvelyk és legalább három ujj is fog
                'support' — amíg a termék valami máshoz ér (letétel)
        guard_mm : ha a termék ennyinél többet mozdul, azonnal megáll

        Miért folytonos: diszkrét pályapontokat ízület-interpolációval összekötve
        a termék MINDEN ponton érintetlen volt, mégis 141 mm-t mozdult — a kár
        a pontok KÖZÖTT keletkezett.
        """
        if not isinstance(target, Pose):
            raise TypeError("a célt a preset() adja, nem nyers koordináta")
        out = _approach_until(self._r, target.xyz, target.R, until=until,
                              guard_mm=guard_mm, verbose=verbose)
        reason = {"kontaktus": "contact", "fogás": "grip",
                  "alátámasztás": "support", "goal": "goal"}.get(
            out["reason"],
            "timeout" if out["reason"].startswith("timeout") else "guard")
        return Result(reason in ("goal", "contact", "grip", "support"), reason,
                      f"{out['travelled_mm']:.0f} mm megtéve, "
                      f"termék {out['product_moved_mm']:.1f} mm", out)

    def follow_plan(self, guard_mm: float = 4.0,
                    verbose: bool = False) -> Result:
        """A BEJEGYZETT közelítési pálya bejárása a fogási pontig.

        ────────────────────────────────────────────────────────────────────
        MIÉRT KELLETT EZ AZ IGE — D1 SZERINT JELENTVE
        ────────────────────────────────────────────────────────────────────
        A `grasp_plan.json` tartalmaz egy ELLENŐRZÖTT pályát (`waypoints`):
        a tervező minden 1 cm-es állomáson igazolta, hogy elérhető,
        ütközésmentes és van ízülettartaléka. A szótár viszont **nem használta**
        — az `approach_until` egyenes vonalban ment a célra, akárhogy is
        kanyarodott az igazolt út. A régi (M1 előtti) `shelflife_grasp.py`
        bejárta a pályát; az M1-es újraírás ezt elejtette.

        Mérve, a szerkesztett tervvel (felülről közelítés):
            egyenes vonalon:   a termék a pre-grasp pontig **19 mm**-t mozdul,
                               záráskor NULLA kontaktus
            a terv pályáján:   ezt méri ez az ige

        Ez tehát nem új képesség, hanem egy MEGLÉVŐ, de elszakadt darab
        visszakötése. A szótár-befagyasztás (D1) óta a második bejelentett
        bővítés.

        ⚠️ A PÁLYÁT NEM SZABAD NYÍLT HUROKBAN LEJÁTSZANI. A tervező
        kinematikában dolgozik, a robot gravitációval fut; mérve 6–7 cm az
        eltérés a parancsolt és a beállt póz között. Ezért minden állomáson
        zárt hurokkal ráhúzunk a TERVEZETT pontra, mielőtt továbbmennénk —
        és a pontok KÖZÖTT is finoman vezetünk át, mert a kár ott keletkezik.
        """
        way = (self._plan or {}).get("waypoints")
        if not way:
            return Result(False, "limit",
                          "ehhez az SKU-hoz nincs bejegyzett pálya")
        box, _ = self._r.product_box()
        p0 = self._r.product_pose().copy()
        moved = lambda: float(np.linalg.norm(
            self._r.product_pose() - p0)) * 1000

        self._r.ramp_to(np.array(way[0]["q"]), n=18, settle=70)
        worst = moved()
        for i, wp in enumerate(way):
            if i:
                self._r.ramp_to(np.array(wp["q"]), n=10, settle=60)
            xyz = box - self._approach_dir * (wp["standoff_cm"] / 100.0)
            self._r.move_grasp_to(xyz, self._R_des, pos_tol=0.006,
                                  rot_tol=0.09,
                                  max_iters=(6 if i == 0 else 4),
                                  slices=(8 if i == 0 else 1))
            worst = max(worst, moved())
            if worst > guard_mm:
                return Result(False, "guard",
                              f"{wp['standoff_cm']:.0f} cm-nél a termék "
                              f"{worst:.1f} mm-t mozdult",
                              {"standoff_cm": wp["standoff_cm"],
                               "product_moved_mm": worst})
            if verbose:
                print(f"      {wp['standoff_cm']:5.1f} cm · termék "
                      f"{moved():5.1f} mm")
        return Result(True, "goal",
                      f"{len(way)} pályapont bejárva, termék "
                      f"legfeljebb {worst:.1f} mm",
                      {"points": len(way), "product_moved_mm": worst})

    def close_until(self, until: str = "grip", verbose: bool = False) -> Result:
        """Ujjzárás feltételig — nem „amennyire csak lehet".

        Teljes záráskor a hüvelyk ÁTHALAD a terméken és a négy ujj mellé kerül:
        nyolc kontaktus, mind ugyanazon az oldalon, nulla szembefogás. Nem az
        számít, milyen erősen zárunk, hanem hogy mikor állunk meg.
        """
        out = _close_until(self._r, until=until, verbose=verbose)
        ok = out["reason"] == "fogás"
        # ⚠️ D2-visszacsatolás (1/2): új leállási ok, `force`. A D1
        # befagyasztás óta ez az ELSŐ interfész-változás, és a terv szerint
        # jelentendő: a `close_until` erőkorlátra is megállhat, mert a
        # pozíció-aktuátorok kontaktusban 51.6 N-t is kifejtettek.
        reason = ("grip" if ok else
                  "force" if out["reason"] == "erőkorlát" else "guard")
        return Result(ok, reason,
                      f"{out['contacts']} kontaktus {out['parts']}, "
                      f"{out['force_N']:.1f} N", out)

    def open_hand(self) -> Result:
        self._r.close_fingers(0.0, settle=200)
        return Result(True, "goal", "kéz nyitva")

    def look_at(self, point: Sequence[float]) -> Result:
        r = self._r.look_at(point)
        return Result(True, "goal", r.detail)

    # ── SKU-tudás (lekérdezhető, de nem az ügynöké) ─────────────────────────

    def sku_info(self) -> dict:
        """Amit az ügynök az SKU-ról TUDHAT — ground truth nélkül."""
        df = self._sku_record().get("date_field", {})
        return {
            "sku": self._sku,
            "has_grasp_plan": True,
            "date_field_known": bool(df),
            "date_location": df.get("location_human"),
            "date_type": df.get("type"),              # use_by / best_before
            "date_format": df.get("format"),
            "decision_rules": self._sku_record().get("decision_rules", {}),
        }

    def reachable(self, pose: Pose) -> Result:
        """Elérhető-e a póz — az ORIENTÁCIÓVAL együtt, mozgás nélkül.

        Olcsó kapu: az ügynök megkérdezheti, mielőtt nekiindulna. A pozíció
        önmagában kevés; a fogási orientációt a kar csak szűk tartományban
        tudja tartani, és az ízülethatáron ülő póz dinamikailag használhatatlan
        (a gravitációs korrekció nem fér bele).
        """
        q, ep, er = self._r.ik6_seed(pose.xyz, pose.R, restarts=10, iters=80)
        mg = self._r.joint_margin(q)
        ok = ep * 1000 < 8 and np.degrees(er) < 4 and mg > 0.15
        return Result(ok, "goal" if ok else "limit",
                      f"IK {ep*1000:.1f} mm / {np.degrees(er):.1f}° · "
                      f"ízülettartalék {mg:.2f} rad",
                      {"ik_mm": ep * 1000, "ik_deg": float(np.degrees(er)),
                       "joint_margin_rad": float(mg)})

    # ── belső ───────────────────────────────────────────────────────────────

    def _reachable(self, pose: Pose) -> bool:
        return bool(self.reachable(pose))

    def _sku_record(self) -> dict:
        import json
        p = (_REPO / "src/envs/assets/shelflife_sku_private" / f"{self._sku}.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def _date_point_world(self) -> Optional[np.ndarray]:
        gid = mujoco.mj_name2id(self._r.model, mujoco.mjtObj.mjOBJ_GEOM,
                                "product_0_date")
        return None if gid < 0 else self._r.data.geom_xpos[gid].copy()


__all__ = ["Robot", "Result", "Pose", "Observation", "STOP_REASONS"]
