"""
shelflife_grasp.py — FOGÁSI réteg a GENE.01-hez (a primitívek kiterjesztése)

    python3 tools/shelflife_grasp.py                 # pózjelöltek végigmérése
    python3 tools/shelflife_grasp.py --pose front_thumb_up --verbose

────────────────────────────────────────────────────────────────────────────
MIÉRT KÜLÖN FÁJL
────────────────────────────────────────────────────────────────────────────
A `shelflife_primitives.py` meglévő fájl, és a munkaszabály szerint meglévőhöz
engedély nélkül nem nyúlunk. Ez a modul ezért LESZÁRMAZTAT (`GraspRobot`),
nem módosít. Ha a fogás beválik, külön kérésre kerülhet be a primitív-rétegbe.

────────────────────────────────────────────────────────────────────────────
HÁROM MÉRT HIBA, AMIT EZ A MODUL JAVÍT
────────────────────────────────────────────────────────────────────────────

1. A „TENYÉR" NEM A TENYÉR — 17 cm-t tévedtünk.
   A primitív-réteg `move_palm_to()`-ja az `r_wrist_3` TEST ORIGÓJÁT viszi a
   célra. Mérve viszont az ujjbegyek a tenyér frame-jében −20…−23 cm-en
   vannak z mentén:

       flex=0  ujjbegyek:  index [ 3.1,  1.3, −22.7] cm
                           hüvelyk [ 8.3,  9.0, −12.6] cm

   Vagyis amikor a `move_palm_to()` „4.9 mm-re a célnál" sikert jelentett, a
   KEZE 20 cm-re volt a terméktől. Innen semmilyen ujjzárás nem érhetett
   célt — ez a magyarázata a korábbi „grasp lefut, de touching_product()
   False" jelenségnek.
   → `grasp_point()`: a fogási középpont a tenyér frame-jében, MÉRVE.
   → `move_grasp_to()`: ezt a pontot viszi a célra, nem a csuklót.

2. NEM VOLT ORIENTÁCIÓ-VEZÉRLÉS.
   Az IK csak pozícióra oldott, a 7-DoF kar 4 redundáns szabadságfokát a
   rácskeresés véletlenszerűen töltötte ki. Mérve a közelítés végén:

       tenyér tengelyei: x=[0.5 0.49 0.71]  y=[−0.75 0.66 0.08]  z=[−0.43 −0.58 0.7]

   Se nem vízszintes, se nem a termék felé néz — a kéz ferdén lefelé állt.
   Fogáshoz az orientáció nem díszítés: eldönti, hogy az ujjak a tárgy KÖRÉ
   zárnak-e vagy mellé.
   → `move_grasp_to()` 6-DoF zárt hurok: pozíció + orientáció együtt,
     csillapított legkisebb négyzetekkel.

3. A HÜVELYKUJJ MÁSODIK ÍZÜLETE SOHA NEM ZÁRT.
   Az `r_thumb_2` ízület tartománya **−1.75…0.00** — nála a hajlítás a
   NEGATÍV irány. A primitív `grasp()` viszont minden `_2`/`_3` végű
   aktuátorra +1.05-öt ír, amit az `inheritrange` 0.00-ra vág. A hüvelykujj
   így fél kézzel zárt.
   → `close_fingers()` az ízület tartományának ELŐJELÉBŐL határozza meg a
     zárási irányt.

────────────────────────────────────────────────────────────────────────────
A MÓDSZER: NEM KITALÁLJUK A JÓ FOGÁST, HANEM VÉGIGMÉRJÜK
────────────────────────────────────────────────────────────────────────────
Több geometriailag értelmes fogási orientáció létezik (elölről tenyérrel,
oldalról, hüvelyk fel / le). Melyik működik, az a kéz tényleges kinematikájától
függ, amit nem ismerünk elég jól ahhoz, hogy megtippeljük — az 1. és 2. pont
épp azt mutatja, mennyire nem.

Ezért a modul PÓZJELÖLTEKET definiál, mindegyiket végigfuttatja, és méri:
kontaktpontok száma, a termék elmozdulása, megtartás emelés közben.
Ez a kimenet megy az SKU-bejegyzés `grasp.recommended` mezőjébe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                     # noqa: E402
from shelflife_primitives import (ShelfLifeRobot, StepResult,  # noqa: E402
                                 FINGER_TOKENS, SETTLE_STEPS, ARM_JOINTS)

# A vezérlési lánc: törzs + kar.
#
# MIÉRT NEM CSAK A KAR: a fogási pózok mind az ízülethatárokon ültek (legjobb
# tartalék 0.14 rad), így a zárt hurok nem tudta korrigálni a gravitációs
# megereszkedést. A törzs bevonásával a lánc 8-DoF lesz, és a pózhoz marad
# mozgástér. Ember is elfordul, ha oldalt nyúl a polcra.
CHAIN_JOINTS = ["torso_yaw"] + ARM_JOINTS

SKU_SCENE = _REPO / "src/envs/assets/shelflife_scene_gene01_sku_v1.xml"


def _plan_path_from_scene() -> Path:
    """A fogási terv útja — a JELENETBEN lévő SKU-hoz, nem fixen az elsőhöz.

    ⚠️ M6-EREDMÉNY, 2026-08-05. Ez a konstans korábban BE VOLT DRÓTOZVA az
    1. SKU-ra:

        PLAN_PATH = .../models/shelflife_sku/alpro_barista_coconut_1l/...

    A 2. SKU (Coca-Cola doboz) felvételekor derült ki: a jelenet a dobozt
    töltötte be, a futásidejű kód viszont a KARTON tervét olvasta hozzá.
    Csendben, hibaüzenet nélkül — a legrosszabb fajta.

    Ez pontosan az, amit az M6 transzfer-tesztnek mérnie kell: „ugyanaz a
    program, új termékkel, kódmódosítás nélkül". A válasz NEM volt — két
    helyen kötötte a kód magát az első SKU-hoz (itt és a jelenetépítő
    ütköző-alakjánál). Mindkettő általánosítás volt, nem új képesség, de
    jelenteni kell, nem elhallgatni.

    A forrás mostantól a jelenet mellé generált `meta.json` `sku_id` mezője —
    egy adat, egy forrás.
    """
    meta = SKU_SCENE.with_suffix(".meta.json")
    if meta.exists():
        import json
        sku = json.loads(meta.read_text()).get("sku_id")
        if sku:
            return _REPO / "models/shelflife_sku" / sku / "grasp_plan.json"
    return _REPO / "models/shelflife_sku/_ismeretlen/grasp_plan.json"


PLAN_PATH = _plan_path_from_scene()


def load_plan() -> dict | None:
    """A `shelflife_grasp_plan.py` által kikeresett fogási terv.

    A futásidejű kód NEM keres újra: a terv geometriát, utat és elérhetőséget
    EGYÜTT szűrve állt elő, és fájlban van, hogy az eval reprodukálható és a
    döntés auditálható legyen.
    """
    import json
    if PLAN_PATH.exists():
        return json.loads(PLAN_PATH.read_text())
    return None

TIP_BODIES = ["r_thumb_distal", "r_index_distal", "r_middle_distal",
              "r_ring_distal", "r_little_distal"]

# A fogási középpont meghatározásához használt ujjhajlítás. Félig zárt kéznél
# a hüvelyk–középsőujj nyílás 9.2 cm, a hüvelyk–mutató 6.6 cm (mérve) — épp a
# 7.9 × 8.0 cm-es karton mérete körül. Teljesen nyitott vagy zárt kézből
# számolva a középpont a tárgyon kívülre esne.
PROBE_FLEX = 0.50

# ── EGYIDEJŰ ÉRINTKEZÉS: ujjankénti fáziskésleltetés ────────────────────────
#
# A fogás azért nem zárt be, mert AMELYIK UJJ ELŐSZÖR HOZZÁÉR, AZ ELTOLJA a
# kartont — semmi nem áll vele szemben. Nem az erő kevés és nem a geometria
# rossz: az érkezések nincsenek szinkronban.
#
# MÉRVE (a kart a fogási pózba állítva, a terméket RÖGZÍTVE, kinematikailag):
# melyik ujj milyen zárási szinten éri el először a kartont —
#
#     hüvelyk 0.13 · mutató 0.28 · középső 0.31 · gyűrűs 0.28 · kisujj 0.28
#
# A hüvelyk 0.15-tel korábban ér oda, mint a többi. Egyenletes záráskor tehát
# egyedül nyomja a kartont ~0.15 szélességű szakaszon — mérve ez 6.6 mm
# elcsúszást okozott, mire a többi ujj odaért, és onnan már kicsúszott.
#
# A késleltetést nem tippeljük: a mért érintkezési szintekből számoljuk úgy,
# hogy MINDEGYIK a SYNC_AT szinten érjen oda.   d = (SYNC_AT − c) / (1 − c)
#
# ⚠️ Ezek SKU-FÜGGŐ értékek (más méretű terméknél mások) — a helyük hosszú
# távon a `grasp_plan.json`-ban van, a fogási terv részeként.
SYNC_AT = 0.60
#
# ⚠️ 2026-08-04: MIND NULLA. A késleltetés arra volt válasz, hogy a digitusok
# 46–73 mm-es, ERŐSEN ELTÉRŐ távolságról indultak. A fogási pont
# újraillesztése után (l. `GRASP_TWEAK_CM`) a rések 6.2 / 7.1 mm-esek és
# kiegyenlítettek — az egyidejűséget a GEOMETRIA adja, nem az ütemezés.
#
# Szoftveres késleltetéssel kompenzálni egy rossz fogási pontot: tünetkezelés.
DIGIT_DELAY = {"thumb": 0.54, "index": 0.44, "middle": 0.42,
               "ring": 0.44, "little": 0.44}

# Korrekció az ujjbegy-centroidhoz képest, a tenyér frame-jében, cm.
#
# MIÉRT NEM ELÉG A CENTROID: a félig zárt kéz ujjbegyeinek középpontja a
# MÁR BEZÁRT térfogat közepe, ami kisebb, mint a 8 cm-es karton. Ha oda
# visszük a termék középpontját, a kéz belemetsz — mérve 11 kontaktus és
# 33 mm behatolás már NYITOTT kézzel is, a fogás előtt.
#
# A helyes offsetet nem tippeltük, hanem KIMÉRTÜK: a kart a névleges pózba
# állítva végigtoltuk a kartont a tenyér frame-jének egy rácsán, és minden
# helyzetben megnéztük a VALÓDI MuJoCo-ütközést. Három feltétel egyszerre:
#   (a) NYITOTT kézzel ütközésmentes a célpozícióban,
#   (b) záráskor a HÜVELYK ÉS legalább három ujj is fog (szembefogás),
#   (c) a 13 cm-es közelítési út VÉGIG ütközésmentes.
# 180 ilyen kombináció van; mindegyik 0.3-as zárásnál mind az öt ujjal fog.
# Ez egyben cáfolja a korábbi feltevésemet, hogy a kéz kicsi lenne.
GRASP_TWEAK_CM = np.array([-4.0, 2.0, -2.0])
#
# ⚠️ FIGYELEM: ez csak TARTALÉK érték. Ha van `grasp_plan.json`, a hatályos
# eltolás ONNAN jön (`tweak_cm`) — l. `_measure_grasp_offset()`. A modul
# konstansának átírása ilyenkor NEM HAT, és ebbe bele is futottunk: a fogási
# pont újraillesztése után semmi nem változott, mert a terv felülírta.
#
# ⚠️ ÚJRAILLESZTVE (M2, `tools/shelflife_grasp_point_fit.py`).
# A korábbi [-4, 2, -2] érték geometriai szerkesztésből jött, és a KÉZ
# NYÍLÁSÁNAK közepére tette a kartont, nem a fogáséba. Mérve, a zárás
# indulásakor a rések: hüvelyk 46 mm, ujjak 69 mm. A doboz egy nála jóval
# nagyobb üregben lebegett, minden ujj 5–7 cm-t vett lendületet, és a
# legközelebbi (a hüvelyk) ért oda elsőnek — az lökte el.
#
# Az új érték kritériuma a HELYES kérdés: nyitott kézben a hüvelyk és a négy
# ujj legyen a doboz SZEMKÖZTI oldalán, KÖZEL és KIEGYENLÍTETTEN.
#     rések 6.4 / 6.5 mm  ·  oppozíciós szög 150°
# (korábban: 46 / 69 mm, a kontaktusok merőlegesek egymásra)

# ── A kéz két póza, ízületnév-végződés szerint (rad) ────────────────────────
#
# NYITOTT: a hüvelyk maximálisan kifordítva és széttárva — MÉRVE ez adja a
# legnagyobb hüvelyk–mutató nyílást (10.7 cm a semleges 7.7 helyett), plusz
# az ujjak enyhe széttárása.
# ZÁRT: teljes hajlítás minden ízületen, a hüvelyknél a HELYES előjellel
# (`r_thumb_2` tartománya −1.75…0.00).
HAND_OPEN = {
    "r_thumb_1_rot": 0.70, "r_thumb_1_add": -0.52, "r_thumb_1_flex": -0.79,
    "r_thumb_2": 0.0, "r_thumb_3": 0.0,
    "r_index_1_add": 0.35, "r_ring_1_add": -0.20, "r_little_1_add": -0.35,
}
HAND_CLOSED = {
    "r_thumb_1_rot": 0.70, "r_thumb_1_add": 0.52, "r_thumb_1_flex": 1.31,
    "r_thumb_2": -1.75, "r_thumb_3": 1.75,
    "r_index_1_add": 0.0, "r_ring_1_add": 0.0, "r_little_1_add": 0.0,
    "r_index_1_flex": 1.38, "r_index_2": 1.75, "r_index_3": 1.75,
    "r_middle_1_flex": 1.38, "r_middle_2": 1.75, "r_middle_3": 1.75,
    "r_ring_1_flex": 1.38, "r_ring_2": 1.75, "r_ring_3": 1.75,
    "r_little_1_flex": 1.38, "r_little_2": 1.75, "r_little_3": 1.75,
}

ROT_WEIGHT = 0.12       # m/rad — 1 rad orientációhiba ~12 cm pozícióhibát ér
IK6_DAMPING = 0.06
IK6_ITERS = 30
IK6_POS_TOL = 0.006     # 6 mm
IK6_ROT_TOL = 0.12      # rad (~7°)


# ═══════════════════════════════════════════════════════════════════════════
# Fogási orientációk
# ═══════════════════════════════════════════════════════════════════════════
#
# A tenyér lokális frame-je, MÉRVE (l. a fájl fejlécében):
#   +x_palm : a kisujjtól a hüvelyk felé      (hüvelyk x=+8.3, kisujj x=−7.9)
#   +y_palm : a tenyér BELSŐ felülete kifelé  (záráskor az ujjak +y-ba jönnek)
#   −z_palm : az ujjak nyúlási iránya         (ujjbegyek z=−20…−23)
#
# Egy erőfogásnál a tárgy hossztengelye az +x_palm-mal párhuzamos (az ujjak
# e körül a tengely körül kunkorodnak), a tenyér belső fele (+y_palm) pedig a
# tárgy középpontja felé néz.
#
# A karton hossztengelye FÜGGŐLEGES (20.4 cm), tehát x_palm-nak függőlegesnek
# kell lennie. Ez adja a „hüvelyk fel" és „hüvelyk le" változatot; a maradék
# szabadság az, hogy melyik lapra kerül a tenyér.

def _frame(x, y):
    """Ortonormált forgatásmátrix két tengelyből (oszlopok = x,y,z)."""
    x = np.asarray(x, float); x /= np.linalg.norm(x)
    y = np.asarray(y, float); y -= (y @ x) * x; y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


GRASP_POSES = {
    # tenyér a termék ELÜLSŐ lapján (a robot felőli oldal), hüvelyk FEL
    "front_thumb_up":   _frame([0, 0, 1], [1, 0, 0]),
    # ugyanaz, hüvelyk LE
    "front_thumb_down": _frame([0, 0, -1], [1, 0, 0]),
    # tenyér a termék JOBB oldalán (a robot jobbja, −y), hüvelyk FEL
    "right_thumb_up":   _frame([0, 0, 1], [0, 1, 0]),
    # tenyér a termék BAL oldalán (+y, testközép felől), hüvelyk FEL
    "left_thumb_up":    _frame([0, 0, 1], [0, -1, 0]),

    # ── HENGERES TERMÉKHEZ (2. SKU, 2026-08-05) ────────────────────────────
    # A fentiek mind a TEJESDOBOZHOZ készültek: álló téglatest, ahol az ujjak
    # függőlegesen simulnak a lapra. Hengeren ez átlós fekvést ad — mérve a
    # hüvelyk a doboz peremén (146 mm), a kisujj a talp közelében (17 mm),
    # a kontaktusok 130 mm-en szétszórva. Ilyenkor a szorítás ELFORGAT.
    #
    # A felhasználó leírása és fotói szerint (data/cola_private/IMG_7379–80)
    # emberi fogásnál a HÜVELYK a MUTATÓVAL (és a középsővel) van szemben,
    # AZONOS magasságban; a gyűrűs és a kisujj csak kísér.
    #
    # Söpréssel mérve (tools/shelflife_pose_cylinder.py): a bütyöksort a
    # függőlegesből −45°-ba forgatva
    #     hüvelyk vs. mutató+középső:  76 mm → 17 mm
    #     ízülettartalék:              0.20 → 0.42 rad
    # A tenyér Z tengelye változatlanul −x, tehát a közelítés iránya ugyanaz.
    "cyl_side_-45": _frame([0.0, -0.7071, 0.7071], [0.0, 0.7071, 0.7071]),
    "cyl_side_-30": _frame([0.0, np.float64(0.5), np.float64(0.866)], [0.0, np.float64(0.866), np.float64(-0.5)]),
    "cyl_side_-60": _frame([0.0, np.float64(0.866), np.float64(0.5)], [0.0, np.float64(0.5), np.float64(-0.866)]),
    "cyl_side_-10": _frame([0.0, np.float64(0.1736), np.float64(0.9848)], [0.0, np.float64(0.9848), np.float64(-0.1736)]),
    "cyl_side_-15": _frame([0.0, np.float64(0.2588), np.float64(0.9659)], [0.0, np.float64(0.9659), np.float64(-0.2588)]),
    "cyl_side_-20": _frame([0.0, np.float64(0.342), np.float64(0.9397)], [0.0, np.float64(0.9397), np.float64(-0.342)]),
    "cyl_side_-25": _frame([0.0, np.float64(0.4226), np.float64(0.9063)], [0.0, np.float64(0.9063), np.float64(-0.4226)]),
    "cyl_side_-75": _frame([0.0, np.float64(0.9659), np.float64(0.2588)], [0.0, np.float64(0.2588), np.float64(-0.9659)]),
}

# A közelítési irány pózonként: a tenyér normálisával szemben állunk be.
# (a pre-grasp pont = fogási cél − közelítési irány × standoff)


class GraspRobot(ShelfLifeRobot):
    """Fogásra képes GENE.01. A primitívek maradnak, ez csak hozzátesz."""

    def __init__(self, scene: Path | None = None, seed: int = 0):
        super().__init__(scene=scene or SKU_SCENE, seed=seed)
        m = self.model
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self._tips = [bid(n) for n in TIP_BODIES]

        # ── ujj-ízületek: melyik aktuátor merre zár ──────────────────────────
        # Az r_thumb_2 tartománya −1.75…0.00 → nála a zárás a NEGATÍV irány.
        # Ezt az ízület tartományából olvassuk ki, nem a névből.
        self._fing = []           # (act_id, closed_value, joint_name, is_flex)
        self._pose = {}           # act_id -> (nyitott, zárt) parancsérték
        self._delay = {}          # act_id -> fáziskésleltetés (egyidejű érkezés)
        for i in range(m.nu):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if not (nm.startswith("act_r_") and any(t in nm for t in FINGER_TOKENS)):
                continue
            jname = nm[4:]                          # "act_" levágva
            jid = m.actuator_trnid[i, 0]
            lo, hi = m.jnt_range[jid]
            o = float(np.clip(HAND_OPEN.get(jname, 0.0), lo, hi))
            c = float(np.clip(HAND_CLOSED.get(jname, 0.0), lo, hi))
            self._pose[i] = (o, c)
            self._delay[i] = next((v for k, v in DIGIT_DELAY.items()
                                   if k in jname), 0.0)
            self._fing.append((i, c, jname, ("flex" in jname
                                             or jname.endswith(("_2", "_3")))))

        # a vezérlési lánc kiterjesztése a törzzsel (ha a jelenetben aktív)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        chain = [n for n in CHAIN_JOINTS if aid(f"act_{n}") >= 0]
        self.chain = chain
        self._arm_q = [m.jnt_qposadr[jid(n)] for n in chain]
        self._arm_v = [m.jnt_dofadr[jid(n)] for n in chain]
        self._arm_a = [aid(f"act_{n}") for n in chain]
        self._arm_range = np.array([m.jnt_range[jid(n)] for n in chain])
        self._cmd = np.zeros(len(chain))

        self.plan = load_plan()
        self._grasp_offset = self._measure_grasp_offset()

    def reset(self, seed: int | None = None) -> StepResult:
        """Az ősosztály `_cmd`-t 7 elemre állítja; nálunk a lánc hosszabb."""
        out = super().reset(seed)
        if hasattr(self, "chain"):
            self._cmd = np.zeros(len(self.chain))
        return out

    def _phase(self, act: int, amount: float) -> float:
        """Egy aktuátor tényleges zárási fázisa. EGY forrás — l. lentebb."""
        d = self._delay[act]
        a = float(np.clip(amount, 0.0, 1.0))
        return 0.0 if a <= d else min(1.0, (a - d) / (1.0 - d))

    def set_hand_qpos(self, data, amount: float, phased: bool = True) -> None:
        """A kéz ízületeit KÖZVETLENÜL állítja be (kinematikai vizsgálatokhoz).

        ⚠️ UGYANAZT a fázisleképezést használja, mint a `close_fingers()`.
        Korábban nem: ez lineárisan interpolált, a `close_fingers()` viszont
        ujjankénti késleltetéssel. Emiatt a fogáspont-illesztés 0.35-ös
        zárásnál öt ujjas szembefogást JÓSOLT, a valódi futásban viszont
        0.35-nél még egyetlen ízület sem mozdult — nulla kontaktus.
        Egy fogalom, egy definíció.
        """
        a = float(np.clip(amount, 0.0, 1.0))
        for act, (o, c) in self._pose.items():
            jid = self.model.actuator_trnid[act, 0]
            ph = self._phase(act, a) if phased else a
            data.qpos[self.model.jnt_qposadr[jid]] = o + (c - o) * ph

    # ── a fogási középpont bemérése ─────────────────────────────────────────

    def _measure_grasp_offset(self, flex: float = PROBE_FLEX) -> np.ndarray:
        """A fogási középpont a TENYÉR frame-jében, méterben — mérve.

        Nem konstans: a hüvelyk és a mutató-/középsőujj begyének felezőpontja
        félig zárt kéznél. Ez az a pont, ahol egy megfogott tárgy középpontja
        ténylegesen van.
        """
        s = self._scratch
        mujoco.mj_resetData(self.model, s)
        # `phased=False`: a fogási KÖZÉPPONT geometriai szerkesztés, nem
        # időbeli lefutás — itt a nyers, késleltetés nélküli félig zárt kéz kell.
        self.set_hand_qpos(s, flex, phased=False)
        mujoco.mj_forward(self.model, s)
        P = s.xpos[self._palm]
        R = s.xmat[self._palm].reshape(3, 3)
        loc = {n: R.T @ (s.xpos[b] - P) for n, b in zip(TIP_BODIES, self._tips)}
        mid = (loc["r_thumb_distal"] + loc["r_index_distal"]
               + loc["r_middle_distal"]) / 3.0
        tweak = (np.array(self.plan["tweak_cm"], float)
                 if getattr(self, "plan", None) else GRASP_TWEAK_CM)
        return mid + tweak / 100.0

    def grasp_point(self) -> np.ndarray:
        """A fogási középpont VILÁGKOORDINÁTÁBAN, az aktuális állapotban."""
        P = self.data.xpos[self._palm]
        R = self.data.xmat[self._palm].reshape(3, 3)
        return P + R @ self._grasp_offset

    def palm_R(self) -> np.ndarray:
        return self.data.xmat[self._palm].reshape(3, 3).copy()

    def product_box(self, idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """A termék ÜTKÖZÉSI DOBOZÁNAK középpontja és félméretei (világ).

        MIÉRT NEM a `product_pose()`: az a TEST FRAME origóját adja, ami a
        szkennelt hálónál nem a tárgy közepe. Mérve: a test origója
        z=1.054-nél van, a doboz közepe z=1.164-nél — 11 cm-rel feljebb,
        vagyis a test origója gyakorlatilag a karton ALJÁN. Erre fogni annyi,
        mint a doboz alsó pereme alá nyúlni.

        ⚠️ ALAKFÜGGŐ, ÉS EZ EGY NAPIG REJTVE MARADT.
        A `geom_size` jelentése alakonként MÁS:

            box       (hx, hy, hz)         → a félméret maga
            cylinder  (r, félmagasság, —)  → a 3. elem HASZNÁLATLAN (0)
            capsule   (r, félhossz, —)     → ua.
            sphere    (r, —, —)            → ua.

        Az első változat mindig a nyers `geom_size`-t adta vissza. A karton
        (box) esetén ez helyes volt, a 2. SKU-nál (henger) viszont a
        FÉLMAGASSÁG **0**-nak látszott: a jelenet szerint a doboz 145 mm
        magas, a fogástervező szerint 0 mm. Minden, ami a termék tetejét
        vagy alját számolta, a KÖZEPÉT kapta.

        Ugyanaz a hibaosztály, mint a jelenetépítő „csak box" ütközőjénél és
        a bedrótozott `PLAN_PATH`-nál: az 1. SKU implicit alapértelmezés lett.
        """
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"product_{idx}_col")
        s = self.model.geom_size[gid].copy()
        t = int(self.model.geom_type[gid])
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([s[0], s[0], s[1]])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.array([s[0], s[0], s[1] + s[0]])
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.array([s[0], s[0], s[0]])
        else:                                     # box — változatlan
            half = s
        return self.data.geom_xpos[gid].copy(), half

    # ── kinematikai kiindulópont a 6-DoF célhoz ─────────────────────────────

    def joint_margin(self, q: np.ndarray) -> float:
        """A legszűkebb ízülethatár-tartalék (rad) egy kar-konfigurációban."""
        return float(np.min(np.minimum(q - self._arm_range[:, 0],
                                       self._arm_range[:, 1] - q)))

    def ik6_seed(self, target: np.ndarray, R_des: np.ndarray,
                 restarts: int = 24, iters: int = 120,
                 feas_m: float = 0.004, feas_rad: float = 0.05
                 ) -> tuple[np.ndarray, float, float]:
        """Kar-konfiguráció a 6-DoF célhoz, TISZTA KINEMATIKÁVAL (scratch).

        Miért kell: a zárt hurok lokálisan javít, de ha rossz medencéből
        indul, 30 iteráció alatt sem talál oda (mérve: 74 mm / 80° maradék).
        Több véletlen indítás + csillapított legkisebb négyzetek megtalálja a
        jó medencét, és a zárt hurok onnan már csak a gravitációs
        megereszkedést korrigálja.

        ⚠️ AMIÉRT NEM ELÉG A NULLA IK-HIBA — ez buktatta el a fogást:
        a keresés talált 0.00 mm / 0.00° megoldást, a zárt hurok mégis
        33 mm-en megállt. Az ok: a megoldás egy ÍZÜLETHATÁRON ült
        (tartalék 0.050 rad = pontosan a beállított margó). A kar
        gravitációs megereszkedését a hurok úgy korrigálná, hogy TÚLMEGY a
        határon — de nem tud, tehát a hiba beragad.

        Kinematikailag tökéletes póz tehát dinamikailag használhatatlan, ha
        nincs benne mozgástér a korrekciónak. Ezért a keresés a megengedett
        pontosságon BELÜL a legnagyobb ízülettartalékot választja, és a
        Jacobian null-terében a tartomány közepe felé húz.
        """
        s = self._scratch
        lo, hi = self._arm_range[:, 0], self._arm_range[:, 1]
        qadr, vadr = np.array(self._arm_q), np.array(self._arm_v)
        q_des = np.zeros(4); q_cur = np.zeros(4)
        q_err = np.zeros(4); q_inv = np.zeros(4); w = np.zeros(3)
        mujoco.mju_mat2Quat(q_des, np.ascontiguousarray(R_des.flatten()))
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))

        q_mid = self._arm_range.mean(axis=1)
        best = ((1, 1e9), None, 1e9, 1e9)
        for t in range(restarts):
            mujoco.mj_resetData(self.model, s)
            q = (self.rng.uniform(lo, hi) if t else self._cmd.copy())
            for _ in range(iters):
                s.qpos[qadr] = q
                mujoco.mj_kinematics(self.model, s)
                mujoco.mj_comPos(self.model, s)
                P = s.xpos[self._palm]; R = s.xmat[self._palm].reshape(3, 3)
                gp = P + R @ self._grasp_offset
                e_pos = target - gp
                mujoco.mju_mat2Quat(q_cur, np.ascontiguousarray(R.flatten()))
                mujoco.mju_negQuat(q_inv, q_cur)
                mujoco.mju_mulQuat(q_err, q_des, q_inv)
                mujoco.mju_quat2Vel(w, q_err, 1.0)
                ep, er = float(np.linalg.norm(e_pos)), float(np.linalg.norm(w))
                # Kétszintű pontozás: amíg nem elég pontos, a hibát
                # minimalizáljuk; amint elég pontos, az ÍZÜLETTARTALÉKOT
                # maximalizáljuk (l. a docstring figyelmeztetését).
                feasible = ep < feas_m and er < feas_rad
                score = ((0, -self.joint_margin(q)) if feasible
                         else (1, ep + ROT_WEIGHT * er))
                if score < best[0]:
                    best = (score, q.copy(), ep, er)
                mujoco.mj_jac(self.model, s, jacp, jacr,
                              np.ascontiguousarray(gp), self._palm)
                J = np.vstack([jacp[:, vadr], ROT_WEIGHT * jacr[:, vadr]])
                e6 = np.concatenate([e_pos, ROT_WEIGHT * w])
                JJt = J @ J.T + (IK6_DAMPING ** 2) * np.eye(6)
                Jpinv = J.T @ np.linalg.inv(JJt)
                dq = Jpinv @ e6
                # null-tér: a feladatot nem rontva húzzuk a tartomány közepe felé
                dq += (np.eye(len(q)) - Jpinv @ J) @ (0.10 * (q_mid - q))
                q = np.clip(q + np.clip(dq, -0.2, 0.2), lo + 0.05, hi - 0.05)
        return best[1], best[2], best[3]

    # ── 6-DoF zárt hurkú pozicionálás ───────────────────────────────────────

    def move_grasp_to(self, target: np.ndarray, R_des: np.ndarray,
                      pos_tol: float = IK6_POS_TOL,
                      rot_tol: float = IK6_ROT_TOL,
                      max_iters: int = IK6_ITERS,
                      slices: int = 1) -> StepResult:
        """A FOGÁSI PONTOT viszi a célra, adott tenyér-orientációval.

        Ugyanaz a zárt hurok, mint a `move_palm_to()`-ban (parancs → beállás →
        mérés → korrekció), két különbséggel:
          · a mért pont a fogási középpont, nem a csukló origója
          · a hiba 6 dimenziós: pozíció + orientáció, súlyozva
        """
        target = np.asarray(target, float)
        if slices > 1:
            start = self.grasp_point().copy()
            for k in range(1, slices):
                wp = start + (target - start) * (k / slices)
                self.move_grasp_to(wp, R_des, pos_tol=0.010, rot_tol=0.20,
                                   max_iters=5, slices=1)
            return self.move_grasp_to(target, R_des, pos_tol=pos_tol,
                                      rot_tol=rot_tol, max_iters=max_iters,
                                      slices=1)

        hist = []
        q_des = np.zeros(4); q_cur = np.zeros(4)
        q_err = np.zeros(4); q_inv = np.zeros(4); w = np.zeros(3)
        mujoco.mju_mat2Quat(q_des, np.ascontiguousarray(R_des.flatten()))

        for _ in range(max_iters):
            for a, v in zip(self._arm_a, self._cmd):
                self.data.ctrl[a] = v
            self.step(SETTLE_STEPS)

            gp = self.grasp_point()
            e_pos = target - gp
            R = self.palm_R()
            mujoco.mju_mat2Quat(q_cur, np.ascontiguousarray(R.flatten()))
            mujoco.mju_negQuat(q_inv, q_cur)
            mujoco.mju_mulQuat(q_err, q_des, q_inv)
            mujoco.mju_quat2Vel(w, q_err, 1.0)      # világ-frame forgásvektor

            ep, er = float(np.linalg.norm(e_pos)), float(np.linalg.norm(w))
            hist.append((ep, er))
            if ep < pos_tol and er < rot_tol:
                return StepResult(True, f"6DoF cél elérve: {ep*1000:.1f} mm / "
                                        f"{np.degrees(er):.1f}°",
                                  {"err_mm": ep * 1000, "err_deg": np.degrees(er),
                                   "iters": len(hist)})

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr,
                          np.ascontiguousarray(gp), self._palm)
            J = np.vstack([jacp[:, self._arm_v], ROT_WEIGHT * jacr[:, self._arm_v]])
            e6 = np.concatenate([e_pos, ROT_WEIGHT * w])
            JJt = J @ J.T + (IK6_DAMPING ** 2) * np.eye(6)
            dq = np.clip(J.T @ np.linalg.solve(JJt, e6), -0.25, 0.25)
            self._cmd = np.clip(self._cmd + dq,
                                self._arm_range[:, 0] + 0.05,
                                self._arm_range[:, 1] - 0.05)

        ep, er = hist[-1]
        return StepResult(False, f"6DoF nem konvergált: {ep*1000:.1f} mm / "
                                 f"{np.degrees(er):.1f}°",
                          {"err_mm": ep * 1000, "err_deg": np.degrees(er),
                           "iters": len(hist)})

    def ramp_to(self, q: np.ndarray, n: int = 14, settle: int = 60) -> None:
        """Lassú átvezetés egy másik kar-konfigurációba.

        MIÉRT KELL: az `ik6_seed()` egy egészen más ízület-medencét adhat, mint
        ahol a kar éppen áll. Ha a parancsot egy lépésben átírjuk, a kar
        ÍVBEN, nagy sebességgel megy át — és mérve LELÖKI a terméket
        (255 mm elmozdulás, vagyis a karton a földön volt, mielőtt az ujjak
        egyáltalán mozdultak). A célpóz jó volt; az odajutás nem.

        Ugyanaz a lecke, mint a Cartesian szeletelésnél: nem a végpont a
        veszélyes, hanem az út.
        """
        q0 = self._cmd.copy()
        for k in range(1, n + 1):
            self._cmd = q0 + (np.asarray(q, float) - q0) * (k / n)
            for a, v in zip(self._arm_a, self._cmd):
                self.data.ctrl[a] = v
            self.step(settle)

    # ── ujjak ───────────────────────────────────────────────────────────────

    def close_fingers(self, amount: float, settle: int = 200) -> None:
        """Kézforma 0…1: NYITOTT → ZÁRT, teljes póz-interpolációval.

        MIÉRT NEM ELÉG A FLEX-ÍZÜLETEKET HAJLÍTANI: a hüvelyk `rot` és `add`
        ízülete nem „flex", a korábbi heurisztika nullán hagyta őket — pedig
        ezek nyitják ki a hüvelyket oppozícióba. Ezért a kézforma két MÉRT
        pózból interpolálódik, nem ízületenkénti előjelszabályból.

        A KÉZ MÉRETE (mérve, mert egyszer rosszul állítottam):
            ujjhossz (bütyök→begy)   8.2 cm
            hüvelyk                 11.0 cm
            tenyérszélesség          9.0 cm
            hüvelyk–mutató nyílás   18.2 cm nyitva, 5.6 cm félig zárva
        Ez teljesen ember méretű kéz, egy 8 cm-es tejesdoboz bőven belefér.
        (Korábban 7.7 cm-es „nyílást" írtam ide — az HIBA volt: a két ujjbegy
        távolságát csak EGY tengelyre vetítve mértem, nem térben.)
        """
        a = float(np.clip(amount, 0.0, 1.0))
        # ── FÁZISOS ZÁRÁS: előbb az ujjak, aztán a hüvelyk ──────────────────
        #
        # MÉRVE, egyenletes záráskor (mind a 21 ízület ugyanazon a 0→1 ütemen):
        #     0.15 → a HÜVELYK megérinti a kartont   (1.2 N)
        #     0.20 → kontaktus elveszve, termék 6.6 mm
        #     0.30 → 3 ujj fog, termék 13.5 mm
        #     0.45 → kicsúszott, termék 17 mm
        #
        # A hüvelyk az OPPOZÍCIÓS ujj: alapból közelebb áll a tárgyhoz, ezért
        # egyenletes ütem mellett ELŐBB ér oda, és kilöki a kartont, mielőtt a
        # négy ujj körbeérne. Nem az erő volt kevés és nem a geometria rossz —
        # a SORREND volt rossz.
        #
        # Ember is így fog: az ujjak körbefonják a tárgyat, a hüvelyk zár utoljára.
        for act, (o, c) in self._pose.items():
            self.data.ctrl[act] = o + (c - o) * self._phase(act, a)
        self.step(settle)

    def contact_count(self, idx: int = 0) -> tuple[int, float]:
        """(kéz–termék kontaktpontok száma, összes normálerő N)."""
        pb = self._products[idx]
        n, f = 0, 0.0
        buf = np.zeros(6)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if y != pb:
                    continue
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, x) or ""
                if nm.startswith("r_") and any(
                        t in nm for t in FINGER_TOKENS + ("wrist", "hand")):
                    mujoco.mj_contactForce(self.model, self.data, c, buf)
                    n += 1
                    f += abs(float(buf[0]))
        return n, f

    def contact_parts(self, idx: int = 0) -> set[str]:
        """Melyik UJJAK érintkeznek a termékkel ({'thumb','index',...})."""
        pb = self._products[idx]
        out: set[str] = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if y != pb:
                    continue
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, x) or ""
                for t in FINGER_TOKENS:
                    if t in nm:
                        out.add(t)
        return out

    def squeeze(self, steps: int = 16, start: float = 0.0,
                stop: float = 1.0, extra: float = 0.08) -> dict:
        """Fokozatos zárás — és MEGÁLLÁS, amint szembefogás jött létre.

        MIÉRT NEM ZÁRUNK VÉGIG:
        teljes záráskor (amount=1.0) a hüvelykujj ÁTHALAD a termék felett és a
        négy ujj mellé kerül — mérve: 8 kontaktus, MIND a karton ugyanazon
        oldalán, hüvelyk sehol. Ez nem fogás, hanem tolás: a termék 63 mm-t
        elcsúszott, majd kiesett a kézből.

        Részleges zárásnál (~0.6) viszont mind az öt ujj érintkezik, a hüvelyk
        a SZEMKÖZTI oldalon — ez az erőzárás. A fogás tehát nem az „amennyire
        csak lehet" záráson múlik, hanem azon, hogy MIKOR ÁLLUNK MEG.

        Ezért a zárás kontaktus-vezérelt: amint a hüvelyk és legalább három
        ujj is fog, még egy kicsit rászorítunk, és ott megállunk.
        """
        p0 = self.product_pose().copy()
        grip_at = None
        for k in range(steps + 1):
            a = start + (stop - start) * k / steps
            self.close_fingers(a, settle=120)
            p = self.contact_parts()
            if "thumb" in p and len(p - {"thumb"}) >= 3:
                grip_at = a
                break
        if grip_at is not None:
            self.close_fingers(min(1.0, grip_at + extra), settle=250)
        n, f = self.contact_count()
        moved = float(np.linalg.norm(self.product_pose() - p0))
        return {"amount": grip_at, "final_n": n, "final_f": f,
                "parts": sorted(self.contact_parts()),
                "moved_mm": moved * 1000}


# ═══════════════════════════════════════════════════════════════════════════
# Egy fogási kísérlet
# ═══════════════════════════════════════════════════════════════════════════

def try_pose(pose: str, standoff: float = 0.06, verbose: bool = False,
             z_offset: float = 0.0, approach: str = 'above') -> dict:
    R_des = GRASP_POSES[pose]
    r = GraspRobot()
    prod0 = r.product_pose().copy()
    box_c, box_h = r.product_box()
    out = {"pose": pose}

    if verbose:
        print(f"    fogási középpont a tenyér frame-jében: "
              f"{np.round(r._grasp_offset*100, 1)} cm")
        print(f"    a karton doboza: közép {np.round(box_c,3)} "
              f"félméret {np.round(box_h*100,1)} cm")

    # 1. NEM hívunk `ready_pose()`-t.
    #
    #    MÉRVE: a primitív `ready_pose()` a szintetikus demó dobozra volt
    #    kalibrálva (8×6×12 cm, teteje z=1.18). Az Alpro karton 20.4 cm magas,
    #    a teteje z=1.27 — 9 cm-rel feljebb. A ready pózba menő ív ezt a
    #    magasságot súrolja, és a termék 33 mm-t elmozdul, MIELŐTT bármi
    #    fogásszerű történne. A pózt tehát nem lehet átvenni: a magasabb
    #    termékhez másik köztes póz kell.
    #
    #    Helyette a lógó alaphelyzetből közvetlenül a fogási orientációba
    #    vezetünk át, a polcon KÍVÜL (x ≤ 0.26 < polc eleje 0.34).
    r.close_fingers(0.15)                    # enyhén nyitott kéz

    # 2. pre-grasp — a KERESETT terv szerint.
    #
    #    A közelítési irányt sem tippeljük többé. A `shelflife_grasp_plan.py`
    #    három feltételt EGYSZERRE szűrt (geometria, út, elérhetőség), mert
    #    külön-külön optimalizálva mindig elromlott a másik kettő valamelyike.
    #    A nyertes: ALULRÓL, 0.35-ös zárással, 5 ujj kontaktussal, és a teljes
    #    12 cm-es közelítési úton 0.0 mm elérhetőségi hibával.
    if r.plan:
        ax, sg = r.plan["approach_palm_axis"]
        approach_dir = sg * R_des[:, ax]
        close_to = r.plan["close_amount"]
    else:
        approach_dir = -R_des[:, 0] if approach == "above" else -R_des[:, 2]
        close_to = 1.0
    target = box_c + np.array([0.0, 0.0, z_offset])
    pre = target - approach_dir * standoff
    if verbose and r.plan:
        print(f"    terv      : {r.plan['approach']} · zárás "
              f"{close_to:.2f} · {r.plan['digits_in_contact']} ujj")

    if r.plan and r.plan.get("waypoints"):
        # A TERV PÁLYÁJÁN haladunk végig, nem a végpontra ugrunk.
        way = r.plan["waypoints"]
        out["seed_mm"], out["seed_deg"] = r.plan["reach_err_mm"], r.plan["reach_err_deg"]
        out["standoff_m"] = way[0]["standoff_cm"] / 100.0
        if verbose:
            print(f"    pálya     : {len(way)} pont, "
                  f"{way[0]['standoff_cm']:.0f} → {way[-1]['standoff_cm']:.0f} cm")
        # ⚠️ A PÁLYÁT NEM SZABAD NYÍLT HUROKBAN LEJÁTSZANI.
        #
        # A tervező KINEMATIKÁBAN dolgozik (`mj_kinematics`), a robot viszont
        # gravitációval fut. Mérve: a parancsolt pálya és a ténylegesen
        # bejárt pálya között **6–7 cm** eltérés van — a kéz y=−0.06…−0.13
        # között haladt a tervezett y=−0.20 helyett. A tervben ellenőrzött
        # 1–2 mm-es tartalékok ekkora megereszkedés mellett semmit nem érnek:
        # a termék 11 cm-nél 68 mm-t ugrott.
        #
        # Ez pontosan a GR1T1-lecke egy szinttel feljebb: **a beállt pózra kell
        # tervezni, nem a parancsoltra.** Ezért minden pályaponton zárt hurokkal
        # ráhúzunk a TERVEZETT pontra, mielőtt továbbmennénk.
        r.ramp_to(np.array(way[0]["q"]), n=18, settle=70)
        m_ramp = float(np.linalg.norm(r.product_pose() - prod0)) * 1000
        worst = m_ramp
        for i, wp in enumerate(way):
            if i:
                # finom átvezetés: a pontok KÖZÖTTI ív a veszélyes, nem a
                # pontok (mérve: n=3-mal 169 mm-t söpört a kéz úgy, hogy
                # egyetlen pályaponton sem volt kontaktus)
                r.ramp_to(np.array(wp["q"]), n=10, settle=60)
            wp_xyz = target - approach_dir * (wp["standoff_cm"] / 100.0)
            # Az ELSŐ ponton a legnagyobb a kezdeti hiba (mérve 61 mm), és
            # ott söpörte le a kéz a terméket. A javítás nem több iteráció,
            # hanem EGYENES VONAL: Cartesian szeletelve a korrekció nem ívben
            # halad. (Távolabbról kezdeni nem lehet: 17 cm-től a kar már nem
            # tartja a fogási orientációt.)
            r.move_grasp_to(wp_xyz, R_des, pos_tol=0.006, rot_tol=0.09,
                            max_iters=(6 if i == 0 else 4),
                            slices=(8 if i == 0 else 1))
            worst = max(worst, float(np.linalg.norm(
                r.product_pose() - prod0)) * 1000)
        out["moved_ramp_mm"] = m_ramp
        r1 = r.move_grasp_to(target, R_des, pos_tol=0.008, rot_tol=0.10,
                             max_iters=10)
        m1 = float(np.linalg.norm(r.product_pose() - prod0)) * 1000
        if verbose:
            print(f"    pálya vége: {r1.detail} · termék {m1:.1f} mm "
                  f"(út közben max {worst:.1f} mm)")
    else:
        qs, ep, er = r.ik6_seed(pre, R_des, restarts=16, iters=110)
        out["seed_mm"], out["seed_deg"] = ep * 1000, float(np.degrees(er))
        out["standoff_m"] = float(standoff)
        r.ramp_to(qs)
        out["moved_ramp_mm"] = float(np.linalg.norm(
            r.product_pose() - prod0)) * 1000
        r1 = r.move_grasp_to(pre, R_des, pos_tol=0.008, rot_tol=0.10,
                             max_iters=15)
        m1 = float(np.linalg.norm(r.product_pose() - prod0)) * 1000

    # 3. rázárás: a fogási pont a termék középpontjába
    r2 = (r1 if (r.plan and r.plan.get('waypoints'))
          else r.move_grasp_to(target, R_des, slices=6))
    m2 = float(np.linalg.norm(r.product_pose() - prod0)) * 1000
    if verbose:
        print(f"    rázárás   : {r2.detail} · termék {m2:.1f} mm")

    # 4. ujjak
    sq = r.squeeze(stop=min(1.0, close_to + 0.25))
    if verbose:
        print(f"    szorítás  : zárás={sq['amount']} · {sq['final_n']} kontakt "
              f"{sq['parts']}, {sq['final_f']:.1f} N, termék {sq['moved_mm']:.1f} mm")

    # 5. emelés — ez a valódi próba
    p_before = r.product_pose().copy()
    lift_target = r.grasp_point() + np.array([0.0, 0.0, 0.10])
    r.move_grasp_to(lift_target, R_des, pos_tol=0.02, rot_tol=0.3, max_iters=8)
    r.step(400)
    dz = float(r.product_pose()[2] - p_before[2])
    n_end, f_end = r.contact_count()
    held = dz > 0.03 and n_end > 0

    out.update({
        "pre_ok": bool(r1), "close_ok": bool(r2),
        "pre_err_mm": r1.data.get("err_mm"), "pre_err_deg": r1.data.get("err_deg"),
        "close_err_mm": r2.data.get("err_mm"), "close_err_deg": r2.data.get("err_deg"),
        "moved_pre_mm": m1, "moved_close_mm": m2,
        "contacts": sq["final_n"], "force_N": sq["final_f"],
        "lift_dz_mm": dz * 1000, "contacts_after_lift": n_end,
        "held": held,
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=None, choices=list(GRASP_POSES))
    ap.add_argument("--standoff", type=float, default=0.06)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if not SKU_SCENE.exists():
        sys.exit(f"nincs jelenet: {SKU_SCENE}\n"
                 f"Futtasd: python3 tools/shelflife_build_scene_sku.py")

    poses = [a.pose] if a.pose else list(GRASP_POSES)
    print("Shelf Life — fogási pózok bemérése (Alpro 1 L, 1.03 kg)\n")
    rows = []
    for p in poses:
        print(f"  ── {p}")
        try:
            rows.append(try_pose(p, a.standoff, verbose=a.verbose or bool(a.pose)))
        except Exception as e:                       # noqa: BLE001
            print(f"     HIBA: {type(e).__name__}: {e}")
            rows.append({"pose": p, "held": False, "error": str(e)})

    print("\n" + "─" * 78)
    print(f"{'póz':<18}{'IK mm/°':>12}{'kontakt':>9}{'erő N':>8}"
          f"{'emelés mm':>11}{'megtart':>9}")
    print("─" * 78)
    for r in rows:
        if "error" in r:
            print(f"{r['pose']:<18}{'HIBA':>12}")
            continue
        print(f"{r['pose']:<18}"
              f"{r['close_err_mm']:>6.0f}/{r['close_err_deg']:>4.0f}"
              f"{r['contacts']:>9}{r['force_N']:>8.1f}"
              f"{r['lift_dz_mm']:>11.0f}{'✅' if r['held'] else '❌':>9}")
    print("─" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
