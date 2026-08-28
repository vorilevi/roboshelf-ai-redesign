"""
shelflife_t1_scene.py — A JELENET A BOOSTER T1-HEZ (29 szabadságfok)

    python3 tools/shelflife_t1_scene.py            # felépítés + minden szám kiírása
    python3 tools/shelflife_t1_scene.py --render   # kép a jelenetről

────────────────────────────────────────────────────────────────────────────
MIÉRT ÚJ JELENET, ÉS MIÉRT NEM A RÉGI ÁTÍRÁSA
────────────────────────────────────────────────────────────────────────────
A GENE.01 jelenete 1,62 m-es robotra készült. A Booster T1 **1,180 m** —
44 centiméterrel alacsonyabb. Egy 1,05 m magas polc a T1-nek a VÁLLA FÖLÖTT
van; azon a magasságon a karja alig nyúlik előre. Ha a polcot nem igazítjuk,
nem a robotot mérjük, hanem egy rosszul felállított kísérletet.

A régi jelenet érintetlen marad — ez külön fájl, külön eszközkészlettel.

────────────────────────────────────────────────────────────────────────────
A MODELL, ÉS AMI VELE JÁR
────────────────────────────────────────────────────────────────────────────
Forrás: BoosterRobotics/booster_assets · `robots/T1/T1_29dof.urdf` · BSD-3.
Helyi másolat: `assets/booster_t1_29dof/`.

    29 ízület:  fej 2 · kar 7×2 · derék 1 · láb 6×2
    magasság    1,1800 m   (MÉRVE a hálókon; a gyártó 1,18 m-t ír)
    tömeg       22,93 kg   (a modell összege — ⚠️ l. lentebb)

⚠️ AZ EREDETI URDF-BEN A `<mujoco>` FORDÍTÓBLOKK KI VAN KOMMENTELVE.
   Enélkül a MuJoCo eldobja a látványhálókat: `nmesh=0`, és a robot 22 db
   hengerré és dobozzá egyszerűsödik. A rendereken ez pont az, amit tavaly
   is elrontottunk — ami nem látszik, azt nem tudjuk ellenőrizni. A helyi
   másolatban a blokk aktív, és `meshdir="."` (az eredeti `"meshes/"` a
   `meshes/meshes/` úthoz vezetett volna, mert a fájlnevek maguk is
   tartalmazzák a mappát). Így `nmesh=31`.

⚠️ A TÖMEG ELLENTMOND — MÁR MEGINT. A modell összege 22,93 kg, a gyártó
   katalógusa ~30 kg-ot ír a T1-re. A GENE.01-nél ugyanez a hiba 152 kg és
   80 kg között feszült. Itt NEM skálázunk: a 23 kg egy 1,18 m-es, 4,3 kg-os
   karú robothoz hihető. De a forrásjelölés `measured (model)`, nem
   `manufacturer` — és minden erőszám ezen áll.

────────────────────────────────────────────────────────────────────────────
A POLC MAGASSÁGA ÉS TÁVOLSÁGA — MÉRVE, NEM BECSÜLVE
────────────────────────────────────────────────────────────────────────────
80 000 véletlen kartartás (derék + a jobb kar hét ízülete), a fogáspont a
csuklótól 160 mm-re (a Robotiq 2F85 mérete a fogós próbapadról):

    magassági sáv    minták   előrenyúlás (medián)
    ─────────────────────────────────────────────
    0,50 – 0,60 m     1 565      0,221 m
    0,70 – 0,80 m     2 467      0,298 m
    0,80 – 0,90 m     2 385      0,333 m   ← itt a legjobb
    0,90 – 1,00 m     2 566      0,337 m   ← és itt
    1,10 – 1,20 m     2 917      0,281 m
    1,30 – 1,40 m     1 321      0,220 m

A doboz fogáspontja tehát 0,85 és 0,95 m közé való. A doboz MÉRT magassága
145,4 mm, a fogáspont a felezőpontban → a polclap teteje 0,800 m.

Az oldalirány sem mindegy: ugyanabban a sávban a jobb kéz a −0,45…−0,25 m
tartományban a legotthonosabb (a robot jobbja a −y). A választott pontot
ellenőriztük: a (0,350 · −0,200 · 0,873) célhoz a legközelebbi minta 6,6 mm.

    ELŐZŐ (GENE.01, 1,62 m)      MOST (T1, 1,18 m)
    polclap teteje  1,050 m       0,800 m      −250 mm
    doboz közepe    1,064 m       0,873 m      −191 mm
    doboz távolság  0,389 m       0,350 m      − 39 mm

────────────────────────────────────────────────────────────────────────────
AMIT EZ A JELENET NEM CSINÁL
────────────────────────────────────────────────────────────────────────────
❌ A medence RÖGZÍTETT. A robot nem áll a lábán, nem egyensúlyoz. Ez
   SZÁNDÉKOS: a kérdésünk a kar és a kamera, nem a járás. Amint járni is
   kell, kerül bele `freejoint`, és az egész erőkép újramérendő.
❌ Nincs rajta a fogó. Az az A1 lépés — de itt már NEM kell ujjakat
   leoperálni: a T1 kezén nincs 21 ujjízület, csak egy csonk.
❌ A kamera helye FELTÉTELEZÉS. Az URDF-ben nincs kamera (`ncam=0`), a
   fejbe mi tettük. Forrásjelölés: `decided`.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

# ⚠️ A SORREND SZÁMÍT. A MuJoCo a `MUJOCO_GL` értékét IMPORTÁLÁSKOR olvassa
#    ki és rögzíti a grafikai hátteret. Ha ez a modul a `mujoco` UTÁN kerül
#    be, a virtuális kijelző már késő — „an OpenGL platform library has not
#    been loaded" hibával áll meg. Ezért van a robotimport előtt.
import shelflife_render_env                      # noqa: E402,F401
import mujoco                                    # noqa: E402

# ── a KAMERA, a gyártó adatlapjából ────────────────────────────────────────
# Intel RealSense D455 · színérzékelő (D450 modul) · D400 adatlap 337029-017,
# Table 3-19: OmniVision OV9782 · 1280×800 aktív képpont · látószög 90°×65° ·
# globális zár · FIX FÓKUSZ · f/2.0 · 1,93 mm · torzítás ≤1,5%
# Mélység: 1280×720 @90 fps · 86°×57° · működési tartomány 0,6–6 m
#
# ⚠️ EDDIG 1920×1080-AT ÉS 45°-OT FELTÉTELEZTÜNK. Mindkettő rossz volt, és
#    ugyanabba az irányba: kevesebb képpont, szélesebb látószög → kevesebb
#    képpont fokonként. Az 5 mm-es karakter olvashatósági határa 982 mm-ről
#    504 mm-re esett. Ez a harmadik alkalom, hogy a kamerafeltevésünk
#    optimista irányba tévedett.
RENDER_W, RENDER_H = 1280, 800
CAM_FOVY_DEG = 65.0
DEPTH_MIN_M = 0.6                                # a mélységérzékelés alsó határa
URDF = _REPO / "assets/booster_t1_29dof/T1_29dof.urdf"
OUT_XML = _REPO / "src/envs/assets/shelflife_scene_t1_v1.xml"

# ── a robot MÉRT geometriája (a hálókon, medence-origóhoz képest) ───────────
SOLE_Z = -0.6869                  # measured: a talp a medence alatt
ROBOT_H = 1.1800                  # measured
SHOULDER_Z = 0.906                # measured, a talajtól
HEAD_Z = 1.102                    # measured, a talajtól (H2 test közepe)

# ── a polc — a MUNKATÉRBŐL SZÁMOLVA ────────────────────────────────────────
SHELF_TOP_Z = 0.800               # derived: a polclap TETEJE a talajtól
SHELF_HALF = (0.175, 0.450, 0.012)
SHELF_X = 0.425                   # a lap közepe → elülső él 0,250 m

# ── a termék — a MÉRT SKU-adatokból, ugyanaz, mint eddig ───────────────────
CAN_R, CAN_H = 0.02905, 0.14540
CAN_MASS = 0.343
CAN_FRICTION = "0.7 0.02 0.002"
CAN_X, CAN_Y = 0.350, -0.200
DATE_R = 0.02324                  # a talp sugara
DATE_PNG = _REPO / "src/envs/assets/shelflife_textures/date_0.png"
DATE_W, DATE_H = 0.0450, 0.0163   # a nyomtatott folt MÉRT mérete (l. a decal-t)

# ── a kar, amivel dolgozunk ────────────────────────────────────────────────
ARM = ["Waist",
       "Right_Shoulder_Pitch", "Right_Shoulder_Roll",
       "Right_Elbow_Pitch", "Right_Elbow_Yaw",
       "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll"]
HEAD = ["AAHead_yaw", "Head_pitch"]
GRIP_OFFSET = 0.160               # csukló → fogáspont (Robotiq 2F85)

# ── alaphelyzet ────────────────────────────────────────────────────────────
# ⚠️ AZ URDF NULLA-POZÍCIÓJA T-TARTÁS: mindkét kar vízszintesen oldalra.
#    Az első renderen ez látszott, és félrevezető: úgy néz ki, mintha a
#    robot beleérne a polcba. Nem ér bele (mérve: 100,6 mm hézag a
#    legszűkebb ponton), de kiindulásnak alkalmatlan. Ez a tartás 40 000
#    minta közül a legjobb „kar a csípő mellett" megoldás, 36,9 mm hibával.
HOME = {
    "Waist": 0.006,
    "Right_Shoulder_Pitch": -3.046, "Right_Shoulder_Roll": -0.959,
    "Right_Elbow_Pitch": -1.621, "Right_Elbow_Yaw": 1.187,
    "Right_Wrist_Pitch": -2.386, "Right_Wrist_Yaw": -1.322,
    "Right_Hand_Roll": -0.709,
    # a bal kar tükrözve: a gördülés- és sodrásjellegű tengelyek előjelet
    # váltanak, a bólintásjellegűek nem
    "Left_Shoulder_Pitch": -3.046, "Left_Shoulder_Roll": 0.959,
    "Left_Elbow_Pitch": -1.621, "Left_Elbow_Yaw": -1.187,
    "Left_Wrist_Pitch": -2.386, "Left_Wrist_Yaw": 1.322,
    "Left_Hand_Roll": 0.709,
}


def apply_home(model, data) -> None:
    """Alaphelyzetbe állítja a robotot, a HATÁROKRA VÁGVA."""
    for nm, v in HOME.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if j < 0:
            continue
        lo_, hi_ = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = float(np.clip(v, lo_, hi_))
    mujoco.mj_forward(model, data)


def urdf_limits() -> dict[str, tuple[float, float]]:
    """A gyártó effort/velocity korlátai — NEM mi találjuk ki őket.

    ⚠️ A GENE.01-nél kp=3000-rel és erőkorlát NÉLKÜL hajtottunk (#10, #17).
    Egy pozíciószabályzó erőkorlát nélkül bármekkora nyomatékot kiad, és
    utána azt mérjük, amit mi magunk írtunk elő. Itt a `forcerange` a
    gyártói `effort` értékből jön.
    """
    out = {}
    for j in ET.parse(URDF).getroot().iter("joint"):
        lim = j.find("limit")
        if lim is not None and j.get("name"):
            out[j.get("name")] = (float(lim.get("effort", 0)),
                                  float(lim.get("velocity", 0)))
    return out


def build_spec() -> mujoco.MjSpec:
    """A robot specifikációja + padló, polc, termék, kamera, aktuátorok.

    ⚠️ NINCS `attach`. A T1 gyökere (`Trunk`) rögzített, ezért a MuJoCo a
    worldbody-ba olvasztja — nincs mit hova csatolni. A jelenetet ezért
    EBBE a specifikációba tesszük bele. A világ origója a MEDENCE, nem a
    padló; minden magasság `SOLE_Z + z` alakban megy be.
    """
    s = mujoco.MjSpec.from_file(str(URDF))
    s.option.timestep = 0.002
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC     # a fogóhoz hangolva
    s.option.impratio = 10.0
    # ⚠️ AZ OFFSCREEN PUFFERT IS MÉRETRE KELL ÁLLÍTANI, különben a MuJoCo
    #    csendben a 640×480-as alapértelmezettre vág vissza — és pont a
    #    felbontáson múlik, olvasható-e a dátum.
    s.visual.global_.offwidth = RENDER_W
    s.visual.global_.offheight = RENDER_H
    w = s.worldbody

    def z(above_floor: float) -> float:
        return SOLE_Z + above_floor

    # ── világítás ──────────────────────────────────────────────────────────
    # ⚠️ EGY LÁMPA KEVÉS — 2026-08-06-on emiatt voltak feketék a renderek.
    for pos, d_, dif in [((-0.6, -0.6, 1.4), (0.4, 0.4, -1), 0.7),
                         ((-0.6, 0.6, 1.4), (0.4, -0.4, -1), 0.5),
                         ((0.9, 0.0, 1.2), (-1, 0, -0.3), 0.6)]:
        L = w.add_light()
        L.pos = [pos[0], pos[1], z(pos[2])]
        L.dir = list(d_)
        L.diffuse = [dif, dif, dif]

    g = w.add_geom()
    g.name = "floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [3, 3, 0.1]
    g.pos = [0, 0, z(0.0)]
    g.rgba = [0.88, 0.88, 0.90, 1]

    # ── EGY POLCLAP, SEMMI TÖBB ────────────────────────────────────────────
    g = w.add_geom()
    g.name = "shelf_board_1"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(SHELF_HALF)
    g.pos = [SHELF_X, 0.0, z(SHELF_TOP_Z - SHELF_HALF[2])]
    g.rgba = [0.72, 0.70, 0.66, 1]

    # ── a termék ───────────────────────────────────────────────────────────
    b = w.add_body()
    b.name = "product_0"
    b.pos = [CAN_X, CAN_Y, z(SHELF_TOP_Z)]
    b.add_freejoint(name="product_0_free")
    g = b.add_geom()
    g.name = "product_0_col"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [CAN_R, CAN_H / 2, 0]
    g.pos = [0, 0, CAN_H / 2]
    g.mass = CAN_MASS
    g.friction = [0.7, 0.02, 0.002]
    g.condim = 4
    g.rgba = [0.85, 0.10, 0.12, 1]
    # a talp: csupasz alumínium korong
    g = b.add_geom()
    g.name = "product_0_base"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    # ⚠️ Z-HARC: a talpkorong és a doboz alaplapja egy síkban volt, és ferde
    #    nézetben foltokban törtek egymásba. A korong a doboz ALÁ kerül.
    g.size = [DATE_R, 0.0003, 0]
    g.pos = [0, 0, -0.0008]
    g.mass = 0.0
    g.contype, g.conaffinity = 0, 0
    g.rgba = [0.78, 0.78, 0.80, 1]

    # ── a DÁTUM, valódi mátrixnyomtatással ─────────────────────────────────
    # ⚠️ EDDIG EGY SIMA FEHÉR KORONG VOLT ITT, felirat nélkül. Emiatt a
    #    „megmutatható-e a dátum" kérdést csak geometriailag lehetett
    #    vizsgálni — hogy OLVASHATÓ-e, nem. Pedig a projekt tétje ez.
    if DATE_PNG.exists():
        t = s.add_texture()
        t.name = "tex_date"
        t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.file = str(DATE_PNG)
        mat = s.add_material()
        mat.name = "mat_date"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tex_date"
        mat.specular, mat.shininess = 0.15, 0.1
        g = b.add_geom()
        g.name = "product_0_date"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        # ⚠️ A talpkorong lejjebb került (z-harc miatt), a dátumlap viszont
        #    fent maradt — és ezzel a DOBOZ BELSEJÉBE szorult, ahol nem
        #    látszik. A render mutatta meg: szép, szabad rálátás egy üres
        #    korongra. A lapnak a korong ALÁ kell kerülnie.
        g.size = [DATE_W / 2, DATE_H / 2, 0.0003]
        g.pos = [0, 0, -0.0014]
        g.mass = 0.0
        g.contype, g.conaffinity = 0, 0
        g.material = "mat_date"

    # ── kamera a fejbe ─────────────────────────────────────────────────────
    # ⚠️ FELTÉTELEZÉS, nem gyártói adat: az URDF-ben ncam=0. A T1 valódi
    #    fejkamerájának helyét és felbontását nem publikálták.
    head = s.body("H2")
    c = head.add_camera()
    c.name = "head_camera"
    c.pos = [0.09, 0.0, 0.02]
    c.fovy = CAM_FOVY_DEG
    c.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    # a fej +x-e az előre irány → ugyanazzal a képlettel, mint a többi kamera
    c.quat = _look_at([0.0, 0.0, 0.0], [1.0, 0.0, -0.35])

    for nm, eye, tgt in [
            ("overview", (1.6, -1.5, z(1.30)), (CAN_X * 0.6, CAN_Y * 0.5, z(0.80))),
            ("side", (0.30, -1.9, z(0.95)), (0.30, 0.0, z(0.80)))]:
        ov = w.add_camera()
        ov.name = nm
        ov.pos = list(eye)
        ov.fovy = 45.0
        ov.quat = _look_at(eye, tgt)

    # ── aktuátorok, GYÁRTÓI erőkorláttal ───────────────────────────────────
    lim = urdf_limits()
    for jname in ARM + HEAD:
        eff, vel = lim.get(jname, (0.0, 0.0))
        a = s.add_actuator()
        a.name = f"act_{jname}"
        a.target = jname
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        kp = 120.0 if jname in HEAD else 300.0
        a.gainprm = [kp] + [0.0] * 9
        a.biasprm = [0.0, -kp, -kp * 0.08] + [0.0] * 7
        if eff > 0:
            a.forcerange = [-eff, eff]
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return s


def _look_at(eye, tgt) -> list[float]:
    """Kameratájolás: a MuJoCo kamerája a SAJÁT −z tengelye felé néz.

    ⚠️ EZT ELRONTOTTAM ELŐSZÖR. `y = cross(x, -f)` volt itt, ami a felfelé
    irányt megfordítja, és a mátrix tükrözötté válik — a `mju_mat2Quat`
    ebből értelmetlen forgatást csinált, az első render a padlót mutatta.
    Ellenőrizhető: f=(1,0,0) esetén x=(0,−1,0), y=(0,0,1), z=(−1,0,0),
    ami pontosan az `xyaxes="0 -1 0  0 0 1"` beállítás.
    """
    f = np.array(tgt, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    # ⚠️ FÜGGŐLEGES NÉZÉSNÉL A KERESZTSZORZAT ELFAJUL. Ha a nézésirány
    #    majdnem párhuzamos a világ „fel" tengelyével — márpedig a doboz
    #    TALPÁRA pontosan alulról nézünk rá —, a `cross(f, z)` nullvektor.
    #    Ilyenkor másik referenciatengely kell. Az `assert` fogta meg.
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    R = np.column_stack([x, y, -f])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, "nem forgatás — tükrözés"
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return list(q)


def build_model():
    s = build_spec()
    model = s.compile()
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    try:
        OUT_XML.write_text(s.to_xml(), encoding="utf-8")
    except Exception:                              # noqa: BLE001
        pass
    return model


def measure(model) -> int:
    """ELLENŐRZÉS. A beállítás nem bizonyíték — minden számot visszamérünk."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    bn = lambda b: mujoco.mj_id2name(          # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    gid = lambda n: mujoco.mj_name2id(         # noqa: E731
        model, mujoco.mjtObj.mjOBJ_GEOM, n)
    ok = True

    print("\n  ── ELLENŐRZÉS ────────────────────────────────────────────")
    print(f"  hálók              {model.nmesh}"
          f"      {'✅' if model.nmesh > 0 else '❌ látványhálók elveszve'}")
    ok &= model.nmesh > 0

    # a robot magassága a hálókon
    V = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[a:a + n].reshape(-1, 3)
        V.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    V = np.vstack(V)
    h = V[:, 2].max() - V[:, 2].min()
    print(f"  robot magassága    {h:.4f} m   "
          f"{'✅' if abs(h - ROBOT_H) < 0.005 else '❌'}")
    ok &= abs(h - ROBOT_H) < 0.005

    floor = V[:, 2].min()
    tomeg = sum(model.body_mass[b] for b in range(model.nbody)
                if not bn(b).startswith(("product", "world")))
    print(f"  tömeg              {tomeg:.2f} kg  [measured (model)]")

    ns = len([g for g in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                  or "").startswith("shelf")])
    print(f"  polcgeomok         {ns}      "
          f"{'✅ egy lap' if ns == 1 else '❌'}")
    ok &= ns == 1

    sb = gid("shelf_board_1")
    top = d.geom_xpos[sb][2] + model.geom_size[sb][2] - floor
    print(f"  polclap teteje     {top:.3f} m  "
          f"{'✅' if abs(top - SHELF_TOP_Z) < 1e-3 else '❌'}")
    ok &= abs(top - SHELF_TOP_Z) < 1e-3

    pb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "product_0")
    pg = gid("product_0_col")
    grip_z = d.geom_xpos[pg][2] - floor
    print(f"  fogáspont          {grip_z:.3f} m  "
          f"× {d.xpos[pb][0]:.3f} m előre  "
          f"{'✅ az édes sávban (0,82–0,95)' if 0.82 <= grip_z <= 0.95 else '❌'}")
    ok &= 0.82 <= grip_z <= 0.95

    print(f"  váll magassága     {SHOULDER_Z:.3f} m  "
          f"→ a polc {SHOULDER_Z - top:+.3f} m-rel a váll alatt")

    # a doboz ÜL-e, vagy lebeg/süllyed
    gap = mujoco.mj_geomDistance(model, d, pg, sb, 0.2, None)
    print(f"  doboz ↔ polclap    {gap*1000:+.2f} mm  "
          f"{'✅ éppen felfekszik' if abs(gap) < 1e-3 else '⚠️ ellenőrizni'}")

    print(f"  aktuátorok         {model.nu}")
    nofl = [a for a in range(model.nu) if not model.actuator_forcelimited[a]]
    print(f"  erőkorlát nélkül   {len(nofl)}      "
          f"{'✅ mind korlátozva' if not nofl else '⚠️ ' + str(len(nofl))}")

    ncam = model.ncam
    print(f"  kamerák            {ncam}      [a fejkamera: decided]")
    ok &= ncam >= 2

    # ⚠️ HOZZÁÉR-E A ROBOT A POLCHOZ? Ezt nem szabad szemre eldönteni: az
    #    első renderen úgy LÁTSZOTT, mintha a polclap átmenne a törzsön.
    gname = lambda g: mujoco.mj_id2name(     # noqa: E731
        model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    sh = [g for g in range(model.ngeom) if gname(g).startswith("shelf")]
    rb = [g for g in range(model.ngeom)
          if not gname(g).startswith(("shelf", "floor", "product"))]
    ft = np.zeros(6)
    for tag, fn in (("T-tartás", lambda: mujoco.mj_forward(model, d)),
                    ("alaphelyzet", lambda: apply_home(model, d))):
        mujoco.mj_resetData(model, d)
        fn()
        gap = min(mujoco.mj_geomDistance(model, d, r, s, 1.0, ft)
                  for r in rb for s in sh)
        print(f"  hézag a polctól    {gap*1000:+7.1f} mm ({tag})  "
              f"{'✅' if gap > 0.01 else '❌ ütközik'}")
        ok &= gap > 0.01

    print("\n  " + ("✅ A JELENET HASZNÁLHATÓ" if ok
                    else "❌ VAN MEGBUKOTT ELLENŐRZÉS — ne építs rá"))
    return 0 if ok else 1


def render(model) -> None:
    """Kép a jelenetről — a beállítás nem bizonyíték, LÁTNI kell.

    ⚠️ A fejlesztői homokozóban nincs ablakkezelő. A `shelflife_render_env`
    importja indít egy virtuális X-kijelzőt; enélkül a MuJoCo „an OpenGL
    platform library has not been loaded" hibával áll meg.
    """
    import imageio.v3 as iio
    shelflife_render_env.ensure(RENDER_W, RENDER_H)
    d = mujoco.MjData(model)
    apply_home(model, d)
    out = _REPO / "results/shelflife_t1"
    out.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(model, RENDER_H, RENDER_W)
    for cam in ("overview", "head_camera", "side"):
        try:
            r.update_scene(d, camera=cam)
        except Exception:                          # noqa: BLE001
            continue
        img = r.render()
        p = out / f"t1_scene_{cam}.png"
        iio.imwrite(p, img)
        print(f"  kép: results/shelflife_t1/{p.name}   "
              f"({img.shape[1]}×{img.shape[0]}, szórás {img.std():.1f}"
              f"{' ⚠️ ÜRES KÉP' if img.std() < 1.0 else ''})")
    r.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    print("Shelf Life — JELENET A BOOSTER T1-HEZ (29 DoF)\n")
    print(f"  robot   {ROBOT_H*100:.1f} cm · váll {SHOULDER_Z*100:.1f} cm · "
          f"fej {HEAD_Z*100:.1f} cm")
    print(f"  polc    teteje {SHELF_TOP_Z*100:.1f} cm · "
          f"elülső él {SHELF_X - SHELF_HALF[0]:.3f} m")
    print(f"  doboz   Ø{CAN_R*2000:.1f} × {CAN_H*1000:.1f} mm · "
          f"{CAN_MASS*1000:.0f} g · x={CAN_X} y={CAN_Y}")
    m = build_model()
    print(f"\n  felépítve: {OUT_XML.name}")
    print(f"  testek {m.nbody} · ízületek {m.njnt} · "
          f"aktuátorok {m.nu} · geomok {m.ngeom} · hálók {m.nmesh}")
    rc = measure(m)
    if a.render:
        render(m)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
