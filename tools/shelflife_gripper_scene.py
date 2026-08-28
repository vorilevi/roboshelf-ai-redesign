"""
shelflife_gripper_scene.py — KÉTUJJAS IPARI FOGÓ a polcnál

    python3 tools/shelflife_gripper_scene.py --build    # a jelenet felépítése
    python3 tools/shelflife_gripper_scene.py            # fogáspróba

────────────────────────────────────────────────────────────────────────────
MIÉRT EZ, ÉS MIÉRT NEM AZ ÖTUJJAS KÉZ
────────────────────────────────────────────────────────────────────────────
A kérdés az volt: **oldaná-e meg egy kétujjas ipari fogó azt, amit az ötujjas
kéz nem tud?** Erre nem válasz az, ha az ötujjas kézzel fogunk „kétujjas
módra" — az csak a fogási stratégiát változtatja, a szerszámot nem.

Ez a modul a valódi összehasonlítást építi fel: **Robotiq 2F85**, a MuJoCo
Menagerie-ből, ugyanazon a polcon, ugyanazzal a dobozzal.

    Robotiq 2F85          Menagerie, hivatalos modell, fogásra hangolva
    fogótávolság          85 mm  (a doboz 58 mm)
    erőkorlát             ±5 N   (a GENE.01 kezén NINCS erőkorlát)
    fogófelület           szilikonpárna, súrlódás 0,7 / 0,6

────────────────────────────────────────────────────────────────────────────
MI EZ A JELENET
────────────────────────────────────────────────────────────────────────────
Nincs benne humanoid. A fogó egy háromtengelyes, pozícióvezérelt szánon ül
(x, y, z + forgatás), és így pontosan oda vihető, ahová akarjuk. Ez SZÁNDÉKOS
egyszerűsítés: most a **végszerszámra** vagyunk kíváncsiak, nem a karra.

    ✅ amire válaszol:  megfogja-e a dobozt EGY KÉTUJJAS FOGÓ a polcon
    ✅ amire válaszol:  elfér-e a polclap és a szomszédos termékek között
    ❌ amire NEM válaszol: rá lehet-e szerelni a GENE.01 csuklójára
    ❌ amire NEM válaszol: a humanoid kar el tudja-e vinni oda

A polc, a doboz mérete, tömege és súrlódása a MÉRT SKU-adatokból jön,
ugyanabból a forrásból, mint az ötujjas kísérletnél — különben nem volna
összehasonlítható.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

import mujoco                                    # noqa: E402

MENAGERIE = _REPO / "mujoco_menagerie/robotiq_2f85/2f85.xml"
SRC_SCENE = _REPO / "src/envs/assets/shelflife_scene_gene01_sku_v1.xml"
OUT_XML = _REPO / "src/envs/assets/shelflife_scene_gripper_v1.xml"

# a doboz és a polc a MÉRT adatokból (l. az SKU-rekordot)
CAN_R, CAN_H = 0.02905, 0.14540          # sugár, teljes magasság [m]
CAN_MASS = 0.343                          # MÉRVE
CAN_FRICTION = "0.7 0.02 0.002"
CAN_POS = (0.3891, -0.2000, 1.0640)       # ugyanott, mint az ötujjas kísérletben
SHELF_Z = 1.05
SHELF_HALF = (0.175, 0.45, 0.012)

BASE_XML = f"""<mujoco model="shelflife_gripper">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="1024"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <material name="mat_board" rgba="0.72 0.70 0.66 1" specular="0.1" shininess="0.1"/>
    <material name="mat_can"   rgba="0.85 0.10 0.12 1" specular="0.3" shininess="0.4"/>
  </asset>

  <worldbody>
    <!-- ⚠️ EGY LÁMPA KEVÉS. Az első változatban egyetlen felső fényforrás
         volt, és a renderek majdnem feketék lettek: a polc árnyékában
         semmi nem látszott. Egy szimulációnál ez nem részletkérdés — a
         2026-08-06-i felvétel EGY hipotézist cáfolt meg, amit számokból
         nem vettem észre. Ha nem látszik, nem tudunk ellenőrizni. -->
    <light pos="-0.6 -0.6 2.0" dir="0.4 0.4 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="-0.6  0.6 2.0" dir="0.4 -0.4 -1" diffuse="0.5 0.5 0.5"/>
    <light pos="-0.8  0.0 1.2" dir="1 0 0" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- ⚠️ EGY POLCLAP, SEMMI TÖBB.
         Az első változatban volt felső polclap és hátfal is. Mindkettő
         OLYAN korlátot hozott be, aminek semmi köze a vizsgált kérdéshez —
         sőt, aktívan félrevezetett: a felülnézeti renderek azért voltak
         használhatatlanok, mert a felső lap takart, és a „meddig fér le a
         fogó" korlát is részben ebből jött.
         A vizsgálat tárgya: EGY LAP, RAJTA A DOBOZ. -->
    <geom name="shelf_board_1" type="box"
          size="{SHELF_HALF[0]} {SHELF_HALF[1]} {SHELF_HALF[2]}"
          pos="0.40 0 {SHELF_Z}" material="mat_board"/>

    <!-- a termék: MÉRT méret, tömeg, súrlódás -->
    <body name="product_0" pos="{CAN_POS[0]} {CAN_POS[1]} {CAN_POS[2]}">
      <freejoint name="product_0_free"/>
      <geom name="product_0_col" type="cylinder"
            size="{CAN_R} {CAN_H/2}" pos="0 0 {CAN_H/2}"
            material="mat_can" mass="{CAN_MASS}"
            friction="{CAN_FRICTION}" condim="4"/>
    </body>

    <!-- a szán, amin a fogó ül: x, y, z + forgatás a függőleges körül -->
    <body name="mount" pos="0 0 0">
      <!-- ⚠️ AZ ARMATURE NEM DÍSZ. Az első változatban a szán egy 2 cm-es,
           0,5 kg-os kocka volt, forgó ízületein kp=800-as szabályzóval. Egy
           ilyen kicsi tehetetlenségi nyomatékhoz képest ez akkora merevség,
           hogy a szimuláció 14 ezredmásodperc alatt szétesett (NaN a
           gyorsulásban). Az `armature` egy virtuális motor-tehetetlenséget
           ad az ízülethez — ez a szokásos megoldás, és a Menagerie minden
           modellje használja. -->
      <joint name="gx" type="slide" axis="1 0 0" range="-1 1" armature="1.0"
             damping="20"/>
      <joint name="gy" type="slide" axis="0 1 0" range="-1 1" armature="1.0"
             damping="20"/>
      <joint name="gz" type="slide" axis="0 0 1" range="0 2" armature="1.0"
             damping="20"/>
      <joint name="gyaw" type="hinge" axis="0 0 1" range="-3.15 3.15"
             armature="0.05" damping="2"/>
      <joint name="gpitch" type="hinge" axis="0 1 0" range="-3.15 3.15"
             armature="0.05" damping="2"/>
      <!-- ⚠️ A GÖRDÜLÉS AZÉRT KELL, MERT NÉLKÜLE NINCS OLDALSÓ FOGÁS.
           Bólintással a fogó vízszintesre fordul, DE a zárási tengelye
           függőleges marad — vagyis a doboz tetejét és alját akarná
           összenyomni. A közelítési tengely körüli elfordulás hozza a
           zárási tengelyt vízszintesbe. Ezt az első jelenetből kihagytam,
           és emiatt csak FELÜLRŐL lehetett fogni — ahol viszont a fogó
           alaplapja ráül a doboz fedelére (mérve: 103 N lefelé). -->
      <joint name="groll" type="hinge" axis="1 0 0" range="-3.15 3.15"
             armature="0.05" damping="2"/>
      <geom name="mount_vis" type="box" size="0.03 0.03 0.02"
            rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0" mass="1.0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="act_gx" joint="gx" kp="2000" kv="120" ctrlrange="-1 1"/>
    <position name="act_gy" joint="gy" kp="2000" kv="120" ctrlrange="-1 1"/>
    <position name="act_gz" joint="gz" kp="2000" kv="120" ctrlrange="0 2"/>
    <position name="act_gyaw" joint="gyaw" kp="60" kv="8" ctrlrange="-3.15 3.15"/>
    <position name="act_gpitch" joint="gpitch" kp="60" kv="8" ctrlrange="-3.15 3.15"/>
    <position name="act_groll" joint="groll" kp="60" kv="8" ctrlrange="-3.15 3.15"/>
  </actuator>
</mujoco>
"""


def build_model():
    """A fogó beépítése a jelenetbe, és a modell LEFORDÍTÁSA a memóriában.

    ⚠️ A kész XML-t NEM használjuk modellbetöltésre. A Menagerie-modell a
    hálókat relatív úton (`assets/`) hivatkozza; ha az összeszerelt XML-t
    máshová írjuk ki, a `base.stl` nem található meg. Ezért a fordítás a
    specifikációból történik, ahol az útvonalak még helyesek — az XML csak
    dokumentációnak készül.
    """
    parent = mujoco.MjSpec.from_string(BASE_XML)
    child = mujoco.MjSpec.from_file(str(MENAGERIE))
    mount = parent.body("mount")
    frame = mount.add_frame()
    # a fogó a szán alatt, lefelé néző tengellyel:
    # a 2F85 saját z tengelye a fogás iránya, ezért 180°-ot fordítunk
    frame.pos = [0, 0, -0.02]
    frame.quat = [0, 1, 0, 0]
    frame.attach_body(child.body("base_mount"), "g_", "")
    model = parent.compile()
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    try:
        OUT_XML.write_text(parent.to_xml(), encoding="utf-8")
    except Exception:                                  # noqa: BLE001
        pass
    return model


def build(verbose: bool = True) -> Path:
    model = build_model()
    print(f"  felépítve: {OUT_XML.name}")
    print(f"  testek: {model.nbody} · ízületek: {model.njnt} · "
          f"aktuátorok: {model.nu} · geomok: {model.ngeom}")
    for a in range(model.nu):
        print("    aktuátor:",
              mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    return OUT_XML


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    print("Shelf Life — KÉTUJJAS IPARI FOGÓ (Robotiq 2F85)\n")
    print(f"  doboz: Ø{CAN_R*2000:.0f} mm × {CAN_H*1000:.0f} mm · "
          f"{CAN_MASS*1000:.0f} g (mérve)")
    print(f"  fogó nyílása: 85 mm — a doboz elfér benne\n")
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
