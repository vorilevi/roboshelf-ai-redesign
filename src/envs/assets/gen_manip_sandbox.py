#!/usr/bin/env python3
"""
Generates scene_manip_sandbox_v2.xml from the original g1_29dof_with_hand.xml.

Changes vs original:
  - pelvis: no freejoint (fixed base), placed at world pos="0 0 0.793"
  - All motor actuators replaced with position actuators (kp tuned)
  - right_hand_palm site added for hand position sensing
  - Environment added: floor, storage table, gondola_A, stock_1, target_shelf
  - Physics: implicitfast integrator, joint damping/armature defaults
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC_XML  = REPO / "unitree_rl_gym/resources/robots/g1_description/g1_29dof_with_hand.xml"
DST_XML  = REPO / "src/envs/assets/scene_manip_sandbox_v2.xml"
MESHDIR  = "../../../unitree_rl_gym/resources/robots/g1_description/meshes"

src = SRC_XML.read_text()

# ── 1. Strip outer <mujoco> tag and extract parts ──────────────────────────
# We rebuild the file from scratch using the robot's worldbody subtree.

# Extract <asset> block (mesh list)
asset_match = re.search(r'<asset>(.*?)</asset>', src, re.DOTALL)
asset_inner = asset_match.group(1).strip() if asset_match else ""

# Extract <worldbody> content
wb_match = re.search(r'<worldbody>(.*?)</worldbody>', src, re.DOTALL)
wb_inner = wb_match.group(1).strip() if wb_match else ""

# ── 2. Fix pelvis: remove freejoint ────────────────────────────────────────
wb_inner = re.sub(
    r'\s*<joint name="floating_base_joint"[^/]*/>\s*',
    '\n      <!-- pelvis rögzített: nincs freejoint -->',
    wb_inner
)

# ── 2b. Passzív jointok: class="passive_joint" hozzáadása ─────────────────
# Minden joint ami NEM jobb kar → passive_joint class → damping=50, armature=0.1
PASSIVE_JOINTS = [
    # lábak
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    # derék
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    # bal kar + kéz
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint",
    "left_hand_middle_0_joint","left_hand_middle_1_joint",
    "left_hand_index_0_joint","left_hand_index_1_joint",
    # jobb csukló + jobb kéz — passzív (numerikusan instabil kp=2/5 mellett)
    # csak a 4 váll+könyök joint aktív (kp=20, stabil)
    "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
    "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
    "right_hand_index_0_joint","right_hand_index_1_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint",
]
for jname in PASSIVE_JOINTS:
    # <joint name="foo_joint" ... /> → hozzáadjuk class="passive_joint"
    wb_inner = re.sub(
        r'(<joint name="' + jname + r'")',
        r'\1 class="passive_joint"',
        wb_inner
    )

# ── 3. Fix pelvis world pos: original is pos="0 0 0.793", keep it.
#    But we want the robot to face +y (toward the table at y=3.2).
#    Original robot faces +x. We rotate pelvis 90° around z: quat = cos45 0 0 sin45 = 0.707 0 0 0.707
#    Actually let's keep original orientation and place the table at x=+1.0 instead.
#    Simpler: don't rotate, put table at x=1.2 (robot's forward = +x in original frame)

# ── 4. Add right_hand_site inside right_wrist_yaw_link ────────────────────
# In original, right_hand_palm_link geoms are directly in right_wrist_yaw_link body.
# We add a site there for hand position sensing.
wb_inner = wb_inner.replace(
    '<body name="right_hand_thumb_0_link" pos="0.067 -0.003 0">',
    '<site name="right_hand_site" size="0.02" pos="0.12 -0.003 0" rgba="0 1 0 0.6"/>\n'
    '                          <body name="right_hand_thumb_0_link" pos="0.067 -0.003 0">'
)

# ── 5. Build new XML ───────────────────────────────────────────────────────
new_xml = f'''<!--
  ROBOSHELF AI — Manipulációs Sandbox Scene v2
  Generált fájl: src/envs/assets/gen_manip_sandbox.py

  Felépítés:
    - G1 robot (g1_29dof_with_hand.xml alapján) — rögzített pelvis
    - Az eredeti kinematika változtatás nélkül (helyes body pos/quat értékek)
    - Lábak és bal kar: passzív (position actuator, nulla célpont)
    - Jobb kar: aktív (position actuator, PPO vezérli)
    - Storage asztal (x=1.0): robot előtt (robot +x irányba néz)
    - Gondola A (x=-0.5): robot mögött, 4 polcszinttel

  Koordináta-rendszer (robot perspektíva):
    - Robot forward = +x irány (az eredeti G1 MJCF-ban)
    - Pelvis: world pos="0 0 0.793" (rögzített)
    - Storage asztal: x=1.2, magasság 0.815m
    - Gondola A: x=-0.5, 4 polcszint

  Jobb kar aktuátorok:
    right_shoulder_pitch/roll/yaw, right_elbow,
    right_wrist_roll/pitch/yaw,
    right_hand_thumb/index/middle (0,1 ill. 0,1,2)
-->
<mujoco model="roboshelf_manip_sandbox_v2">

  <compiler angle="radian" autolimits="true" meshdir="{MESHDIR}"/>

  <!-- timestep=0.002 → 500 Hz, implicitfast = stabil position control -->
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <global azimuth="120" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="20"/>
  </visual>

  <default>
    <!-- Passzív jointok (lábak, derék, bal kar): nagy damping, nincs actuator
         A pelvis rögzített → lábak lógnak, high damping megakadályozza az oszcillációt -->
    <default class="passive_joint">
      <joint damping="50.0" armature="0.1"/>
    </default>
    <!-- Aktív jobb kar váll/könyök -->
    <default class="arm_joint">
      <joint damping="2.0" armature="0.01"/>
    </default>
    <default class="arm_motor">
      <position kp="20" forcerange="-25 25"/>
    </default>
    <!-- Csukló -->
    <default class="wrist_joint">
      <joint damping="0.5" armature="0.005"/>
    </default>
    <default class="wrist_motor">
      <position kp="5" forcerange="-5 5"/>
    </default>
    <!-- Ujjak -->
    <default class="finger_joint">
      <joint damping="0.1" armature="0.001"/>
    </default>
    <default class="finger_motor">
      <position kp="2" forcerange="-2.45 2.45"/>
    </default>
    <!-- Termék doboz -->
    <default class="product_box">
      <geom friction="0.8 0.01 0.001" condim="4" solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </default>
  </default>

  <asset>
    <texture name="tex_floor" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.82 0.82 0.78" rgb2="0.78 0.78 0.74"/>
    <material name="mat_floor"  texture="tex_floor" texrepeat="6 6" reflectance="0.1"/>
    <material name="mat_shelf"  rgba="0.7 0.7 0.7 1" reflectance="0.3" shininess="0.5"/>
    <material name="mat_board"  rgba="0.88 0.86 0.82 1" reflectance="0.05"/>
    <material name="mat_wall"   rgba="0.95 0.94 0.92 1"/>
    <material name="mat_red"    rgba="0.85 0.15 0.15 1"/>
    <material name="mat_green"  rgba="0.15 0.70 0.25 1"/>
{asset_inner}
  </asset>

  <worldbody>

    <light name="ceil_1" pos="0 0 3.5" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="true" cutoff="60"/>
    <light name="ceil_2" pos="1 0 3.0" dir="0 0 -1" diffuse="0.7 0.7 0.7" castshadow="false" cutoff="60"/>
    <geom name="floor" type="plane" size="4 4 0.01" material="mat_floor"/>

    <!-- ===== G1 ROBOT (rögzített alap, eredeti kinematika) ===== -->
{wb_inner}

    <!-- ===== STORAGE ASZTAL (x=0.45, workspace-en belül) ===== -->
    <!-- Váll world x≈0.004, max reach≈0.58m → asztal max x≈0.5
         x=0.45: váll→doboz ≈ 0.50m → éppen elérhető -->
    <body name="storage" pos="0.45 0 0">
      <geom name="st_legs" type="box" size="0.20 0.30 0.35" pos="0 0 0.35" rgba="0.50 0.45 0.35 1"/>
      <geom name="st_top"  type="box" size="0.20 0.30 0.015" pos="0 0 0.715" rgba="0.60 0.55 0.45 1"/>
    </body>

    <!-- ===== GONDOLA A (robot mögött, x=-0.5) ===== -->
    <body name="gondola_A" pos="-0.5 0 0">
      <geom name="gA_sL" type="box" size="0.02 0.35 0.60" pos="0 -0.35 0.60" material="mat_shelf"/>
      <geom name="gA_sR" type="box" size="0.02 0.35 0.60" pos="0  0.35 0.60" material="mat_shelf"/>
      <geom name="gA_bk" type="box" size="0.01 0.33 0.60" pos="-0.20 0 0.60" material="mat_shelf"/>
      <geom name="gA_b1" type="box" size="0.18 0.33 0.012" pos="0 0 0.10" material="mat_board"/>
      <geom name="gA_b2" type="box" size="0.18 0.33 0.012" pos="0 0 0.40" material="mat_board"/>
      <geom name="gA_b3" type="box" size="0.18 0.33 0.012" pos="0 0 0.70" material="mat_board"/>
      <geom name="gA_b4" type="box" size="0.18 0.33 0.012" pos="0 0 1.00" material="mat_board"/>
    </body>

    <!-- ===== STOCK TERMÉK (doboz az asztalon, x=0.45) ===== -->
    <!-- Asztalfelszín z=0.715, doboz félmagasság=0.04 → z=0.755 -->
    <body name="stock_1" pos="0.45 0.0 0.755">
      <freejoint/>
      <geom type="box" size="0.05 0.04 0.04" class="product_box" material="mat_red" mass="0.35"/>
      <site name="stock_1_site" size="0.01" rgba="1 0 0 0.8"/>
    </body>

    <!-- ===== TARGET (gondola b2, x=-0.41) ===== -->
    <site name="target_shelf" type="box" size="0.065 0.045 0.002"
          pos="-0.41 0.0 0.415" rgba="0 1 0 0.4"/>

  </worldbody>

  <!-- ===== AKTUÁTOROK ===== -->
  <!-- Csak a jobb kar aktív. A passzív jointok (lábak, derék, bal kar)
       damping=50 + armature=0.1 által rögzítve, actuator nélkül.
       ctrl mérete = 14 (nu=14), ARM_QPOS_START=0 az env-ben! -->
  <actuator>
    <!-- Jobb kar — AKTÍV (PPO vezérli, ctrl[0..3], nu=4)
         Csak váll + könyök: stabil kp=20, actuatorfrcrange=±25
         Csukló és ujjak: passzív (equality constraint rögzíti) -->
    <position name="right_shoulder_pitch" joint="right_shoulder_pitch_joint" class="arm_motor"/>
    <position name="right_shoulder_roll"  joint="right_shoulder_roll_joint"  class="arm_motor"/>
    <position name="right_shoulder_yaw"   joint="right_shoulder_yaw_joint"   class="arm_motor"/>
    <position name="right_elbow"          joint="right_elbow_joint"          class="arm_motor"/>
  </actuator>

  <!-- ===== SZENZOR ===== -->
  <sensor>
    <framepos name="s_stock_1"    objtype="body" objname="stock_1"/>
    <framepos name="s_right_hand" objtype="site" objname="right_hand_site"/>
    <framepos name="s_target"     objtype="site" objname="target_shelf"/>
    <touch    name="touch_palm"   site="right_hand_site"/>
  </sensor>

</mujoco>
'''

# ── 6. equality constraints: passzív jointok rögzítése 0-ban ──────────────
# weld helyett joint equality: qpos=0 kényszer, numerikusan stabil
equality_block = '''
  <!-- ===== EQUALITY CONSTRAINTS: passzív jointok rögzítve 0-ban ===== -->
  <!-- Ez stabil alternatíva a high-damping actuatorral szemben -->
  <equality>
    <joint name="fix_l_hip_p"   joint1="left_hip_pitch_joint"    polycoef="0 0 0 0 0"/>
    <joint name="fix_l_hip_r"   joint1="left_hip_roll_joint"     polycoef="0 0 0 0 0"/>
    <joint name="fix_l_hip_y"   joint1="left_hip_yaw_joint"      polycoef="0 0 0 0 0"/>
    <joint name="fix_l_knee"    joint1="left_knee_joint"         polycoef="0 0 0 0 0"/>
    <joint name="fix_l_ank_p"   joint1="left_ankle_pitch_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_l_ank_r"   joint1="left_ankle_roll_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_r_hip_p"   joint1="right_hip_pitch_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_r_hip_r"   joint1="right_hip_roll_joint"    polycoef="0 0 0 0 0"/>
    <joint name="fix_r_hip_y"   joint1="right_hip_yaw_joint"     polycoef="0 0 0 0 0"/>
    <joint name="fix_r_knee"    joint1="right_knee_joint"        polycoef="0 0 0 0 0"/>
    <joint name="fix_r_ank_p"   joint1="right_ankle_pitch_joint" polycoef="0 0 0 0 0"/>
    <joint name="fix_r_ank_r"   joint1="right_ankle_roll_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_w_yaw"     joint1="waist_yaw_joint"         polycoef="0 0 0 0 0"/>
    <joint name="fix_w_roll"    joint1="waist_roll_joint"        polycoef="0 0 0 0 0"/>
    <joint name="fix_w_pitch"   joint1="waist_pitch_joint"       polycoef="0 0 0 0 0"/>
    <joint name="fix_ls_p"      joint1="left_shoulder_pitch_joint" polycoef="0 0 0 0 0"/>
    <joint name="fix_ls_r"      joint1="left_shoulder_roll_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_ls_y"      joint1="left_shoulder_yaw_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_le"        joint1="left_elbow_joint"          polycoef="0 0 0 0 0"/>
    <joint name="fix_lwr_r"     joint1="left_wrist_roll_joint"     polycoef="0 0 0 0 0"/>
    <joint name="fix_lwr_p"     joint1="left_wrist_pitch_joint"    polycoef="0 0 0 0 0"/>
    <joint name="fix_lwr_y"     joint1="left_wrist_yaw_joint"      polycoef="0 0 0 0 0"/>
    <joint name="fix_lt0"       joint1="left_hand_thumb_0_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_lt1"       joint1="left_hand_thumb_1_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_lt2"       joint1="left_hand_thumb_2_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_lm0"       joint1="left_hand_middle_0_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_lm1"       joint1="left_hand_middle_1_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_li0"       joint1="left_hand_index_0_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_li1"       joint1="left_hand_index_1_joint"   polycoef="0 0 0 0 0"/>
    <!-- Jobb csukló + ujjak rögzítve (passzív, kp instabilitás miatt) -->
    <joint name="fix_rwr_r"     joint1="right_wrist_roll_joint"    polycoef="0 0 0 0 0"/>
    <joint name="fix_rwr_p"     joint1="right_wrist_pitch_joint"   polycoef="0 0 0 0 0"/>
    <joint name="fix_rwr_y"     joint1="right_wrist_yaw_joint"     polycoef="0 0 0 0 0"/>
    <joint name="fix_rt0"       joint1="right_hand_thumb_0_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_rt1"       joint1="right_hand_thumb_1_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_rt2"       joint1="right_hand_thumb_2_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_ri0"       joint1="right_hand_index_0_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_ri1"       joint1="right_hand_index_1_joint"  polycoef="0 0 0 0 0"/>
    <joint name="fix_rm0"       joint1="right_hand_middle_0_joint" polycoef="0 0 0 0 0"/>
    <joint name="fix_rm1"       joint1="right_hand_middle_1_joint" polycoef="0 0 0 0 0"/>
  </equality>
'''

# Equality block a </worldbody> után, </actuator> előtt
new_xml = new_xml.replace(
    '\n  <!-- ===== AKTUÁTOROK =====',
    equality_block + '\n  <!-- ===== AKTUÁTOROK ====='
)

DST_XML.write_text(new_xml)
print(f"✅ Generálva: {DST_XML}")
print(f"   Forrás: {SRC_XML}")
