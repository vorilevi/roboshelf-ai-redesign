#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 2: G1 lokomóció a retail boltban (v22)

Gymnasium wrapper a Unitree G1 navigációjához a kiskereskedelmi környezetben.
A robot megtanulja a boltban való járást: egyensúly, akadálykerülés, célnavigáció.

Feladat: A G1 a start pozícióból (y=0.5) eljut a raktárhoz (y=3.8).
Reward:
  - Előre haladás jutalma (y irányban)
  - Egyensúly megtartása (nem esik el)
  - Kontroll költség (simább mozgás)
  - Akadály-ütközés büntetés
  - Célba érkezés bonus

Observation: G1 propriocepció (qpos, qvel) + cél relatív pozíciója
Action: 29 DoF nyomaték parancsok

Használat:
  # Regisztrálás után:
  env = gym.make('RoboshelfRetailNav-v0')

  # Vagy közvetlenül:
  env = RoboshelfRetailNavEnv()
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import mujoco


# --- Környezet regisztráció ---
try:
    register(
        id='RoboshelfRetailNav-v0',
        entry_point='roboshelf_retail_nav_env:RoboshelfRetailNavEnv',
        max_episode_steps=1000,
    )
except gym.error.Error:
    pass  # Már regisztrálva


class RoboshelfRetailNavEnv(gym.Env):
    """
    Unitree G1 navigáció a Roboshelf retail bolt környezetben.

    A robot a bolt elején indul (y=0.5) és el kell jutnia a raktárhoz (y=3.8).
    Közben ki kell kerülnie a gondola polcokat és meg kell tartania az egyensúlyt.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Curriculum paraméter előre definiálva (_load_model() printje már használja)
        self.buoyancy_force = 103.0  # N — CurriculumCallback felülírja

        # --- MJCF betöltés ---
        self._load_model()

        # --- Spaces ---
        # Observation: robot qpos (29 joint + 7 freejoint) + qvel (29+6) + target rel pos (2)
        obs_size = self._get_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: 29 aktuátor (G1 joint position targets)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )

        # --- Feladat paraméterek ---
        self.target_pos = np.array([0.0, 3.8])  # Raktár pozíció (x, y)
        self.start_pos = np.array([0.0, 0.5])   # Robot start (x, y)

        # --- Reward súlyok (v22) ---
        # v21 diagnózis: ep=86, reward=-317, dist=3.18m (csak 12cm haladás!)
        #   Fine-tune a v20-ból — a beégett álló/forgó viselkedés nem volt kitanulható
        #   Gyökér ok #1: w_healthy=0.05 > w_dist×max_step=0.04 → álló robot jobban fizet!
        #   Gyökér ok #2: ep=86 lépésnél mindig összeesik → fizikai instabilitás
        #     Unitree hivatalos unitree_rl_gym: G1 soha nem indul egyenes lábakkal!
        #     Egyenes lábból lábemelés → teljes torzó billenés → esés
        #   Gyökér ok #3: csúszó támaszkodó láb → nincs talajerő → esés
        #   Gyökér ok #4: lábak keresztezik egymást → robot saját magában megbotlik
        #
        # v22 változtatások (3 fizikai fix + reward rebalance):
        #
        #   A. GUGGOLÓ ALAPPÓZ [ÚJ v22] — Unitree unitree_rl_gym alapján
        #      hip_pitch: -0.1 rad, knee: +0.3 rad, ankle_pitch: -0.2 rad
        #      Hatás: lejjebb viszi a súlypontot, lábemelés lehetséges torzó-billenés nélkül
        #
        #   B. LÁBCSÚSZÁS BÜNTETÉS [ÚJ v22] — modern legged_gym szerint
        #      w_feet_slip: -0.1 × (talajon lévő láb lineáris sebessége²)
        #      Hatás: a támaszkodó láb nem csúszik hátra, stabil talajerő
        #
        #   C. LÁBAK TÁVOLSÁG + ÜTKÖZÉS BÜNTETÉS [ÚJ v22] — legged_gym szerint
        #      w_feet_distance: -1.0 × max(0, min_dist - actual_dist)² ha túl közel
        #      Hatás: lábak nem keresztezik egymást, robot nem botlik saját magában
        #
        #   D. REWARD REBALANCE [ÚJ v22]
        #      w_healthy: 0.05 → 0.01 (ne fizessen többet az állás mint a haladás)
        #      w_dist: 2.0 → 8.0 (per-step mozgás jutalom > egészség jutalom)
        #      w_orientation: -1.0 → -2.0 (erősebb irányvezérlés)
        #
        #   v21 örökölt változtatások (megmaradnak):
        #      Skálázatlan iránybüntetések (yaw_rate, lateral, hip_yaw, orientation)
        #      Gait reward: w_gait=0.5, vel_air_threshold=0.02

        # Pozitív jutalmak
        self.w_forward = 8.0         # Cél irányú sebesség (max(0, v) clipping)
        self.w_dist = 8.0            # Per-lépés távolságcsökkentés (PBRS) [v22: 2.0→8.0]
        self.w_dist_final = 200.0    # Epizód végi távolság bonus
        self.w_proximity = 2.0       # Közelségi bonus (3.5m küszöbön belül)
        self.w_air_time = 3.0        # Lábemeléses jutalom (haladás közben)
        self.vel_air_threshold = 0.02 # m/s — v20: 0.1 → v21: 0.02 (alacsonyabb küszöb)
        self.w_healthy = 0.01        # Minimális életben-maradás jutalom [v22: 0.05→0.01]
        self.w_goal = 100.0          # Célba érés bonus
        self.w_gait = 0.5            # Gait timing [v21: BE — v20-ban 0.0 volt]

        # Kontroll / fizikai büntetések (NEM skálázottak)
        self.w_ctrl = -0.001         # Aktuátor erőfeszítés (action²)
        self.w_contact = -0.0001     # Kontakt erők (cfrc_ext clip[-1,1]²)

        # Skálázatlan iránybüntetések [v21 — azonnal hatnak, penalty_scale NÉLKÜL]
        self.w_orientation = -2.0    # Yaw eltérés (1-cos) [v22: -1.0→-2.0, erősebb]
        self.w_yaw_rate = -0.5       # Z-tengely körüli szögsebesség: pörgés büntetése
        self.w_lateral = -1.0        # Cél-merőleges (oldalazó) sebesség büntetése
        self.w_hip_yaw = -0.3        # Hip yaw ízületek alapállapottól való eltérése

        # Lábstabilitás büntetések [ÚJ v22 — azonnal hatnak, penalty_scale NÉLKÜL]
        self.w_feet_slip = -0.1      # Talajon lévő láb csúszása (lin_vel² × contact)
        self.w_feet_distance = -1.0  # Lábak közötti min. távolság kényszer (0.15m alatt büntet)
        self.feet_min_distance = 0.15  # m — ennél közelebb ne legyenek a lábak

        # Skálázott simasági büntetések [penalty_scale-lel, 0→1 curriculum]
        self.w_action_rate = -0.005  # Hirtelen akcióváltás büntetése
        self.w_dof_acc = -1e-7       # Ízületi gyorsulás büntetése
        self.w_dof_vel = -5e-5       # Ízületi sebesség [v21: visszakapcsolva, de gyengébb]

        # Terminálási büntetések
        self.w_fall = -20.0          # Esés büntetés
        self.w_stuck = -20.0         # Helyben állás büntetés
        self.w_backward = -20.0      # Tartós hátrafelé haladás büntetés

        # --- Penalty curriculum skálázó ---
        # Csak a simasági büntetésekre vonatkozik (action_rate, dof_acc, dof_vel)
        # Az iránybüntetések (orientation, yaw_rate, lateral, hip_yaw) NEM skálázottak
        # FONTOS: alapértelmezés 1.0 — standalone futásnál teljes büntetés
        self.penalty_scale = 1.0     # CurriculumCallback írja (0→1 a tanítás során)

        # --- No-backward detekció (grace period + ablak) ---
        self.backward_window = 30    # lépés ablak (~0.6s) — a grace period után aktív
        self.backward_threshold = -0.2  # m/s — ez alatt "hátrafelé megy"
        self.grace_period = 150      # [ÚJ v20] lépések (~3s) — nincs backward check
        self._backward_history = []
        self._episode_steps = 0      # [ÚJ v20] lépésszámláló a grace period-hoz

        # --- Curriculum paraméterek (CurriculumCallback frissíti) ---
        # buoyancy_force: Z-irányú külső erő a pelvis-en [N], 0 = kikapcsolva
        # stuck_window: CurriculumCallback írja felül lépésszám alapján
        # G1 tömeg ~35 kg, g=9.81 → súly ≈ 343N
        # FIGYELEM: G1 URDF tartalmazhat virtuális torzó linket is!
        #   Ha csak a pelvis-re hat: 206N = 60% → biztonságos
        #   Ha a MuJoCo mindkét torzóra szétosztja: 206N × 2 = 412N > súly → repül!
        # Konzervatív beállítás: 103N (30%) — elég a kikönnyítéshez, nem repül fel
        # Az első futásnál ellenőrizd: torso_z > 1.5m? → repül, csökkenteni kell
        self.buoyancy_force = 103.0  # N (gravitáció 30%-a, konzervatív) — CurriculumCallback nullázza
        self.vel_stuck_threshold = 0.15  # m/s — küszöb változatlan
        self.stuck_window = 9999     # lépés — CurriculumCallback csökkenti (kezdetben: ki)
        self._vel_history = []

        # --- Gait paraméterek (ciklikus lépésminta) ---
        # 0.8s periódus = 1.25 lépés/mp, 50% offset = szimmetrikus bal-jobb váltás
        self._gait_period = 0.8    # másodperc
        self._gait_offset = 0.5   # jobb láb fázis offsetje (50% = szimmetrikus)
        self._stance_threshold = 0.55  # gait ciklus 55%-a stance, 45% swing
        self._episode_time = 0.0  # epizód eltelt ideje (másodperc)

        # --- Láb body indexek (betöltés után keressük meg) ---
        self._left_foot_id = None
        self._right_foot_id = None

        # --- Állapot ---
        self._healthy_z_range = (0.5, 1.5)  # G1 törzs magasság tartomány
        self._prev_dist_to_target = None

        # Renderer
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, width=640, height=480)

    def _load_model(self):
        """Betölti a G1 + retail bolt kombinált MJCF modellt."""
        # Keressük a G1 modellt
        g1_candidates = [
            # Környezeti változó — Kaggle/Colab esetén ez az elsődleges
            os.environ.get("G1_MODEL_PATH"),
            # Kaggle klón (manuálisan klónozott menagerie)
            "/kaggle/working/mujoco_menagerie/unitree_g1",
            # MacBook Homebrew miniforge
            "/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
            # pip-pel telepített mujoco_playground
            self._find_in_site_packages("mujoco_playground/external_deps/mujoco_menagerie/unitree_g1"),
            # Standalone menagerie
            self._find_in_site_packages("mujoco_menagerie/unitree_g1"),
            # Lokális klón
            os.path.expanduser("~/mujoco_menagerie/unitree_g1"),
        ]

        g1_dir = None
        for path in g1_candidates:
            if path and os.path.exists(os.path.join(str(path), "g1.xml")):
                g1_dir = str(path)
                break

        if g1_dir is None:
            # Fallback: MuJoCo Playground telepítése szükséges
            raise FileNotFoundError(
                "Unitree G1 MJCF nem található! Telepítsd: pip install playground\n"
                "Vagy klónozd: git clone https://github.com/google-deepmind/mujoco_menagerie.git"
            )

        # Retail bolt XML keresése
        store_xml = self._find_store_xml()

        # Kombinált XML létrehozása (a G1 mappából, hogy a meshdir helyes legyen)
        import tempfile
        combined_xml = f"""<mujoco model="roboshelf_g1_retail_nav">
  <include file="g1.xml"/>
  <include file="{store_xml}"/>
</mujoco>"""

        # Process-specifikus temp fájlnév: elkerüli a párhuzamos env-ek konfliktusát
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', dir=g1_dir,
            prefix='_roboshelf_nav_', delete=False
        )
        tmp.write(combined_xml)
        tmp.close()
        self._combined_xml_path = tmp.name

        self.model = mujoco.MjModel.from_xml_path(self._combined_xml_path)
        self.data = mujoco.MjData(self.model)

        # G1 torso body index megkeresése
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self._torso_id == -1:
            # Fallback: első nem-world body
            self._torso_id = 1

        # G1 robot tömege (fix spec érték — body_mass az egész scene-t tartalmazza!)
        G1_MASS_KG = 35.0  # Unitree G1 spec: ~35 kg
        g1_weight_N = G1_MASS_KG * 9.81  # ~343 N
        print(f"  ✅ Retail Nav env betöltve: {self.model.nbody} body, {self.model.nu} actuators")
        print(f"  ℹ️  G1 tömeg (spec): {G1_MASS_KG} kg, súly: {g1_weight_N:.1f} N | "
              f"Felhajtóerő: {self.buoyancy_force:.1f} N ({self.buoyancy_force/g1_weight_N*100:.1f}%)")

        # Láb body indexek megkeresése (G1: ankle_roll_link)
        foot_candidates = [
            ("left_ankle_roll_link",  "right_ankle_roll_link"),   # G1
            ("left_ankle_link",       "right_ankle_link"),         # alternatív
            ("left_foot",             "right_foot"),               # generikus
        ]
        for left_name, right_name in foot_candidates:
            l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, left_name)
            r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, right_name)
            if l != -1 and r != -1:
                self._left_foot_id = l
                self._right_foot_id = r
                print(f"  ✅ Láb body-k: [{l}]{left_name}, [{r}]{right_name}")
                break
        if self._left_foot_id is None:
            print("  ⚠️  Láb body-k nem találhatók — contact pattern reward kikapcsolva")

    def _find_in_site_packages(self, subpath):
        """Keres egy útvonalat a site-packages-ben."""
        import sys
        for p in sys.path:
            if "site-packages" in p:
                full = os.path.join(p, subpath)
                if os.path.exists(full):
                    return full
        return None

    def _find_store_xml(self):
        """Megkeresi a retail bolt XML-t."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "assets", "roboshelf_retail_store.xml"),
            os.path.join(os.getcwd(), "src", "envs", "assets", "roboshelf_retail_store.xml"),
            os.path.join(os.getcwd(), "roboshelf_retail_store.xml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)

        raise FileNotFoundError(
            "roboshelf_retail_store.xml nem található!\n"
            f"Keresett helyek: {candidates}"
        )

    def _get_obs_size(self):
        """Kiszámítja az observation vektor méretét."""
        # qpos (nq) + qvel (nv) + target_rel_xy (2)
        return self.model.nq + self.model.nv + 2

    def _get_obs(self):
        """Összeállítja az observation vektort."""
        # Robot pozíció (x,y a padlón)
        robot_xy = self.data.body(self._torso_id).xpos[:2]
        target_rel = self.target_pos - robot_xy

        obs = np.concatenate([
            self.data.qpos.flat.copy(),
            self.data.qvel.flat.copy(),
            target_rel,
        ]).astype(np.float32)
        return obs

    def _is_healthy(self):
        """Ellenőrzi, hogy a robot áll-e (nem esett el)."""
        torso_z = self.data.body(self._torso_id).xpos[2]
        return self._healthy_z_range[0] < torso_z < self._healthy_z_range[1]

    def _compute_reward(self):
        """Kiszámítja a reward-ot (v21: skálázatlan iránybüntetések + gait bekapcsolva)."""
        robot_xy = self.data.body(self._torso_id).xpos[:2]
        dist_to_target = np.linalg.norm(self.target_pos - robot_xy)

        # Cél irány és sebesség komponensek
        direction_to_target = self.target_pos - robot_xy
        dist_norm = np.linalg.norm(direction_to_target) + 1e-6
        dir_unit = direction_to_target / dist_norm
        # cvel formátum: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
        cvel = self.data.body(self._torso_id).cvel
        lin_vel_xy = cvel[3:5]              # vízszintes sebesség vektor (x, y)
        ang_vel_z  = cvel[2]                # Z-tengely körüli szögsebesség (pörgés)
        forward_component = np.dot(lin_vel_xy, dir_unit)  # cél irányú sebesség [m/s]
        # Laterális (oldalazó) sebesség: teljes vízszintes − cél irányú komponens
        lateral_vel = lin_vel_xy - forward_component * dir_unit
        lateral_speed = np.linalg.norm(lateral_vel)

        # 1. Előre haladás jutalma (max(0,v) clipping — negatív forward = 0, nem negatív)
        vel_tracking = self.w_forward * max(0.0, forward_component)

        # --- SKÁLÁZATLAN IRÁNYBÜNTETÉSEK [ÚJ v21] ---
        # v20 hiba: orientation * penalty_scale = 0 korai fázisban → pörgés beégett
        # v21 fix: ezek azonnal hatnak, penalty_scale NÉLKÜL

        # 1b. Orientáció büntetés: yaw eltérés a cél iránytól (NEM skálázott)
        torso_xmat = self.data.body(self._torso_id).xmat.reshape(3, 3)
        robot_forward = torso_xmat[:2, 1]  # robot Y-tengelye (előre néző irány)
        robot_forward_unit = robot_forward / (np.linalg.norm(robot_forward) + 1e-6)
        cos_yaw_error = np.dot(robot_forward_unit, dir_unit)  # 1=célra néz, -1=háttal áll
        orientation_penalty = self.w_orientation * (1.0 - cos_yaw_error)  # max: -2.0

        # 1c. Yaw szögsebesség büntetés: Z-tengely körüli forgás (NEM skálázott)
        # Pörgés közvetlen büntetése — nem az eredményt (rossz irány), hanem a mozgást
        yaw_rate_penalty = self.w_yaw_rate * abs(ang_vel_z)  # pl. -0.5 × |ω_z|

        # 1d. Laterális sebesség büntetés: oldalazás büntetése (NEM skálázott)
        # Kiszámítva a cél-merőleges sebességkomponensből (irányfüggetlen)
        lateral_penalty = self.w_lateral * lateral_speed  # pl. -1.0 × |v_lateral|

        # 1e. Hip yaw büntetés: csípő-kifordító ízületek alapállapottól való eltérése
        # G1 lábak sorrendje: hip_pitch[0/6], hip_roll[1/7], hip_yaw[2/8], ...
        # qpos[7:] = lábak (freejoint = qpos[0:7])
        # Bal hip_yaw: qpos[9] (7 + 2), Jobb hip_yaw: qpos[15] (7 + 6 + 2)
        hip_yaw_penalty = 0.0
        if self.model.nq >= 16:
            q_hip_yaw_bal  = self.data.qpos[9]   # bal hip_yaw
            q_hip_yaw_jobb = self.data.qpos[15]  # jobb hip_yaw
            hip_yaw_penalty = self.w_hip_yaw * (q_hip_yaw_bal**2 + q_hip_yaw_jobb**2)

        # No-backward detekció előzmény frissítése
        self._backward_history.append(forward_component)
        if len(self._backward_history) > self.backward_window:
            self._backward_history.pop(0)

        # 2. Potential-based távolságcsökkentés (Ng et al. 1999)
        dist_delta = (self._prev_dist_to_target - dist_to_target)
        dist_shaping = self.w_dist * dist_delta
        self._prev_dist_to_target = dist_to_target

        # 3. Közelségi bonus (3.5m küszöbön belül, lineáris)
        PROXIMITY_THRESHOLD = 3.5
        proximity_bonus = 0.0
        if dist_to_target < PROXIMITY_THRESHOLD:
            proximity_bonus = self.w_proximity * (PROXIMITY_THRESHOLD - dist_to_target) / PROXIMITY_THRESHOLD

        # 4. Életben-maradás jutalom
        healthy_reward = self.w_healthy if self._is_healthy() else 0.0

        # 5. Kontroll költség (NEM skálázott)
        ctrl_cost = self.w_ctrl * np.sum(np.square(self._last_action))

        # 6. Kontakt erő büntetés (cfrc_ext clip[-1,1] — Gymnasium Humanoid-v4 szerint)
        contact_forces = np.clip(self.data.cfrc_ext.flat.copy(), -1.0, 1.0)
        contact_cost = self.w_contact * np.sum(np.square(contact_forces))

        # 7. Célba érés bonus
        goal_bonus = 0.0
        if dist_to_target < 0.5:
            goal_bonus = self.w_goal

        # 8. Esés büntetés (termináláskor egyszer)
        fall_penalty = self.w_fall if not self._is_healthy() else 0.0

        # 9. Lábemeléses jutalom [v21: vel_air_threshold 0.1→0.02 m/s]
        # Alacsony küszöb: lassú járás esetén is aktiválódik (v20-ban sosem aktiválódott)
        air_time_reward = 0.0
        if self._left_foot_id is not None and self.w_air_time > 0.0:
            if forward_component > self.vel_air_threshold:
                left_contact  = self.data.cfrc_ext[self._left_foot_id,  2] > 1.0
                right_contact = self.data.cfrc_ext[self._right_foot_id, 2] > 1.0
                n_air = (not left_contact) + (not right_contact)
                air_time_reward = self.w_air_time * n_air

        # 10. Skálázott simasági büntetések (penalty_scale 0→1 a curriculum során)
        # a) Hirtelen akcióváltás büntetése
        action_rate_cost = self.w_action_rate * np.sum(
            np.square(self._last_action - self._prev_action)
        ) * self.penalty_scale
        # b) Ízületi gyorsulás büntetése
        dof_acc_cost = self.w_dof_acc * np.sum(
            np.square(self.data.qacc[6:])
        ) * self.penalty_scale
        # c) Ízületi sebesség büntetése [v21: visszakapcsolva, de gyengébb: -5e-5]
        dof_vel_cost = self.w_dof_vel * np.sum(
            np.square(self.data.qvel[6:])
        ) * self.penalty_scale

        # 11. Gait timing jutalom [v21: BEKAPCSOLVA — w_gait=0.5]
        # v20: kikapcsolva (0.0) → nem volt ösztönzés valódi járásra
        # v21: bekapcsolva — robot elég stabil (ep=138) ahhoz, hogy gait-et tanuljon
        gait_reward = 0.0
        if self._left_foot_id is not None and self.w_gait > 0.0:
            phase = (self._episode_time % self._gait_period) / self._gait_period
            left_should_contact  = phase < self._stance_threshold
            right_should_contact = (phase + self._gait_offset) % 1.0 < self._stance_threshold
            left_contact  = self.data.cfrc_ext[self._left_foot_id,  2] > 1.0
            right_contact = self.data.cfrc_ext[self._right_foot_id, 2] > 1.0
            left_match  = not (left_contact  ^ left_should_contact)
            right_match = not (right_contact ^ right_should_contact)
            gait_reward = self.w_gait * (float(left_match) + float(right_match))

        # 12. Lábcsúszás büntetés [ÚJ v22 — penalty_scale NÉLKÜL, azonnal hat]
        # Ha a láb érinti a talajt (kontakt erő > 1N) ÉS mozog → csúszás → büntetés
        # Forrás: legged_gym feet_slip komponens
        feet_slip_penalty = 0.0
        if self._left_foot_id is not None and self.w_feet_slip != 0.0:
            for foot_id in (self._left_foot_id, self._right_foot_id):
                foot_contact = self.data.cfrc_ext[foot_id, 2] > 1.0  # z-irányú erő > 1N
                if foot_contact:
                    # Láb lineáris sebessége (world frame) — cvel [3:6] = lin vel
                    foot_vel = self.data.body(foot_id).cvel[3:5]  # x,y komponens
                    feet_slip_penalty += self.w_feet_slip * np.sum(np.square(foot_vel))

        # 13. Lábak közötti távolság büntetés [ÚJ v22 — penalty_scale NÉLKÜL]
        # Ha a két lábfej egymáshoz túl közel kerül → lábkeresztezés/ütközés büntetése
        # Forrás: legged_gym feet_distance komponens
        feet_distance_penalty = 0.0
        if self._left_foot_id is not None and self.w_feet_distance != 0.0:
            left_pos  = self.data.body(self._left_foot_id).xpos[:2]   # x,y
            right_pos = self.data.body(self._right_foot_id).xpos[:2]  # x,y
            foot_dist = np.linalg.norm(left_pos - right_pos)
            if foot_dist < self.feet_min_distance:
                violation = self.feet_min_distance - foot_dist
                feet_distance_penalty = self.w_feet_distance * (violation ** 2)

        total_reward = (vel_tracking
                        + orientation_penalty + yaw_rate_penalty
                        + lateral_penalty + hip_yaw_penalty
                        + dist_shaping + proximity_bonus
                        + healthy_reward + ctrl_cost + contact_cost
                        + goal_bonus + fall_penalty + air_time_reward
                        + action_rate_cost + dof_acc_cost + dof_vel_cost
                        + gait_reward
                        + feet_slip_penalty + feet_distance_penalty)

        info = {
            "forward_reward": vel_tracking,
            "forward_component": forward_component,
            "orientation_penalty": orientation_penalty,
            "yaw_rate_penalty": yaw_rate_penalty,
            "lateral_penalty": lateral_penalty,
            "hip_yaw_penalty": hip_yaw_penalty,
            "cos_yaw_error": cos_yaw_error,
            "ang_vel_z": ang_vel_z,
            "lateral_speed": lateral_speed,
            "dist_shaping": dist_shaping,
            "proximity_bonus": proximity_bonus,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "contact_cost": contact_cost,
            "action_rate_cost": action_rate_cost,
            "dof_acc_cost": dof_acc_cost,
            "dof_vel_cost": dof_vel_cost,
            "goal_bonus": goal_bonus,
            "fall_penalty": fall_penalty,
            "air_time_reward": air_time_reward,
            "gait_reward": gait_reward,
            "feet_slip_penalty": feet_slip_penalty,
            "feet_distance_penalty": feet_distance_penalty,
            "dist_to_target": dist_to_target,
            "dist_delta": dist_delta,
            "torso_z": self.data.body(self._torso_id).xpos[2],
            "robot_xy": robot_xy.copy(),
        }

        return total_reward, info

    def _compute_final_dist_bonus(self, dist_to_target):
        """Epizód végi egyszeri dist bonus (v14).

        A per-lépés dist_shaping túl gyenge (0.002m/lép → semmi).
        Epizód végén egyszer: w_dist_final × (start_dist - final_dist)
        Ha robot 0.5m-t ment: +100 bonus → erősebb mint az összes tracking reward.
        """
        dist_improvement = self._start_dist_to_target - dist_to_target
        return self.w_dist_final * max(dist_improvement, 0.0)  # csak közeledésért!

    def reset(self, seed=None, options=None):
        """Környezet resetelése."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # G1 kezdő pozíció beállítása — a G1 XML "stand" keyframe alapján!
        # Forrás: mujoco_menagerie/unitree_g1/g1.xml <keyframe name="stand">
        # freejoint: [x, y, z, qw, qx, qy, qz]
        # Reset zaj: Humanoid-v4 mintájára (reset_noise_scale=1e-2)
        # Nélküle: determinisztikus reset → policy mindig ugyanazt csinálja (±0 szórás eval-ban)
        noise_scale = 0.01
        noise = self.np_random.uniform(-noise_scale, noise_scale, self.model.nq)

        if self.model.nq > 7:
            self.data.qpos[0] = 0.0 + noise[0]   # x
            self.data.qpos[1] = 0.5 + noise[1]   # y (bolt eleje)
            self.data.qpos[2] = 0.79              # z — nem zajos, maradjon stabil magasság
            self.data.qpos[3] = 1.0               # qw — nem zajos
            self.data.qpos[4:7] = noise[4:7] * 0.001  # kis quaternion zaj
            # Lábak: guggoló alappóz [v22: unitree_rl_gym értékek]
            # qpos[7:19] = 12 láb ízület (bal+jobb: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
            self.data.qpos[7:19] = noise[7:19]    # kis zaj alap
            if self.model.nq >= 19:
                # Bal láb: hip_pitch[7], knee[10], ankle_pitch[11]
                self.data.qpos[7]  = -0.1 + noise[7]   # bal hip_pitch
                self.data.qpos[10] =  0.3 + noise[10]  # bal knee
                self.data.qpos[11] = -0.2 + noise[11]  # bal ankle_pitch
                # Jobb láb: hip_pitch[13], knee[16], ankle_pitch[17]
                self.data.qpos[13] = -0.1 + noise[13]  # jobb hip_pitch
                self.data.qpos[16] =  0.3 + noise[16]  # jobb knee
                self.data.qpos[17] = -0.2 + noise[17]  # jobb ankle_pitch
            # Derék: nulla + kis zaj
            self.data.qpos[19:22] = noise[19:22]
            # Karok: keyframe értékek + kis zaj
            if self.model.nq >= 36:
                self.data.qpos[22] = 0.2  + noise[22]
                self.data.qpos[23] = 0.2  + noise[23]
                self.data.qpos[24] = 0.0  + noise[24]
                self.data.qpos[25] = 1.28 + noise[25]
                self.data.qpos[26:29] = noise[26:29]
                self.data.qpos[29] = 0.2  + noise[29]
                self.data.qpos[30] = -0.2 + noise[30]
                self.data.qpos[31] = 0.0  + noise[31]
                self.data.qpos[32] = 1.28 + noise[32]
                self.data.qpos[33:36] = noise[33:36]

        # ctrl alapértékek beállítása a keyframe alapján
        # Position control: ctrl = célpozíció. Nulla ctrl ≠ egyensúly!
        # A G1 keyframe ctrl értékei tartják fenn az egyensúlyi pozíciót.
        self._default_ctrl = np.zeros(self.model.nu)
        # G1 aktuátor sorrend (lábak): hip_pitch[0/6], hip_roll[1/7], hip_yaw[2/8],
        #   knee[3/9], ankle_pitch[4/10], ankle_roll[5/11]
        # Derék (3 aktuátor, index 12-14): nulla
        # Bal kar (7 aktuátor, index 15-21): [0.2, 0.2, 0, 1.28, 0, 0, 0]
        # Jobb kar (7 aktuátor, index 22-28): [0.2, -0.2, 0, 1.28, 0, 0, 0]

        # [v22: GUGGOLÓ ALAPPÓZ — Unitree unitree_rl_gym alapján]
        # v19 hiba: hip_pitch=+0.1 rad destabilizált (elesett ~30 lépésen belül)
        # v22 fix: unitree_rl_gym hivatalos értékek — kisebb szögek, konzisztens irány
        #   hip_pitch_joint: -0.1 rad (enyhe előre dőlés)
        #   knee_joint:      +0.3 rad (enyhén hajlított térd)
        #   ankle_pitch_joint: -0.2 rad (boka kompenzáció)
        # Hatás: súlypont lejjebb, lábemelés lehetséges torzó-billenés nélkül
        # Forrás: unitree_rl_gym/envs/g1_config.py default_joint_angles
        if self.model.nu >= 12:
            # Bal láb: hip_pitch[0], knee[3], ankle_pitch[4]
            self._default_ctrl[0] = -0.1   # bal hip_pitch
            self._default_ctrl[3] =  0.3   # bal knee
            self._default_ctrl[4] = -0.2   # bal ankle_pitch
            # Jobb láb: hip_pitch[6], knee[9], ankle_pitch[10]
            self._default_ctrl[6] = -0.1   # jobb hip_pitch
            self._default_ctrl[9] =  0.3   # jobb knee
            self._default_ctrl[10] = -0.2  # jobb ankle_pitch

        if self.model.nu >= 29:
            self._default_ctrl[15] = 0.2
            self._default_ctrl[16] = 0.2
            self._default_ctrl[17] = 0.0
            self._default_ctrl[18] = 1.28
            self._default_ctrl[22] = 0.2
            self._default_ctrl[23] = -0.2
            self._default_ctrl[24] = 0.0
            self._default_ctrl[25] = 1.28

        self.data.ctrl[:] = self._default_ctrl.copy()

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist_to_target = np.linalg.norm(
            self.target_pos - self.data.body(self._torso_id).xpos[:2]
        )
        self._start_dist_to_target = self._prev_dist_to_target  # epizód végi dist bonus
        self._episode_time = 0.0  # Gait fázis időszámláló nullázása
        self._last_action = np.zeros(self.model.nu)   # ctrl cost inicializálás
        self._prev_action = np.zeros(self.model.nu)   # v18: action_rate cost inicializálás
        self._vel_history = []       # stuck-detection előzmény nullázása
        self._backward_history = []  # no-backward detekció nullázása
        self._episode_steps = 0      # [ÚJ v20] grace period számláló nullázása

        obs = self._get_obs()
        info = {"dist_to_target": self._prev_dist_to_target}

        return obs, info

    def step(self, action):
        """Egy szimuláció lépés végrehajtása."""
        # Akció skálázás: [-1, 1] → aktuátor tartomány
        action = np.clip(action, -1.0, 1.0)
        self._episode_steps += 1  # [ÚJ v20] grace period számláló
        self._prev_action = self._last_action.copy()  # v18: action_rate cost (előző lépés akciója)
        self._last_action = action.copy()             # ctrl cost + action_rate alapja

        # Aktuátor vezérlés beállítása
        # Az akció a keyframe default_ctrl körüli offset (nem a tartomány közepe!)
        # Ezzel nulla akció = egyensúlyi pozíció, nem random ctrl_mean
        if self.model.nu > 0:
            ctrl_range = self.model.actuator_ctrlrange
            # Akció skála: 0.3 radian max eltérés az egyensúlyi pozíciótól
            # Teljes tartomány helyett kis perturbációk → stabilabb tanulás
            ACTION_SCALE = 0.3
            raw_ctrl = self._default_ctrl + action * ACTION_SCALE
            self.data.ctrl[:] = np.clip(raw_ctrl, ctrl_range[:, 0], ctrl_range[:, 1])

        # Felhajtóerő alkalmazása (v17 curriculum)
        # xfrc_applied shape: (nbody, 6) — [:3] torque, [3:] force (N, world frame)
        # Z-irány (index 5): felfelé pozitív
        # CurriculumCallback lineárisan csökkenti buoyancy_force-t 0-ra a tanítás során
        # FONTOS: minden lépés előtt be kell állítani (mj_step nullázza)
        if self.buoyancy_force > 0.0:
            self.data.xfrc_applied[self._torso_id, 5] = self.buoyancy_force

        # Fizikai szimuláció (2 sub-step)
        # v13 tanulság: sub-step=1 → tracking reward negatív lesz (cvel kisebb/más irányú)
        # A v11-es policy 2 sub-step fizikán tanult és jól működött (+133 reward)
        # → Visszaállítás 2 sub-stepre
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)

        # Gait időszámláló léptetése
        self._episode_time += 2 * self.model.opt.timestep

        # Observation
        obs = self._get_obs()

        # Reward
        reward, info = self._compute_reward()

        # Terminated: robot elesett
        terminated = not self._is_healthy()

        # v15: Stuck-detection — ha robot 1.5mp-ig áll (v < 0.15 m/s) → terminál
        # Csúszóablak a cél irányú sebességre (forward_component)
        forward_component = info["forward_component"]
        self._vel_history.append(abs(forward_component))
        if len(self._vel_history) > self.stuck_window:
            self._vel_history.pop(0)

        stuck = False
        if (len(self._vel_history) == self.stuck_window
                and np.mean(self._vel_history) < self.vel_stuck_threshold):
            stuck = True
            terminated = True  # korai terminálás

        # No-backward terminálás [v19: bevezetett, v20: grace period hozzáadva]
        # Ha az utolsó backward_window lépésben az átlag sebesség < backward_threshold
        #   → robot tartósan hátrafelé megy → korai terminálás + büntetés
        # v20 változás: grace_period (150 lép, ~3s) — ez alatt NEM ellenőrzi
        #   v19 hiba: 30 lépéses ablak grace period nélkül túl hamar triggerelt
        #   A robot az első 3 másodpercben egyensúlyoz/stabilizálódik → normális hátra sways
        #   Csak grace_period UTÁN: ha robot TÉNYLEG tartósan hátrafelé megy → terminál
        # _backward_history már frissítve van _compute_reward-ban (minden lépésben)
        backward_triggered = False
        if (not terminated
                and self._episode_steps > self.grace_period  # [ÚJ v20] grace period letelt
                and len(self._backward_history) == self.backward_window
                and np.mean(self._backward_history) < self.backward_threshold):
            backward_triggered = True
            terminated = True

        # Truncated: max lépésszám (Gymnasium kezeli)
        truncated = False

        # Epizód végi dist bonus (termináláskor vagy truncation-kor egyszer)
        if terminated or truncated:
            dist_to_target = info["dist_to_target"]
            final_bonus = self._compute_final_dist_bonus(dist_to_target)
            reward += final_bonus
            info["final_dist_bonus"] = final_bonus
            info["dist_improvement"] = self._start_dist_to_target - dist_to_target
            # v15: stuck büntetés az esés penaltyn felül
            if stuck and self._is_healthy():
                reward += self.w_stuck
                info["stuck_penalty"] = self.w_stuck
            else:
                info["stuck_penalty"] = 0.0
            # v19: no-backward büntetés
            if backward_triggered:
                reward += self.w_backward
                info["backward_penalty"] = self.w_backward
            else:
                info["backward_penalty"] = 0.0
        else:
            info["final_dist_bonus"] = 0.0
            info["dist_improvement"] = 0.0
            info["stuck_penalty"] = 0.0
            info["backward_penalty"] = 0.0

        info["stuck"] = stuck
        info["backward_triggered"] = backward_triggered

        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering."""
        if self.render_mode == "rgb_array" and self._renderer is not None:
            mujoco.mj_forward(self.model, self.data)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        """Cleanup."""
        if self._renderer is not None:
            self._renderer.close()
        # Temp fájl törlése
        if hasattr(self, '_combined_xml_path') and os.path.exists(self._combined_xml_path):
            try:
                os.remove(self._combined_xml_path)
            except:
                pass


# === Standalone teszt ===
if __name__ == "__main__":
    print("Roboshelf Retail Nav Env — teszt")
    print("=" * 40)

    env = RoboshelfRetailNavEnv()
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Kezdő távolság a céltól: {info['dist_to_target']:.2f}m")
    print()

    # Random policy teszt
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  Robot elesett a {step}. lépésnél")
            break

    print(f"\n  100 lépés reward: {total_reward:.1f}")
    print(f"  Végső távolság a céltól: {info['dist_to_target']:.2f}m")
    print(f"  Robot magasság: {info['torso_z']:.2f}m")
    print(f"\n✅ Env működik!")
    env.close()
