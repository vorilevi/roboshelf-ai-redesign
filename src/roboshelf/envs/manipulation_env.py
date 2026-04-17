#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 3: G1 Polcfeltöltés (Manipuláció)

FELADAT:
  A G1 robot a raktárasztalról felvesz egy terméket (doboz/henger),
  és a megjelölt üres polchelyre helyezi (pl. pA1_3 — "out-of-stock" slot).

REWARD struktúra:
  1. Megközelítés:   robot keze → termék távolság csökkentése (+)
  2. Megfogás:       termék kontaktusa a kézzel → bónusz (+)
  3. Szállítás:      termék magassága nő (felemelik) (+)
  4. Elhelyezés:     termék a célpozícióba kerül (+++)
  5. Egyensúly:      robot nem esik el (+)
  6. Kontroll cost:  sima mozgásra ösztönzés (-)

OBSERVATION (propriocepció + task):
  - G1 joint qpos, qvel
  - Jobb kéz vége (end-effector) pozíciója
  - Termék pozíciója + orientációja
  - Célpozíció relatív helyzetei
  - Kontakt szenzor (kéz-termék érintkezés)

ACTION:
  - 29 DoF: G1 joint position targets (kar + ujj aktuátorok)

ÁLLAPOTGÉP:
  APPROACH  → közelíts a termékhez
  GRASP     → fogd meg
  LIFT      → emeld fel
  TRANSPORT → vidd a célpozícióba
  PLACE     → tedd le a polcra

TODO (Fázis 3 implementáció):
  [ ] G1 ujj/kéz aktuátorok ellenőrzése a g1.xml-ben
  [ ] Inverse kinematics (IK) vagy end-effector space action
  [ ] Domain randomization (termék pozíció, súly, súrlódás)
  [ ] Demonstrációs adatok (opcionális, imitation learning-hez)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import mujoco
from pathlib import Path

# --- Regisztráció ---
try:
    register(
        id='RoboshelfManipulation-v0',
        entry_point='roboshelf_manipulation_env:RoboshelfManipulationEnv',
        max_episode_steps=2000,
    )
except gym.error.Error:
    pass


# --- Állapotgép ---
class TaskState:
    APPROACH  = 0   # Közelíts a termékhez
    GRASP     = 1   # Fogd meg
    LIFT      = 2   # Emeld fel (min. 0.2m a polc felett)
    TRANSPORT = 3   # Vidd a célpozícióba
    PLACE     = 4   # Tedd le
    DONE      = 5   # Siker


class RoboshelfManipulationEnv(gym.Env):
    """
    Unitree G1 polcfeltöltés a Roboshelf retail bolt környezetben.

    A robot a raktárasztalról (y=3.8) felvesz egy terméket és
    a megjelölt üres polchelyre (gondola A, 1. polc, 3. slot) viszi.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Fizikai paraméterek
    GRASP_DIST_THRESHOLD = 0.05    # 5 cm — "megfogva" ha a kéz ennyire van
    PLACE_DIST_THRESHOLD = 0.06    # 6 cm — "lerakva" ha a termék a célnál van
    LIFT_HEIGHT_MIN      = 0.20    # 20 cm — minimális felemelésmag

    # Reward súlyok
    W_APPROACH   =  2.0   # Közelítési jutalom (per lépés delta távolság)
    W_GRASP      = 20.0   # Megfogás bónusz (egyszer)
    W_LIFT       =  5.0   # Emelési jutalom (per lépés delta magasság)
    W_TRANSPORT  =  3.0   # Szállítási jutalom (termék → cél delta)
    W_PLACE      = 200.0  # Elhelyezési bónusz (egyszer)
    W_HEALTHY    =  1.0   # Egyensúly jutalom (per lépés)
    W_CTRL       = -0.005 # Kontroll cost

    def __init__(self, render_mode=None, randomize=False):
        super().__init__()
        self.render_mode = render_mode
        self.randomize = randomize  # Domain randomization (Fázis 3b)

        self._load_model()

        # --- Observation space ---
        # qpos(nq) + qvel(nv) + ee_pos(3) + obj_pos(3) + obj_quat(4) + target_pos(3) + contact(1)
        obs_dim = self.model.nq + self.model.nv + 3 + 3 + 4 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # --- Action space: 29 DoF joint position targets ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )

        # --- Feladat pozíciók ---
        # Termék: raktárasztal tetején (kb. x=-0.7, y=3.8+0.25, z=0.815+0.15+0.055)
        self.obj_start_pos  = np.array([-0.70, 4.05,  1.02])
        # Cél: Gondola A, 1. polc, 3. slot (az üres hely a store.xml-ben: x=-1.25, y=2.05, z=0.167)
        self.obj_target_pos = np.array([-1.25, 2.05, 0.167])

        # --- Állapot ---
        self.task_state = TaskState.APPROACH
        self._prev_approach_dist = None
        self._prev_obj_height    = None
        self._prev_transport_dist = None
        self._grasped = False
        self._placed  = False

        # Body / site indexek (betöltés után töltjük ki)
        self._torso_id   = None
        self._obj_id     = None
        self._ee_site_id = None   # jobb kéz end-effector site

        self._resolve_ids()

        # Renderer
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, width=640, height=480)

    # ─────────────────────────────────────────────────────────────
    # Modell betöltés
    # ─────────────────────────────────────────────────────────────

    def _load_model(self):
        """Betölti a G1 + retail bolt kombinált MJCF modellt."""
        # G1 keresés (ugyanaz mint a nav env-ben)
        g1_dir = self._find_g1()
        if g1_dir is None:
            raise FileNotFoundError(
                "Unitree G1 MJCF nem található!\n"
                "pip install mujoco-playground  VAGY\n"
                "git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie"
            )

        store_xml = self._find_store_xml()

        # Kombinált XML — G1 mappájából töltjük, hogy a mesh útvonalak helyesek legyenek
        combined = f"""<mujoco model="roboshelf_g1_manipulation">
  <include file="g1.xml"/>
  <include file="{store_xml}"/>
</mujoco>"""

        tmp_path = os.path.join(g1_dir, "_roboshelf_manip_tmp.xml")
        with open(tmp_path, "w") as f:
            f.write(combined)

        self.model = mujoco.MjModel.from_xml_path(tmp_path)
        self.data  = mujoco.MjData(self.model)
        print(f"  ✅ Manipulation env betöltve: {self.model.nbody} body, {self.model.nu} aktuátor")

    def _find_g1(self):
        candidates = []
        for p in sys.path:
            if "site-packages" in p:
                for sub in [
                    "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
                    "mujoco_menagerie/unitree_g1",
                ]:
                    candidates.append(os.path.join(p, sub))
        candidates += [
            os.path.expanduser("~/mujoco_menagerie/unitree_g1"),
            os.path.expanduser("~/Documents/mujoco_menagerie/unitree_g1"),
        ]
        for c in candidates:
            if c and os.path.exists(os.path.join(c, "g1.xml")):
                return c
        return None

    def _find_store_xml(self):
        candidates = [
            os.path.join(os.path.dirname(__file__), "assets", "roboshelf_retail_store.xml"),
            os.path.join(os.getcwd(), "src", "envs", "assets", "roboshelf_retail_store.xml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        raise FileNotFoundError(f"roboshelf_retail_store.xml nem található!")

    def _resolve_ids(self):
        """Megkeresi a szükséges body/site indexeket."""
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self._torso_id < 0:
            self._torso_id = 1

        # A pick target termék neve a store XML-ben: "pA1_3" lenne az üres slot,
        # de egy meglévő terméket fogunk — crate_1 a raktárasztalon
        # (a freejoint-os termékek közül az egyiket pick-eljük)
        # TODO: dinamikus termékválasztás, most hardkódolt
        obj_candidates = ["crate_1", "pA1_1", "pA2_1"]
        for name in obj_candidates:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx >= 0:
                self._obj_id = idx
                break
        if self._obj_id is None:
            # fallback: az utolsó freejoint-os body
            self._obj_id = self.model.nbody - 1

        # End-effector site: G1-ben "right_wrist" vagy hasonló
        ee_candidates = ["right_wrist", "right_hand", "right_palm", "R_wrist_yaw_link"]
        self._ee_site_id = None
        for name in ee_candidates:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx >= 0:
                self._ee_site_id = idx
                break
        if self._ee_site_id is None:
            # fallback: jobbkéz body keresés név alapján
            for i in range(self.model.nbody):
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                if "right" in bname.lower() and ("wrist" in bname.lower() or "hand" in bname.lower()):
                    self._ee_site_id = i
                    break
        if self._ee_site_id is None:
            self._ee_site_id = self._torso_id  # végső fallback

    # ─────────────────────────────────────────────────────────────
    # Observation + Reward
    # ─────────────────────────────────────────────────────────────

    def _get_ee_pos(self):
        return self.data.body(self._ee_site_id).xpos.copy()

    def _get_obj_pos(self):
        return self.data.body(self._obj_id).xpos.copy()

    def _get_obj_quat(self):
        return self.data.body(self._obj_id).xquat.copy()

    def _is_healthy(self):
        torso_z = self.data.body(self._torso_id).xpos[2]
        return 0.5 < torso_z < 1.5

    def _has_contact_with_obj(self):
        """Egyszerűsített: a kéz és a termék érintkeznek-e."""
        ee_pos  = self._get_ee_pos()
        obj_pos = self._get_obj_pos()
        return np.linalg.norm(ee_pos - obj_pos) < self.GRASP_DIST_THRESHOLD

    def _get_obs(self):
        ee_pos  = self._get_ee_pos()
        obj_pos = self._get_obj_pos()
        obj_quat = self._get_obj_quat()
        contact = float(self._has_contact_with_obj())

        obs = np.concatenate([
            self.data.qpos.flat.copy(),           # robot joint pozíciók
            self.data.qvel.flat.copy(),           # robot joint sebességek
            ee_pos,                               # jobb kéz pozíciója
            obj_pos,                              # termék pozíciója
            obj_quat,                             # termék orientációja
            self.obj_target_pos,                  # célpozíció
            [contact],                            # kéz-termék érintkezés
        ])
        return obs

    def _compute_reward(self):
        ee_pos  = self._get_ee_pos()
        obj_pos = self._get_obj_pos()
        reward  = 0.0
        info    = {}

        # 1. Egyensúly
        if self._is_healthy():
            reward += self.W_HEALTHY

        # 2. Kontroll cost
        ctrl_cost = self.W_CTRL * np.sum(np.square(self.data.ctrl))
        reward += ctrl_cost

        # 3. Állapotgép-alapú reward
        if self.task_state == TaskState.APPROACH:
            dist = np.linalg.norm(ee_pos - obj_pos)
            if self._prev_approach_dist is not None:
                reward += self.W_APPROACH * (self._prev_approach_dist - dist)
            self._prev_approach_dist = dist
            # Átmenet: ha elég közel
            if dist < self.GRASP_DIST_THRESHOLD:
                self.task_state = TaskState.GRASP

        elif self.task_state == TaskState.GRASP:
            if self._has_contact_with_obj():
                reward += self.W_GRASP
                self._grasped = True
                self.task_state = TaskState.LIFT
                self._prev_obj_height = obj_pos[2]

        elif self.task_state == TaskState.LIFT:
            dh = obj_pos[2] - (self._prev_obj_height or obj_pos[2])
            reward += self.W_LIFT * max(dh, 0)
            self._prev_obj_height = obj_pos[2]
            if obj_pos[2] > self.obj_start_pos[2] + self.LIFT_HEIGHT_MIN:
                self.task_state = TaskState.TRANSPORT
                self._prev_transport_dist = np.linalg.norm(obj_pos - self.obj_target_pos)

        elif self.task_state == TaskState.TRANSPORT:
            dist = np.linalg.norm(obj_pos - self.obj_target_pos)
            if self._prev_transport_dist is not None:
                reward += self.W_TRANSPORT * (self._prev_transport_dist - dist)
            self._prev_transport_dist = dist
            if dist < self.PLACE_DIST_THRESHOLD * 3:
                self.task_state = TaskState.PLACE

        elif self.task_state == TaskState.PLACE:
            dist = np.linalg.norm(obj_pos - self.obj_target_pos)
            reward += self.W_TRANSPORT * max(self._prev_transport_dist or dist - dist, 0)
            self._prev_transport_dist = dist
            if dist < self.PLACE_DIST_THRESHOLD:
                reward += self.W_PLACE
                self._placed = True
                self.task_state = TaskState.DONE

        info = {
            "task_state": self.task_state,
            "ee_pos": ee_pos,
            "obj_pos": obj_pos,
            "target_pos": self.obj_target_pos,
            "grasped": self._grasped,
            "placed": self._placed,
            "healthy": self._is_healthy(),
        }
        return reward, info

    # ─────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Robot start: bolt eleje
        if self.model.nq > 7:
            self.data.qpos[0] = 0.0
            self.data.qpos[1] = 0.5
            self.data.qpos[2] = 0.75
            self.data.qpos[3] = 1.0
            self.data.qpos[4:7] = 0.0

        # Domain randomization (Fázis 3b)
        if self.randomize and seed is not None:
            rng = np.random.default_rng(seed)
            # Termék pozíció ±3 cm véletlen zaj
            noise = rng.uniform(-0.03, 0.03, size=3)
            noise[2] = 0  # z-t nem randomizáljuk (asztal felett marad)
            self.obj_start_pos = np.array([-0.70, 4.05, 1.02]) + noise

        # Állapotgép reset
        self.task_state = TaskState.APPROACH
        self._prev_approach_dist  = None
        self._prev_obj_height     = None
        self._prev_transport_dist = None
        self._grasped = False
        self._placed  = False

        mujoco.mj_forward(self.model, self.data)

        obs  = self._get_obs()
        info = {"task_state": self.task_state}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        if self.model.nu > 0:
            ctrl_range = self.model.actuator_ctrlrange
            ctrl_mean  = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2
            ctrl_half  = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2
            self.data.ctrl[:] = ctrl_mean + action * ctrl_half

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._compute_reward()

        terminated = (not self._is_healthy()) or (self.task_state == TaskState.DONE)
        truncated  = False

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            mujoco.mj_forward(self.model, self.data)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
