"""
Locomotion policy adapterek.

Három adapter érhető el:

1. DummyLocomotionAdapter
   Nulla akciót ad — fejlesztési fázishoz, ha nincs betanított modell.

2. UnitreeRLGymAdapter  ← ELSŐDLEGES
   A unitree_rl_gym motion.pt LSTM policy-t futtatja MuJoCo-ban,
   pontosan reprodukálva a deploy_mujoco.py logikáját.
   - 47 dim obs vektor (omega, gravity, cmd, qj, dqj, action, gait_phase)
   - 12 dim output (lábak PD target pozíciói)
   - LSTM hidden state kezelés epizódonként
   - 50 Hz control (control_decimation=10, sim_dt=0.002)

3. LocomotionPolicyAdapter
   SB3 PPO modellt futtat — saját tanítású policy-hoz.

Használat:
    from roboshelf_ai.locomotion.policy_adapter import UnitreeRLGymAdapter
    adapter = UnitreeRLGymAdapter("~/unitree_rl_gym/deploy/pre_train/g1/motion.pt")
    ctrl = adapter.step(mj_data, command)
    mj_data.ctrl[:12] = ctrl
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from roboshelf_ai.core.interfaces.locomotion_command import (
    LocomotionCommand,
    LocomotionCommandSpace,
    COMMAND_SPACE_BASIC,
    validate_command,
)

logger = logging.getLogger(__name__)

# G1 teljes DoF
G1_DOF = 29
# unitree_rl_gym motion.pt csak a lábakat vezérli
G1_LEG_DOF = 12

# G1 láb default szögek (unitree_rl_gym g1.yaml default_angles)
# Sorrend: bal [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
#          jobb [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
G1_DEFAULT_ANGLES = np.array([
    -0.1, 0.0, 0.0,  0.3, -0.2, 0.0,   # bal láb
    -0.1, 0.0, 0.0,  0.3, -0.2, 0.0,   # jobb láb
], dtype=np.float32)

# PD gain-ek (g1.yaml)
G1_KPS = np.array([100, 100, 100, 150, 40, 40,
                    100, 100, 100, 150, 40, 40], dtype=np.float32)
G1_KDS = np.array([2, 2, 2, 4, 2, 2,
                    2, 2, 2, 4, 2, 2], dtype=np.float32)

# Skálázási faktorok (g1.yaml)
ANG_VEL_SCALE  = 0.25
DOF_POS_SCALE  = 1.0
DOF_VEL_SCALE  = 0.05
ACTION_SCALE   = 0.25
CMD_SCALE      = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Gait fázis
GAIT_PERIOD = 0.8   # másodperc

# Control decimation (sim_dt=0.002, control_dt=0.02 → 50 Hz policy)
CONTROL_DECIMATION = 10
SIM_DT = 0.002


# ---------------------------------------------------------------------------
# Alap adapter interfész
# ---------------------------------------------------------------------------

class BaseLocomotionAdapter:
    """Közös interfész minden locomotion adapter számára."""

    def step_mujoco(self, data, command: LocomotionCommand) -> np.ndarray:
        """MuJoCo MjData-ból olvas, visszaad aktuátorparancsot.

        Args:
            data:    mujoco.MjData — aktuális szimulációs állapot
            command: LocomotionCommand — high-level mozgási parancs

        Returns:
            ctrl (np.ndarray): aktuátorparancs, shape a konkrét adaptertől függ
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Adapter belső állapotának nullázása (epizód reset előtt hívandó)."""
        pass

    @property
    def is_dummy(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Dummy adapter
# ---------------------------------------------------------------------------

class DummyLocomotionAdapter(BaseLocomotionAdapter):
    """Nulla ctrl-t ad vissza — fejlesztési fázishoz."""

    def __init__(self, dof: int = G1_LEG_DOF) -> None:
        self._dof = dof
        logger.warning(
            "DummyLocomotionAdapter aktív — nulla akciót ad. "
            "Cseréld le UnitreeRLGymAdapter-re."
        )

    def step_mujoco(self, data, command: LocomotionCommand) -> np.ndarray:
        return np.zeros(self._dof, dtype=np.float32)

    @property
    def is_dummy(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# UnitreeRLGymAdapter — motion.pt LSTM policy
# ---------------------------------------------------------------------------

class UnitreeRLGymAdapter(BaseLocomotionAdapter):
    """Unitree RL Gym LSTM policy futtatása MuJoCo-ban.

    Pontosan reprodukálja a deploy_mujoco.py logikáját:
    - 47 dim obs vektor összerakása
    - LSTM hidden/cell state kezelés
    - PD control: torque = (target_q - q)*kp + (0 - dq)*kd
    - control_decimation=10 (50 Hz policy, 500 Hz szimuláció)

    Obs vektor (47 dim):
        [0:3]   omega * 0.25              (szögsebesség)
        [3:6]   gravity_orientation       (gravitáció iránya)
        [6:9]   cmd * [2.0, 2.0, 0.25]   (parancs)
        [9:21]  (qj - default) * 1.0     (láb joint pozíciók, 12D)
        [21:33] dqj * 0.05               (láb joint sebességek, 12D)
        [33:45] prev_action              (előző akció, 12D)
        [45:47] [sin_phase, cos_phase]   (gait fázis)

    Output (12 dim): láb joint pozíció target-ek
    ctrl = action * 0.25 + default_angles

    Args:
        model_path: motion.pt útvonala
        leg_joint_indices: G1 MjData-ban a 12 láb joint qpos indexei
                           (default: qpos[7:19])
        leg_ctrl_indices:  A 12 láb aktuátor ctrl indexei
                           (default: ctrl[0:12])
    """

    def __init__(
        self,
        model_path: str | Path,
        leg_joint_qpos_start: int = 7,    # qpos[7:19] = 12 láb joint
        leg_joint_qvel_start: int = 6,    # qvel[6:18] = 12 láb joint vel
        leg_ctrl_start: int = 0,          # ctrl[0:12] = 12 láb aktuátor
        device: str = "cpu",
    ) -> None:
        self._device = device
        self._qpos_s = leg_joint_qpos_start
        self._qvel_s = leg_joint_qvel_start
        self._ctrl_s = leg_ctrl_start

        self._policy = None
        self._hidden = None
        self._cell   = None
        self._prev_action = np.zeros(G1_LEG_DOF, dtype=np.float32)
        self._sim_counter = 0   # MuJoCo step számláló (gait fázishoz és decimation-hoz)
        self._obs = np.zeros(47, dtype=np.float32)

        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            logger.warning(f"motion.pt nem található: {model_path} → DummyAdapter módban fut.")
            self._dummy = DummyLocomotionAdapter()
            return
        self._dummy = None
        self._load_policy(model_path)

    def _load_policy(self, path: Path) -> None:
        try:
            import torch
            self._torch = torch
            self._policy = torch.jit.load(str(path), map_location=self._device)
            self._policy.eval()
            # LSTM hidden/cell state inicializálás (shape: [1, 1, 64])
            self._hidden = torch.zeros(1, 1, 64)
            self._cell   = torch.zeros(1, 1, 64)
            logger.info(f"motion.pt betöltve: {path}")
        except Exception as e:
            logger.error(f"motion.pt betöltési hiba: {e}")
            self._policy = None
            self._dummy = DummyLocomotionAdapter()

    def reset(self) -> None:
        """Epizód reset — LSTM state és számlálók nullázása."""
        if self._policy is not None:
            self._hidden = self._torch.zeros(1, 1, 64)
            self._cell   = self._torch.zeros(1, 1, 64)
        self._prev_action = np.zeros(G1_LEG_DOF, dtype=np.float32)
        self._sim_counter = 0

    def step_mujoco(self, data, command: LocomotionCommand) -> np.ndarray:
        """Egy MuJoCo szimulációs lépés.

        A policy 50 Hz-en fut (control_decimation=10):
        minden 10. MuJoCo lépésnél frissül az akció.
        Közbülső lépéseknél az előző PD target marad érvényes.

        Returns:
            torque (np.ndarray, shape=(12,)): nyomaték a 12 láb aktuátorra
        """
        if self._dummy is not None:
            return self._dummy.step_mujoco(data, command)

        self._sim_counter += 1

        # Aktuális joint állapot
        qj  = data.qpos[self._qpos_s : self._qpos_s + G1_LEG_DOF].copy().astype(np.float32)
        dqj = data.qvel[self._qvel_s : self._qvel_s + G1_LEG_DOF].copy().astype(np.float32)

        # Policy frissítés minden control_decimation-dik lépésben
        if self._sim_counter % CONTROL_DECIMATION == 0:
            # Obs vektor összerakása (deploy_mujoco.py alapján)
            quat  = data.qpos[3:7].astype(np.float32)   # [qw, qx, qy, qz]
            omega = data.qvel[3:6].astype(np.float32)   # szögsebesség

            gravity = self._get_gravity_orientation(quat)
            cmd_vec = np.array([command.v_forward, command.v_lateral, command.yaw_rate],
                               dtype=np.float32)

            # Gait fázis
            t = self._sim_counter * SIM_DT
            phase = (t % GAIT_PERIOD) / GAIT_PERIOD
            sin_ph = math.sin(2 * math.pi * phase)
            cos_ph = math.cos(2 * math.pi * phase)

            # Obs feltöltés
            self._obs[0:3]   = omega * ANG_VEL_SCALE
            self._obs[3:6]   = gravity
            self._obs[6:9]   = cmd_vec * CMD_SCALE
            self._obs[9:21]  = (qj - G1_DEFAULT_ANGLES) * DOF_POS_SCALE
            self._obs[21:33] = dqj * DOF_VEL_SCALE
            self._obs[33:45] = self._prev_action
            self._obs[45]    = sin_ph
            self._obs[46]    = cos_ph

            # Policy inference (LSTM)
            obs_t = self._torch.from_numpy(self._obs).unsqueeze(0)
            with self._torch.no_grad():
                action_t = self._policy(obs_t)

            action = action_t.detach().numpy().squeeze().astype(np.float32)
            self._prev_action = action.copy()
            self._target_dof_pos = action * ACTION_SCALE + G1_DEFAULT_ANGLES

        # PD control → nyomaték
        target_q = getattr(self, '_target_dof_pos', G1_DEFAULT_ANGLES)
        torque = (target_q - qj) * G1_KPS + (0.0 - dqj) * G1_KDS
        return torque

    @staticmethod
    def _get_gravity_orientation(quat: np.ndarray) -> np.ndarray:
        """Gravitáció iránya a robot test frame-jében (deploy_mujoco.py alapján).
        quat = [qw, qx, qy, qz]
        """
        qw, qx, qy, qz = quat
        g = np.zeros(3, dtype=np.float32)
        g[0] =  2 * (-qz * qx + qw * qy)
        g[1] = -2 * ( qz * qy + qw * qx)
        g[2] =  1 - 2 * (qw * qw + qz * qz)
        return g

    @property
    def is_dummy(self) -> bool:
        return self._policy is None

    @property
    def target_dof_pos(self) -> np.ndarray:
        """Aktuális PD target pozíció (12D) — debugginghoz."""
        return getattr(self, '_target_dof_pos', G1_DEFAULT_ANGLES).copy()


# ---------------------------------------------------------------------------
# SB3 PPO adapter — saját tanítású policy-hoz
# ---------------------------------------------------------------------------

class LocomotionPolicyAdapter(BaseLocomotionAdapter):
    """SB3 PPO locomotion policy adapter — saját tanítású modellhez.

    Ha nincs betanított modell, DummyAdapter módban fut.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        command_space: LocomotionCommandSpace = COMMAND_SPACE_BASIC,
        device: str = "cpu",
    ) -> None:
        self._command_space = command_space
        self._device = device
        self._model = None
        self._last_action = np.zeros(G1_DOF, dtype=np.float32)

        if model_path is None:
            logger.info("model_path=None → DummyAdapter módban fut.")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Modell nem található: {model_path} → DummyAdapter módban fut.")
            return

        self._load_model(model_path)

    def _load_model(self, model_path: Path) -> None:
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(str(model_path), device=self._device)
            logger.info(f"SB3 PPO modell betöltve: {model_path}")
        except Exception as e:
            logger.error(f"Modell betöltési hiba: {e}")

    def step_mujoco(self, data, command: LocomotionCommand) -> np.ndarray:
        cmd = validate_command(command, self._command_space)
        if self._model is None:
            return np.zeros(G1_DOF, dtype=np.float32)
        # SB3 obs: MjData-ból összerak egy obs vektort
        qpos = data.qpos.astype(np.float32)
        qvel = data.qvel.astype(np.float32)
        cmd_vec = np.array(cmd.to_vector(), dtype=np.float32)
        obs = np.concatenate([qpos[2:], qvel, cmd_vec])
        action, _ = self._model.predict(obs, deterministic=True)
        self._last_action = action.astype(np.float32)
        return self._last_action

    def reset(self) -> None:
        self._last_action = np.zeros(G1_DOF, dtype=np.float32)

    @property
    def is_dummy(self) -> bool:
        return self._model is None
