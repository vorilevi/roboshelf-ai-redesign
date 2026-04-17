"""
Policy interfészek — szimulátortól független absztrakt osztályok.

Hierarchia:
    BasePolicy
    ├── BaseLocomotionPolicy   (low-level: qpos/qvel → aktuátorparancs)
    └── BaseTaskPolicy         (high-level: obs + goal → locomotion command)
        ├── BaseNavPolicy
        └── BaseManipPolicy

Minden konkrét policy (UnitreeRLGymAdapter, NavPPOPolicy, stb.)
valamelyik alap osztályból öröklődik. Így a szimulátorcsere (MuJoCo → Isaac Lab)
nem érinti a felsőbb policy-ket, csak a from_mujoco / from_isaac adaptereket.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from roboshelf_ai.core.interfaces.robot_state import RobotState
from roboshelf_ai.core.interfaces.locomotion_command import LocomotionCommand


# ---------------------------------------------------------------------------
# Alap policy
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    """Minden policy közös őse."""

    @abstractmethod
    def reset(self) -> None:
        """Epizód reset — belső állapot (pl. LSTM hidden state) nullázása."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


# ---------------------------------------------------------------------------
# Low-level locomotion policy
# ---------------------------------------------------------------------------

class BaseLocomotionPolicy(BasePolicy):
    """Low-level policy: robot állapot + parancs → aktuátorparancs.

    A kimenet közvetlenül a szimulátorba megy (ctrl vektor vagy torque).
    Nem tud semmit a retail feladatról, csak a mozgásért felelős.

    Implementálni kell:
        step(state, command) → np.ndarray (aktuátorparancs)
    """

    @abstractmethod
    def step(
        self,
        state: RobotState,
        command: LocomotionCommand,
    ) -> np.ndarray:
        """Egy lépés végrehajtása.

        Args:
            state:   A robot aktuális állapota (szimulátorfüggetlen)
            command: A high-level mozgási parancs

        Returns:
            ctrl (np.ndarray): Aktuátorparancs vagy torque vektor
        """

    @property
    def is_dummy(self) -> bool:
        """True ha nincs betöltött modell (fejlesztési módban fut)."""
        return False


# ---------------------------------------------------------------------------
# High-level task policy alap
# ---------------------------------------------------------------------------

class BaseTaskPolicy(BasePolicy):
    """High-level policy: megfigyelés + cél → locomotion parancs.

    Nem tudja hogyan kell járni — ezt a locomotion policy-re bízza.
    Csak azt dönti el: merre menj, milyen sebességgel, mikor fordulj.

    Implementálni kell:
        act(obs, goal_info) → LocomotionCommand
    """

    @abstractmethod
    def act(
        self,
        obs: np.ndarray,
        goal_info: Dict[str, Any],
    ) -> LocomotionCommand:
        """Egy lépés: megfigyelésből locomotion parancsot számít.

        Args:
            obs:       Az env observation vektora
            goal_info: Feladatspecifikus cél információ
                       Nav-nál: {'goal_xy': np.array([x, y]), 'dist': float}
                       Manip-nál: {'target_pos': np.array([x,y,z]), ...}

        Returns:
            LocomotionCommand — a locomotion policy-nek adott parancs
        """

    def act_deterministic(
        self,
        obs: np.ndarray,
        goal_info: Dict[str, Any],
    ) -> LocomotionCommand:
        """Determinisztikus akció (eval-hoz). Default: ugyanaz mint act()."""
        return self.act(obs, goal_info)


# ---------------------------------------------------------------------------
# Nav policy interfész
# ---------------------------------------------------------------------------

class BaseNavPolicy(BaseTaskPolicy):
    """Retail navigációs policy interfész.

    Bemenete: robot pozíció, célpont iránya/távolsága, akadály info.
    Kimenete: LocomotionCommand (v_forward, yaw_rate).

    Nem tud járni — ezt a locomotion policy-re bízza.
    Nem tud tárgyat fogni — ezt a manipulation policy-re bízza.
    """

    @abstractmethod
    def act(
        self,
        obs: np.ndarray,
        goal_info: Dict[str, Any],
    ) -> LocomotionCommand:
        """Nav-specifikus act: obs tartalmazza a robot pozíciót és cél irányt."""


# ---------------------------------------------------------------------------
# Manipulation policy interfész
# ---------------------------------------------------------------------------

class BaseManipPolicy(BaseTaskPolicy):
    """Manipulációs policy interfész (pick & place).

    Bemenete: kar ízületi állapotok, tárgy pozíció, gripper állapot.
    Kimenete: LocomotionCommand (vagy direkt kar akció — tbd Fázis C-ben).
    """

    @abstractmethod
    def act(
        self,
        obs: np.ndarray,
        goal_info: Dict[str, Any],
    ) -> LocomotionCommand:
        """Manip-specifikus act."""


# ---------------------------------------------------------------------------
# SB3-alapú task policy wrapper
# ---------------------------------------------------------------------------

class SB3TaskPolicy(BaseTaskPolicy):
    """Stable-Baselines3 PPO modellt csomagolja BaseTaskPolicy-ba.

    Így bármely SB3-ban tanított nav vagy manipulation policy
    felcserélhető anélkül hogy a hívó kód változna.
    """

    def __init__(
        self,
        model_path: str,
        command_space,
        device: str = "cpu",
    ) -> None:
        self._model = None
        self._command_space = command_space
        self._device = device
        self._load(model_path)

    def _load(self, model_path: str) -> None:
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(model_path, device=self._device)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"SB3 modell betöltési hiba: {e}")

    def reset(self) -> None:
        pass  # SB3 PPO-nak nincs belső epizód state-je

    def act(
        self,
        obs: np.ndarray,
        goal_info: Dict[str, Any],
    ) -> LocomotionCommand:
        if self._model is None:
            return LocomotionCommand()  # stop
        action, _ = self._model.predict(obs, deterministic=False)
        # Action → LocomotionCommand konverzió
        # Feltételezzük: action = [v_forward, yaw_rate] (2D)
        v_fwd = float(np.clip(action[0], *self._command_space.v_forward_range))
        yaw   = float(np.clip(action[1], *self._command_space.yaw_rate_range))
        return LocomotionCommand(v_forward=v_fwd, yaw_rate=yaw)

    def act_deterministic(
        self,
        obs: np.ndarray,
        goal_info: Dict[str, Any],
    ) -> LocomotionCommand:
        if self._model is None:
            return LocomotionCommand()
        action, _ = self._model.predict(obs, deterministic=True)
        v_fwd = float(np.clip(action[0], *self._command_space.v_forward_range))
        yaw   = float(np.clip(action[1], *self._command_space.yaw_rate_range))
        return LocomotionCommand(v_forward=v_fwd, yaw_rate=yaw)
