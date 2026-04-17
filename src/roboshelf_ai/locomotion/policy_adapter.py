"""
LocomotionPolicyAdapter — egy betanított locomotion policy-t
meghívható szolgáltatásként tesz elérhetővé a high-level réteg számára.

Használat (betanított modellel):
    adapter = LocomotionPolicyAdapter("roboshelf-results/loco/v1/best_model.zip")
    action = adapter.step(obs, command)

Használat fejlesztési fázisban (nincs még modell):
    adapter = LocomotionPolicyAdapter(model_path=None)  # DummyAdapter módban fut
    action = adapter.step(obs, command)  # nulla akciót ad vissza
"""

from __future__ import annotations

import logging
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

# G1 aktuátorok száma
G1_DOF = 29


# ---------------------------------------------------------------------------
# Alap adapter interfész
# ---------------------------------------------------------------------------

class BaseLocomotionAdapter:
    """Közös interfész minden locomotion adapter számára."""

    def step(self, obs: np.ndarray, command: LocomotionCommand) -> np.ndarray:
        """Egy lépés végrehajtása.

        Args:
            obs:     A robot aktuális proprioceptív megfigyelése.
            command: A high-level mozgási parancs.

        Returns:
            action (np.ndarray, shape=(G1_DOF,)): Aktuátorparancs.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Adapter állapotának visszaállítása (epizód reset előtt hívandó)."""
        pass

    @property
    def is_dummy(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Dummy adapter — fejlesztési fázishoz, ha még nincs betanított modell
# ---------------------------------------------------------------------------

class DummyLocomotionAdapter(BaseLocomotionAdapter):
    """Nulla akciót ad vissza. Fejlesztési fázisban használandó,
    amikor még nincs betanított locomotion policy.

    Figyelmeztetést logol, hogy ne felejtsd el lecserélni.
    """

    def __init__(self, dof: int = G1_DOF) -> None:
        self._dof = dof
        logger.warning(
            "DummyLocomotionAdapter aktív — nulla akciókat ad vissza. "
            "Cseréld le LocomotionPolicyAdapter-re, ha van betanított modell."
        )

    def step(self, obs: np.ndarray, command: LocomotionCommand) -> np.ndarray:
        return np.zeros(self._dof, dtype=np.float32)

    @property
    def is_dummy(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fő adapter — betanított SB3 PPO modellel
# ---------------------------------------------------------------------------

class LocomotionPolicyAdapter(BaseLocomotionAdapter):
    """Betanított SB3 PPO locomotion policy-t futtat egy lépésenként.

    A command vektort az obs végéhez fűzi, pontosan úgy, ahogy a tanítás
    során a locomotion env csinálja. Ez fontos: az obs formátumnak egyeznie
    kell a train env observation space-ével.

    Args:
        model_path: A betanított .zip modell útvonala. Ha None vagy nem létezik,
                    automatikusan DummyAdapter módba vált.
        command_space: A megengedett parancstartomány (clip-hez).
        device: "cpu" vagy "cuda" (default: "cpu", M2-n MPS float64 problémák miatt).
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
        self._last_action: np.ndarray = np.zeros(G1_DOF, dtype=np.float32)
        self._dummy = DummyLocomotionAdapter()

        if model_path is None:
            logger.info("model_path=None → DummyAdapter módban fut.")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(
                f"Modell nem található: {model_path} → DummyAdapter módban fut."
            )
            return

        self._load_model(model_path)

    def _load_model(self, model_path: Path) -> None:
        """SB3 PPO modell betöltése. Lazy import: csak itt töltjük be az SB3-at."""
        try:
            from stable_baselines3 import PPO  # type: ignore
            self._model = PPO.load(str(model_path), device=self._device)
            logger.info(f"Locomotion modell betöltve: {model_path}")
        except ImportError:
            logger.error(
                "stable_baselines3 nem elérhető. "
                "Futtasd: pip install stable-baselines3"
            )
        except Exception as e:
            logger.error(f"Modell betöltési hiba: {e} → DummyAdapter módban fut.")

    def step(self, obs: np.ndarray, command: LocomotionCommand) -> np.ndarray:
        """Egy locomotion lépés végrehajtása.

        A command-ot validálja, az obs-hoz fűzi, majd a policy-val akciót számít.
        Ha nincs betöltött modell, DummyAdapter-t használ.
        """
        cmd = validate_command(command, self._command_space)
        cmd_vec = np.array(cmd.to_vector(), dtype=np.float32)

        if self._model is None:
            return self._dummy.step(obs, cmd)

        # Command hozzáfűzése az obs-hoz (egyezik a train env obs struktúrával)
        obs_with_cmd = np.concatenate([obs.astype(np.float32), cmd_vec])

        action, _ = self._model.predict(obs_with_cmd, deterministic=True)
        self._last_action = action.astype(np.float32)
        return self._last_action

    def reset(self) -> None:
        self._last_action = np.zeros(G1_DOF, dtype=np.float32)

    @property
    def is_dummy(self) -> bool:
        return self._model is None

    @property
    def last_action(self) -> np.ndarray:
        """Az utolsó kiadott aktuátorparancs (debugginghoz)."""
        return self._last_action.copy()
