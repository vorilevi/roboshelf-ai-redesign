"""
roboshelf_ai
============
Roboshelf AI redesign — hierarchikus RL csomag Unitree G1 robothoz.

Top-level convenience importok a leggyakrabban használt szimbólumokhoz:

    from roboshelf_ai import RobotState, LocomotionCommand
    from roboshelf_ai import UnitreeRLGymAdapter
    from roboshelf_ai import BaseNavPolicy, BaseManipPolicy

A teljes interfész réteg:
    roboshelf_ai.core.interfaces
    roboshelf_ai.locomotion
    roboshelf_ai.mujoco.envs.*
"""

__version__ = "0.2.0"  # Fázis A kész, Fázis B folyamatban

# --- Core interfészek ---
from roboshelf_ai.core.interfaces import (
    RobotState,
    LocomotionCommand,
    LocomotionCommandSpace,
    COMMAND_SPACE_BASIC,
    COMMAND_SPACE_FULL,
    validate_command,
    BasePolicy,
    BaseLocomotionPolicy,
    BaseTaskPolicy,
    BaseNavPolicy,
    BaseManipPolicy,
    SB3TaskPolicy,
)

# --- Locomotion adapterek ---
from roboshelf_ai.locomotion import (
    UnitreeRLGymAdapter,
    DummyLocomotionAdapter,
    LocomotionPolicyAdapter,
)

__all__ = [
    "__version__",
    # Core
    "RobotState",
    "LocomotionCommand",
    "LocomotionCommandSpace",
    "COMMAND_SPACE_BASIC",
    "COMMAND_SPACE_FULL",
    "validate_command",
    "BasePolicy",
    "BaseLocomotionPolicy",
    "BaseTaskPolicy",
    "BaseNavPolicy",
    "BaseManipPolicy",
    "SB3TaskPolicy",
    # Locomotion
    "UnitreeRLGymAdapter",
    "DummyLocomotionAdapter",
    "LocomotionPolicyAdapter",
]
