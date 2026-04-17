"""
roboshelf_ai.core.interfaces
============================
Szimulátorfüggetlen absztrakt interfészek és adatosztályok.

Gyors import:
    from roboshelf_ai.core.interfaces import (
        RobotState,
        LocomotionCommand, LocomotionCommandSpace,
        COMMAND_SPACE_BASIC, COMMAND_SPACE_FULL,
        validate_command,
        BasePolicy, BaseLocomotionPolicy, BaseTaskPolicy,
        BaseNavPolicy, BaseManipPolicy,
        SB3TaskPolicy,
    )
"""

from roboshelf_ai.core.interfaces.robot_state import RobotState

from roboshelf_ai.core.interfaces.locomotion_command import (
    LocomotionCommand,
    LocomotionCommandSpace,
    COMMAND_SPACE_BASIC,
    COMMAND_SPACE_FULL,
    validate_command,
)

from roboshelf_ai.core.interfaces.base_policy import (
    BasePolicy,
    BaseLocomotionPolicy,
    BaseTaskPolicy,
    BaseNavPolicy,
    BaseManipPolicy,
    SB3TaskPolicy,
)

from roboshelf_ai.core.interfaces.demonstration import (
    DemoStep,
    DemoCollector,
    DemoDataset,
)

__all__ = [
    # Robot állapot
    "RobotState",
    # Parancsok
    "LocomotionCommand",
    "LocomotionCommandSpace",
    "COMMAND_SPACE_BASIC",
    "COMMAND_SPACE_FULL",
    "validate_command",
    # Policy interfészek
    "BasePolicy",
    "BaseLocomotionPolicy",
    "BaseTaskPolicy",
    "BaseNavPolicy",
    "BaseManipPolicy",
    "SB3TaskPolicy",
    # Demonstráció / IL
    "DemoStep",
    "DemoCollector",
    "DemoDataset",
]
