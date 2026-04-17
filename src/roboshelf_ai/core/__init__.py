"""
roboshelf_ai.core
=================
Core layer: szimulátorfüggetlen interfészek és közös callback-ek.

Re-exportálja a core.interfaces összes publikus szimbólumát,
hogy a hívók ne kelljen mélyebbre menni a csomagfában.

    from roboshelf_ai.core import RobotState, LocomotionCommand
    from roboshelf_ai.core.callbacks import EpisodeStatsCallback  # SB3 szükséges
"""

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

# callbacks lazy-importálva: stable_baselines3 nem mindig elérhető
# (pl. tesztkörnyezetben, Isaac Lab-on). Explicit importálandó:
#   from roboshelf_ai.core.callbacks import EpisodeStatsCallback

__all__ = [
    # interfaces
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
    # callbacks (lazy — external import szükséges)
    "EpisodeStatsCallback",
    "LinearCurriculumCallback",
    "make_vec_normalize",
    "check_acceptance",
]


def __getattr__(name: str):
    """Lazy import a callbacks modulból — elkerüli az SB3 hard dependenciát."""
    _callbacks = {
        "EpisodeStatsCallback",
        "LinearCurriculumCallback",
        "make_vec_normalize",
        "check_acceptance",
    }
    if name in _callbacks:
        from roboshelf_ai.core import callbacks as _cb
        return getattr(_cb, name)
    raise AttributeError(f"module 'roboshelf_ai.core' has no attribute {name!r}")
