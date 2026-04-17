"""
LocomotionCommand — a hierarchikus rendszer kommunikációs szerződése.

Ez a modul definiálja azt a parancsstruktúrát, amelyen keresztül a high-level
navigation policy parancsol a low-level locomotion policy-nek.

Egységek:
  v_forward    : m/s   — előre irányú sebesség (0 = állj, pozitív = előre)
  v_lateral    : m/s   — oldal irányú sebesség (pozitív = balra)
  yaw_rate     : rad/s — forgási sebesség függőleges tengely körül (pozitív = balra)
  stance_width : m     — talpak távolsága (0 = default)
  step_height  : m     — lépés magassága (0 = default)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import math


# ---------------------------------------------------------------------------
# Parancs dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocomotionCommand:
    """Egy high-level mozgási parancs a locomotion policy számára.

    Minden mező SI-egységben van megadva. A default értékek "állj" állapotot
    jelentenek. A `clip()` metódus gondoskodik arról, hogy a parancs a megengedett
    tartományon belül maradjon.
    """

    v_forward: float = 0.0    # m/s
    v_lateral: float = 0.0    # m/s
    yaw_rate: float = 0.0     # rad/s
    stance_width: float = 0.0 # m, 0 = default G1 stance
    step_height: float = 0.0  # m, 0 = default gait

    def clip(self, space: "LocomotionCommandSpace") -> "LocomotionCommand":
        """Visszaad egy új LocomotionCommand-ot, amelynek mezői a space határain belül vannak."""
        return LocomotionCommand(
            v_forward=_clip(self.v_forward, space.v_forward_range),
            v_lateral=_clip(self.v_lateral, space.v_lateral_range),
            yaw_rate=_clip(self.yaw_rate, space.yaw_rate_range),
            stance_width=_clip(self.stance_width, space.stance_width_range),
            step_height=_clip(self.step_height, space.step_height_range),
        )

    def to_vector(self) -> Tuple[float, ...]:
        """Parancsot float tuple-ként adja vissza (obs-ba beilleszthető)."""
        return (self.v_forward, self.v_lateral, self.yaw_rate,
                self.stance_width, self.step_height)

    def __repr__(self) -> str:
        return (
            f"LocomotionCommand("
            f"v_fwd={self.v_forward:.2f} m/s, "
            f"v_lat={self.v_lateral:.2f} m/s, "
            f"yaw={math.degrees(self.yaw_rate):.1f} °/s)"
        )


# ---------------------------------------------------------------------------
# Parancs-tér (határok és defaultok)
# ---------------------------------------------------------------------------

@dataclass
class LocomotionCommandSpace:
    """A LocomotionCommand megengedett tartománya.

    Minden range egy (min, max) tuple. A `default` egy LocomotionCommand,
    amellyel a policy alaphelyzetbe hozható.
    """

    v_forward_range: Tuple[float, float] = (-0.5, 1.5)   # m/s
    v_lateral_range: Tuple[float, float] = (-0.5, 0.5)   # m/s
    yaw_rate_range: Tuple[float, float] = (-1.0, 1.0)    # rad/s
    stance_width_range: Tuple[float, float] = (0.0, 0.0) # m (most fixált)
    step_height_range: Tuple[float, float] = (0.0, 0.0)  # m (most fixált)

    @property
    def default(self) -> LocomotionCommand:
        """Visszaad egy nullvektoros (állj) parancsot."""
        return LocomotionCommand()

    @property
    def dim(self) -> int:
        """A parancsvektor dimenziója."""
        return 5

    def validate(self, cmd: LocomotionCommand) -> bool:
        """True, ha a parancs a megengedett tartományon belül van."""
        return (
            _in_range(cmd.v_forward, self.v_forward_range)
            and _in_range(cmd.v_lateral, self.v_lateral_range)
            and _in_range(cmd.yaw_rate, self.yaw_rate_range)
            and _in_range(cmd.stance_width, self.stance_width_range)
            and _in_range(cmd.step_height, self.step_height_range)
        )

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        """Visszaadja a határokat dict formában (pl. config exporthoz)."""
        return {
            "v_forward": self.v_forward_range,
            "v_lateral": self.v_lateral_range,
            "yaw_rate": self.yaw_rate_range,
            "stance_width": self.stance_width_range,
            "step_height": self.step_height_range,
        }


# ---------------------------------------------------------------------------
# Előre definiált command space-ek
# ---------------------------------------------------------------------------

# Alap navigációhoz: előre/hátra + kanyar, nincs lateral
COMMAND_SPACE_BASIC = LocomotionCommandSpace(
    v_forward_range=(0.0, 1.5),
    v_lateral_range=(0.0, 0.0),
    yaw_rate_range=(-1.0, 1.0),
)

# Teljes mozgástér: lateral is engedélyezett
COMMAND_SPACE_FULL = LocomotionCommandSpace(
    v_forward_range=(-0.5, 1.5),
    v_lateral_range=(-0.5, 0.5),
    yaw_rate_range=(-1.5, 1.5),
)


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _clip(value: float, range_: Tuple[float, float]) -> float:
    return max(range_[0], min(range_[1], value))


def _in_range(value: float, range_: Tuple[float, float]) -> bool:
    return range_[0] <= value <= range_[1]


def validate_command(cmd: LocomotionCommand,
                     space: LocomotionCommandSpace) -> LocomotionCommand:
    """Ellenőrzi és levágja a parancsot a megengedett tartományra.

    Mindig LocomotionCommand-ot ad vissza — soha nem dob kivételt.
    """
    return cmd.clip(space)
