"""
RobotState — szimulátorfüggetlen robot állapot reprezentáció.

Ez a dataclass az egyetlen híd a szimulátor-specifikus kód (MuJoCo MjData,
Isaac Lab obs tensor) és a szimulátortól független policy/adapter réteg között.

MuJoCo-ból töltsd ki így:
    state = RobotState.from_mujoco(data, torso_id=1, leg_qpos_start=7)

Isaac Lab-ból töltsd ki így (a jövőben):
    state = RobotState.from_isaac(env_obs, env_idx=0)

Az adapterek és a policy-k mindig RobotState-et kapnak, soha nem MjData-t
vagy Isaac tensor-t közvetlenül — így a szimulátorcsere nem érinti a felsőbb rétegeket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RobotState:
    """Egy G1 robot aktuális kinematikai állapota.

    Minden mező numpy array, float32, world frame-ben kivéve ahol jelezve.

    Attribútumok:
        qpos:   Teljes joint pozíció vektor (nq,) — freejoint [x,y,z,qw,qx,qy,qz] + ízületek
        qvel:   Teljes joint sebesség vektor (nv,) — freejoint [vx,vy,vz,wx,wy,wz] + ízületek
        quat:   Torzó quaternion [qw, qx, qy, qz] — world frame
        omega:  Torzó szögsebesség [wx, wy, wz] — body frame (rad/s)
        xpos:   Torzó pozíció [x, y, z] — world frame (m)
        xmat:   Torzó rotációs mátrix (9,) row-major — world frame
        # Opcionálisak (nem minden szimulátornál elérhető azonnal)
        foot_contact:   Lábkontakt jelzők [left, right] — bool
        foot_pos:       Lábfej pozíciók (2, 3) — world frame
        sim_time:       Szimulációs idő (s)
        step_count:     Szimulációs lépésszám
    """

    qpos:  np.ndarray                          # (nq,)
    qvel:  np.ndarray                          # (nv,)
    quat:  np.ndarray                          # (4,) [qw, qx, qy, qz]
    omega: np.ndarray                          # (3,) szögsebesség body frame
    xpos:  np.ndarray                          # (3,) torzó world pos
    xmat:  np.ndarray                          # (9,) torzó rotációs mátrix

    foot_contact: Optional[np.ndarray] = None  # (2,) bool
    foot_pos:     Optional[np.ndarray] = None  # (2, 3)
    sim_time:     float = 0.0
    step_count:   int = 0

    # ------------------------------------------------------------------
    # Kényelmi property-k
    # ------------------------------------------------------------------

    @property
    def torso_z(self) -> float:
        """Torzó magassága (m)."""
        return float(self.xpos[2])

    @property
    def torso_xy(self) -> np.ndarray:
        """Torzó x,y pozíció (m)."""
        return self.xpos[:2].copy()

    @property
    def upright(self) -> float:
        """Torzó függőlegességének mértéke: rotációs mátrix Z-Z eleme.
        1.0 = teljesen függőleges, 0.0 = vízszintes, negatív = fejjel lefelé."""
        return float(self.xmat[8])  # xmat[2,2] row-major indexelésben

    @property
    def leg_qpos(self) -> np.ndarray:
        """12 láb joint pozíció (qpos[7:19])."""
        return self.qpos[7:19].copy()

    @property
    def leg_qvel(self) -> np.ndarray:
        """12 láb joint sebesség (qvel[6:18])."""
        return self.qvel[6:18].copy()

    @property
    def lin_vel(self) -> np.ndarray:
        """Torzó lineáris sebesség world frame-ben [vx, vy, vz] (m/s)."""
        return self.qvel[0:3].copy()

    # ------------------------------------------------------------------
    # Gyártó metódusok szimulátorhoz
    # ------------------------------------------------------------------

    @classmethod
    def from_mujoco(
        cls,
        data,                        # mujoco.MjData
        torso_body_id: int = 1,
        left_foot_id:  Optional[int] = None,
        right_foot_id: Optional[int] = None,
    ) -> "RobotState":
        """MuJoCo MjData-ból tölt ki egy RobotState-et.

        Args:
            data:          mujoco.MjData
            torso_body_id: pelvis body index (általában 1)
            left_foot_id:  bal lábfej body index (None = nem elérhető)
            right_foot_id: jobb lábfej body index (None = nem elérhető)
        """
        torso = data.body(torso_body_id)

        # Lábkontakt
        foot_contact = None
        foot_pos = None
        if left_foot_id is not None and right_foot_id is not None:
            lc = data.cfrc_ext[left_foot_id,  2] > 1.0
            rc = data.cfrc_ext[right_foot_id, 2] > 1.0
            foot_contact = np.array([lc, rc], dtype=bool)
            foot_pos = np.array([
                data.body(left_foot_id).xpos.copy(),
                data.body(right_foot_id).xpos.copy(),
            ], dtype=np.float32)

        return cls(
            qpos=data.qpos.astype(np.float32),
            qvel=data.qvel.astype(np.float32),
            quat=data.qpos[3:7].astype(np.float32),   # [qw, qx, qy, qz]
            omega=data.qvel[3:6].astype(np.float32),  # body angular vel
            xpos=torso.xpos.astype(np.float32),
            xmat=torso.xmat.astype(np.float32),
            foot_contact=foot_contact,
            foot_pos=foot_pos,
            sim_time=float(data.time),
            step_count=int(data.time / 0.002),  # approx, sim_dt=0.002
        )

    @classmethod
    def from_isaac(cls, obs: np.ndarray, dt: float = 0.02) -> "RobotState":
        """Isaac Lab observation tensor-ból tölt ki egy RobotState-et.

        PLACEHOLDER — implementálandó amikor Isaac Lab-ra váltunk.
        Az obs tensor formátumát az Isaac Lab G1 env határozza meg.
        """
        raise NotImplementedError(
            "Isaac Lab adapter még nincs implementálva. "
            "Implementáld amikor Isaac Lab env-re váltasz."
        )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RobotState("
            f"z={self.torso_z:.2f}m, "
            f"upright={self.upright:.2f}, "
            f"lin_vel={self.lin_vel[:2]}, "
            f"t={self.sim_time:.2f}s)"
        )
