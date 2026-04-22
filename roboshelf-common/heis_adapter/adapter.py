"""
HEIS (Humanoid Robots and Embodied Intelligence Standardization) adapter.

Szabvány: HEIS 2026 Q1 v1.0
Kiadó:    Kínai Nemzeti Szabványügyi Bizottság, elnök: Wang Xingxing (Unitree)
Forrás:   roboshelf_two_track_redesign_2026-04-22.md — 2.3. fejezet

Ez a modul a Roboshelf AI saját obs/action sémáját konvertálja HEIS-kompatibilis
formátumra (és vissza), valamint EIbench-kompatibilis metrikákat exportál.

Használat:
    from roboshelf_common.heis_adapter import HEISAdapter

    adapter = HEISAdapter(track="humanoid")
    heis_obs = adapter.obs_to_heis(my_obs_dict)
    metrics  = adapter.export_eibench_metrics(episode_log)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HEISVersion(str, Enum):
    """HEIS szabvány verziók — a verziót explicit paraméterként kezeljük,
    hogy a 2027-es revízió ne okozzon silent breakage-t."""
    V1_0 = "1.0"


@dataclass
class HEISObservation:
    """HEIS-szabványos observation struktúra.

    Mezők (HEIS 2026 Q1 v1.0 szerint):
      proprioceptive_27dim  — joint pozíciók, sebességek, IMU
      command_12dim         — navigációs + manipulációs parancsok
      sensor_3dim           — talpnyomás / proximity szenzorok aggregált értékei
    """
    proprioceptive_27dim: list[float] = field(default_factory=lambda: [0.0] * 27)
    command_12dim: list[float]        = field(default_factory=lambda: [0.0] * 12)
    sensor_3dim: list[float]          = field(default_factory=lambda: [0.0] * 3)
    heis_version: str                 = HEISVersion.V1_0.value
    track: str                        = "humanoid"  # "humanoid" | "ean"

    def to_dict(self) -> dict[str, Any]:
        return {
            "heis_version": self.heis_version,
            "track": self.track,
            "proprioceptive_27dim": self.proprioceptive_27dim,
            "command_12dim": self.command_12dim,
            "sensor_3dim": self.sensor_3dim,
        }


@dataclass
class EIBenchMetrics:
    """EIbench-kompatibilis metrikák egy epizód után.

    Referencia: HEIS EIbench specifikáció 2026 Q1.
    """
    success: bool          = False
    success_rate: float    = 0.0   # 0.0–1.0
    completion_time_s: float = 0.0
    energy_efficiency: float = 0.0  # torque × displacement integrál (alacsonyabb = jobb)
    min_human_distance_m: float = float("inf")  # Human-Safe Nav metrika
    heis_version: str      = HEISVersion.V1_0.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "heis_version": self.heis_version,
            "success": self.success,
            "success_rate": self.success_rate,
            "completion_time_s": self.completion_time_s,
            "energy_efficiency": self.energy_efficiency,
            "min_human_distance_m": self.min_human_distance_m,
        }


class HEISAdapter:
    """Konvertálja a Roboshelf saját obs/action sémáját HEIS-kompatibilis formátumra.

    Args:
        track:        "humanoid" (MuJoCo/unitree_rl_mjlab) vagy "ean" (Isaac Lab)
        heis_version: HEIS szabvány verzió — default: v1.0
        strict:       Ha True, figyelmeztetés helyett kivételt dob ismeretlen mezőnél
    """

    def __init__(
        self,
        track: str = "humanoid",
        heis_version: HEISVersion = HEISVersion.V1_0,
        strict: bool = False,
    ) -> None:
        if track not in ("humanoid", "ean"):
            raise ValueError(f"Ismeretlen track: '{track}'. Elfogadott: 'humanoid', 'ean'.")
        self.track = track
        self.heis_version = heis_version
        self.strict = strict

    # ------------------------------------------------------------------
    # Observation konverzió
    # ------------------------------------------------------------------

    def obs_to_heis(self, obs: dict[str, Any]) -> HEISObservation:
        """Saját observation dict → HEIS observation struktúra.

        A konkrét mezők track-függők és egyelőre stub-ok.
        Implementálni a 2. Fázisban, amikor az unitree_rl_mjlab obs dim rögzített.

        Args:
            obs: Saját observation dict (track-specifikus kulcsok)

        Returns:
            HEISObservation — HEIS-kompatibilis struktúra
        """
        # TODO (Fázis 2): konkrét mezők leképezése track-specifikusan
        # humanoid: joint_pos(12) + joint_vel(12) + imu(3) = 27 dim proprioceptive
        # ean: manip_joint_pos(7) + grasp_force(3) + shelf_proximity(3) + ... = 27 dim

        if not obs:
            warnings.warn(
                "obs_to_heis: üres observation dict érkezett — HEIS stub értékeket adunk vissza.",
                stacklevel=2,
            )

        return HEISObservation(
            proprioceptive_27dim=[0.0] * 27,  # TODO: obs["joint_pos"] + obs["joint_vel"] + obs["imu"]
            command_12dim=[0.0] * 12,          # TODO: obs["command"][:12]
            sensor_3dim=[0.0] * 3,             # TODO: obs["foot_contact"] aggregált
            heis_version=self.heis_version.value,
            track=self.track,
        )

    # ------------------------------------------------------------------
    # Action konverzió
    # ------------------------------------------------------------------

    def action_to_heis(self, action: Any) -> dict[str, Any]:
        """Aktuátor kimenet → HEIS action format.

        Args:
            action: np.ndarray vagy list — saját action vektor

        Returns:
            dict HEIS-kompatibilis action mezőkkel
        """
        # TODO (Fázis 2): track-specifikus leképezés
        # humanoid: 29 DoF aktuátor → HEIS joint_torques_Nm (SI)
        # ean: 7 DoF manipulátor → HEIS arm_joint_torques_Nm

        return {
            "heis_version": self.heis_version.value,
            "track": self.track,
            "joint_torques_Nm": list(action) if action is not None else [],
            # TODO: Nm konverzió, ha a saját action nem SI egységben van
        }

    # ------------------------------------------------------------------
    # EIbench export
    # ------------------------------------------------------------------

    def export_eibench_metrics(self, episode_log: dict[str, Any]) -> EIBenchMetrics:
        """EIbench-kompatibilis metrika export egy epizód logból.

        Args:
            episode_log: dict alábbi (opcionális) kulcsokkal:
                - "success": bool
                - "episode_length": int (lépések száma)
                - "dt": float (szimulációs lépésköz, másodpercben)
                - "torques": list[list[float]] — minden lépés torque vektora
                - "displacements": list[float] — minden lépés elmozdulása (m)
                - "min_human_distance_m": float — legközelebbi ember-távolság

        Returns:
            EIBenchMetrics — közvetlenül JSON-ba serializálható
        """
        success = bool(episode_log.get("success", False))
        ep_len = episode_log.get("episode_length", 0)
        dt = episode_log.get("dt", 0.02)  # default 50 Hz → 0.02s

        completion_time_s = ep_len * dt

        # Energiahatékonyság: torque × displacement szumma minden lépésen
        torques = episode_log.get("torques", [])
        displacements = episode_log.get("displacements", [])
        if torques and displacements and len(torques) == len(displacements):
            energy = sum(
                sum(abs(t) for t in torque_vec) * disp
                for torque_vec, disp in zip(torques, displacements)
            )
        else:
            energy = 0.0  # nem mért, vagy hiányos adat

        return EIBenchMetrics(
            success=success,
            success_rate=1.0 if success else 0.0,  # egyedi epizód: bináris
            completion_time_s=completion_time_s,
            energy_efficiency=energy,
            min_human_distance_m=episode_log.get("min_human_distance_m", float("inf")),
            heis_version=self.heis_version.value,
        )
