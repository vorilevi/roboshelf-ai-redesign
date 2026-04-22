"""
VLA inference kliens — közös interface a három támogatott modellhez.

Támogatott modellek:
  wall-oss      — X-Square Robot WALL-OSS (Qwen2.5-VL MoE + flow-matching fej)
  unifolm-vla-0 — Unitree UnifoLM-VLA-0 (G1-specifikus, 12 feladatkategória)
  groot-n1.6    — NVIDIA GR00T N1.6-3B (Dual-System, diffúziós transzformer)

Modell-swap: egyetlen `model` attribútum vagy konstruktor arg változtatásával.
Minden modell ugyanazt az interface-t exportálja — a track-specifikus kódnak
nem kell tudnia, melyik VLA fut alatta.

Használat:
    from roboshelf_common.vla_client import VLAClient

    client = VLAClient(model="wall-oss", dtype="bfloat16")
    client.load()
    result = client.predict(
        observation={"image": img_array, "joint_pos": joint_pos},
        language_instruction="place the milk on the top shelf"
    )
    action = result["action"]   # np.ndarray
    conf   = result["confidence"]

Phase: 030 — stub implementáció, valós betöltés a 1. Fázisban (fork + sanity check után)
"""

from __future__ import annotations

import time
import warnings
from enum import Enum
from typing import Any


class VLAModel(str, Enum):
    """Támogatott VLA modellek."""
    WALL_OSS    = "wall-oss"
    UNIFOLM_VLA0 = "unifolm-vla-0"
    GROOT_N16   = "groot-n1.6"


# Modell-specifikus metaadatok (Fázis 1-ben frissítendő tényleges értékekkel)
_MODEL_METADATA: dict[str, dict[str, Any]] = {
    VLAModel.WALL_OSS: {
        "architecture": "Qwen2.5-VL MoE + flow-matching head",
        "action_dim": 7,          # TODO: pontosítani fake_inference.py futtatás után
        "checkpoint_hf": "XSquareRobot/wall-oss-flow-v0.1",
        "leerobot_v3_compatible": True,
        "recommended_dtype": "bfloat16",
        "notes": "10-20% alacsonyabb inference latencia diffúziós transzformernél",
    },
    VLAModel.UNIFOLM_VLA0: {
        "architecture": "UnifoLM VLA (Unitree G1-specifikus)",
        "action_dim": 12,         # G1 dexterous hand: 3 ujj × 2 ízület = 6 + kar 6
        "checkpoint_hf": "unitreerobotics/UnifoLM-VLA-0",
        "leerobot_v3_compatible": True,
        "recommended_dtype": "bfloat16",
        "notes": "12 manipulációs feladatkategória, G1 dexterous hand-re optimalizált",
    },
    VLAModel.GROOT_N16: {
        "architecture": "GR00T N1.6-3B Dual-System (diffúziós transzformer)",
        "action_dim": 7,
        "checkpoint_hf": "nvidia/GR00T-N1.6-3B",
        "leerobot_v3_compatible": True,
        "recommended_dtype": "bfloat16",
        "notes": "Referencia baseline az A/B/C teszthez",
    },
}


class VLAClient:
    """Közös VLA inference kliens.

    Args:
        model:   VLA modell neve — "wall-oss", "unifolm-vla-0", vagy "groot-n1.6"
        device:  "cpu" | "cuda" | "mps" (Apple Silicon)
        dtype:   "bfloat16" | "float16" | "float32"
        stub_mode: Ha True, valós betöltés nélkül dummy outputot ad (fejlesztés/tesztelés)
    """

    def __init__(
        self,
        model: str | VLAModel = VLAModel.WALL_OSS,
        device: str = "cpu",
        dtype: str = "bfloat16",
        stub_mode: bool = True,
    ) -> None:
        if isinstance(model, str):
            try:
                model = VLAModel(model)
            except ValueError:
                valid = [m.value for m in VLAModel]
                raise ValueError(
                    f"Ismeretlen modell: '{model}'. Elfogadott értékek: {valid}"
                )

        self.model = model
        self.device = device
        self.dtype = dtype
        self.stub_mode = stub_mode

        self._loaded = False
        self._model_obj: Any = None
        self._processor: Any = None

        if stub_mode:
            warnings.warn(
                f"VLAClient stub_mode=True — '{model.value}' valós inferenciát nem végez. "
                "Fázis 1-ben kapcsold ki: stub_mode=False.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Betöltés
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Modell betöltés — lusta inicializálás.

        stub_mode=True esetén csak logol, nem tölt be semmit.
        stub_mode=False esetén a track-specifikus betöltő logikát hívja
        (implementálandó Fázis 1-ben, fork + sanity check után).
        """
        if self._loaded:
            return

        if self.stub_mode:
            print(f"[VLAClient] stub_mode — '{self.model.value}' nem töltve be valósan.")
            self._loaded = True
            return

        # TODO (Fázis 1): valós betöltés, model-specifikusan
        # wall-oss:      from wallx.model import Qwen2_5_VLMoEForAction; ...
        # unifolm-vla-0: from unifolm.model import UnifoLMVLA; ...
        # groot-n1.6:    from gr00t.model import GrootN16Policy; ...
        raise NotImplementedError(
            f"Valós betöltés a '{self.model.value}' modellhez még nincs implementálva. "
            "Fázis 1-ben implementálni a fork + sanity check alapján."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: dict[str, Any],
        language_instruction: str,
    ) -> dict[str, Any]:
        """VLA forward pass.

        Args:
            observation:          dict — legalább "image" (np.ndarray) kulccsal,
                                  opcionálisan "joint_pos", "joint_vel", "imu" stb.
            language_instruction: természetes nyelvi feladatleírás (pl. "place milk on top shelf")

        Returns:
            dict:
              "action":     np.ndarray — aktuátorparancs vektor
              "confidence": float — modell konfidencia (0.0–1.0), ha elérhető
              "latency_ms": float — forward pass ideje ms-ban
              "model":      str   — melyik VLA adta a választ
        """
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        if self.stub_mode:
            action_dim = _MODEL_METADATA[self.model]["action_dim"]
            action = [0.0] * action_dim  # dummy nulla akció
            confidence = 0.0
        else:
            # TODO (Fázis 1): valós forward pass
            raise NotImplementedError("Valós predict() Fázis 1-ben implementálni.")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "action": action,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model": self.model.value,
        }

    # ------------------------------------------------------------------
    # Segédmetódusok
    # ------------------------------------------------------------------

    def model_info(self) -> dict[str, Any]:
        """Metaadatok az aktív modellről."""
        return {
            "model": self.model.value,
            "device": self.device,
            "dtype": self.dtype,
            "stub_mode": self.stub_mode,
            "loaded": self._loaded,
            **_MODEL_METADATA.get(self.model, {}),
        }

    def swap_model(self, new_model: str | VLAModel) -> None:
        """Modell csere futásidőben (config-flag módosítás).

        A betöltött modell unload-olódik, az új betöltése lusta (load() hívásakor).
        """
        if isinstance(new_model, str):
            new_model = VLAModel(new_model)
        print(f"[VLAClient] Modell váltás: {self.model.value} → {new_model.value}")
        self.model = new_model
        self._loaded = False
        self._model_obj = None
        self._processor = None

    def __repr__(self) -> str:
        status = "betöltve" if self._loaded else "nem betöltve"
        mode = "STUB" if self.stub_mode else "VALÓS"
        return f"VLAClient(model={self.model.value}, device={self.device}, {mode}, {status})"
