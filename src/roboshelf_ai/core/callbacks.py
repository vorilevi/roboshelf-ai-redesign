"""
Common SB3 callbacks és training utilities.

Ezek a callbackek minden tanítási scriptnél újrafelhasználhatók
(locomotion, nav, manipulation). Nem kell minden train script-ben
újraimplementálni őket.
"""

from __future__ import annotations

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


# ---------------------------------------------------------------------------
# EpisodeStatsCallback — részletes reward komponens logging
# ---------------------------------------------------------------------------

class EpisodeStatsCallback(BaseCallback):
    """TensorBoard-ra logolja az info dict reward komponenseit.

    Az env step() info dict-jéből bármely kulcsot logolja,
    amelynek neve 'r_'-vel kezdődik (pl. r_forward, r_upright).

    Használat:
        callback = EpisodeStatsCallback(verbose=0)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_info_buffer: list[dict] = []

    def _on_step(self) -> bool:
        # SB3 infos: lista, egy elem per env
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                # Epizód végi reward komponensek logolása
                for k, v in info.items():
                    if k.startswith("r_") and isinstance(v, (int, float)):
                        self.logger.record_mean(f"ep_info/{k}", v)
        return True


# ---------------------------------------------------------------------------
# CurriculumCallback — lineáris curriculum paraméter skálázás
# ---------------------------------------------------------------------------

class LinearCurriculumCallback(BaseCallback):
    """Egy env attribútumot lineárisan változtat a tanítás során.

    Példa: buoyancy_force 103 → 0 az első 3M lépésen át.

    Használat:
        callback = LinearCurriculumCallback(
            attr_name="buoyancy_force",
            start_value=103.0,
            end_value=0.0,
            anneal_timesteps=3_000_000,
        )
    """

    def __init__(
        self,
        attr_name: str,
        start_value: float,
        end_value: float,
        anneal_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.attr_name = attr_name
        self.start_value = start_value
        self.end_value = end_value
        self.anneal_timesteps = anneal_timesteps

    def _on_step(self) -> bool:
        t = min(self.num_timesteps / self.anneal_timesteps, 1.0)
        value = self.start_value + t * (self.end_value - self.start_value)

        # Beállítás minden env-en
        env = self.training_env
        if hasattr(env, "env_method"):
            try:
                env.env_method("__setattr__", self.attr_name, value)
            except Exception:
                pass

        if self.verbose > 0 and self.num_timesteps % 100_000 == 0:
            self.logger.record(f"curriculum/{self.attr_name}", value)

        return True


# ---------------------------------------------------------------------------
# VecNormalize wrapper factory
# ---------------------------------------------------------------------------

def make_vec_normalize(env: VecEnv, load_path: str | None = None,
                       norm_obs: bool = True, norm_reward: bool = True,
                       clip_obs: float = 10.0, gamma: float = 0.99):
    """VecNormalize wrapper létrehozása vagy betöltése.

    A VecNormalize futó átlagot tart fenn az obs és reward normalizáláshoz.
    Tanítás után mentsd el: vec_env.save('vec_normalize.pkl')
    Eval-nál töltsd be: make_vec_normalize(..., load_path='vec_normalize.pkl')

    Args:
        env:         VecEnv amit be kell csomagolni
        load_path:   Ha megadott, betölti a mentett normalizálási statisztikákat
        norm_obs:    Obs normalizálás (ajánlott locomotion-hoz)
        norm_reward: Reward normalizálás (ajánlott, de eval-nál kapcsold ki)
        clip_obs:    Obs clip határok (±10 tipikus)
        gamma:       Reward normalizáláshoz használt gamma

    Returns:
        VecNormalize-ba csomagolt env
    """
    from stable_baselines3.common.vec_env import VecNormalize

    if load_path and os.path.exists(load_path):
        vec_env = VecNormalize.load(load_path, env)
        vec_env.training = False   # eval módban ne frissítse a statisztikákat
        vec_env.norm_reward = False
        if norm_reward:
            vec_env.training = True
            vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            gamma=gamma,
        )
    return vec_env


# ---------------------------------------------------------------------------
# Acceptance check utility
# ---------------------------------------------------------------------------

def check_acceptance(
    eval_results: dict,
    min_ep_length: int = 300,
    min_success_rate: float = 0.0,
) -> tuple[bool, str]:
    """Elfogadási feltételek ellenőrzése eval eredményekből.

    Args:
        eval_results:     {'mean_ep_length': float, 'success_rate': float, ...}
        min_ep_length:    Minimum átlag epizód hossz (lépés)
        min_success_rate: Minimum célpont-elérési arány [0, 1]

    Returns:
        (accepted: bool, reason: str)
    """
    mean_len = eval_results.get("mean_ep_length", 0)
    success  = eval_results.get("success_rate", 0)

    if mean_len < min_ep_length:
        return False, f"ep_len {mean_len:.0f} < {min_ep_length}"
    if success < min_success_rate:
        return False, f"success_rate {success:.2f} < {min_success_rate:.2f}"
    return True, f"OK (ep_len={mean_len:.0f}, success={success:.2f})"
