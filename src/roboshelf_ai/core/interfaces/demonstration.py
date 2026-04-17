"""
Demonstrációs adat interfész — Imitációs tanulás (BC) csatornához.

Adatformátum:
    Minden demonstráció egy epizód, amely obs-action párokból áll.
    Az adatokat .npz fájlban tároljuk (numpy compressed).

    Fájl struktúra (data/demonstrations/<name>.npz):
        obs      : float32 [N, obs_dim]   — megfigyelések
        actions  : float32 [N, act_dim]   — expert akciók
        rewards  : float32 [N]            — lépésenkénti reward (opcionális)
        dones    : bool    [N]            — epizód vége jelzők
        ep_starts: bool    [N]            — epizód kezdetek (BC DataLoader-hez)
        infos    : dict                   — metaadatok (config, dátum, forrás)

Gyűjtési módok:
    1. ScriptedExpert  — deterministikus gépi vezérlés (PD, rule-based)
       → gyors, reprodukálható, korlátozott fedettség
    2. KeyboardTeleop  — kézi irányítás MuJoCo viewer-en keresztül
       → változatos, de lassú és fárasztó
    3. PolicyRollout   — betanított (részleges) policy önkiértékelése
       → bootstrapping: BC init → PPO → jobb policy → több BC adat

BC tanítás:
    from roboshelf_ai.locomotion.train_loco_bc import train_bc
    train_bc("data/demonstrations/scripted_loco_v1.npz", output_dir="...")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Egyetlen lépés adata
# ---------------------------------------------------------------------------

@dataclass
class DemoStep:
    """Egy szimulációs lépés megfigyelés-akció párja."""
    obs: np.ndarray       # float32 [obs_dim]
    action: np.ndarray    # float32 [act_dim]
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Demonstráció gyűjtő
# ---------------------------------------------------------------------------

class DemoCollector:
    """Epizódonként gyűjti a demonstrációs lépéseket, majd .npz-be menti.

    Használat:
        collector = DemoCollector(obs_dim=9, act_dim=2)
        collector.start_episode()
        for step in episode:
            collector.record(obs, action, reward, done)
        collector.end_episode()
        collector.save("data/demonstrations/scripted_nav_v1.npz")
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._obs: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._ep_starts: List[bool] = []
        self._n_episodes = 0
        self._in_episode = False

    def start_episode(self) -> None:
        """Új epizód kezdése — a következő record() ep_start=True lesz."""
        self._in_episode = True
        self._next_is_start = True

    def record(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        """Egy lépés rögzítése."""
        self._obs.append(obs.astype(np.float32))
        self._actions.append(action.astype(np.float32))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._ep_starts.append(self._next_is_start)
        self._next_is_start = False

    def end_episode(self) -> None:
        """Epizód lezárása."""
        if self._dones and not self._dones[-1]:
            # Ha az utolsó lépés nem done, jelöljük lezártnak
            self._dones[-1] = True
        self._n_episodes += 1
        self._in_episode = False

    @property
    def n_steps(self) -> int:
        return len(self._obs)

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    def save(
        self,
        path: str | Path,
        source: str = "scripted",
        config_name: str = "",
    ) -> Path:
        """Összegyűjtött adat mentése .npz fájlba.

        Args:
            path:        Kimeneti .npz fájl útvonala
            source:      'scripted' | 'teleop' | 'policy_rollout'
            config_name: Melyik config alapján gyűjtöttük

        Returns:
            Az elmentett fájl Path objektuma
        """
        if not self._obs:
            raise ValueError("Nincs rögzített adat — hívj start_episode() + record() előbb.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        obs_arr     = np.stack(self._obs,    axis=0)   # [N, obs_dim]
        actions_arr = np.stack(self._actions, axis=0)  # [N, act_dim]
        rewards_arr = np.array(self._rewards, dtype=np.float32)
        dones_arr   = np.array(self._dones,   dtype=bool)
        ep_starts   = np.array(self._ep_starts, dtype=bool)

        np.savez_compressed(
            str(path),
            obs=obs_arr,
            actions=actions_arr,
            rewards=rewards_arr,
            dones=dones_arr,
            ep_starts=ep_starts,
            # Metaadatok skalárként
            n_episodes=np.array(self._n_episodes),
            obs_dim=np.array(self.obs_dim),
            act_dim=np.array(self.act_dim),
            source=np.array(source),
            config_name=np.array(config_name),
            collected_at=np.array(time.strftime("%Y-%m-%dT%H:%M:%S")),
        )

        print(
            f"✅ Demonstráció mentve: {path}\n"
            f"   Epizódok: {self._n_episodes}, lépések: {len(self._obs)}, "
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim}"
        )
        return path

    def clear(self) -> None:
        """Puffer törlése — új gyűjtés előtt."""
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._ep_starts.clear()
        self._n_episodes = 0
        self._in_episode = False


# ---------------------------------------------------------------------------
# Betöltő
# ---------------------------------------------------------------------------

class DemoDataset:
    """Mentett .npz demonstráció betöltése és ellenőrzése.

    Használat:
        ds = DemoDataset("data/demonstrations/scripted_nav_v1.npz")
        print(ds.summary())
        obs, actions = ds.obs, ds.actions   # numpy tömbök
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Demonstráció fájl nem található: {path}")
        data = np.load(str(path), allow_pickle=True)
        self.obs:       np.ndarray = data["obs"]
        self.actions:   np.ndarray = data["actions"]
        self.rewards:   np.ndarray = data["rewards"]
        self.dones:     np.ndarray = data["dones"]
        self.ep_starts: np.ndarray = data["ep_starts"]
        self.n_episodes: int = int(data["n_episodes"])
        self.source:    str  = str(data.get("source", "unknown"))
        self.config_name: str = str(data.get("config_name", ""))
        self.collected_at: str = str(data.get("collected_at", ""))
        self._path = path

    @property
    def n_steps(self) -> int:
        return len(self.obs)

    def summary(self) -> str:
        ep_lengths = np.where(self.ep_starts)[0]
        ep_lengths = np.diff(
            np.append(ep_lengths, self.n_steps)
        )
        return (
            f"DemoDataset: {self._path.name}\n"
            f"  Epizódok:   {self.n_episodes}\n"
            f"  Lépések:    {self.n_steps}\n"
            f"  obs_dim:    {self.obs.shape[1]}\n"
            f"  act_dim:    {self.actions.shape[1]}\n"
            f"  Ep hossz:   min={ep_lengths.min()}, "
            f"max={ep_lengths.max()}, "
            f"átlag={ep_lengths.mean():.1f}\n"
            f"  Forrás:     {self.source}\n"
            f"  Gyűjtve:    {self.collected_at}"
        )
