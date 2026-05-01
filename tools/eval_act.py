"""
ACT Policy Evaluation — F3d Phase 030.

Betölti a train_act.py által mentett best_model.pt-t, majd 50 epizódot futtat
a MuJoCo scene_manip_sandbox_v2.xml környezetben, és méri a success rate-et.

Architektúra: a ScriptedExpert env infrastruktúráját újra felhasználja
(reset, step, obs kinyerés), de az expert action helyett az ACT policy outputja
megy az env-be.

Action chunking végrehajtás:
    - ACT modell minden exec_horizon lépésenként újra fut (re-query)
    - Alapértelmezett: exec_horizon = chunk_size (= 20, teljes chunk egyszerre)
    - --exec-horizon 1 → lépésenkénti re-query (legreaktívabb, de lassabb)
    - --exec-horizon 5 → ajánlott kompromisszum push task esetén

Normalizáció:
    - Obs:    z-score   (mean, std — stats.json)
    - Action: minmax → [-1, 1] (min, max — stats.json); visszanormalizálás a step() előtt

Futtatás (repo gyökeréből):
    python3 tools/eval_act.py \\
        --ckpt    results/bc_checkpoints_act_v1 \\
        --stats   data/lerobot/scripted_v1/meta/stats.json \\
        --n-eval  50 \\
        --exec-horizon 5

Kimenet:
    Success rate, átlag epizódhossz, átlag végső place_dist
    Ha SR < 40% → F3d PPF fine-tune ajánlott
    Ha SR ≥ 60% → F3e kihagyható, Phase 040 jöhet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_TOOLS_DIR = _HERE.parent

# ScriptedExpert importja (env infrastruktúra)
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import scripted_expert as _exp   # noqa: E402
from scripted_expert import (    # noqa: E402
    ScriptedExpert,
    _get_obs,
    _JOINT_RANGES,
    _GRIPPER_CLOSED,
    _GRIPPER_OPEN,
    _DEFAULT_ARM_POS,
    ARM_QPOS_START,
    ARM_CTRL_START,
    GRIPPER_CTRL_START,
    N_ARM_DOF,
    DECIMATION,
    GOAL_RADIUS,
    STOCK_QPOS_START,
    STOCK_RESET_Z,
    MIN_SUCCESS_STEP,
)

import mujoco

# train_act importja (model + load_policy)
from train_act import load_policy, ACTModel


# ─── Normalizáció ────────────────────────────────────────────────────────────

class StatsNormalizer:
    """Obs z-score + action minmax normalizáció / denormalizáció."""

    def __init__(self, stats: dict):
        obs    = stats["obs"]
        action = stats["action"]

        self.obs_mean = np.array(obs["mean"], dtype=np.float32)
        self.obs_std  = np.array(obs["std"],  dtype=np.float32) + 1e-8
        self.act_min  = np.array(action["min"], dtype=np.float32)
        self.act_max  = np.array(action["max"], dtype=np.float32)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        rng = (self.act_max - self.act_min) + 1e-8
        return (action_norm + 1.0) * 0.5 * rng + self.act_min


# ─── Env wrapper ─────────────────────────────────────────────────────────────

class PushTaskEnv:
    """
    Minimális MuJoCo env wrapper a push task evaluálásához.
    A ScriptedExpert infrastruktúráját újrafelhasználja, de saját action-t hajt végre.
    """

    def __init__(self, xml_path: Path, seed: int = 0):
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data  = mujoco.MjData(self._model)
        self._rng   = np.random.default_rng(seed)

        # Site / body ID-k (ScriptedExpert-tel azonos névkonvenció)
        self._hand_site_id   = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
        self._stock_body_id  = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

        # Hand body ID-k (contact detection)
        self._hand_body_ids: set = set()
        for name in _exp._HAND_BODY_NAMES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._hand_body_ids.add(bid)

        self._arm_dof_ids = list(range(ARM_QPOS_START, ARM_QPOS_START + N_ARM_DOF))
        self._step_count  = 0

    def reset(self, stock_x: float | None = None, stock_y: float | None = None) -> np.ndarray:
        mujoco.mj_resetData(self._model, self._data)

        # Kar alapállás
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = _DEFAULT_ARM_POS
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF]  = _DEFAULT_ARM_POS
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7]  = _GRIPPER_OPEN

        # Stock reset — F3c tartomány
        lo_x, hi_x = _exp.STOCK_RESET_X_RANGE
        lo_y, hi_y = _exp.STOCK_RESET_Y_RANGE

        for _ in range(50):
            sx = float(stock_x if stock_x is not None else self._rng.uniform(lo_x, hi_x))
            sy = float(stock_y if stock_y is not None else self._rng.uniform(lo_y, hi_y))
            self._data.qpos[STOCK_QPOS_START + 0] = sx
            self._data.qpos[STOCK_QPOS_START + 1] = sy
            self._data.qpos[STOCK_QPOS_START + 2] = STOCK_RESET_Z
            self._data.qpos[STOCK_QPOS_START + 3:STOCK_QPOS_START + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            h = self._data.site_xpos[self._hand_site_id]
            s = self._data.xpos[self._stock_body_id]
            if np.linalg.norm(h - s) >= 0.12:
                break

        self._step_count = 0
        mujoco.mj_forward(self._model, self._data)
        return self._get_obs()

    def step(self, action_norm: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        """
        Végrehajt egy policy lépést.

        Args:
            action_norm: (5,) float32 — [4 arm normalizált + 1 gripper] ∈ [-1,1]

        Returns:
            obs:     (24,) következő megfigyelés
            done:    epizód vége-e
            info:    {success, place_dist, timeout}
        """
        arm_action     = np.array(action_norm[:4], dtype=np.float32)
        gripper_signal = float(np.clip(action_norm[4], -1.0, 1.0))

        # Kar: position control (denorm: [-1,1] → joint range)
        lo, hi = _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
        target_qpos = lo + (arm_action + 1.0) * 0.5 * (hi - lo)
        target_qpos = np.clip(target_qpos, lo, hi)
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_qpos

        # Gripper
        t = (gripper_signal + 1.0) / 2.0
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = (
            (1.0 - t) * _GRIPPER_OPEN + t * _GRIPPER_CLOSED
        )

        for _ in range(DECIMATION):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = self._get_obs()

        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        place_dist = float(np.linalg.norm(stock_pos - target_pos))

        success = (place_dist < GOAL_RADIUS) and (self._step_count >= MIN_SUCCESS_STEP)
        timeout = self._step_count >= 300
        done    = success or timeout

        return obs, done, {
            "success":   success,
            "place_dist": place_dist,
            "timeout":   timeout,
            "step":      self._step_count,
        }

    def _get_obs(self) -> np.ndarray:
        return _get_obs(
            self._model, self._data,
            self._hand_site_id, self._target_site_id,
            self._stock_body_id, self._hand_body_ids,
        )


# ─── Policy futtatás ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    env:          PushTaskEnv,
    model:        ACTModel,
    normalizer:   StatsNormalizer,
    device:       torch.device,
    exec_horizon: int,
    max_steps:    int = 300,
    verbose:      bool = False,
) -> dict:
    """
    Egyetlen epizód: ACT policy vezérli.

    Action chunking: a modell exec_horizon lépésenként fut újra.
    Az ACT paper a teljes chunk-ot hajtja végre; push task esetén
    kisebb exec_horizon reaktívabb viselkedést ad.
    """
    obs = env.reset()
    model.eval()

    chunk_buf    = []   # aktuális chunk végrehajtásra váró lépései
    buf_idx      = 0    # melyik chunk-lépésnél tartunk

    total_steps  = 0
    last_info    = {}

    for _ in range(max_steps):
        # Re-query ha a chunk kimerült
        if buf_idx >= len(chunk_buf):
            obs_norm = normalizer.normalize_obs(obs)
            obs_t    = torch.from_numpy(obs_norm).unsqueeze(0).to(device)  # (1, 24)

            actions_pred, _ = model(obs_t, actions=None)   # (1, chunk, 5)
            actions_np = actions_pred[0].cpu().numpy()     # (chunk, 5)

            # Action denormalizáció: a modell [-1,1]-re normalizált outputot ad,
            # de a step() szintén [-1,1] input-ot vár (joint range denorm ott van).
            # Tehát nincs szükség külső denormalizációra — a step() kezeli.
            # KIVÉTEL: ha a tréning során a target action már denorm volt → nem kell.
            # A mi esetünkben: scripted_expert → _norm_action() → [-1,1] mentve.
            # A model output tehát elvileg [-1,1] → step() elfogadja.
            chunk_buf = actions_np[:exec_horizon]   # (exec_horizon, 5)
            buf_idx   = 0

            if verbose:
                print(f"  [re-query @ step {total_steps}] "
                      f"chunk mean abs: {np.abs(actions_np).mean():.3f}")

        # Következő chunk-lépés végrehajtása
        action_dataset_norm = chunk_buf[buf_idx]
        buf_idx += 1

        # FONTOS: a modell a dataset-norm térben ad outputot.
        # step() az eredeti [-1,1] (scripted_expert _norm_action) térben vár inputot.
        # Denormalizáció: dataset-norm → eredeti [-1,1].
        action = normalizer.denormalize_action(action_dataset_norm)

        obs, done, info = env.step(action)
        last_info    = info
        total_steps += 1

        if done:
            break

    return {
        "success":    last_info.get("success", False),
        "place_dist": last_info.get("place_dist", float("inf")),
        "steps":      total_steps,
        "timeout":    last_info.get("timeout", False),
    }


# ─── Main eval ───────────────────────────────────────────────────────────────

def evaluate(
    ckpt_dir:     str,
    stats_path:   str | None,
    n_eval:       int,
    exec_horizon: int,
    seed:         int,
    verbose:      bool,
):
    ckpt_dir  = Path(_REPO_ROOT / ckpt_dir) if not Path(ckpt_dir).is_absolute() \
                else Path(ckpt_dir)

    # --- Model betöltése ---
    print(f"Checkpoint: {ckpt_dir}")
    model, cfg = load_policy(str(ckpt_dir))
    device     = next(model.parameters()).device
    print(f"Device: {device} | chunk_size: {cfg['model']['chunk_size']}")

    if exec_horizon < 0:
        exec_horizon = cfg["model"]["chunk_size"]
    print(f"exec_horizon: {exec_horizon}")

    # --- Normalizációs statisztikák ---
    if stats_path is None:
        ds_path    = Path(_REPO_ROOT) / cfg["dataset"]["path"]
        stats_path = str(ds_path / "meta" / "stats.json")

    stats_path = Path(stats_path) if Path(stats_path).is_absolute() \
                 else _REPO_ROOT / stats_path

    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json nem található: {stats_path}\n"
                                f"Add meg: --stats data/lerobot/scripted_v1/meta/stats.json")

    with open(stats_path) as f:
        stats = json.load(f)
    normalizer = StatsNormalizer(stats)
    print(f"Stats: {stats_path.name} betöltve ✅")

    # --- Env ---
    xml_path = _REPO_ROOT / cfg["eval"]["env"]["xml_path"]
    env      = PushTaskEnv(xml_path, seed=seed)

    # --- Eval loop ---
    print(f"\n{'─'*60}")
    print(f"Eval: {n_eval} epizód | exec_horizon={exec_horizon} | seed={seed}")
    print(f"{'─'*60}")

    results = []
    successes = 0

    for ep in range(n_eval):
        res = run_episode(env, model, normalizer, device,
                          exec_horizon=exec_horizon, verbose=verbose)
        results.append(res)
        if res["success"]:
            successes += 1

        sr_so_far = 100.0 * successes / (ep + 1)
        status    = "✅" if res["success"] else "❌"
        print(f"[{ep+1:3d}/{n_eval}] {status}  "
              f"steps={res['steps']:3d}  "
              f"place_dist={res['place_dist']:.3f}m  "
              f"SR={sr_so_far:.1f}%")

    # --- Összesítés ---
    sr          = 100.0 * successes / n_eval
    avg_steps   = float(np.mean([r["steps"]      for r in results]))
    avg_dist    = float(np.mean([r["place_dist"]  for r in results]))
    n_timeout   = sum(r["timeout"] for r in results)
    succ_steps  = [r["steps"] for r in results if r["success"]]
    avg_succ_steps = float(np.mean(succ_steps)) if succ_steps else float("nan")

    print(f"\n{'═'*60}")
    print(f"EREDMÉNY: {successes}/{n_eval} siker  →  SR = {sr:.1f}%")
    print(f"  Átlag lépés (összes):   {avg_steps:.1f}")
    print(f"  Átlag lépés (sikeresek): {avg_succ_steps:.1f}")
    print(f"  Átlag place_dist:        {avg_dist:.3f} m")
    print(f"  Timeout:                 {n_timeout}/{n_eval}")
    print(f"{'═'*60}")

    # --- Elfogadási feltétel ---
    if sr >= 80.0:
        verdict = "✅ KIVÁLÓ — F3e (UnifoLM LoRA) kihagyható; Phase 040 jöhet"
    elif sr >= 60.0:
        verdict = "✅ ELFOGADVA — F3e (BC+PPO PPF fine-tune) ajánlott a ≥75% célhoz"
    elif sr >= 40.0:
        verdict = "⚠️  GYENGE — F3e PPF fine-tune szükséges"
    elif sr >= 20.0:
        verdict = "❌ ELÉGTELEN — F3d újrafuttatás több adattal / más hyperparaméterekkel"
    else:
        verdict = "❌ SIKERTELEN — modell nem tanult; hibakeresés szükséges"

    print(f"\n{verdict}")
    print(f"{'─'*60}")

    return sr, results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACT policy eval — Phase 030 F3d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példa:
  python3 tools/eval_act.py \\
      --ckpt          results/bc_checkpoints_act_v1 \\
      --stats         data/lerobot/scripted_v1/meta/stats.json \\
      --n-eval        50 \\
      --exec-horizon  5
        """,
    )
    parser.add_argument("--ckpt",         required=True,
                        help="Checkpoint könyvtár (train_act.py kimenete)")
    parser.add_argument("--stats",        default=None,
                        help="stats.json (alapból: dataset.path/meta/stats.json a config-ból)")
    parser.add_argument("--n-eval",       type=int, default=50,
                        help="Eval epizódok száma (alapért.: 50)")
    parser.add_argument("--exec-horizon", type=int, default=5,
                        help="Lépések száma re-query előtt (alapért.: 5; -1 → chunk_size)")
    parser.add_argument("--seed",         type=int, default=123,
                        help="RNG seed az env reset-hez (alapért.: 123)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Re-query eseményeket is kiírja")
    args = parser.parse_args()

    evaluate(
        ckpt_dir     = args.ckpt,
        stats_path   = args.stats,
        n_eval       = args.n_eval,
        exec_horizon = args.exec_horizon,
        seed         = args.seed,
        verbose      = args.verbose,
    )


if __name__ == "__main__":
    main()
