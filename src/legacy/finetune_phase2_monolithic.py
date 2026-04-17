#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 2: G1 Retail Navigáció PPO Fine-tune

A már betanított 3M lépéses modell folytatása további lépésekkel.
A TensorBoard grafikon folytonos marad (reset_num_timesteps=False).

Használat:
  python src/training/roboshelf_phase2_finetune.py              # +3M lépés (default)
  python src/training/roboshelf_phase2_finetune.py --steps 6000000
  python src/training/roboshelf_phase2_finetune.py --model path/to/model.zip
"""

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ENVS_DIR = _THIS_DIR.parent / "envs"
if str(_ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(_ENVS_DIR))

import numpy as np
import torch
import shutil

# --- Output mappa ---
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = Path("/kaggle/working/roboshelf-phase2-results")
elif os.path.exists("/content"):
    OUTPUT_DIR = Path("/content/roboshelf-phase2-results")
else:
    _REPO_ROOT = _THIS_DIR.parent.parent
    _RESULTS_DIR = _REPO_ROOT / "roboshelf-results" / "phase2"
    if _RESULTS_DIR.exists() or (_REPO_ROOT / "roboshelf-results").exists():
        OUTPUT_DIR = _RESULTS_DIR
    else:
        OUTPUT_DIR = Path.home() / "Documents" / "roboshelf-ai-dev" / "roboshelf-results" / "phase2"

MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"


def make_retail_env(n_envs=4, seed=42):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    def make_single(rank):
        def _init():
            env = RoboshelfRetailNavEnv()
            env.reset(seed=seed + rank)
            return env
        return _init

    set_random_seed(seed)
    if n_envs > 1:
        env = SubprocVecEnv([make_single(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_single(0)])

    # norm_reward=True: tréning közben normalizált reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env


def make_eval_env():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    env = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    return env


def find_best_model():
    """Megkeresi a best_model.zip-et az OUTPUT_DIR-ben."""
    best = MODELS_DIR / "best" / "best_model.zip"
    if best.exists():
        return best
    # Fallback: legújabb final model
    finals = sorted(MODELS_DIR.glob("*_final.zip"))
    if finals:
        return finals[-1]
    return None


def finetune(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

    # v21 fine-tune: futtatáskor a v20 modellből indul
    # Env frissítések (v21): skálázatlan iránybüntetések + gait=0.5 + vel_air_threshold=0.02
    # Konfiguráció: 0N buoyancy (v20 már leépítette), penalty_scale=1.0-ról indul
    # LR: 1e-4 (v20-nál 3e-4 volt — kisebb lépések a finomhangoláshoz)
    # Clip: 0.15 (v20-nál 0.2 volt — konzervatívabb frissítések)

    timestamp = int(time.time())
    run_name = f"g1_retail_nav_m2_20m_v21_{timestamp}"

    # Device
    if torch.cuda.is_available():
        device = "cuda"
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU (M2)"

    # Modell és VecNormalize útvonalak
    model_path = Path(args.model) if args.model else find_best_model()
    if model_path is None:
        print("❌ Nem találok betanított modellt!")
        print(f"   Keresés: {MODELS_DIR}")
        print("   Add meg kézzel: --model <path/to/model.zip>")
        sys.exit(1)

    # VecNormalize: model mellé keresünk .pkl-t, vagy best/ mappában
    vecnorm_path = Path(str(model_path).replace(".zip", "_vecnormalize.pkl"))
    if not vecnorm_path.exists():
        vecnorm_path = MODELS_DIR / "best" / "best_model_vecnormalize.pkl"
    if not vecnorm_path.exists():
        # Utolsó final vecnormalize
        pkls = sorted(MODELS_DIR.glob("*_final_vecnormalize.pkl"))
        vecnorm_path = pkls[-1] if pkls else None

    print(f"\n{'='*60}")
    print(f"  ROBOSHELF AI — Fázis 2: Fine-tune")
    print(f"  Alap modell: {model_path.name}")
    print(f"  VecNormalize: {vecnorm_path.name if vecnorm_path else 'NEM TALÁLVA — új stats'}")
    print(f"  +{args.steps:,} lépés")
    print(f"  LR: {args.lr:.0e}  |  Clip: {args.clip}")
    print(f"  Device: {device_label}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = MODELS_DIR / "best"
    best_dir.mkdir(exist_ok=True)

    # Env létrehozás
    print("  Környezetek létrehozása...")
    train_env_raw = make_retail_env(n_envs=4)
    eval_env = make_eval_env()

    # VecNormalize stats betöltése a régi tanításból
    if vecnorm_path and vecnorm_path.exists():
        print(f"  VecNormalize stats betöltése: {vecnorm_path.name}")
        # Betöltjük a régi statisztikát az új env-be
        old_stats = VecNormalize.load(vecnorm_path, train_env_raw.venv if hasattr(train_env_raw, 'venv') else train_env_raw)
        # Átmásoljuk a running mean/var-t
        train_env_raw.obs_rms = old_stats.obs_rms
        train_env_raw.ret_rms = old_stats.ret_rms
        # Eval env is kapja a statsot
        eval_env.obs_rms = old_stats.obs_rms
        eval_env.ret_rms = old_stats.ret_rms
        print("  ✅ VecNormalize stats betöltve")
    else:
        print("  ⚠️  VecNormalize stats nem találhatók — új statisztikával indul")

    # Modell betöltése, env cserélve
    print(f"  Modell betöltése: {model_path.name}")
    model = PPO.load(
        str(model_path),
        env=train_env_raw,
        device=device,
        # LR és clip frissítése
        learning_rate=args.lr,
        clip_range=args.clip,
    )
    print(f"  ✅ Modell betöltve ({model.num_timesteps:,} lépés volt eddig)")

    # VecNormalize sync callback
    class SyncVecNormalizeCallback(BaseCallback):
        def __init__(self, train_env, eval_env_ref):
            super().__init__()
            self.train_env = train_env
            self.eval_env_ref = eval_env_ref

        def _on_step(self):
            return True

        def on_eval_start(self):
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env_ref, VecNormalize):
                self.eval_env_ref.obs_rms = self.train_env.obs_rms
                self.eval_env_ref.ret_rms = self.train_env.ret_rms

    sync_cb = SyncVecNormalizeCallback(train_env_raw, eval_env)

    eval_freq = max(args.steps // 10, 50_000)

    # evaluations.npz mentés előtt backupoljuk a meglévőt
    # Minden futásnál frissítjük a backupot a jelenlegi teljes históriával
    eval_npz = LOGS_DIR / "evaluations.npz"
    eval_npz_backup = LOGS_DIR / "evaluations_before_finetune.npz"
    if eval_npz.exists():
        shutil.copy2(eval_npz, eval_npz_backup)
        print(f"  💾 evaluations.npz backup frissítve ({eval_npz_backup.name})")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(LOGS_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    # AppendEvalCallback: csak eval után fut, nem minden lépésnél
    class AppendEvalCallback(BaseCallback):
        """Eval után (eval_freq-enként egyszer) mergeli a backup + új npz-t."""
        def __init__(self, backup_npz, output_npz, freq):
            super().__init__()
            self.backup_npz = backup_npz
            self.output_npz = output_npz
            self.freq = freq
            self._last_merge = 0

        def _on_step(self):
            # Csak eval_freq-enként fut, nem minden lépésnél
            if self.num_timesteps - self._last_merge >= self.freq:
                self._merge()
                self._last_merge = self.num_timesteps
            return True

        def _merge(self):
            if self.output_npz.exists() and self.backup_npz.exists():
                old = np.load(self.backup_npz)
                new = np.load(self.output_npz)
                all_ts = np.concatenate([old['timesteps'], new['timesteps']])
                all_r  = np.concatenate([old['results'],   new['results']])
                all_l  = np.concatenate([old['ep_lengths'], new['ep_lengths']])
                _, unique_idx = np.unique(all_ts, return_index=True)
                np.savez(self.output_npz,
                         timesteps=all_ts[unique_idx],
                         results=all_r[unique_idx],
                         ep_lengths=all_l[unique_idx])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.steps // 5, 100_000),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=run_name,
    )

    # Fine-tune tanítás — reset_num_timesteps=False: folytonos TensorBoard görbe
    print(f"\n  🚀 Fine-tune indítása (+{args.steps:,} lépés)...\n")
    start = time.time()

    append_cb = AppendEvalCallback(eval_npz_backup, eval_npz, eval_freq)

    model.learn(
        total_timesteps=args.steps,
        callback=[sync_cb, eval_callback, checkpoint_callback, append_cb],
        tb_log_name=run_name,
        reset_num_timesteps=False,   # ← folytonos görbe a korábbi tanítás után
        progress_bar=True,
    )

    # Végső merge (ha az utolsó eval után még nem futott le)
    if eval_npz.exists() and eval_npz_backup.exists():
        old = np.load(eval_npz_backup)
        new = np.load(eval_npz)
        # Deduplikálás timestep alapján
        all_ts = np.concatenate([old['timesteps'], new['timesteps']])
        all_r  = np.concatenate([old['results'],   new['results']])
        all_l  = np.concatenate([old['ep_lengths'], new['ep_lengths']])
        # Egyedi timestepek megtartása, sorrendben
        _, unique_idx = np.unique(all_ts, return_index=True)
        np.savez(eval_npz,
                 timesteps=all_ts[unique_idx],
                 results=all_r[unique_idx],
                 ep_lengths=all_l[unique_idx])
        print(f"  ✅ evaluations.npz kiegészítve: {len(unique_idx)} összesített eval pont")

    elapsed = time.time() - start
    print(f"\n  ⏱️  Tanítás befejezve: {elapsed/60:.1f} perc")

    # Mentés
    final_model = str(MODELS_DIR / f"{run_name}_final")
    model.save(f"{final_model}.zip")
    train_env_raw.save(f"{final_model}_vecnormalize.pkl")
    train_env_raw.save(str(best_dir / "best_model_vecnormalize.pkl"))
    print(f"  💾 {final_model}.zip")
    print(f"  💾 {final_model}_vecnormalize.pkl")

    # Kiértékelés
    print(f"\n  📊 Kiértékelés (10 epizód)...")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize as VN
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    ev = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    ev = VN.load(f"{final_model}_vecnormalize.pkl", ev)
    ev.training = False
    ev.norm_reward = False

    rewards, lengths, dists = [], [], []
    for ep in range(10):
        obs = ev.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = ev.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        if 'dist_to_target' in info[0]:
            dists.append(info[0]['dist_to_target'])
        dist_str = f"{dists[-1]:.2f}m" if dists else "?"
        print(f"    Ep {ep+1:2d}: reward={total_reward:7.1f}, lépés={steps:4d}, táv={dist_str}")

    print(f"\n    📈 Átlag reward: {np.mean(rewards):.1f} (±{np.std(rewards):.1f})")
    print(f"    📏 Átlag hossz:  {np.mean(lengths):.0f} lépés")
    if dists:
        print(f"    📍 Átlag távolság céltól: {np.mean(dists):.2f}m (start: 3.3m)")

    train_env_raw.close()
    eval_env.close()
    ev.close()

    print(f"\n  ✅ Fine-tune kész! Eredmények: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboshelf AI — G1 Fine-tune")
    parser.add_argument(
        "--steps", type=int, default=20_000_000,
        help="További lépések száma (default: 20M — v21 fine-tune)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Alap modell .zip útvonala (default: legfrissebb *_final.zip)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4 — v21 fine-tune)"
    )
    parser.add_argument(
        "--clip", type=float, default=0.15,
        help="PPO clip range (default: 0.15 — v21 fine-tune)"
    )
    args = parser.parse_args()
    finetune(args)
