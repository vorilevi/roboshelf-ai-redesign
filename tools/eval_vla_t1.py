"""
UnifoLM-VLA-0 Policy Evaluation — T1 Vendor-independence track.

Betölti a Kaggle-on fine-tune-olt VLA checkpointot (step_10000.pt),
majd N epizódot futtat a T1ShelfStockEnv MuJoCo push task környezetben.

G1 eval_vla.py-tól való különbségek:
  - ACTION_DIM = 4  (T1 4-DOF jobb kar, nincs gripper)
  - Nincs kamera — state-only, dummy fekete kép a VLM inputhoz
  - Checkpoint formátum: trainable_state dict (LoRA + action model együtt)
  - T1ShelfStockEnv (gymnasium) az eval env

Futtatás (repo gyökeréből):
    # Mac M2 CPU-n:
    python3 tools/eval_vla_t1.py \\
        --ckpt results/vla_ckpt/unifolm_vla_t1_roboshelf/step_10000.pt \\
        --n-eval 50 --no-quantize

    # CUDA-val (gyorsabb):
    python3 tools/eval_vla_t1.py \\
        --ckpt results/vla_ckpt/unifolm_vla_t1_roboshelf/step_10000.pt \\
        --n-eval 50

Kimenet:
    Success rate, átlag epizódhossz, átlag place_dist
    Elfogadás: ≥70% SR → T1 VLA validált, vendor-independence track Phase 040 jöhet

Referenciák:
    G1 eval:   tools/eval_vla.py
    T1 train:  notebooks/kaggle_unifolm_vla_finetune_t1.py
    T1 env:    src/roboshelf_ai/mujoco/envs/manipulation/t1_shelf_stock_env.py
    Checkpoint: results/vla_ckpt/unifolm_vla_t1_roboshelf/step_10000.pt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_TOOLS_DIR = _HERE.parent

sys.path.insert(0, str(_REPO_ROOT / "src"))

from roboshelf_ai.mujoco.envs.manipulation.t1_shelf_stock_env import (
    T1ShelfStockEnv,
    ACTION_DIM as _ENV_ACTION_DIM,
)


# ── Konstansok (tréninggel azonos!) ──────────────────────────────────────────

ACTION_DIM  = 4      # T1: 4-DOF jobb kar (nincs gripper)
STATE_DIM   = 24
CHUNK_SIZE  = 10
LANGUAGE    = "Push the stock to the target position on the shelf."
HF_MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
IMG_SIZE    = 224
MAX_STEPS   = 300    # max policy lépés per epizód

assert _ENV_ACTION_DIM == ACTION_DIM, \
    f"T1ShelfStockEnv ACTION_DIM={_ENV_ACTION_DIM} != {ACTION_DIM}"


# ── unifolm-vla repo setup ────────────────────────────────────────────────────

def setup_unifolm_repo(repo_dir: Path) -> None:
    """Klónoz + patchel + telepít — azonos logika mint a Kaggle notebookban."""
    if not repo_dir.exists():
        print(f"Klónozás: {repo_dir} ...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/unitreerobotics/unifolm-vla.git",
            str(repo_dir)
        ], check=True)
    else:
        print(f"Repo megvan: {repo_dir}")

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-deps", "-e", str(repo_dir)
    ], check=True)

    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))

    # Dimenzió patch
    DIM_PATTERNS = [
        (re.compile(r'\bACTION_DIM\s*=\s*\d+'),        f'ACTION_DIM = {ACTION_DIM}'),
        (re.compile(r'\bPROPRIO_DIM\s*=\s*\d+'),       f'PROPRIO_DIM = {STATE_DIM}'),
        (re.compile(r'\bNUM_ACTIONS_CHUNK\s*=\s*\d+'), f'NUM_ACTIONS_CHUNK = {CHUNK_SIZE}'),
    ]
    DIM_KEYWORDS = {'ACTION_DIM','PROPRIO_DIM','NUM_ACTIONS_CHUNK','G1_EE_6D'}
    patched = []
    for py in repo_dir.rglob("*.py"):
        try:
            src = py.read_text(errors='ignore')
        except Exception:
            continue
        if not any(kw in src for kw in DIM_KEYWORDS):
            continue
        new = src
        for pattern, repl in DIM_PATTERNS:
            new = pattern.sub(repl, new)
        if new != src:
            py.write_text(new)
            patched.append(py.relative_to(repo_dir))
    print(f"  Dimenzió patch: {len(patched)} fájl" if patched else "  Dimenzió patch: nem volt szükséges")

    # eager attention patch
    for qp in repo_dir.rglob("QWen2_5.py"):
        src = qp.read_text()
        new = src.replace('attn_implementation="flash_attention_2"', 'attn_implementation="eager"')
        new = new.replace("attn_implementation='flash_attention_2'", "attn_implementation='eager'")
        if new != src:
            qp.write_text(new)
            print("  QWen2_5.py patchelve (eager attn)")


# ── Modell betöltés ───────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, repo_dir: Path, no_quantize: bool):
    """
    Betölti a T1 VLA modelt a step_10000.pt checkpointból.

    A checkpoint trainable_state dict-et tartalmaz (LoRA + action_model),
    nem külön lora_final/ + final.pt mint a G1 verziónál.

    Returns: (model, processor, device)
    """
    import gc, yaml, torch
    import torch.nn as nn
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from peft import get_peft_model, LoraConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.1f} GB")
    else:
        print("CPU módban fut — lassú (~30s/lépés)")

    # ── 1. VLM base betöltés ──
    print("\n[1/5] Qwen2.5-VL-7B betöltés...")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    load_kwargs: dict = dict(attn_implementation="eager", low_cpu_mem_usage=True)
    if device == "cuda" and not no_quantize:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"]   = "auto"
        print("  8-bit quantizáció (CUDA)")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"]  = "cpu"
        print("  bf16 CPU (no quantize)")

    vlm       = Qwen2_5_VLForConditionalGeneration.from_pretrained(HF_MODEL, **load_kwargs)
    processor = Qwen2_5_VLProcessor.from_pretrained(HF_MODEL)
    print("  VLM OK")

    # ── 2. LoRA config (tréninggel azonos: r=32) ──
    print("\n[2/5] LoRA config alkalmazása (r=32)...")
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    vlm = get_peft_model(vlm, lora_cfg)
    print("  LoRA OK")

    # ── 3. Monkey-patch + config ──
    print("\n[3/5] Monkey-patch + config...")
    from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface
    _c_vlm, _c_proc = vlm, processor
    def _patched_init(self, config, **kwargs):
        nn.Module.__init__(self)
        self.model     = _c_vlm
        self.processor = _c_proc
    _QWen_VL_Interface.__init__ = _patched_init

    from unifolm_vla.model.framework.share_tools import dict_to_namespace
    config_yaml = repo_dir / "src/unifolm_vla/config/training/unifolm_vla_train.yaml"
    with open(config_yaml) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))
    cfg.framework.qwenvl.base_vlm                        = HF_MODEL
    cfg.framework.qwenvl.attn_implementation             = "eager"
    cfg.framework.action_model.action_dim                = ACTION_DIM
    cfg.framework.action_model.state_dim                 = STATE_DIM
    cfg.framework.action_model.action_horizon            = CHUNK_SIZE
    cfg.framework.action_model.future_action_window_size = CHUNK_SIZE - 1
    if not hasattr(cfg, "trainer") or cfg.trainer is None:
        cfg.trainer = dict_to_namespace({"repeated_diffusion_steps": 4})

    # ── 4. Unifolm_VLA + runtime konstans override ──
    print("\n[4/5] Unifolm_VLA instantiálás...")
    from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA

    _overrides = {'ACTION_DIM': ACTION_DIM, 'PROPRIO_DIM': STATE_DIM,
                  'NUM_ACTIONS_CHUNK': CHUNK_SIZE}
    for _mn in list(sys.modules.keys()):
        if 'unifolm' not in _mn:
            continue
        _m = sys.modules[_mn]
        for _k, _v in _overrides.items():
            if hasattr(_m, _k) and getattr(_m, _k) != _v:
                setattr(_m, _k, _v)

    model = Unifolm_VLA(config=cfg)
    model.action_model = model.action_model.to(device, torch.float32)

    # ── 5. Checkpoint betöltés (trainable_state) ──
    print(f"\n[5/5] Checkpoint betöltés: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nem található: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)

    if "trainable_state" in ckpt:
        state = ckpt["trainable_state"]
        step  = ckpt.get("step", "?")
        loss  = ckpt.get("loss", "?")
        print(f"  step={step}, loss={loss:.4f}" if isinstance(loss, float) else f"  step={step}")
    elif "model_state" in ckpt:
        # Régi formátum fallback
        state = ckpt["model_state"]
        print("  ⚠️ Régi model_state formátum")
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    print(f"  Betöltött kulcsok: {n_loaded}/{len(state)}")
    if unexpected:
        print(f"  ⚠️  Ismeretlen kulcsok: {len(unexpected)}")

    model.eval()
    print("✅ load_model kész")
    return model, processor, device


# ── VLA inference ─────────────────────────────────────────────────────────────

def build_vla_input(state: np.ndarray, processor, device: str) -> dict:
    """
    T1 VLA input — nincs kamera, dummy fekete kép (tréninggel azonos).
    state: (24,) float32
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Dummy fekete kép (tréninggel azonos: np.zeros 224x224x3)
    image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    msg = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  LANGUAGE}
    ]}]
    text    = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    img_inp, _ = process_vision_info(msg)

    out = processor(text=[text], images=[img_inp], padding=True, return_tensors="pt")
    out["state"] = torch.FloatTensor(state).unsqueeze(0)   # (1, 24)

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in out.items()}


@torch.no_grad()
def predict_actions(model, inputs: dict, device: str) -> np.ndarray:
    """VLA inference → (CHUNK_SIZE, ACTION_DIM) normalizált akciók."""
    output = model(qwen_inputs=inputs)

    if isinstance(output, dict):
        for key in ("action_pred", "actions", "pred_actions", "action", "output"):
            if key in output and output[key] is not None:
                acts = output[key]
                if isinstance(acts, torch.Tensor):
                    return acts[0].cpu().float().numpy()

        print(f"\n  ⚠️  Ismeretlen output kulcsok: {list(output.keys())}")
        raise RuntimeError("VLA output nem tartalmaz action tenzort.")

    elif isinstance(output, torch.Tensor):
        return output[0].cpu().float().numpy()

    raise RuntimeError(f"Ismeretlen VLA output típus: {type(output)}")


# ── Epizód futtatás ───────────────────────────────────────────────────────────

def run_episode(env: T1ShelfStockEnv, model, processor,
                device: str, exec_horizon: int,
                seed: int, verbose: bool = False) -> dict:
    """
    Egyetlen T1 epizód VLA vezérléssel.
    T1ShelfStockEnv: step() visszaad obs, reward, terminated, truncated, info
    """
    obs, _info = env.reset(seed=seed)
    chunk_buf = np.zeros((0, ACTION_DIM), dtype=np.float32)
    buf_idx   = 0
    last_info: dict = {}
    t_query   = 0.0
    step_count = 0

    for _ in range(MAX_STEPS):
        if buf_idx >= len(chunk_buf):
            t0        = time.time()
            inputs    = build_vla_input(obs, processor, device)
            chunk_buf = predict_actions(model, inputs, device)  # (CHUNK_SIZE, 4)
            buf_idx   = 0
            t_query   = time.time() - t0
            if verbose:
                print(f"    [query step={step_count}] {t_query:.2f}s | "
                      f"chunk mean abs: {np.abs(chunk_buf).mean():.3f}")

        action = chunk_buf[buf_idx]
        buf_idx += 1

        obs, _reward, terminated, truncated, info = env.step(action)
        last_info  = info
        step_count += 1
        done = terminated or truncated
        if done:
            break

    success = last_info.get("placed", False)
    dist    = last_info.get("stock_target_dist", float("inf"))

    return {
        "success":    success,
        "place_dist": dist,
        "steps":      step_count,
        "timeout":    not success,
        "t_query_s":  t_query,
    }


# ── Eval fő logika ────────────────────────────────────────────────────────────

def evaluate(args):
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() \
                else _REPO_ROOT / args.ckpt

    repo_dir = Path(args.repo) if args.repo else \
               Path.home() / "roboshelf-ai-dev" / "unifolm_roboshelf"

    print("=" * 65)
    print("UnifoLM-VLA-0 Eval — T1 Vendor-independence track")
    print("=" * 65)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Repo:       {repo_dir}")
    print(f"ACTION_DIM={ACTION_DIM} | STATE_DIM={STATE_DIM} | CHUNK={CHUNK_SIZE}")

    setup_unifolm_repo(repo_dir)
    model, processor, device = load_model(ckpt_path, repo_dir, args.no_quantize)

    exec_horizon = args.exec_horizon
    if exec_horizon < 1 or exec_horizon > CHUNK_SIZE:
        exec_horizon = CHUNK_SIZE

    env = T1ShelfStockEnv()

    print(f"\nexec_horizon={exec_horizon} | n_eval={args.n_eval} | seed={args.seed}")
    print(f"\n{'─'*65}")
    print(f"Eval indítás: {args.n_eval} epizód")
    print(f"{'─'*65}")

    results   = []
    successes = 0
    t0_total  = time.time()

    for ep in range(args.n_eval):
        ep_seed = args.seed + ep
        try:
            res = run_episode(env, model, processor, device,
                              exec_horizon=exec_horizon,
                              seed=ep_seed, verbose=args.verbose)
        except Exception as e:
            print(f"\n[{ep+1:3d}/{args.n_eval}] ❌ HIBA: {e}")
            if args.verbose:
                import traceback; traceback.print_exc()
            results.append({"success": False, "place_dist": float("inf"),
                             "steps": 0, "timeout": True, "t_query_s": 0})
            continue

        results.append(res)
        if res["success"]:
            successes += 1

        sr_now  = 100.0 * successes / (ep + 1)
        status  = "✅" if res["success"] else "❌"
        elapsed = time.time() - t0_total
        print(f"[{ep+1:3d}/{args.n_eval}] {status}  "
              f"steps={res['steps']:3d}  dist={res['place_dist']:.3f}m  "
              f"SR={sr_now:.1f}%  ({elapsed/60:.1f}min)")

    # ── Összesítés ──
    sr         = 100.0 * successes / args.n_eval
    avg_steps  = float(np.mean([r["steps"]     for r in results]))
    valid_dist = [r["place_dist"] for r in results if r["place_dist"] < 1e6]
    avg_dist   = float(np.mean(valid_dist)) if valid_dist else float("inf")
    n_timeout  = sum(r["timeout"] for r in results)
    succ_steps = [r["steps"] for r in results if r["success"]]
    avg_q_time = float(np.mean([r["t_query_s"] for r in results if r["t_query_s"] > 0]))
    total_min  = (time.time() - t0_total) / 60

    print(f"\n{'═'*65}")
    print(f"EREDMÉNY — T1 UnifoLM-VLA-0: {successes}/{args.n_eval}  SR = {sr:.1f}%")
    print(f"  Átlag lépés (összes):    {avg_steps:.1f}")
    if succ_steps:
        print(f"  Átlag lépés (sikeresek): {float(np.mean(succ_steps)):.1f}")
    print(f"  Átlag place_dist:        {avg_dist:.3f} m")
    print(f"  Timeout:                 {n_timeout}/{args.n_eval}")
    print(f"  Átlag VLA query idő:     {avg_q_time:.2f}s")
    print(f"  Teljes futásidő:         {total_min:.1f} perc")
    print(f"{'═'*65}")

    if sr >= 70.0:
        verdict = "✅ ELFOGADVA — T1 UnifoLM-VLA-0 ≥70% SR, vendor-independence validált"
    elif sr >= 50.0:
        verdict = "⚠️  RÉSZLEGES — további fine-tune vagy több demo ajánlott"
    elif sr >= 20.0:
        verdict = "❌ GYENGE — több adat / hosszabb tréning szükséges"
    else:
        verdict = "❌ SIKERTELEN — inference hiba vagy modell nem tanult"

    print(f"\n{verdict}")
    print(f"{'─'*65}")

    return sr, results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="UnifoLM-VLA-0 eval — T1 vendor-independence track"
    )
    p.add_argument("--ckpt", type=str,
                   default="results/vla_ckpt/unifolm_vla_t1_roboshelf/step_10000.pt",
                   help="Checkpoint path (step_10000.pt)")
    p.add_argument("--repo", type=str, default=None,
                   help="unifolm-vla repo helye (default: ~/roboshelf-ai-dev/unifolm_roboshelf)")
    p.add_argument("--n-eval", type=int, default=50,
                   help="Epizódok száma (default: 50)")
    p.add_argument("--exec-horizon", type=int, default=10,
                   help=f"Lépések száma egy VLA query után (1-{CHUNK_SIZE}, default: {CHUNK_SIZE})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-quantize", action="store_true",
                   help="Ne használjon 8-bit quantizációt (Mac M2 / CPU eval)")
    p.add_argument("--verbose", action="store_true",
                   help="VLA query részletek kiírása")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
