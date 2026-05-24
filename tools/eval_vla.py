"""
UnifoLM-VLA-0 Policy Evaluation — F3e Phase 030.

Betölti a Kaggle-on fine-tune-olt VLA checkpointot (final.pt + lora_final/),
majd N epizódot futtat a MuJoCo push task környezetben, és méri a success rate-et.

Architektúra:
  - Qwen2.5-VL-7B-Instruct (8-bit CUDA / bf16 CPU) + LoRA adapter
  - DiT flow-matching action head (action_model, 588M param)
  - Input: front_cam kép (224x224) + state (24-dim) + language
  - Output: action chunk (CHUNK_SIZE=10, ACTION_DIM=5)

Futtatás (repo gyökeréből):
    # CUDA-val (ajánlott, gyors):
    python3 tools/eval_vla.py \\
        --ckpt    results/vla_ckpt \\
        --n-eval  50

    # CPU-n (Mac M2, lassú ~30s/lépés):
    python3 tools/eval_vla.py \\
        --ckpt    results/vla_ckpt \\
        --n-eval  10 \\
        --no-quantize

    # unifolm-vla repo helye (alapért. ~/roboshelf-ai-dev/unifolm_roboshelf):
    python3 tools/eval_vla.py \\
        --ckpt    results/vla_ckpt \\
        --repo    /path/to/unifolm-vla

Kimenet:
    Success rate, átlag epizódhossz, átlag place_dist
    Acceptance: ≥70% SR → F3e elfogadva, Phase 040 jöhet
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

if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import scripted_expert as _exp
from scripted_expert import (
    _get_obs, _JOINT_RANGES, _GRIPPER_CLOSED, _GRIPPER_OPEN,
    _DEFAULT_ARM_POS, ARM_QPOS_START, ARM_CTRL_START, GRIPPER_CTRL_START,
    N_ARM_DOF, DECIMATION, GOAL_RADIUS, STOCK_QPOS_START, STOCK_RESET_Z,
    MIN_SUCCESS_STEP,
)
import mujoco


# ── Konstansok (tréninggel azonos) ───────────────────────────────────────────

ACTION_DIM   = 5
STATE_DIM    = 24
CHUNK_SIZE   = 10
LANGUAGE     = "Push the box to the target position on the shelf."
HF_MODEL     = "Qwen/Qwen2.5-VL-7B-Instruct"
IMG_SIZE     = 224
XML_PATH     = _REPO_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"
CAMERA_NAME  = "front_cam"
MAX_STEPS    = 300


# ── unifolm-vla repo setup ────────────────────────────────────────────────────

def setup_unifolm_repo(repo_dir: Path) -> None:
    """
    Klónozza a unifolm-vla repot ha nincs meg, majd patcheli és telepíti.
    Ugyanaz a logika mint a Kaggle notebookban (Cella 2).
    """
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

    # ── PATCH 1: dimenzió konstansok ──
    DIM_PATTERNS = [
        (re.compile(r'\bACTION_DIM\s*=\s*\d+'),        f'ACTION_DIM = {ACTION_DIM}'),
        (re.compile(r'\bPROPRIO_DIM\s*=\s*\d+'),       f'PROPRIO_DIM = {STATE_DIM}'),
        (re.compile(r'\bNUM_ACTIONS_CHUNK\s*=\s*\d+'), f'NUM_ACTIONS_CHUNK = {CHUNK_SIZE}'),
        (re.compile(r'"action_dim"\s*:\s*\d+'),         f'"action_dim": {ACTION_DIM}'),
        (re.compile(r'"proprio_dim"\s*:\s*\d+'),        f'"proprio_dim": {STATE_DIM}'),
        (re.compile(r'"num_actions_chunk"\s*:\s*\d+'),  f'"num_actions_chunk": {CHUNK_SIZE}'),
        (re.compile(r"'action_dim'\s*:\s*\d+"),         f"'action_dim': {ACTION_DIM}"),
        (re.compile(r"'proprio_dim'\s*:\s*\d+"),        f"'proprio_dim': {STATE_DIM}"),
        (re.compile(r"'num_actions_chunk'\s*:\s*\d+"),  f"'num_actions_chunk': {CHUNK_SIZE}"),
    ]
    DIM_KEYWORDS = {'ACTION_DIM','PROPRIO_DIM','NUM_ACTIONS_CHUNK',
                    'action_dim','proprio_dim','num_actions_chunk','G1_EE_6D'}
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
    if patched:
        for f in patched:
            print(f"  Patchelve: {f}")
    else:
        print("  Dimenzió patch: nem volt szükséges")

    # ── PATCH 2: eager attention ──
    for qp in repo_dir.rglob("QWen2_5.py"):
        src = qp.read_text()
        new = src.replace('attn_implementation="flash_attention_2"', 'attn_implementation="eager"')
        new = new.replace("attn_implementation='flash_attention_2'", "attn_implementation='eager'")
        if new != src:
            qp.write_text(new)
            print("  QWen2_5.py patchelve (eager attn)")


# ── Modell betöltés ───────────────────────────────────────────────────────────

def load_model(ckpt_dir: Path, repo_dir: Path, no_quantize: bool):
    """
    Betölti a VLA modelt:
      1. Qwen2.5-VL-7B base (8-bit CUDA / bf16 CPU) + LoRA adapter
      2. DiT action_model (final.pt)
      3. Monkey-patch + runtime konstans override

    Returns: (model, processor, device)
    """
    import gc, yaml, torch
    import torch.nn as nn
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from peft import PeftModel

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

    load_kwargs: dict = dict(
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if device == "cuda" and not no_quantize:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"]   = "auto"
        print("  8-bit quantizáció (CUDA)")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"]  = "cpu"
        print("  bf16 CPU (no quantize)")

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(HF_MODEL, **load_kwargs)
    processor = Qwen2_5_VLProcessor.from_pretrained(HF_MODEL)

    if device == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VLM OK — VRAM: {used:.1f}/{total:.1f} GB")

    # ── 2. LoRA adapter betöltés ──
    lora_dir = ckpt_dir / "lora_final"
    print(f"\n[2/5] LoRA adapter: {lora_dir}")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter nem található: {lora_dir}")
    vlm = PeftModel.from_pretrained(vlm, str(lora_dir))
    vlm = vlm.merge_and_unload()   # LoRA súlyokat beolvasztja → gyorsabb inference
    print("  LoRA beolvasztva (merge_and_unload)")

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

    # ── 4. Unifolm_VLA + runtime patch ──
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

    model     = Unifolm_VLA(config=cfg)
    model_proc = model.processor
    model.action_model = model.action_model.to(device, torch.float32)

    # Dimenzió ellenőrzés + fallback réteg-csere
    layer1 = model.action_model.action_encoder.layer1
    if layer1.in_features != ACTION_DIM:
        print(f"  ⚠️ Réteg-csere ({layer1.in_features}→{ACTION_DIM})...")
        OLD = layer1.in_features
        for name, lyr in list(model.action_model.named_modules()):
            if not isinstance(lyr, nn.Linear):
                continue
            nl = name.lower()
            is_state = ('state' in nl or 'proprio' in nl) and 'action' not in nl
            ni = (STATE_DIM if is_state else ACTION_DIM) if lyr.in_features  == OLD else lyr.in_features
            no = (STATE_DIM if is_state else ACTION_DIM) if lyr.out_features == OLD else lyr.out_features
            if ni == lyr.in_features and no == lyr.out_features:
                continue
            parts = name.split('.')
            parent = model.action_model
            for p_name in parts[:-1]:
                parent = getattr(parent, p_name)
            new_lyr = nn.Linear(ni, no, bias=(lyr.bias is not None)).to(device, torch.float32)
            setattr(parent, parts[-1], new_lyr)
        model.action_model = model.action_model.to(device)
        layer1 = model.action_model.action_encoder.layer1

    assert layer1.in_features == ACTION_DIM, \
        f"Réteg-csere sikertelen: {layer1.in_features} != {ACTION_DIM}"
    print(f"  layer1: ({ACTION_DIM} → {layer1.out_features}) ✓")

    # ── 5. action_model súlyok betöltése ──
    print("\n[5/5] action_model betöltés (final.pt)...")
    final_pt = ckpt_dir / "final.pt"
    if not final_pt.exists():
        raise FileNotFoundError(f"final.pt nem található: {final_pt}")
    ckpt = torch.load(str(final_pt), map_location=device)

    # A final.pt tartalmazhatja a state_dict-et direktben vagy 'action_model' kulcs alatt
    if "action_model" in ckpt:
        state_dict = ckpt["action_model"]
        step  = ckpt.get("step", "?")
        loss  = ckpt.get("loss", "?")
        print(f"  step={step}, loss={loss:.4f}" if isinstance(loss, float) else f"  step={step}")
    else:
        state_dict = ckpt

    model.action_model.load_state_dict(state_dict, strict=False)
    print("  action_model súlyok betöltve ✅")

    model.eval()
    return model, model_proc, device


# ── MuJoCo Env ────────────────────────────────────────────────────────────────

class VLAPushEnv:
    """Push task env VLA inferenciához: obs + kamera frame."""

    def __init__(self, seed: int = 0):
        self._model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self._data  = mujoco.MjData(self._model)
        self._rng   = np.random.default_rng(seed)

        self._renderer    = mujoco.Renderer(self._model, height=IMG_SIZE, width=IMG_SIZE)
        self._cam_id      = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

        self._hand_body_ids: set = set()
        for name in _exp._HAND_BODY_NAMES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._hand_body_ids.add(bid)

        self._step_count = 0

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns: (obs 24-dim, frame RGB 224×224×3)"""
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = _DEFAULT_ARM_POS
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF]  = _DEFAULT_ARM_POS
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7]  = _GRIPPER_OPEN

        lo_x, hi_x = _exp.STOCK_RESET_X_RANGE
        lo_y, hi_y = _exp.STOCK_RESET_Y_RANGE
        for _ in range(50):
            sx = float(self._rng.uniform(lo_x, hi_x))
            sy = float(self._rng.uniform(lo_y, hi_y))
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
        return self._get_obs(), self._render()

    def step(self, action_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        action_raw: (5,) — [4 kar joint pos ∈ joint_range + 1 gripper ∈ [-1,1]]
        A VLA outputja a tréningbeli normalised action tér [-1,1].
        Konverzió: [-1,1] → tényleges joint pozíció (azonos mint eval_act.py step()).
        """
        arm_action     = np.array(action_raw[:4], dtype=np.float32)
        gripper_signal = float(np.clip(action_raw[4], -1.0, 1.0))

        lo, hi = _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
        target_qpos = lo + (arm_action + 1.0) * 0.5 * (hi - lo)
        target_qpos = np.clip(target_qpos, lo, hi)
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = target_qpos

        t = (gripper_signal + 1.0) / 2.0
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = (
            (1.0 - t) * _GRIPPER_OPEN + t * _GRIPPER_CLOSED
        )

        for _ in range(DECIMATION):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs   = self._get_obs()
        frame = self._render()

        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        place_dist = float(np.linalg.norm(stock_pos - target_pos))
        success = (place_dist < GOAL_RADIUS) and (self._step_count >= MIN_SUCCESS_STEP)
        timeout = self._step_count >= MAX_STEPS
        done    = success or timeout

        return obs, frame, done, {"success": success, "place_dist": place_dist,
                                  "timeout": timeout, "step": self._step_count}

    def _render(self) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=CAMERA_NAME)
        return self._renderer.render().copy()   # RGB uint8 (H, W, 3)

    def _get_obs(self) -> np.ndarray:
        return _get_obs(self._model, self._data,
                        self._hand_site_id, self._target_site_id,
                        self._stock_body_id, self._hand_body_ids)

    def close(self):
        self._renderer.close()


# ── VLA inference ─────────────────────────────────────────────────────────────

def build_vla_input(frame: np.ndarray, state: np.ndarray,
                    processor, device: str) -> dict:
    """
    Ugyanaz a collate logika mint a Kaggle tréning notebookban (Cella 4).
    frame: (H, W, 3) uint8 RGB
    state: (24,) float32
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    image = Image.fromarray(frame)
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  LANGUAGE}
    ]}]
    text    = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    img_inp, _ = process_vision_info(msg)

    out = processor(text=[text], images=[img_inp], padding=True, return_tensors="pt")
    out["state"] = torch.FloatTensor(state).unsqueeze(0)   # (1, 24)

    # Eszközre mozgatás
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in out.items()}


@torch.no_grad()
def predict_actions(model, inputs: dict, device: str) -> np.ndarray:
    """
    VLA inference: visszaad (CHUNK_SIZE, ACTION_DIM) normalizált akciókat.

    A Unifolm_VLA.forward() training módban action_loss-t számol (ha van 'action' a dict-ben).
    Inference módban (nincs 'action') a flow matching ODE-t futtatja és predicted action-t ad.

    Ha az API különbözik az itt feltételezettől, ellenőrizd:
        from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
        import inspect; print(inspect.getsource(Unifolm_VLA.forward))
    """
    output = model(qwen_inputs=inputs)

    # ── API discovery: keressük az action outputot ──
    if isinstance(output, dict):
        for key in ("action_pred", "actions", "pred_actions", "action", "output"):
            if key in output and output[key] is not None:
                acts = output[key]
                if isinstance(acts, torch.Tensor):
                    return acts[0].cpu().float().numpy()   # (CHUNK_SIZE, ACTION_DIM)

        # Ha semmilyen action kulcs nincs — kiírjuk mit kaptunk
        print(f"\n  ⚠️  Ismeretlen output kulcsok: {list(output.keys())}")
        print("      Módosítsd a predict_actions() függvényt a helyes kulccsal.")
        print("      A model forward() forráskód megnézéséhez:")
        print("        from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA")
        print("        import inspect; print(inspect.getsource(Unifolm_VLA.forward))")
        raise RuntimeError("VLA output nem tartalmaz action tenzort. Lásd a fenti útmutatót.")

    elif isinstance(output, torch.Tensor):
        return output[0].cpu().float().numpy()

    else:
        raise RuntimeError(f"Ismeretlen VLA output típus: {type(output)}")


# ── Epizód futtatás ───────────────────────────────────────────────────────────

def run_episode(env: VLAPushEnv, model, processor,
                device: str, exec_horizon: int,
                verbose: bool = False) -> dict:
    """
    Egyetlen epizód: VLA vezérli a kart.
    exec_horizon: hány lépést hajt végre egy query között (1–CHUNK_SIZE).
    """
    obs, frame = env.reset()
    chunk_buf  = np.zeros((0, ACTION_DIM), dtype=np.float32)
    buf_idx    = 0
    last_info  = {}
    t_query    = 0.0

    for _ in range(MAX_STEPS):
        # Re-query ha a chunk kimerült
        if buf_idx >= len(chunk_buf):
            t0 = time.time()
            inputs    = build_vla_input(frame, obs, processor, device)
            chunk_buf = predict_actions(model, inputs, device)  # (CHUNK_SIZE, ACTION_DIM)
            buf_idx   = 0
            t_query   = time.time() - t0
            if verbose:
                print(f"    [query {env._step_count}] {t_query:.2f}s | "
                      f"chunk mean abs: {np.abs(chunk_buf).mean():.3f}")

        action = chunk_buf[buf_idx]
        buf_idx += 1

        obs, frame, done, info = env.step(action)
        last_info = info
        if done:
            break

    return {
        "success":    last_info.get("success", False),
        "place_dist": last_info.get("place_dist", float("inf")),
        "steps":      last_info.get("step", MAX_STEPS),
        "timeout":    last_info.get("timeout", True),
        "t_query_s":  t_query,
    }


# ── Eval fő logika ────────────────────────────────────────────────────────────

def evaluate(args):
    ckpt_dir = Path(args.ckpt) if Path(args.ckpt).is_absolute() \
               else _REPO_ROOT / args.ckpt

    repo_dir = Path(args.repo) if args.repo else \
               Path.home() / "roboshelf-ai-dev" / "unifolm_roboshelf"

    print("=" * 65)
    print("UnifoLM-VLA-0 Eval — F3e Phase 030")
    print("=" * 65)
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Repo:       {repo_dir}")

    # ── Setup ──
    setup_unifolm_repo(repo_dir)
    model, processor, device = load_model(ckpt_dir, repo_dir, args.no_quantize)

    exec_horizon = args.exec_horizon
    if exec_horizon < 1 or exec_horizon > CHUNK_SIZE:
        exec_horizon = CHUNK_SIZE
    print(f"\nexec_horizon={exec_horizon} | n_eval={args.n_eval} | seed={args.seed}")

    # ── Env ──
    env = VLAPushEnv(seed=args.seed)

    # ── Eval loop ──
    print(f"\n{'─'*65}")
    print(f"Eval indítás: {args.n_eval} epizód")
    print(f"{'─'*65}")

    results   = []
    successes = 0
    t0_total  = time.time()

    for ep in range(args.n_eval):
        try:
            res = run_episode(env, model, processor, device,
                              exec_horizon=exec_horizon, verbose=args.verbose)
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

        sr_now = 100.0 * successes / (ep + 1)
        status = "✅" if res["success"] else "❌"
        elapsed = time.time() - t0_total
        print(f"[{ep+1:3d}/{args.n_eval}] {status}  "
              f"steps={res['steps']:3d}  dist={res['place_dist']:.3f}m  "
              f"SR={sr_now:.1f}%  "
              f"({elapsed/60:.1f}min)")

    env.close()

    # ── Összesítés ──
    sr          = 100.0 * successes / args.n_eval
    avg_steps   = float(np.mean([r["steps"]      for r in results]))
    avg_dist    = float(np.mean([r["place_dist"]  for r in results if r["place_dist"] < 1e6]))
    n_timeout   = sum(r["timeout"] for r in results)
    succ_steps  = [r["steps"] for r in results if r["success"]]
    avg_q_time  = float(np.mean([r["t_query_s"] for r in results if r["t_query_s"] > 0]))

    total_min = (time.time() - t0_total) / 60

    print(f"\n{'═'*65}")
    print(f"EREDMÉNY: {successes}/{args.n_eval} siker  →  SR = {sr:.1f}%")
    print(f"  Átlag lépés (összes):    {avg_steps:.1f}")
    print(f"  Átlag lépés (sikeresek): {float(np.mean(succ_steps)):.1f}" if succ_steps else "  Sikeresek: 0")
    print(f"  Átlag place_dist:        {avg_dist:.3f} m")
    print(f"  Timeout:                 {n_timeout}/{args.n_eval}")
    print(f"  Átlag VLA query idő:     {avg_q_time:.2f}s")
    print(f"  Teljes futásidő:         {total_min:.1f} perc")
    print(f"{'═'*65}")

    # ── Verdict ──
    if sr >= 70.0:
        verdict = "✅ ELFOGADVA — F3e UnifoLM-VLA-0 ≥70% SR, Phase 040 jöhet"
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
        description="UnifoLM-VLA-0 eval — F3e Phase 030",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  python3 tools/eval_vla.py --ckpt results/vla_ckpt --n-eval 50
  python3 tools/eval_vla.py --ckpt results/vla_ckpt --n-eval 10 --no-quantize
  python3 tools/eval_vla.py --ckpt results/vla_ckpt --exec-horizon 3 --verbose
        """,
    )
    p.add_argument("--ckpt",         required=True,
                   help="Checkpoint könyvtár (final.pt + lora_final/ benne)")
    p.add_argument("--repo",         default=None,
                   help="unifolm-vla repo helye (alapért.: ~/roboshelf-ai-dev/unifolm_roboshelf)")
    p.add_argument("--n-eval",       type=int, default=50,
                   help="Eval epizódok száma (alapért.: 50)")
    p.add_argument("--exec-horizon", type=int, default=CHUNK_SIZE,
                   help=f"Lépések száma VLA query között (1–{CHUNK_SIZE}, alapért.: {CHUNK_SIZE})")
    p.add_argument("--seed",         type=int, default=123)
    p.add_argument("--no-quantize",  action="store_true",
                   help="8-bit helyett bf16 CPU (Mac M2 kompatibilis, de lassú)")
    p.add_argument("--verbose",      action="store_true")
    args = p.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
