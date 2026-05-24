# %% [markdown]
# # Roboshelf — UnifoLM-VLA-0 Eval (F3e acceptance teszt)
# **Kaggle T4 | MuJoCo push task | 50 epizód | SR ≥ 70% = elfogadva**
#
# **Előfeltételek:**
# - Input 1: `roboshelf-vla-v1` dataset (ugyanaz mint tréningnél)
# - Input 2: előző tréning notebook outputja (roboshelf_vla_ckpt/ — final.pt + lora_final/)
# - GPU: T4 x1 | Internet: ON
#
# **Hogyan add hozzá a checkpoint inputot Kaggle-on:**
# 1. Nyisd meg ezt a notebookot Kaggle-on
# 2. Jobb panel → Input → Add Input → Notebook Outputs
# 3. Keresd meg a tréning notebookot → add hozzá
# 4. A checkpoint elérési útja: /kaggle/input/<notebook-neve>/roboshelf_vla_ckpt/
#
# **Futtatás:** Save and run all (commit mód)

# %% [markdown]
# ## Cella 1 — Függőségek

# %%
import os, subprocess, sys

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# EGL: headless offscreen rendering Kaggle szerveren (nincs X11 display)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def _apt(*args):
    subprocess.check_call(["apt-get", "install", "-y", "-q", *args])

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# EGL/osmesa könyvtárak headless rendereléshez
_apt("libegl1-mesa-dev", "libgl1-mesa-glx", "libgles2-mesa-dev")

_pip("transformers==4.52.3")
_pip("accelerate==1.5.2")
_pip("bitsandbytes")
_pip("peft>=0.10.0")
_pip("einops", "tiktoken", "scipy")
_pip("pillow==11.3.0")
_pip("opencv-python", "pyarrow")
_pip("qwen-vl-utils")
_pip("mujoco")

print("✅ Függőségek telepítve")

# %% [markdown]
# ## Cella 2 — Repo clone + patchek (azonos a tréninggel)

# %%
import re, sys, subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/unifolm-vla")
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

ACTION_DIM = 5
STATE_DIM  = 24
CHUNK_SIZE = 10
LANGUAGE   = "Push the box to the target position on the shelf."

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth=1",
        "https://github.com/unitreerobotics/unifolm-vla.git", str(REPO_DIR)],
        check=True)
    print(f"Repo klónozva: {REPO_DIR}")
else:
    print(f"Repo megvan: {REPO_DIR}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "-e", str(REPO_DIR)], check=True)

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

DIM_PATTERNS = [
    (re.compile(r'\bACTION_DIM\s*=\s*\d+'),        'ACTION_DIM = 5'),
    (re.compile(r'\bPROPRIO_DIM\s*=\s*\d+'),       'PROPRIO_DIM = 24'),
    (re.compile(r'\bNUM_ACTIONS_CHUNK\s*=\s*\d+'), 'NUM_ACTIONS_CHUNK = 10'),
    (re.compile(r'"action_dim"\s*:\s*\d+'),         '"action_dim": 5'),
    (re.compile(r'"proprio_dim"\s*:\s*\d+'),        '"proprio_dim": 24'),
    (re.compile(r'"num_actions_chunk"\s*:\s*\d+'),  '"num_actions_chunk": 10'),
    (re.compile(r"'action_dim'\s*:\s*\d+"),         "'action_dim': 5"),
    (re.compile(r"'proprio_dim'\s*:\s*\d+"),        "'proprio_dim': 24"),
    (re.compile(r"'num_actions_chunk'\s*:\s*\d+"),  "'num_actions_chunk': 10"),
]
DIM_KEYWORDS = {'ACTION_DIM','PROPRIO_DIM','NUM_ACTIONS_CHUNK',
                'action_dim','proprio_dim','num_actions_chunk','G1_EE_6D'}

for py in REPO_DIR.rglob("*.py"):
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
        print(f"  Patchelve: {py.relative_to(REPO_DIR)}")

for qp in REPO_DIR.rglob("QWen2_5.py"):
    src = qp.read_text()
    new = src.replace('attn_implementation="flash_attention_2"', 'attn_implementation="eager"')
    new = new.replace("attn_implementation='flash_attention_2'", "attn_implementation='eager'")
    if new != src:
        qp.write_text(new)
        print("  QWen2_5.py patchelve (eager)")

print("✅ Repo + patchek kész")

# %% [markdown]
# ## Cella 3 — Checkpoint és roboshelf repo keresése

# %%
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")

# ── Checkpoint (final.pt + lora_final/) ──────────────────────────────────────
# A tréning notebook outputjából kerül ide (Input → Notebook Outputs)
_ckpt_candidates = list(KAGGLE_INPUT.rglob("final.pt"))
if not _ckpt_candidates:
    print("❌ final.pt nem található /kaggle/input/ alatt!")
    print("\nElérhető input könyvtárak:")
    for d in sorted(KAGGLE_INPUT.iterdir()):
        print(f"  {d.name}/")
    raise FileNotFoundError(
        "Add hozzá a tréning notebook outputját inputként!\n"
        "Kaggle → Input → Add Input → Notebook Outputs → <tréning notebook>"
    )

CKPT_DIR  = _ckpt_candidates[0].parent
FINAL_PT  = CKPT_DIR / "final.pt"
LORA_DIR  = CKPT_DIR / "lora_final"

print(f"Checkpoint: {CKPT_DIR}")
print(f"  final.pt:   {FINAL_PT.stat().st_size/1e6:.0f} MB ✅")
print(f"  lora_final: {'✅' if LORA_DIR.exists() else '❌ HIÁNYZIK'}")

# ── Roboshelf repo (XML + env infrastruktúra) ─────────────────────────────────
_roboshelf_candidates = list(KAGGLE_INPUT.rglob("scene_manip_sandbox_v2.xml"))
if _roboshelf_candidates:
    ROBOSHELF_ROOT = _roboshelf_candidates[0].parent.parent.parent.parent
    print(f"Roboshelf repo: {ROBOSHELF_ROOT}")
else:
    # Ha nincs input-ként hozzáadva, klónozzuk
    ROBOSHELF_ROOT = Path("/kaggle/working/roboshelf-ai-redesign")
    if not ROBOSHELF_ROOT.exists():
        print("Roboshelf repo klónozása...")
        subprocess.run(["git", "clone", "--depth=1",
            "https://github.com/vorilevi/roboshelf-ai-redesign.git",
            str(ROBOSHELF_ROOT)], check=True)
    print(f"Roboshelf repo: {ROBOSHELF_ROOT}")

XML_PATH  = ROBOSHELF_ROOT / "src/envs/assets/scene_manip_sandbox_v2.xml"
TOOLS_DIR = ROBOSHELF_ROOT / "tools"

if not XML_PATH.exists():
    raise FileNotFoundError(f"scene_manip_sandbox_v2.xml nem található: {XML_PATH}")
print(f"XML: {XML_PATH} ✅")

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

print("✅ Elérési utak OK")

# %% [markdown]
# ## Cella 4 — MuJoCo env + scripted_expert infrastruktúra

# %%
import numpy as np
import mujoco
from PIL import Image

import scripted_expert as _exp
from scripted_expert import (
    _get_obs, _JOINT_RANGES, _GRIPPER_CLOSED, _GRIPPER_OPEN,
    _DEFAULT_ARM_POS, ARM_QPOS_START, ARM_CTRL_START, GRIPPER_CTRL_START,
    N_ARM_DOF, DECIMATION, GOAL_RADIUS, STOCK_QPOS_START, STOCK_RESET_Z,
    MIN_SUCCESS_STEP,
)

IMG_SIZE  = 224
MAX_STEPS = 300

IMG_SIZE  = 224
MAX_STEPS = 300

# Ha az XML-ből hiányoznak a kamerák (régi GitHub verzió), patcheljük
_CAMERA_PATCH = """
    <camera name="front_cam"
            pos="0.35 -1.4 0.9"
            xyaxes="1 0 0  0 0 1"/>
    <camera name="top_cam"
            pos="0.35 0.0 2.2"
            xyaxes="1 0 0  0 1 0"/>
"""

def _load_model_patched(xml_path) -> mujoco.MjModel:
    """XML betöltés — ha nincs front_cam, patcheli a stringbe."""
    xml_str = Path(xml_path).read_text()
    if 'name="front_cam"' not in xml_str:
        print("  ⚠️ front_cam nem volt az XML-ben — patchelés...")
        xml_str = xml_str.replace("</worldbody>", _CAMERA_PATCH + "\n  </worldbody>")
    return mujoco.MjModel.from_xml_string(xml_str)

# Ellenőrzés
_test_model = _load_model_patched(XML_PATH)
_cams = [mujoco.mj_id2name(_test_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
         for i in range(_test_model.ncam)]
print(f"  Elérhető kamerák: {_cams}")
CAMERA_NAME = "front_cam"

class PushTaskEnv:
    """MuJoCo push task env VLA inferenciához — kamera + state obs."""

    def __init__(self, seed: int = 0):
        self._model    = _load_model_patched(XML_PATH)
        self._data     = mujoco.MjData(self._model)
        self._rng      = np.random.default_rng(seed)
        self._renderer = mujoco.Renderer(self._model, height=IMG_SIZE, width=IMG_SIZE)

        self._hand_site_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_shelf")
        self._stock_body_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "stock_1")

        self._hand_body_ids = set()
        for name in _exp._HAND_BODY_NAMES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._hand_body_ids.add(bid)

        self._step_count = 0

    def reset(self):
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARM_DOF] = _DEFAULT_ARM_POS
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF]  = _DEFAULT_ARM_POS
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7]  = _GRIPPER_OPEN

        lo_x, hi_x = _exp.STOCK_RESET_X_RANGE
        lo_y, hi_y = _exp.STOCK_RESET_Y_RANGE
        for _ in range(50):
            sx = float(self._rng.uniform(lo_x, hi_x))
            sy = float(self._rng.uniform(lo_y, hi_y))
            self._data.qpos[STOCK_QPOS_START]     = sx
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
        return self._get_obs(), self._render_pil()

    def step(self, action: np.ndarray):
        lo, hi = _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]
        arm_target = lo + (np.clip(action[:4], -1, 1) + 1.0) * 0.5 * (hi - lo)
        self._data.ctrl[ARM_CTRL_START:ARM_CTRL_START + N_ARM_DOF] = np.clip(arm_target, lo, hi)

        t = (float(np.clip(action[4], -1, 1)) + 1.0) / 2.0
        self._data.ctrl[GRIPPER_CTRL_START:GRIPPER_CTRL_START + 7] = (
            (1.0 - t) * _GRIPPER_OPEN + t * _GRIPPER_CLOSED
        )
        for _ in range(DECIMATION):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs   = self._get_obs()
        frame = self._render_pil()

        stock_pos  = self._data.xpos[self._stock_body_id].copy()
        target_pos = self._data.site_xpos[self._target_site_id].copy()
        dist       = float(np.linalg.norm(stock_pos - target_pos))
        success    = (dist < GOAL_RADIUS) and (self._step_count >= MIN_SUCCESS_STEP)
        timeout    = self._step_count >= MAX_STEPS
        return obs, frame, success or timeout, {"success": success, "dist": dist, "timeout": timeout}

    def _render_pil(self) -> Image.Image:
        self._renderer.update_scene(self._data, camera=CAMERA_NAME)
        return Image.fromarray(self._renderer.render().copy())

    def _get_obs(self) -> np.ndarray:
        return _get_obs(self._model, self._data,
                        self._hand_site_id, self._target_site_id,
                        self._stock_body_id, self._hand_body_ids)

    def close(self):
        self._renderer.close()

env_test = PushTaskEnv(seed=0)
obs, img = env_test.reset()
env_test.close()
print(f"Env OK — obs: {obs.shape}, kép: {img.size} ✅")

# %% [markdown]
# ## Cella 5 — Modell betöltés (azonos a tréninggel)

# %%
import gc, yaml, torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/7] Device: {DEVICE}")
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"      GPU: {p.name} | VRAM: {p.total_memory/1e9:.1f} GB")

gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    print(f"[2/7] VRAM szabad: {free:.1f} GB")

print("[3/7] Qwen2.5-VL-7B betöltés (8-bit)...")
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    HF_MODEL, load_in_8bit=True, device_map="auto",
    attn_implementation="eager", low_cpu_mem_usage=True)
_processor = Qwen2_5_VLProcessor.from_pretrained(HF_MODEL)

if DEVICE == "cuda":
    print(f"      VLM: {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print("[4/7] LoRA adapter betöltés + beolvasztás...")
from peft import PeftModel
_vlm = PeftModel.from_pretrained(_vlm, str(LORA_DIR))
_vlm = _vlm.merge_and_unload()
print("      LoRA merge_and_unload ✅")

print("[5/7] Monkey-patch + config...")
from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface
_c_vlm, _c_proc = _vlm, _processor
def _patched_init(self, config, **kwargs):
    nn.Module.__init__(self)
    self.model     = _c_vlm
    self.processor = _c_proc
_QWen_VL_Interface.__init__ = _patched_init

from unifolm_vla.model.framework.share_tools import dict_to_namespace
config_yaml = REPO_DIR / "src/unifolm_vla/config/training/unifolm_vla_train.yaml"
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

print("[6/7] Unifolm_VLA instantiálás...")
from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA

_overrides = {'ACTION_DIM': ACTION_DIM, 'PROPRIO_DIM': STATE_DIM, 'NUM_ACTIONS_CHUNK': CHUNK_SIZE}
for _mn in list(sys.modules.keys()):
    if 'unifolm' not in _mn: continue
    _m = sys.modules[_mn]
    for _k, _v in _overrides.items():
        if hasattr(_m, _k) and getattr(_m, _k) != _v:
            setattr(_m, _k, _v)

model     = Unifolm_VLA(config=cfg)
processor = model.processor
model.action_model = model.action_model.to(DEVICE, torch.float32)

layer1 = model.action_model.action_encoder.layer1
if layer1.in_features != ACTION_DIM:
    print(f"  ⚠️ Réteg-csere ({layer1.in_features}→{ACTION_DIM})...")
    OLD = layer1.in_features
    for name, lyr in list(model.action_model.named_modules()):
        if not isinstance(lyr, nn.Linear): continue
        nl = name.lower()
        is_state = ('state' in nl or 'proprio' in nl) and 'action' not in nl
        ni = (STATE_DIM if is_state else ACTION_DIM) if lyr.in_features  == OLD else lyr.in_features
        no = (STATE_DIM if is_state else ACTION_DIM) if lyr.out_features == OLD else lyr.out_features
        if ni == lyr.in_features and no == lyr.out_features: continue
        parts = name.split('.')
        parent = model.action_model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], nn.Linear(ni, no, bias=(lyr.bias is not None)).to(DEVICE, torch.float32))
    model.action_model = model.action_model.to(DEVICE)
    layer1 = model.action_model.action_encoder.layer1

assert layer1.in_features == ACTION_DIM
print(f"  layer1: ({ACTION_DIM}→{layer1.out_features}) ✓")

print("[7/7] action_model súlyok betöltése (final.pt)...")
ckpt = torch.load(str(FINAL_PT), map_location=DEVICE)
state_dict = ckpt["action_model"] if "action_model" in ckpt else ckpt
step_saved = ckpt.get("step", "?")
loss_saved = ckpt.get("loss", "?")
model.action_model.load_state_dict(state_dict, strict=False)
print(f"  step={step_saved}, loss={loss_saved:.4f}" if isinstance(loss_saved, float) else f"  step={step_saved}")

model.eval()
if DEVICE == "cuda":
    print(f"\n✅ Modell kész — VRAM: {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("\n✅ Modell kész")

# %% [markdown]
# ## Cella 6 — Inference függvények

# %%
from qwen_vl_utils import process_vision_info

def build_input(image: Image.Image, state: np.ndarray) -> dict:
    """Ugyanaz a collate logika mint tréningnél."""
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  LANGUAGE}
    ]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    img_inp, _ = process_vision_info(msg)
    out = processor(text=[text], images=[img_inp], padding=True, return_tensors="pt")
    out["state"] = torch.FloatTensor(state).unsqueeze(0)
    return {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in out.items()}


@torch.no_grad()
def predict_chunk(image: Image.Image, state: np.ndarray) -> np.ndarray:
    """
    VLA inference: (CHUNK_SIZE, ACTION_DIM) normalizált akció visszaadása.

    Inference módban (nincs 'action' kulcs az inputban) a flow matching
    ODE-t futtatja és predicted action chunk-ot ad vissza.
    """
    inputs = build_input(image, state)
    output = model(qwen_inputs=inputs)

    # Output kulcs keresése
    if isinstance(output, dict):
        for key in ("action_pred", "actions", "pred_actions", "action", "output"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key][0].cpu().float().numpy()   # (CHUNK_SIZE, ACTION_DIM)
        # Ha nem ismerjük a kulcsot — debug info
        print(f"\n⚠️ Ismeretlen output kulcsok: {list(output.keys())}")
        print("   Tartalmak:")
        for k, v in output.items():
            print(f"     {k}: {type(v)} {getattr(v, 'shape', '')}")
        raise RuntimeError("Ismeretlen VLA output — lásd fent")
    elif isinstance(output, torch.Tensor):
        return output[0].cpu().float().numpy()
    else:
        raise RuntimeError(f"Ismeretlen output típus: {type(output)}")


# Teszt: egy üres inference
print("Inference teszt...")
_env_test = PushTaskEnv(seed=99)
_obs_test, _img_test = _env_test.reset()
_env_test.close()

import time
t0 = time.time()
_chunk = predict_chunk(_img_test, _obs_test)
t_infer = time.time() - t0

print(f"✅ Inference OK — chunk shape: {_chunk.shape} | idő: {t_infer:.2f}s")
print(f"   Action range: [{_chunk.min():.3f}, {_chunk.max():.3f}]")

# Becsült futásidő
queries_per_ep = MAX_STEPS // CHUNK_SIZE
est_min = queries_per_ep * t_infer * 50 / 60
print(f"\n⏱  Becsült eval idő (50 ep): {est_min:.0f} perc")

# %% [markdown]
# ## Cella 7 — Eval loop (50 epizód)

# %%
import time
from collections import deque

N_EVAL       = 50
EXEC_HORIZON = CHUNK_SIZE   # teljes chunk végrehajtása re-query előtt
SEED         = 123

env      = PushTaskEnv(seed=SEED)
results  = []
successes = 0
t0_total = time.time()

print(f"Eval: {N_EVAL} epizód | exec_horizon={EXEC_HORIZON} | seed={SEED}")
print("─" * 65)

for ep in range(N_EVAL):
    obs, img    = env.reset()
    chunk_buf   = np.zeros((0, ACTION_DIM), dtype=np.float32)
    buf_idx     = 0
    last_info   = {}
    t_ep        = time.time()

    for _ in range(MAX_STEPS):
        # Re-query ha a chunk kimerült
        if buf_idx >= len(chunk_buf):
            chunk_buf = predict_chunk(img, obs)   # (CHUNK_SIZE, ACTION_DIM)
            buf_idx   = 0

        obs, img, done, info = env.step(chunk_buf[buf_idx])
        buf_idx  += 1
        last_info = info
        if done:
            break

    success = last_info.get("success", False)
    dist    = last_info.get("dist", float("inf"))
    timeout = last_info.get("timeout", True)

    if success:
        successes += 1

    results.append({"success": success, "dist": dist, "timeout": timeout})
    sr_now  = 100.0 * successes / (ep + 1)
    elapsed = time.time() - t0_total
    status  = "✅" if success else "❌"
    print(f"[{ep+1:3d}/{N_EVAL}] {status}  dist={dist:.3f}m  SR={sr_now:.1f}%  "
          f"({elapsed/60:.1f}min)")

env.close()

# %% [markdown]
# ## Cella 8 — Eredmények + verdict

# %%
import json
from pathlib import Path

sr        = 100.0 * successes / N_EVAL
avg_dist  = float(np.mean([r["dist"] for r in results if r["dist"] < 1e6]))
n_timeout = sum(r["timeout"] for r in results)
total_min = (time.time() - t0_total) / 60

print(f"\n{'═'*65}")
print(f"EREDMÉNY — F3e UnifoLM-VLA-0 Acceptance Teszt")
print(f"{'═'*65}")
print(f"  Sikeresek:        {successes}/{N_EVAL}")
print(f"  Success Rate:     {sr:.1f}%")
print(f"  Átlag place_dist: {avg_dist:.3f} m")
print(f"  Timeout:          {n_timeout}/{N_EVAL}")
print(f"  Futásidő:         {total_min:.1f} perc")
print(f"{'═'*65}")

if sr >= 70.0:
    verdict = "✅ ELFOGADVA — F3e ≥70% SR, Phase 040 mehet"
elif sr >= 50.0:
    verdict = "⚠️  RÉSZLEGES — több demo vagy hosszabb tréning ajánlott"
elif sr >= 20.0:
    verdict = "❌ GYENGE — az acceptance feltétel nem teljesült"
else:
    verdict = "❌ SIKERTELEN — modell nem tanult vagy inference hiba"

print(f"\n{verdict}")

# Mentés
out = {
    "sr": sr, "successes": successes, "n_eval": N_EVAL,
    "avg_dist": avg_dist, "n_timeout": n_timeout,
    "total_min": total_min, "verdict": verdict,
    "results": results,
}
result_path = Path("/kaggle/working/vla_eval_results.json")
result_path.write_text(json.dumps(out, indent=2))
print(f"\nEredmény mentve: {result_path}")
print("Töltsd le a Kaggle Output panelből!")
