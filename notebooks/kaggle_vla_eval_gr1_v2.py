# ============================================================
# ⚠️  V2 ÁG — MEREV-TEST JAVÍTÁS (2026-08-02)
#     Külön ág. A v1 notebookok VÁLTOZATLANOK, azokon fut a
#     korábbi betanítás (20% eval SR baseline).
#     v2 scene: scene_manip_sandbox_gr1_v2.xml
#     v2 env:   gr1_shelf_stock_env_v2.py
#     v2 data:  vorilevi/roboshelf-gr1-push-v2 (1000 ep, 100% scripted SR)
#     Megj.: a v2 geometriája MÁS (asztal x=0.36 vs 0.45), ezért a
#     v1 20%-ával NEM azonos feladaton mért eredmény.
# ============================================================
# %% [markdown]
# # Roboshelf — GR1T1 VLA Eval
# **Notebook neve: roboshelf-gr1-eval-v1**
# **Input: unifolm_vla_gr1_roboshelf_final.zip (fine-tune output)**
# **GPU: T4 x1 | Internet: ON**

# %% [markdown]
# ## Cella 1 — Függőségek

# %%
import os, subprocess, sys
from pathlib import Path

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MUJOCO_GL'] = 'egl'

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("transformers==4.52.3")
_pip("accelerate==1.5.2")
_pip("bitsandbytes")
_pip("peft>=0.10.0")
_pip("einops", "tiktoken", "scipy")
_pip("pillow==11.3.0")
_pip("qwen-vl-utils")
_pip("mujoco", "gymnasium[mujoco]")

print("✅ Függőségek telepítve")

# %% [markdown]
# ## Cella 2 — Repók + GR1T1 mesh fájlok

# %%
import re, sys, subprocess
from pathlib import Path

WORK          = Path("/kaggle/working")
UNIFOLM_DIR   = WORK / "unifolm-vla"
ROBOSHELF_DIR = WORK / "roboshelf-ai-redesign"

ACTION_DIM = 4
STATE_DIM  = 24
CHUNK_SIZE = 10
LANGUAGE   = "Push the stock to the target position on the shelf."
HF_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- unifolm-vla ---
if not UNIFOLM_DIR.exists():
    print("unifolm-vla klónozása...")
    subprocess.run(["git", "clone", "--depth=1",
        "https://github.com/unitreerobotics/unifolm-vla.git",
        str(UNIFOLM_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "-e", str(UNIFOLM_DIR)], check=True)
if str(UNIFOLM_DIR / "src") not in sys.path:
    sys.path.insert(0, str(UNIFOLM_DIR / "src"))
print("  unifolm-vla ✅")

# --- roboshelf-ai-redesign ---
if not ROBOSHELF_DIR.exists():
    print("roboshelf klónozása...")
    subprocess.run(["git", "clone", "--depth=1",
        "https://github.com/vorilevi/roboshelf-ai-redesign.git",
        str(ROBOSHELF_DIR)], check=True)
if str(ROBOSHELF_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROBOSHELF_DIR / "src"))
print("  roboshelf ✅")

# --- GR1T1 mesh fájlok (FFTAI/Wiki-GRx-Models, GPL-3.0) ---
# T1-gyel ellentétben: NEM mujoco_menagerie sparse-checkout, hanem
# saját setup_gr1_assets.sh script (a menagerie-ben nincs fourier_gr1t1)
print("GR1T1 mesh letöltés...")
subprocess.run(
    ["bash", "setup_gr1_assets.sh"],
    cwd=str(ROBOSHELF_DIR),
    check=True,
)

_stl_dir = ROBOSHELF_DIR / "mujoco_menagerie/fourier_gr1t1/assets"
stl_count = len(list(_stl_dir.glob("*.STL")))
print(f"  STL fájlok: {stl_count}")
assert stl_count >= 35, f"❌ STL fájlok hiányoznak — setup_gr1_assets.sh sikertelen! ({stl_count})"
print("  GR1T1 meshes ✅")

# --- ACTION_DIM patch ---
DIM_PATTERNS = [
    (re.compile(r'\bACTION_DIM\s*=\s*\d+'),        f'ACTION_DIM = {ACTION_DIM}'),
    (re.compile(r'\bPROPRIO_DIM\s*=\s*\d+'),       f'PROPRIO_DIM = {STATE_DIM}'),
    (re.compile(r'\bNUM_ACTIONS_CHUNK\s*=\s*\d+'), f'NUM_ACTIONS_CHUNK = {CHUNK_SIZE}'),
]
DIM_KEYWORDS = {'ACTION_DIM', 'PROPRIO_DIM', 'NUM_ACTIONS_CHUNK', 'G1_EE_6D'}
for py in UNIFOLM_DIR.rglob("*.py"):
    try:
        src = py.read_text(errors='ignore')
    except Exception:
        continue
    if not any(kw in src for kw in DIM_KEYWORDS):
        continue
    new = src
    for pat, repl in DIM_PATTERNS:
        new = pat.sub(repl, new)
    if new != src:
        py.write_text(new)

# --- flash_attention_2 → eager patch ---
for qp in UNIFOLM_DIR.rglob("QWen2_5.py"):
    src = qp.read_text()
    new = src.replace('attn_implementation="flash_attention_2"', 'attn_implementation="eager"')
    new = new.replace("attn_implementation='flash_attention_2'", "attn_implementation='eager'")
    if new != src:
        qp.write_text(new)
        print("  QWen2_5.py patchelve (eager)")

print("✅ Repók + patchek kész")

# %% [markdown]
# ## Cella 3 — Checkpoint + XML ellenőrzés

# %%
KAGGLE_INPUT = Path("/kaggle/input")

_ckpt_candidates = list(KAGGLE_INPUT.rglob("step_10000.pt"))

if not _ckpt_candidates:
    print("❌ step_10000.pt nem található!")
    print("Elérhető inputok:")
    for d in sorted(KAGGLE_INPUT.iterdir()):
        print(f"  {d}/")
        for f in sorted(d.rglob("*"))[:10]:
            print(f"    {f}")
    raise FileNotFoundError(
        "Add hozzá a fine-tune notebook outputját inputként!\n"
        "  → unifolm_vla_gr1_roboshelf_final.zip tartalmát"
    )

CKPT_PATH = _ckpt_candidates[0]
print(f"Checkpoint: {CKPT_PATH}")
print(f"  Méret: {CKPT_PATH.stat().st_size / 1e9:.2f} GB ✅")

XML_PATH = ROBOSHELF_DIR / "src/envs/assets/scene_manip_sandbox_gr1_v2.xml"
assert XML_PATH.exists(), f"❌ Scene XML nem található: {XML_PATH}"
print(f"XML: {XML_PATH} ✅")

print("✅ Elérési utak OK")

# %% [markdown]
# ## Cella 4 — MuJoCo env teszt

# %%
import numpy as np
import mujoco
from PIL import Image

from roboshelf_ai.mujoco.envs.manipulation.gr1_shelf_stock_env_v2 import (
    GR1ShelfStockEnv, N_ARM_DOF,
)

IMG_SIZE  = 224
MAX_STEPS = 300

_env_test = GR1ShelfStockEnv()
_obs, _info = _env_test.reset(seed=0)
print(f"GR1T1 Env OK — obs shape: {_obs.shape} ✅")
del _env_test

print("✅ MuJoCo env kész")

# %% [markdown]
# ## Cella 5 — Modell betöltés

# %%
import gc, yaml, torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import get_peft_model, LoraConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.1f} GB")

gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# [1] VLM base betöltés
print("\n[1/5] Qwen2.5-VL-7B betöltés (8-bit)...")
_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    HF_MODEL, load_in_8bit=True, device_map="auto",
    attn_implementation="eager", low_cpu_mem_usage=True)
_processor = Qwen2_5_VLProcessor.from_pretrained(HF_MODEL)
if DEVICE == "cuda":
    print(f"  VLM OK — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
else:
    print("  VLM OK")

# [2] LoRA (tréninggel azonos: r=32)
print("\n[2/5] LoRA config (r=32)...")
_lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM",
)
_vlm = get_peft_model(_vlm, _lora_cfg)

# [3] Monkey-patch
print("\n[3/5] Monkey-patch...")
from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface
_c_vlm, _c_proc = _vlm, _processor
def _patched_init(self, config, **kwargs):
    nn.Module.__init__(self)
    self.model     = _c_vlm
    self.processor = _c_proc
_QWen_VL_Interface.__init__ = _patched_init

# [4] Unifolm_VLA
print("\n[4/5] Unifolm_VLA instantiálás...")
from unifolm_vla.model.framework.share_tools import dict_to_namespace
from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
import yaml as _yaml

_config_yaml = UNIFOLM_DIR / "src/unifolm_vla/config/training/unifolm_vla_train.yaml"
with open(_config_yaml) as f:
    _cfg = dict_to_namespace(_yaml.safe_load(f))
_cfg.framework.qwenvl.base_vlm                        = HF_MODEL
_cfg.framework.qwenvl.attn_implementation             = "eager"
_cfg.framework.action_model.action_dim                = ACTION_DIM
_cfg.framework.action_model.state_dim                 = STATE_DIM
_cfg.framework.action_model.action_horizon            = CHUNK_SIZE
_cfg.framework.action_model.future_action_window_size = CHUNK_SIZE - 1
if not hasattr(_cfg, "trainer") or _cfg.trainer is None:
    _cfg.trainer = dict_to_namespace({"repeated_diffusion_steps": 4})

_overrides = {'ACTION_DIM': ACTION_DIM, 'PROPRIO_DIM': STATE_DIM, 'NUM_ACTIONS_CHUNK': CHUNK_SIZE}
for _mn in list(sys.modules.keys()):
    if 'unifolm' not in _mn:
        continue
    _m = sys.modules[_mn]
    for _k, _v in _overrides.items():
        if hasattr(_m, _k) and getattr(_m, _k) != _v:
            setattr(_m, _k, _v)

model = Unifolm_VLA(config=_cfg)
model.action_model = model.action_model.to(DEVICE, torch.float32)

# [5] Checkpoint betöltés
print(f"\n[5/5] Checkpoint betöltés: {CKPT_PATH.name}")
_ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
if "trainable_state" in _ckpt:
    _state = _ckpt["trainable_state"]
    print(f"  step={_ckpt.get('step','?')} | loss={_ckpt.get('loss', float('nan')):.4f}")
    _meta = _ckpt.get("config", {})
    if _meta:
        print(f"  robot={_meta.get('robot','?')} | action_dim={_meta.get('action_dim','?')}")
else:
    _state = _ckpt.get("model_state", _ckpt)
    print("  ⚠️ Régi formátum (model_state)")

_missing, _unexpected = model.load_state_dict(_state, strict=False)
print(f"  Betöltött: {len(_state) - len(_unexpected)}/{len(_state)} kulcs ✅")

model.eval()
print("\n✅ Modell betöltve")

# %% [markdown]
# ## Cella 6 — Eval loop (50 epizód)

# %%
import time
from qwen_vl_utils import process_vision_info

@torch.no_grad()
def _predict(obs: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  LANGUAGE}
    ]}]
    text       = _processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    img_inp, _ = process_vision_info(msg)
    inp = _processor(text=[text], images=[img_inp], padding=True, return_tensors="pt")
    inp["state"] = torch.FloatTensor(obs).unsqueeze(0)
    inp = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}

    out = model.predict_action(qwen_inputs=inp)
    return out["normalized_actions"][0]  # (CHUNK_SIZE, ACTION_DIM) numpy array


N_EVAL       = 50
EXEC_HORIZON = CHUNK_SIZE
SEED         = 42

env = GR1ShelfStockEnv()
results   = []
successes = 0
t0_total  = time.time()

print(f"Eval: {N_EVAL} epizód | max_steps={MAX_STEPS} | exec_horizon={EXEC_HORIZON}")
print("─" * 65)

for ep in range(N_EVAL):
    obs, _ = env.reset(seed=SEED + ep)
    chunk_buf = np.zeros((0, ACTION_DIM), dtype=np.float32)
    buf_idx   = 0
    step_count = 0
    last_info  = {}

    for _ in range(MAX_STEPS):
        if buf_idx >= len(chunk_buf):
            chunk_buf = _predict(obs)
            buf_idx   = 0

        obs, _rew, terminated, truncated, info = env.step(chunk_buf[buf_idx])
        buf_idx += 1
        last_info   = info
        step_count += 1
        if terminated or truncated:
            break

    success = last_info.get("placed", False)
    dist    = last_info.get("stock_target_dist", float("inf"))
    results.append({"success": success, "dist": dist, "steps": step_count})
    if success:
        successes += 1

    sr_now  = 100.0 * successes / (ep + 1)
    status  = "✅" if success else "❌"
    elapsed = time.time() - t0_total
    print(f"[{ep+1:3d}/{N_EVAL}] {status}  steps={step_count:3d}  "
          f"dist={dist:.3f}m  SR={sr_now:.1f}%  ({elapsed/60:.1f}min)")

env.close()

sr        = 100.0 * successes / N_EVAL
avg_steps = float(np.mean([r["steps"] for r in results]))
avg_dist  = float(np.mean([r["dist"]  for r in results if r["dist"] < 1e6]))
total_min = (time.time() - t0_total) / 60

print(f"\n{'═'*65}")
print(f"EREDMÉNY — GR1T1 UnifoLM-VLA-0  (Track 3 — Fourier GR1T1)")
print(f"  SR:        {successes}/{N_EVAL} = {sr:.1f}%")
print(f"  avg steps: {avg_steps:.1f}")
print(f"  avg dist:  {avg_dist:.3f}m")
print(f"  Idő:       {total_min:.1f} perc")
print(f"  Elfogadva: {'✅ IGEN (≥70%)' if sr >= 70 else '❌ NEM (<70%)'}")
print(f"{'═'*65}")
print()
print(f"Track 3 státusz (vendor-independence ≥70%):")
print(f"  G1   (Unitree):  80%  ✅")
print(f"  T1   (Booster):  86%  ✅")
print(f"  GR1T1 (Fourier): {sr:.0f}%  {'✅' if sr >= 70 else '❌'}")
