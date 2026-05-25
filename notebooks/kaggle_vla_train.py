# %% [markdown]
# # Roboshelf — UnifoLM-VLA-0 Fine-tune v2
# **Kaggle T4 | LeRobot parquet+MP4 | flow-matching action head**
#
# **Előfeltételek:**
# - Dataset: `roboshelf-vla-v2` hozzáadva input-ként (1000 demo, ~8.7% SR)
# - GPU: T4 x1 | Internet: ON
# - Becsült futásidő: ~4 óra (10 000 lépés)
#
# **Futtatás:** Runtime → Run All

# %% [markdown]
# ## Cella 1 — Függőségek

# %%
import os, subprocess, sys

# CUDA memory allocator: expandable segments csökkenti a fragmentációt
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("transformers==4.52.3")
_pip("accelerate==1.5.2")
_pip("bitsandbytes")
_pip("peft>=0.10.0")
_pip("einops", "tiktoken", "scipy", "tensorboard")
_pip("pillow==11.3.0", "albumentations==1.4.18")
_pip("opencv-python", "pyarrow")
_pip("qwen-vl-utils")

print("✅ Függőségek telepítve")

# %% [markdown]
# ## Cella 2 — Repo clone + fájl patchek
#
# ⚠️ **Kritikus:** Ez fut le ELŐBB mint bármilyen `unifolm_vla` import.
# A konstans-patchek csak importálás előtt érvényesek.

# %%
import re, sys, subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/unifolm-vla")
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Konstansok (a mi datasetünkhöz)
ACTION_DIM = 5
STATE_DIM  = 24
CHUNK_SIZE = 10
LANGUAGE   = "Push the box to the target position on the shelf."

# ── Clone ──────────────────────────────────────────────────────────────
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth=1",
        "https://github.com/unitreerobotics/unifolm-vla.git", str(REPO_DIR)],
        check=True)
    print(f"Repo klónozva: {REPO_DIR}")
else:
    print(f"Repo már megvan: {REPO_DIR}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "-e", str(REPO_DIR)], check=True)

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

# ── PATCH 1: dimenzió konstansok — MINDEN .py fájlban ─────────────────
# Az unifolm_vla legalább két helyen hardcode-olja ACTION_DIM=23-t:
#   rlds_dataloader/constants.py  (assignment stílus: ACTION_DIM = 23)
#   training/vla/constants.py     (lehet dict stílus: "action_dim": 23)
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

patched = []
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
        patched.append(py.relative_to(REPO_DIR))

if patched:
    for f in patched: print(f"  Patchelve: {f}")
else:
    print("  ⚠️ Egyetlen fájlban sem volt patchelendő konstans")

# ── PATCH 2: QWen2_5.py — eager attention (T4 nem támogatja flash_attn2) ──
for qp in REPO_DIR.rglob("QWen2_5.py"):
    src = qp.read_text()
    new = src.replace('attn_implementation="flash_attention_2"', 'attn_implementation="eager"')
    new = new.replace("attn_implementation='flash_attention_2'", "attn_implementation='eager'")
    if new != src:
        qp.write_text(new)
        print("  QWen2_5.py patchelve (eager)")

print("✅ Repo + patchek kész")

# %% [markdown]
# ## Cella 3 — Dataset ellenőrzés

# %%
import json
from pathlib import Path

# Auto-discovery: megkeresi az info.json-t bárhol /kaggle/input/ alatt
KAGGLE_INPUT = Path("/kaggle/input")
_info_candidates = list(KAGGLE_INPUT.rglob("meta/info.json"))

if not _info_candidates:
    print("❌ Nem található info.json /kaggle/input/ alatt!")
    print("\nElérhető input könyvtárak:")
    for d in sorted(KAGGLE_INPUT.iterdir()):
        print(f"  {d.name}/")
    raise FileNotFoundError("Dataset nem található. Add hozzá a 'roboshelf-vla-v1' datasetet input-ként.")

DS_ROOT = _info_candidates[0].parent.parent
print(f"Dataset megtalálva: {DS_ROOT}")

meta        = json.loads((_info_candidates[0]).read_text())
CAMERA_NAME = meta.get("cameras", ["front_cam"])[0]
DATA_DIR    = DS_ROOT / "data"  / "chunk-000"
VIDEO_DIR   = DS_ROOT / "videos" / "chunk-000" / f"observation.images.{CAMERA_NAME}"
N_EPISODES  = meta["total_episodes"]

mp4s     = list(DS_ROOT.rglob("*.mp4"))
parquets = list(DS_ROOT.rglob("*.parquet"))
print(f"Parquet: {len(parquets)} | MP4: {len(mp4s)}")
print(f"Epizódok: {N_EPISODES} | action_dim={meta['action_dim']} | obs_dim={meta['obs_dim']}")
print(f"Kamera: {CAMERA_NAME} | FPS: {meta['fps']}")
print(f"Task: {meta['task']}")
print("✅ Dataset OK")

# %% [markdown]
# ## Cella 4 — Dataset + DataLoader

# %%
import cv2, numpy as np, torch
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 224

class RoboshelfDataset(Dataset):
    def __init__(self, data_dir, video_dir, n_episodes, chunk_size=CHUNK_SIZE):
        self.data_dir   = data_dir
        self.video_dir  = video_dir
        self.chunk_size = chunk_size
        self.samples    = []
        for ep in range(n_episodes):
            p = data_dir / f"episode_{ep:06d}.parquet"
            if not p.exists(): continue
            n = len(pq.read_table(p, columns=["frame_index"]))
            for s in range(n):
                self.samples.append((ep, s, n))
        print(f"  {len(self.samples)} sample ({n_episodes} ep)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ep, step, n_steps = self.samples[idx]
        # Kép
        cap = cv2.VideoCapture(str(self.video_dir / f"episode_{ep:06d}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ok, frame = cap.read(); cap.release()
        if not ok:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        image = Image.fromarray(frame)
        # State + action chunk
        tbl   = pq.read_table(self.data_dir / f"episode_{ep:06d}.parquet")
        state = np.array([tbl[f"obs_{i}"][step].as_py() for i in range(STATE_DIM)], dtype=np.float32)
        acts  = [np.array([tbl[f"action_{j}"][min(step+k, n_steps-1)].as_py()
                           for j in range(ACTION_DIM)], dtype=np.float32)
                 for k in range(self.chunk_size)]
        return {"image": image,
                "state": torch.FloatTensor(state),
                "action_chunk": torch.FloatTensor(np.stack(acts)),
                "language": LANGUAGE}

def make_collate(processor):
    def collate(batch):
        from qwen_vl_utils import process_vision_info
        images    = [b["image"]    for b in batch]
        states    = torch.stack([b["state"]        for b in batch])
        actions   = torch.stack([b["action_chunk"] for b in batch])
        languages = [b["language"] for b in batch]
        msgs = [[{"role":"user","content":[
                    {"type":"image","image":img},
                    {"type":"text","text":lang}]}]
                for img, lang in zip(images, languages)]
        texts   = [processor.apply_chat_template(m, tokenize=False,
                   add_generation_prompt=True) for m in msgs]
        img_inp = []
        for m in msgs:
            ii, _ = process_vision_info(m)
            img_inp.append(ii[0] if ii else None)
        out = processor(text=texts, images=img_inp, padding=True, return_tensors="pt")
        out["state"]  = states
        out["action"] = actions
        return out
    return collate

dataset = RoboshelfDataset(DATA_DIR, VIDEO_DIR, N_EPISODES)
print("✅ Dataset kész")

# %% [markdown]
# ## Cella 5 — Modell betöltés
#
# 4-bit VLM inject. Újrafuttatható: `_vlm` globálisan cache-elve van.

# %%
import sys, gc, yaml, torch
import torch.nn as nn
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/8] Device: {DEVICE}")
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    total_vram = p.total_memory / 1e9
    print(f"      GPU: {p.name} | VRAM: {total_vram:.1f} GB")

# ── 1. Memóriatisztítás ────────────────────────────────────────────────
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"[2/8] VRAM szabad betöltés előtt: {free:.1f} GB")

# ── 2. VLM betöltés (cache — újrafuttatásnál nem tölt le újra) ────────
print("[3/8] VLM betöltés (4-bit NF4, ~10-20 perc első futáskor)...")
if "_vlm" not in globals():
    _vlm, _processor = None, None

if _vlm is None:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    _vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL,
        load_in_8bit=True,
        device_map="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    print("      VLM weights betöltve, processor...")
    _processor = Qwen2_5_VLProcessor.from_pretrained(HF_MODEL)
    if DEVICE == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        print(f"      VLM OK — VRAM használat: {used:.1f}/{total_vram:.1f} GB")
else:
    print("      VLM cache-ből újrafelhasználva")

# ── 3. _QWen_VL_Interface monkey-patch ────────────────────────────────
print("[4/8] _QWen_VL_Interface monkey-patch...")
from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface
_c_vlm, _c_proc = _vlm, _processor
def _patched_init(self, config, **kwargs):
    nn.Module.__init__(self)
    self.model     = _c_vlm
    self.processor = _c_proc
_QWen_VL_Interface.__init__ = _patched_init
print("      patch OK")

# ── 4. Config ─────────────────────────────────────────────────────────
print("[5/8] Config betöltés...")
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
print(f"      action_dim={ACTION_DIM}, state_dim={STATE_DIM}, chunk={CHUNK_SIZE}")

# ── 5. Unifolm_VLA import ─────────────────────────────────────────────
print("[6/8] Unifolm_VLA import...")
from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
print("      import OK")

# ── 6. Runtime modul-patch — IMPORT UTÁN, instantiálás ELŐTT ──────────
print("[7/8] Runtime modul-patch...")
_overrides = {'ACTION_DIM': ACTION_DIM, 'PROPRIO_DIM': STATE_DIM,
              'NUM_ACTIONS_CHUNK': CHUNK_SIZE}
for _mn in list(sys.modules.keys()):
    if 'unifolm' not in _mn: continue
    _m = sys.modules[_mn]
    for _k, _v in _overrides.items():
        if hasattr(_m, _k) and getattr(_m, _k) != _v:
            print(f"      {_mn}.{_k}: {getattr(_m,_k)} → {_v}")
            setattr(_m, _k, _v)

# ── 7. Modell instantiálás ─────────────────────────────────────────────
print("[8/8] Unifolm_VLA instantiálás...")
model     = Unifolm_VLA(config=cfg)
processor = model.processor
model.action_model = model.action_model.to(DEVICE)

# ── 7. Dimenzió ellenőrzés + fallback réteg-csere ─────────────────────
layer1 = model.action_model.action_encoder.layer1
if layer1.in_features != ACTION_DIM:
    print(f"\n⚠️ Dimenzió hiba ({layer1.in_features}→{ACTION_DIM}), réteg-csere...")
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
        new_lyr = nn.Linear(ni, no, bias=(lyr.bias is not None)).to(DEVICE, torch.float32)
        setattr(parent, parts[-1], new_lyr)
        print(f"  {name}: ({lyr.in_features},{lyr.out_features}) → ({ni},{no})")
    model.action_model = model.action_model.to(DEVICE)
    layer1 = model.action_model.action_encoder.layer1

assert layer1.in_features == ACTION_DIM, \
    f"❌ Sikertelen javítás: {layer1.in_features} != {ACTION_DIM}"

if DEVICE == "cuda":
    used = torch.cuda.memory_allocated()/1e9
    total = torch.cuda.get_device_properties(0).total_memory/1e9
    print(f"\n✅ Modell kész — VRAM: {used:.1f}/{total:.1f} GB")
print(f"  layer1: ({ACTION_DIM} → {layer1.out_features}) ✓")

# %% [markdown]
# ## Cella 6 — LoRA + Optimizer + DataLoader

# %%
from peft import get_peft_model, LoraConfig

lora_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
model.qwen_vl_interface.model = get_peft_model(
    model.qwen_vl_interface.model, lora_cfg)
model.qwen_vl_interface.model.print_trainable_parameters()

collate_fn = make_collate(processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
    num_workers=0, collate_fn=collate_fn, pin_memory=(DEVICE=="cuda"))

# AdamW8bit: optimizer states 8-biten tárolva → 4.7 GB helyett ~1.2 GB (T4 fér el)
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit([
    {"params": model.action_model.parameters(),            "lr": 1e-4},
    {"params": model.qwen_vl_interface.model.parameters(), "lr": 1e-5},
], weight_decay=0.01)

MAX_STEPS = 10000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_STEPS, eta_min=1e-6)

# Memóriatisztítás training előtt
import gc; gc.collect(); torch.cuda.empty_cache()
if DEVICE == "cuda":
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"VRAM szabad training előtt: {free:.1f} GB")

print(f"\n✅ LoRA + optimizer kész | {len(dataloader)} batch/epoch")

# %% [markdown]
# ## Cella 7 — Training (~4h)

# %%
import time
from collections import deque
from pathlib import Path

CKPT_DIR  = Path("/kaggle/working/roboshelf_vla_ckpt")
CKPT_DIR.mkdir(exist_ok=True)
LOG_EVERY  = 100
SAVE_EVERY = 2000  # köztes checkpoint minden 2000 lépésnél (session crash ellen)

model.train()
loss_buf  = deque(maxlen=50)
step      = 0
t0        = time.time()
data_iter = iter(dataloader)

print(f"Training — {MAX_STEPS} lépés | device={DEVICE}")
print("─" * 55)

while step < MAX_STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    qwen_inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
    optimizer.zero_grad()
    loss = model(qwen_inputs=qwen_inputs)["action_loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_buf.append(loss.item())
    step += 1

    if step % LOG_EVERY == 0:
        print(f"[{step:4d}/{MAX_STEPS}] loss={sum(loss_buf)/len(loss_buf):.4f}"
              f" | lr={optimizer.param_groups[0]['lr']:.1e}"
              f" | {(time.time()-t0)/60:.1f}min")

    if step % SAVE_EVERY == 0:
        p = CKPT_DIR / f"step_{step:05d}.pt"
        torch.save({
            "step": step,
            "loss": sum(loss_buf)/len(loss_buf),
            "action_model": model.action_model.state_dict(),
        }, p)
        lora_dir = CKPT_DIR / f"lora_step_{step:05d}"
        model.qwen_vl_interface.model.save_pretrained(str(lora_dir))
        print(f"  💾 {p.name} + {lora_dir.name}/")

elapsed = time.time() - t0
print(f"\n✅ Training kész — {elapsed/60:.1f} perc | "
      f"final loss={sum(loss_buf)/len(loss_buf):.4f}")

# %% [markdown]
# ## Cella 8 — Checkpoint mentés + letöltés

# %%
import shutil

# Action model mentés
torch.save({
    "step": step,
    "loss": sum(loss_buf)/len(loss_buf),
    "action_model": model.action_model.state_dict(),
    "config": {"action_dim": ACTION_DIM, "state_dim": STATE_DIM,
               "chunk_size": CHUNK_SIZE, "camera": CAMERA_NAME},
}, CKPT_DIR / "final.pt")

# LoRA adapter mentés
lora_final_dir = CKPT_DIR / "lora_final"
model.qwen_vl_interface.model.save_pretrained(str(lora_final_dir))
print(f"LoRA adapter mentve: {lora_final_dir}/")

print("✅ Checkpointok elérhetők a Kaggle Output panelben:")
print(f"   {CKPT_DIR}/final.pt")
print(f"   {CKPT_DIR}/lora_final/")
