"""
kaggle_unifolm_vla_finetune_t1.py — Importálható modul
=======================================================
Roboshelf T1 | UnifoLM-VLA-0 fine-tune | Kaggle T4

G1 verziótól való eltérések:
  - ACTION_DIM = 4  (T1 4-DOF jobb kar, vs G1 5-DOF)
  - Nincs kamera — state-only, dummy fekete kép a VLM inputhoz
  - Dataset: vorilevi/roboshelf-t1-push-v1 (HF, 982 ep)
  - download_dataset() új lépés (HF-ről tölti le)
  - LoRA r=32 (G1 F3e: 80% SR @ 10k steps, r=32)
  - Checkpoint: unifolm_vla_t1_roboshelf

Feltöltési mód:
  1. Kaggle Dataset-ként töltsd fel ezt a fájlt
     (pl. "roboshelf-vla-scripts-t1" névvel, leventevrss account)
  2. Az új notebook-ban:
       import sys
       sys.path.insert(0, "/kaggle/input/datasets/leventevrss/roboshelf-vla-scripts-t1")
       import kaggle_unifolm_vla_finetune_t1 as vla

  FONTOS — Kaggle path formátum (AI-6 known issue):
    scripts: /kaggle/input/datasets/leventevrss/roboshelf-vla-scripts-t1
    dataset: /kaggle/input/datasets/leventevrss/roboshelf-t1-push-v1/t1_push_v1

API (sorrendben):
  vla.install_deps()                              # először! (AI-7: import csak utána)
  import torch
  vla.clone_and_patch_repo()
  DS_ROOT = Path("/kaggle/input/datasets/leventevrss/roboshelf-t1-push-v1/t1_push_v1")
  info = vla.check_dataset(ds_root=DS_ROOT)       # dataset Kaggle inputból jön, nem HF-ről
  dataset = vla.build_dataset(info)
  model, processor = vla.load_model(info)
  model, dataloader, optimizer, scheduler = vla.setup_training(model, processor, dataset)
  # setup_training automatikusan: gradient_checkpointing + AdamW8bit + batch_size=1 (AI-8)
  vla.train(model, dataloader, optimizer, scheduler)
  vla.save_final(model, info)

Referenciák:
  G1 verzió:  notebooks/kaggle_unifolm_vla_finetune.py  (F3e: 80% SR)
  T1 dataset: vorilevi/roboshelf-t1-push-v1  (982 ep, 98.2% scripted SR)
  T1 env:     src/roboshelf_ai/mujoco/envs/manipulation/t1_shelf_stock_env.py
"""

from __future__ import annotations

import re
import sys
import subprocess
import json
import time
import shutil
from collections import deque
from pathlib import Path
from typing import Optional

# ── Modul szintű cache ────────────────────────────────────────────────────────
_vlm       = None
_processor = None

# ── Alapértelmezett útvonalak ─────────────────────────────────────────────────
DEFAULT_REPO_DIR = Path("/kaggle/working/unifolm-vla")
# Kaggle path formátum: /kaggle/input/datasets/{username}/{dataset-slug}/
# (AI-6 known issue: soha ne használj /kaggle/input/{slug} rövidített formátumot)
DEFAULT_DS_ROOT  = Path("/kaggle/input/datasets/leventevrss/roboshelf-t1-push-v1/t1_push_v1")
DEFAULT_CKPT_DIR = Path("/kaggle/working/unifolm_vla_t1_roboshelf")
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_DATASET_ID    = "vorilevi/roboshelf-t1-push-v1"  # fallback HF letöltéshez

# ── Robotika konstansok ───────────────────────────────────────────────────────
CHUNK_SIZE = 10
ACTION_DIM = 4    # T1: 4-DOF jobb kar (vs G1: 5)
STATE_DIM  = 24
IMG_SIZE   = 224
LANGUAGE   = "Push the stock to the target position on the shelf."


# =============================================================================
# 1. install_deps
# =============================================================================

def install_deps() -> None:
    """Telepíti az összes szükséges csomagot."""
    def pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    pip("transformers==4.52.3")
    pip("accelerate==1.5.2")
    pip("bitsandbytes")
    pip("peft>=0.10.0")
    pip("einops", "tiktoken", "scipy", "tensorboard")
    pip("pillow==11.3.0")
    pip("albumentations==1.4.18")
    pip("opencv-python", "pyarrow")
    pip("qwen-vl-utils")
    pip("huggingface_hub>=0.23.0")
    print("✅ install_deps kész")


# =============================================================================
# 2. clone_and_patch_repo
# =============================================================================

def clone_and_patch_repo(
    repo_dir: Optional[Path] = None,
) -> Path:
    """
    Klónozza az unifolm-vla repót és patcheli:
      - ACTION_DIM → 4  (T1 4-DOF)
      - PROPRIO_DIM → 24
      - flash_attention_2 → eager  (T4 nem támogatja)
    """
    repo_dir = repo_dir or DEFAULT_REPO_DIR

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/unitreerobotics/unifolm-vla.git",
             str(repo_dir)],
            check=True,
        )
        print(f"  Repo klónozva: {repo_dir}")
    else:
        print(f"  Repo már megvan: {repo_dir}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo_dir)],
        check=True,
    )

    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))

    # ── PATCH 1: dimenzió konstansok (ACTION_DIM=4, PROPRIO_DIM=24) ──────────
    DIM_PATTERNS = [
        (re.compile(r'\bACTION_DIM\s*=\s*\d+'),        'ACTION_DIM = 4'),
        (re.compile(r'\bPROPRIO_DIM\s*=\s*\d+'),       'PROPRIO_DIM = 24'),
        (re.compile(r'\bNUM_ACTIONS_CHUNK\s*=\s*\d+'), 'NUM_ACTIONS_CHUNK = 10'),
        (re.compile(r'"action_dim"\s*:\s*\d+'),         '"action_dim": 4'),
        (re.compile(r'"proprio_dim"\s*:\s*\d+'),        '"proprio_dim": 24'),
        (re.compile(r'"num_actions_chunk"\s*:\s*\d+'),  '"num_actions_chunk": 10'),
        (re.compile(r"'action_dim'\s*:\s*\d+"),         "'action_dim': 4"),
        (re.compile(r"'proprio_dim'\s*:\s*\d+"),        "'proprio_dim': 24"),
        (re.compile(r"'num_actions_chunk'\s*:\s*\d+"),  "'num_actions_chunk': 10"),
    ]

    DIM_KEYWORDS = {
        'ACTION_DIM', 'PROPRIO_DIM', 'NUM_ACTIONS_CHUNK',
        'action_dim', 'proprio_dim', 'num_actions_chunk',
        'G1_EE_6D',
    }

    patched_files = []
    for py_file in repo_dir.rglob("*.py"):
        try:
            src = py_file.read_text(errors='ignore')
        except Exception:
            continue
        if not any(kw in src for kw in DIM_KEYWORDS):
            continue
        new_src = src
        for pattern, replacement in DIM_PATTERNS:
            new_src = pattern.sub(replacement, new_src)
        if new_src != src:
            py_file.write_text(new_src)
            patched_files.append(py_file.relative_to(repo_dir))

    if patched_files:
        for f in patched_files:
            print(f"  Patchelve: {f}")
    else:
        print("  ⚠️  Egyetlen fájlban sem találtunk patchelendő dimenzió konstanst!")

    # ── PATCH 2: QWen2_5.py — eager attention ─────────────────────────────────
    patched = False
    for qp in repo_dir.rglob("QWen2_5.py"):
        src = qp.read_text()
        changed = False
        for old, new in [
            ('attn_implementation="flash_attention_2"', 'attn_implementation="eager"'),
            ("attn_implementation='flash_attention_2'", "attn_implementation='eager'"),
        ]:
            if old in src:
                src = src.replace(old, new)
                changed = True
        if changed:
            qp.write_text(src)
            print(f"  QWen2_5.py patchelve (flash_attention_2 → eager)")
        patched = True
    if not patched:
        print("  ⚠️  QWen2_5.py nem található!")

    print("✅ clone_and_patch_repo kész")
    return repo_dir


# =============================================================================
# 3. download_dataset  (új lépés — T1-specifikus)
# =============================================================================

def download_dataset(
    ds_root: Optional[Path] = None,
    hf_dataset_id: str = HF_DATASET_ID,
) -> Path:
    """
    FALLBACK — csak akkor szükséges, ha a dataset NEM lett hozzáadva
    Kaggle input-ként.

    Normál workflow:
      1. Kaggle-on add hozzá a datasetet: leventevrss/roboshelf-t1-push-v1
      2. DEFAULT_DS_ROOT = /kaggle/input/roboshelf-t1-push-v1
      → Ez a függvény NEM kell.

    Fallback (ha nincs Kaggle input):
      Letölti HF-ről: vorilevi/roboshelf-t1-push-v1 → /kaggle/working/t1_push_v1/
    """
    from huggingface_hub import snapshot_download

    fallback_dir = Path("/kaggle/working/t1_push_v1")
    ds_root = ds_root or fallback_dir

    if (DEFAULT_DS_ROOT / "meta" / "info.json").exists():
        print(f"  Kaggle input dataset megvan: {DEFAULT_DS_ROOT}")
        return DEFAULT_DS_ROOT

    if ds_root.exists() and (ds_root / "meta" / "info.json").exists():
        print(f"  Dataset már megvan: {ds_root}")
        return ds_root

    print(f"  HF fallback letöltés: {hf_dataset_id} → {ds_root}")
    snapshot_download(
        repo_id   = hf_dataset_id,
        repo_type = "dataset",
        local_dir = str(ds_root),
    )
    print(f"✅ download_dataset kész: {ds_root}")
    return ds_root


# =============================================================================
# 4. check_dataset
# =============================================================================

def check_dataset(
    ds_root: Optional[Path] = None,
) -> dict:
    """
    Ellenőrzi a LeRobot dataset elérési útját és visszaadja a metaadatokat.
    """
    ds_root = ds_root or DEFAULT_DS_ROOT

    parquets = list(ds_root.rglob("*.parquet"))
    print(f"  Parquet: {len(parquets)} fájl | Kamera: nincs (state-only)")

    meta     = json.loads((ds_root / "meta" / "info.json").read_text())
    data_dir = ds_root / "data" / "chunk-000"

    info = {
        "ds_root":     ds_root,
        "data_dir":    data_dir,
        "video_dir":   None,          # T1: nincs videó
        "camera_name": None,
        "n_episodes":  meta["total_episodes"],
        "fps":         meta["fps"],
        "action_dim":  meta["action_dim"],
        "obs_dim":     meta["obs_dim"],
        "task":        meta["task"],
    }

    print(f"  Dataset: {info['n_episodes']} epizód | "
          f"action_dim={info['action_dim']} | obs_dim={info['obs_dim']}")
    print(f"  Task: {info['task']}")
    print("✅ check_dataset kész")
    return info


# =============================================================================
# 5. build_dataset
# =============================================================================

def build_dataset(info: dict) -> "RoboshelfVLADataset":
    """Létrehozza a RoboshelfVLADataset objektumot."""
    dataset = RoboshelfVLADataset(
        data_dir   = info["data_dir"],
        n_episodes = info["n_episodes"],
        chunk_size = CHUNK_SIZE,
    )
    print("✅ build_dataset kész")
    return dataset


# =============================================================================
# 6. load_model
# =============================================================================

def load_model(
    info: Optional[dict] = None,
    repo_dir: Optional[Path] = None,
    hf_model: str = DEFAULT_HF_MODEL,
) -> tuple:
    """
    Betölti a VLM-et 8-bit-ben, majd összerakja az Unifolm_VLA modellt.
    Azonos a G1 verzióval; action_dim=4 kerül a config-ba.
    """
    import yaml
    import torch
    import torch.nn as nn

    global _vlm, _processor

    repo_dir   = repo_dir or DEFAULT_REPO_DIR
    action_dim = info["action_dim"] if info else ACTION_DIM
    obs_dim    = info["obs_dim"]    if info else STATE_DIM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        free  = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
        print(f"  GPU: {props.name} | VRAM: {total:.1f} GB (szabad: {free:.1f} GB)")

    if _vlm is None:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )
        torch.cuda.empty_cache()
        print(f"\n  VLM betöltés 8-bit: {hf_model}")
        print("  Ez ~3-5 percet vesz igénybe...")

        _vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model,
            load_in_8bit=True,
            device_map="auto",
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        _processor = Qwen2_5_VLProcessor.from_pretrained(hf_model)

        if device == "cuda":
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VLM OK — VRAM: {used:.1f} / {total:.1f} GB")
    else:
        print("  VLM már betöltve (cache)")

    from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface

    _captured_vlm       = _vlm
    _captured_processor = _processor

    def _patched_init(self, config, **kwargs):
        nn.Module.__init__(self)
        self.model     = _captured_vlm
        self.processor = _captured_processor

    _QWen_VL_Interface.__init__ = _patched_init
    print("  _QWen_VL_Interface.__init__ patchelve")

    from unifolm_vla.model.framework.share_tools import dict_to_namespace

    config_yaml = repo_dir / "src/unifolm_vla/config/training/unifolm_vla_train.yaml"
    with open(config_yaml) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    config.framework.qwenvl.base_vlm                        = hf_model
    config.framework.qwenvl.attn_implementation             = "eager"
    config.framework.action_model.action_dim                = action_dim
    config.framework.action_model.state_dim                 = obs_dim
    config.framework.action_model.action_horizon            = CHUNK_SIZE
    config.framework.action_model.future_action_window_size = CHUNK_SIZE - 1

    if not hasattr(config, "trainer") or config.trainer is None:
        config.trainer = dict_to_namespace({"repeated_diffusion_steps": 4})

    from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA

    _overrides = {
        'ACTION_DIM':        action_dim,
        'PROPRIO_DIM':       obs_dim,
        'NUM_ACTIONS_CHUNK': CHUNK_SIZE,
    }
    for mod_name in list(sys.modules.keys()):
        if 'unifolm' not in mod_name:
            continue
        mod = sys.modules[mod_name]
        for attr, val in _overrides.items():
            if hasattr(mod, attr) and getattr(mod, attr) != val:
                print(f"  [runtime] {mod_name}.{attr}: {getattr(mod, attr)} → {val}")
                setattr(mod, attr, val)

    model     = Unifolm_VLA(config=config)
    processor = model.processor
    model.action_model = model.action_model.to(device)

    layer1 = model.action_model.action_encoder.layer1
    if layer1.in_features != action_dim:
        print(f"\n  ⚠️  Dimenzió mismatch (layer1.in={layer1.in_features} != {action_dim})")
        model.action_model = _rebuild_action_model(
            model.action_model,
            old_action_dim = layer1.in_features,
            new_action_dim = action_dim,
            new_state_dim  = obs_dim,
            device         = device,
        )
        assert model.action_model.action_encoder.layer1.in_features == action_dim
        print(f"  ✓ action_model javítva")

    if device == "cuda":
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ load_model kész — VRAM: {used:.1f} / {total:.1f} GB")
    else:
        print("\n✅ load_model kész")
    print(f"  action_dim={action_dim}  state_dim={obs_dim}  chunk={CHUNK_SIZE}")

    return model, processor


# =============================================================================
# 7. setup_training
# =============================================================================

def setup_training(
    model,
    processor,
    dataset,
    batch_size: int = 1,       # T4 16GB: batch=1 kötelező (AI-8 known issue)
    lr_action:  float = 1e-4,
    lr_lora:    float = 1e-5,
    max_steps:  int = 10000,   # G1 F3e: 10k steps → 80% SR
) -> tuple:
    """
    LoRA r=32 (G1 F3e konfigurációval azonos).

    T4 memória fix-ek (AI-8 known issue):
      - batch_size=1 (alapértelmezett)
      - gradient_checkpointing_enable() a VLM-re
      - AdamW8bit (bitsandbytes) — optimizer state ~75% kisebb
    """
    import torch
    import bitsandbytes as bnb   # install_deps() után hívandó (AI-7 known issue)
    from torch.utils.data import DataLoader
    from peft import get_peft_model, LoraConfig

    lora_cfg = LoraConfig(
        r=32,              # G1 F3e: r=32 → 80% SR
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.qwen_vl_interface.model = get_peft_model(
        model.qwen_vl_interface.model,
        lora_cfg,
    )
    model.qwen_vl_interface.model.print_trainable_parameters()

    # Gradient checkpointing — aktivációk újraszámítása backward-ban (AI-8)
    model.qwen_vl_interface.model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()

    collate_fn = _make_qwen_collate(processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # 8-bit Adam — optimizer state ~75% kisebb (AI-8)
    optimizer = bnb.optim.AdamW8bit([
        {"params": model.action_model.parameters(),            "lr": lr_action},
        {"params": model.qwen_vl_interface.model.parameters(), "lr": lr_lora},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-6,
    )

    print(f"\n✅ setup_training kész")
    print(f"  LoRA r=32 | batch={batch_size} | {len(dataloader)} batch/epoch")
    print(f"  AdamW8bit + gradient_checkpointing (T4 OOM fix)")
    print(f"  max_steps={max_steps} (~456 perc T4-en, G1 referencia alapján)")
    return model, dataloader, optimizer, scheduler


# =============================================================================
# 8. train
# =============================================================================

def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    max_steps:  int = 10000,
    log_every:  int = 100,
    save_every: int = 1000,
    ckpt_dir: Optional[Path] = None,
) -> float:
    """Training loop (~456 perc T4-en, 10k lépés)."""
    import torch

    ckpt_dir = ckpt_dir or DEFAULT_CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()

    loss_buf  = deque(maxlen=100)
    step      = 0
    t0        = time.time()
    data_iter = iter(dataloader)

    print(f"\nTraining — {max_steps} lépés | device={device}")
    print("─" * 60)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        qwen_inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }

        optimizer.zero_grad()
        out  = model(qwen_inputs=qwen_inputs)
        loss = out["action_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_buf.append(loss.item())
        step += 1

        if step % log_every == 0:
            avg_loss = sum(loss_buf) / len(loss_buf)
            elapsed  = time.time() - t0
            lr_now   = optimizer.param_groups[0]["lr"]
            eta_min  = (max_steps - step) / max(step, 1) * elapsed / 60
            print(f"[{step:5d}/{max_steps}] loss={avg_loss:.4f} | "
                  f"lr={lr_now:.1e} | {elapsed/60:.1f}min | ETA: {eta_min:.0f}min")

        if step % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            import torch as _torch
            _torch.save({
                "step":        step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":        sum(loss_buf) / len(loss_buf),
                "action_dim":  ACTION_DIM,
                "state_dim":   STATE_DIM,
                "chunk_size":  CHUNK_SIZE,
                "robot":       "booster_t1",
            }, ckpt_path)
            print(f"  💾 Checkpoint → {ckpt_path.name}")

    elapsed    = time.time() - t0
    final_loss = sum(loss_buf) / len(loss_buf)
    print(f"\n✅ Training kész! {elapsed/60:.1f} perc | final_loss={final_loss:.4f}")
    return final_loss


# =============================================================================
# 9. save_final
# =============================================================================

def save_final(
    model,
    info: Optional[dict] = None,
    step: int = 10000,
    ckpt_dir: Optional[Path] = None,
) -> Path:
    """Elmenti a végső checkpointot és zip-eli."""
    import torch

    ckpt_dir = ckpt_dir or DEFAULT_CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    final_path = ckpt_dir / "final.pt"
    torch.save({
        "step":        step,
        "model_state": model.state_dict(),
        "config": {
            "action_dim":  info["action_dim"] if info else ACTION_DIM,
            "state_dim":   info["obs_dim"]    if info else STATE_DIM,
            "chunk_size":  CHUNK_SIZE,
            "robot":       "booster_t1",
            "hf_model":    DEFAULT_HF_MODEL,
            "hf_dataset":  HF_DATASET_ID,
        },
    }, final_path)

    zip_path = Path("/kaggle/working/unifolm_vla_t1_roboshelf_final.zip")
    shutil.make_archive(
        str(zip_path).replace(".zip", ""), "zip", str(ckpt_dir)
    )
    print(f"✅ save_final kész: {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)")
    print("   Töltsd le a Kaggle Output panelből!")
    return zip_path


# =============================================================================
# Belső segédosztályok
# =============================================================================

class RoboshelfVLADataset:
    """
    T1 LeRobot parquet → (dummy_image, state, action_chunk, language).
    Nincs videó — fekete kép a VLM inputhoz (state-only tanítás).
    """

    def __init__(
        self,
        data_dir:   Path,
        n_episodes: int,
        chunk_size: int = CHUNK_SIZE,
    ):
        import pyarrow.parquet as pq

        self._pq        = pq
        self.data_dir   = data_dir
        self.chunk_size = chunk_size

        self.samples = []
        for ep in range(n_episodes):
            pq_path = data_dir / f"episode_{ep:06d}.parquet"
            if not pq_path.exists():
                continue
            tbl     = pq.read_table(pq_path, columns=["frame_index"])
            n_steps = len(tbl)
            for s in range(n_steps):
                self.samples.append((ep, s, n_steps))

        print(f"  Dataset: {len(self.samples)} sample "
              f"({n_episodes} ep × ~{len(self.samples)//max(n_episodes,1)} lépés)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        import numpy as np
        from PIL import Image

        ep, step, n_steps = self.samples[idx]

        # Dummy fekete kép (nincs kamera) — VLM language featurekre támaszkodik
        image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        # State + action chunk
        tbl = self._pq.read_table(self.data_dir / f"episode_{ep:06d}.parquet")
        state = np.array(
            [tbl[f"obs_{i}"][step].as_py() for i in range(STATE_DIM)],
            dtype=np.float32,
        )
        actions = []
        for k in range(self.chunk_size):
            i = min(step + k, n_steps - 1)
            actions.append(np.array(
                [tbl[f"action_{j}"][i].as_py() for j in range(ACTION_DIM)],
                dtype=np.float32,
            ))
        action_chunk = np.stack(actions)  # (10, 4)

        return {
            "image":        image,
            "state":        torch.FloatTensor(state),         # (24,)
            "action_chunk": torch.FloatTensor(action_chunk),  # (10, 4)
            "language":     LANGUAGE,
        }


def _make_qwen_collate(processor):
    """Collate fn: PIL Image lista + language → Qwen2.5-VL inputs dict."""
    import torch

    def collate(batch):
        images    = [b["image"]    for b in batch]
        states    = torch.stack([b["state"]        for b in batch])  # (B, 24)
        actions   = torch.stack([b["action_chunk"] for b in batch])  # (B, 10, 4)
        languages = [b["language"] for b in batch]

        messages_batch = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  lang},
            ]}]
            for img, lang in zip(images, languages)
        ]
        texts = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            for msgs in messages_batch
        ]

        from qwen_vl_utils import process_vision_info
        img_list = []
        for msgs in messages_batch:
            img_inp, _ = process_vision_info(msgs)
            img_list.append(img_inp[0] if img_inp else None)

        processed = processor(
            text=texts, images=img_list, padding=True, return_tensors="pt",
        )
        processed["state"]  = states
        processed["action"] = actions
        return processed

    return collate


def _rebuild_action_model(action_model, old_action_dim, new_action_dim, new_state_dim, device):
    """Fallback: rossz méretű Linear rétegek cseréje az action_model-ben."""
    import torch
    import torch.nn as nn

    replaced = 0
    for name, layer in list(action_model.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue

        name_lower = name.lower()
        has_action = 'action' in name_lower
        has_state  = 'state' in name_lower or 'proprio' in name_lower

        new_in  = layer.in_features
        new_out = layer.out_features
        changed = False

        if layer.in_features == old_action_dim:
            new_in  = new_state_dim if (has_state and not has_action) else new_action_dim
            changed = True
        if layer.out_features == old_action_dim:
            new_out = new_state_dim if (has_state and not has_action) else new_action_dim
            changed = True

        if not changed:
            continue

        parts  = name.split('.')
        parent = action_model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]

        new_layer = nn.Linear(new_in, new_out, bias=(layer.bias is not None))
        new_layer = new_layer.to(device=device, dtype=torch.float32)
        setattr(parent, attr, new_layer)
        print(f"    {name}: {layer.in_features}→{new_in}, {layer.out_features}→{new_out}")
        replaced += 1

    action_model = action_model.to(device)
    print(f"  {replaced} réteg cserélve")
    return action_model
