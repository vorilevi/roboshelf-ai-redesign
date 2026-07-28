"""
kaggle_unifolm_vla_finetune.py — Importálható modul
====================================================
Roboshelf F3e | UnifoLM-VLA-0 fine-tune | Kaggle T4

Feltöltési mód:
  1. Kaggle Dataset-ként töltsd fel ezt a fájlt
     (pl. "roboshelf-vla-scripts" névvel, egyetlen fájlként)
  2. Az új notebook-ban add hozzá input-ként, majd:
       import sys
       sys.path.insert(0, "/kaggle/input/roboshelf-vla-scripts")
       import kaggle_unifolm_vla_finetune as vla

API (az új notebook által hívandó sorrendben):
  vla.install_deps()
  vla.clone_and_patch_repo()
  info = vla.check_dataset()
  dataset = vla.build_dataset(info)
  model, processor = vla.load_model(info)
  model, dataloader, optimizer, scheduler = vla.setup_training(model, processor, dataset)
  vla.train(model, dataloader, optimizer, scheduler)
  vla.save_final(model, info)

Ismert hibák és megoldások (beépítve):
  - transformers 4.52.x regression #43032 → load_in_8bit=True
  - constants.py hardcode ACTION_DIM=23 → clone UTÁN patcheljük
  - T4 flash_attention_2 nem támogatott → eager patchelés
  - _QWen_VL_Interface újra letölti VLM-et → inject + globals cache
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

# ── Modul szintű cache (VLM csak egyszer töltődik le) ─────────────────────
_vlm       = None
_processor = None

# ── Alapértelmezett útvonalak ─────────────────────────────────────────────
DEFAULT_REPO_DIR = Path("/kaggle/working/unifolm-vla")
DEFAULT_DS_ROOT  = Path("/kaggle/input/datasets/leventevrss/roboshelf-vla-v1/vla_v1")
DEFAULT_CKPT_DIR = Path("/kaggle/working/unifolm_vla_roboshelf")
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# ── Robotika konstansok (a mi datasetünkre) ───────────────────────────────
CHUNK_SIZE = 10
ACTION_DIM = 5
STATE_DIM  = 24
IMG_SIZE   = 224
LANGUAGE   = "Push the box to the target position on the shelf."


# =============================================================================
# 1. install_deps
# =============================================================================

def install_deps() -> None:
    """
    Telepíti az összes szükséges Python csomagot.
    Kaggle-n az első cellában hívd meg.
    """
    def pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    pip("transformers==4.52.3")   # Qwen2.5-VL >= 4.48; 8-bit workaround a 4.52.x regression-re
    pip("accelerate==1.5.2")
    pip("bitsandbytes")           # 8-bit INT8 quantization
    pip("peft>=0.10.0")           # LoRA / QLoRA
    pip("einops", "tiktoken", "scipy", "tensorboard")
    pip("pillow==11.3.0")
    pip("albumentations==1.4.18")
    pip("opencv-python", "pyarrow")
    pip("qwen-vl-utils")          # Qwen2.5-VL képfeldolgozó helper

    print("✅ install_deps kész")


# =============================================================================
# 2. clone_and_patch_repo
# =============================================================================

def clone_and_patch_repo(
    repo_dir: Optional[Path] = None,
) -> Path:
    """
    Klónozza az unifolm-vla repót és azonnal patcheli a kritikus fájlokat.

    FONTOS: Ezt a függvényt MIELŐTT bármilyen unifolm_vla import hívod meg!
    A fájl patch csak importálás előtt érvényes (Python module cache).

    Patch-ek:
      - MINDEN .py fájl: ACTION_DIM/PROPRIO_DIM/NUM_ACTIONS_CHUNK
        → assignment (= N) ÉS dict/yaml (": N") formátumban is
        → G1_EE_6D tartalmú fájlok külön figyelve
        (a DiT model a training/vla/constants.py-ból olvas, ami dict-stílusú lehet)
      - QWen2_5.py: flash_attention_2 → eager (T4 nem támogatja)

    Returns:
        repo_dir (Path): a klónozott repo gyökere
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

    # ── PATCH 1: dimenzió konstansok — MINDEN .py fájlban ────────────────
    # A unifolm_vla-ban LEGALÁBB KÉT különböző helyen van hardcode-olva:
    #   - rlds_dataloader/constants.py  (assignment stílus: ACTION_DIM = 23)
    #   - training/vla/constants.py     (lehet dict stílus: "action_dim": 23)
    #   - G1_EE_6D preset fájlok        (bármilyen stílus)
    # Mindkét formátumot kezeljük, minden .py fájlban.

    # Regex minták (assignment ÉS dict/yaml stílus)
    DIM_PATTERNS = [
        # Python assignment: ACTION_DIM = 23
        (re.compile(r'\bACTION_DIM\s*=\s*\d+'),        'ACTION_DIM = 5'),
        (re.compile(r'\bPROPRIO_DIM\s*=\s*\d+'),       'PROPRIO_DIM = 24'),
        (re.compile(r'\bNUM_ACTIONS_CHUNK\s*=\s*\d+'), 'NUM_ACTIONS_CHUNK = 10'),
        # Dict/yaml double-quote: "action_dim": 23
        (re.compile(r'"action_dim"\s*:\s*\d+'),         '"action_dim": 5'),
        (re.compile(r'"proprio_dim"\s*:\s*\d+'),        '"proprio_dim": 24'),
        (re.compile(r'"num_actions_chunk"\s*:\s*\d+'),  '"num_actions_chunk": 10'),
        # Dict/yaml single-quote: 'action_dim': 23
        (re.compile(r"'action_dim'\s*:\s*\d+"),         "'action_dim': 5"),
        (re.compile(r"'proprio_dim'\s*:\s*\d+"),        "'proprio_dim': 24"),
        (re.compile(r"'num_actions_chunk'\s*:\s*\d+"),  "'num_actions_chunk': 10"),
    ]

    # Kulcsszavak: ezeket tartalmazó fájlokat vizsgáljuk
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

    # ── PATCH 2: QWen2_5.py — eager attention ────────────────────────────
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

    print("✅ clone_and_patch_repo kész — most már biztonságos az unifolm_vla import")
    return repo_dir


# =============================================================================
# 3. check_dataset
# =============================================================================

def check_dataset(
    ds_root: Optional[Path] = None,
) -> dict:
    """
    Ellenőrzi a LeRobot dataset elérési útját és visszaadja a metaadatokat.

    Returns:
        info dict:
            ds_root, data_dir, video_dir, camera_name,
            n_episodes, fps, action_dim, obs_dim, task
    """
    ds_root = ds_root or DEFAULT_DS_ROOT

    mp4s     = list(ds_root.rglob("*.mp4"))
    parquets = list(ds_root.rglob("*.parquet"))
    print(f"  Parquet: {len(parquets)} fájl | MP4: {len(mp4s)} videó")

    meta      = json.loads((ds_root / "meta" / "info.json").read_text())
    camera    = meta.get("cameras", ["front_cam"])[0]
    data_dir  = ds_root / "data"  / "chunk-000"
    video_dir = ds_root / "videos" / "chunk-000" / f"observation.images.{camera}"

    info = {
        "ds_root":     ds_root,
        "data_dir":    data_dir,
        "video_dir":   video_dir,
        "camera_name": camera,
        "n_episodes":  meta["total_episodes"],
        "fps":         meta["fps"],
        "action_dim":  meta["action_dim"],
        "obs_dim":     meta["obs_dim"],
        "task":        meta["task"],
    }

    print(f"  Dataset: {info['n_episodes']} epizód | "
          f"action_dim={info['action_dim']} | obs_dim={info['obs_dim']}")
    print(f"  Kamera: {camera} | FPS: {info['fps']}")
    print(f"  Task: {info['task']}")
    print("✅ check_dataset kész")
    return info


# =============================================================================
# 4. build_dataset
# =============================================================================

def build_dataset(info: dict) -> "RoboshelfVLADataset":
    """
    Létrehozza a RoboshelfVLADataset objektumot.
    A collate_fn-t a setup_training() állítja össze (kell hozzá processor).

    Args:
        info: check_dataset() által visszaadott dict

    Returns:
        dataset: RoboshelfVLADataset instance
    """
    dataset = RoboshelfVLADataset(
        data_dir   = info["data_dir"],
        video_dir  = info["video_dir"],
        n_episodes = info["n_episodes"],
        chunk_size = CHUNK_SIZE,
    )
    print("✅ build_dataset kész")
    return dataset


# =============================================================================
# 5. load_model
# =============================================================================

def load_model(
    info: Optional[dict] = None,
    repo_dir: Optional[Path] = None,
    hf_model: str = DEFAULT_HF_MODEL,
) -> tuple:
    """
    Betölti a VLM-et 8-bit-ben, majd összerakja az Unifolm_VLA modellt.

    Stratégia:
      - load_in_8bit=True: régi, stabil code path (NEM érinti a transformers
        4.52.x regression-t, GitHub #43032)
      - _QWen_VL_Interface monkey-patch: pre-loadolt VLM inject
        (elkerüli a Unifolm_VLA init-beli from_pretrained újrahívást)
      - Modul szintű cache (_vlm, _processor): újrafuttatáskor nem tölt le újra

    Returns:
        (model, processor): Unifolm_VLA instance + Qwen2.5-VL processor
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

    # ── VLM betöltés (cache-elve) ──────────────────────────────────────────
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
            load_in_8bit=True,           # régi stabil code path
            device_map="auto",
            attn_implementation="eager", # T4: nincs flash_attention_2
            low_cpu_mem_usage=True,
        )
        _processor = Qwen2_5_VLProcessor.from_pretrained(hf_model)

        if device == "cuda":
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VLM OK — VRAM: {used:.1f} / {total:.1f} GB (szabad: {total-used:.1f} GB)")
    else:
        print("  VLM már betöltve (cache), újrafelhasználás...")

    # ── _QWen_VL_Interface monkey-patch ────────────────────────────────────
    from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface

    _captured_vlm       = _vlm
    _captured_processor = _processor

    def _patched_init(self, config, **kwargs):
        nn.Module.__init__(self)
        self.model     = _captured_vlm
        self.processor = _captured_processor

    _QWen_VL_Interface.__init__ = _patched_init
    print("  _QWen_VL_Interface.__init__ patchelve (VLM inject)")

    # ── Config betöltés ────────────────────────────────────────────────────
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

    # ── Unifolm_VLA import (ez tölti be a DiT action model modult is) ──────
    # FONTOS: az importnak a runtime patch ELŐTT kell történnie!
    # A `from unifolm_vla.training.vla.constants import ACTION_DIM` stílusú
    # import lokális integer kötést hoz létre a DiT modul névterében.
    # Ezért a patch-nek IMPORT UTÁN, de INSTANTIÁLÁS ELŐTT kell futnia.
    from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA

    # ── 2. védelmi vonal: runtime modul-szintű konstans override ──────────
    # Most már az összes unifolm_vla modul betöltve van sys.modules-ban,
    # beleértve a DiT action model modult is.
    # setattr(modul, 'ACTION_DIM', 5) felülírja a lokális kötést is,
    # mert a `from x import y` a modult saját __dict__-jébe írja.
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

    # ── Modell instantiálás ────────────────────────────────────────────────
    # Most már minden betöltött modul ACTION_DIM-je helyes (5),
    # ezért a DiT __init__ nn.Linear(5, 1536)-ot fog létrehozni.
    model     = Unifolm_VLA(config=config)
    processor = model.processor
    model.action_model = model.action_model.to(device)

    # ── 3. védelmi vonal: közvetlen réteg-csere ha a dimenzió még mindig rossz
    # Akkor aktiválódik, ha a constants dict-lookup stílusú volt ÉS
    # a runtime patch sem fogta meg (pl. a binding nem modul-szintű,
    # hanem closure-ban él). Ez a fallback garantáltan működik.
    layer1 = model.action_model.action_encoder.layer1
    if layer1.in_features != action_dim:
        print(f"\n  ⚠️  Dimenzió mismatch (layer1.in={layer1.in_features} != {action_dim})")
        print(f"  → action_model réteg-csere...")
        model.action_model = _rebuild_action_model(
            model.action_model,
            old_action_dim = layer1.in_features,
            new_action_dim = action_dim,
            new_state_dim  = obs_dim,
            device         = device,
        )
        layer1_new = model.action_model.action_encoder.layer1
        assert layer1_new.in_features == action_dim, \
            f"❌ Réteg-csere sikertelen: {layer1_new.in_features} != {action_dim}"
        print(f"  ✓ action_model javítva: action_dim={action_dim}, state_dim={obs_dim}")

    if device == "cuda":
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ load_model kész — VRAM: {used:.1f} / {total:.1f} GB")
    else:
        print("\n✅ load_model kész")
    print(f"  VLM:         Qwen2.5-VL-7B-Instruct [8-bit INT8]")
    print(f"  Action head: action_dim={action_dim}  state_dim={obs_dim}  chunk={CHUNK_SIZE}")
    print(f"  layer1 shape: ({action_dim} → {model.action_model.action_encoder.layer1.out_features})")

    return model, processor


# =============================================================================
# 6. setup_training
# =============================================================================

def setup_training(
    model,
    processor,
    dataset,
    batch_size: int = 2,
    lr_action:  float = 1e-4,
    lr_lora:    float = 1e-5,
    max_steps:  int = 2000,
) -> tuple:
    """
    LoRA a VLM-re, DataLoader és optimizer összerakás.

    FONTOS: prepare_model_for_kbit_training() NINCS meghívva!
      - 8-bit esetén felesleges és kettős híváskor "already a PEFT model"
        warningot/hibát okoz ha a cellát újrafuttatod.

    Returns:
        (model, dataloader, optimizer, scheduler)
    """
    import torch
    from torch.utils.data import DataLoader
    from peft import get_peft_model, LoraConfig

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
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

    collate_fn = _make_qwen_collate(processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW([
        {"params": model.action_model.parameters(),           "lr": lr_action},
        {"params": model.qwen_vl_interface.model.parameters(), "lr": lr_lora},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-6,
    )

    print(f"\n✅ setup_training kész")
    print(f"  batch_size={batch_size} | {len(dataloader)} batch/epoch")
    print(f"  lr_action={lr_action} | lr_lora={lr_lora}")
    return model, dataloader, optimizer, scheduler


# =============================================================================
# 7. train
# =============================================================================

def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    max_steps: int = 2000,
    log_every:  int = 50,
    save_every: int = 500,
    ckpt_dir: Optional[Path] = None,
) -> float:
    """
    Training loop (~1h, 2000 lépés T4-en).

    Returns:
        final_loss (float): utolsó 50 lépés átlagos loss-a
    """
    import torch

    ckpt_dir = ckpt_dir or DEFAULT_CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()

    loss_buf  = deque(maxlen=50)
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
            print(f"[{step:4d}/{max_steps}] loss={avg_loss:.4f} | "
                  f"lr={lr_now:.1e} | {elapsed/60:.1f}min")

        if step % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            import torch as _torch
            _torch.save({
                "step":            step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            sum(loss_buf) / len(loss_buf),
                "action_dim":      ACTION_DIM,
                "state_dim":       STATE_DIM,
                "chunk_size":      CHUNK_SIZE,
            }, ckpt_path)
            print(f"  💾 Checkpoint → {ckpt_path.name}")

    elapsed    = time.time() - t0
    final_loss = sum(loss_buf) / len(loss_buf)
    print(f"\n✅ Training kész! {elapsed/60:.1f} perc | final_loss={final_loss:.4f}")
    return final_loss


# =============================================================================
# 8. save_final
# =============================================================================

def save_final(
    model,
    info: Optional[dict] = None,
    step: int = 2000,
    ckpt_dir: Optional[Path] = None,
) -> Path:
    """
    Elmenti a végső checkpointot és zip-eli a letöltéshez.

    Returns:
        zip_path (Path): a kész zip fájl útvonala
    """
    import torch

    ckpt_dir = ckpt_dir or DEFAULT_CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    final_path = ckpt_dir / "final.pt"
    torch.save({
        "step":        step,
        "model_state": model.state_dict(),
        "config": {
            "action_dim":  info["action_dim"]  if info else ACTION_DIM,
            "state_dim":   info["obs_dim"]      if info else STATE_DIM,
            "chunk_size":  CHUNK_SIZE,
            "camera":      info["camera_name"]  if info else "front_cam",
            "hf_model":    DEFAULT_HF_MODEL,
        },
    }, final_path)

    zip_path = Path("/kaggle/working/unifolm_vla_roboshelf_final.zip")
    shutil.make_archive(
        str(zip_path).replace(".zip", ""), "zip", str(ckpt_dir)
    )
    print(f"✅ save_final kész: {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)")
    print("   Töltsd le a Kaggle Output panelből!")
    return zip_path


# =============================================================================
# Belső segédosztályok (nem API, az import után is elérhetők)
# =============================================================================

class RoboshelfVLADataset:
    """
    LeRobot parquet + MP4 → (image, state, action_chunk, language) tuple.
    Lazy: minden lépésnél olvassa a parquet-et és a videót.
    """

    def __init__(
        self,
        data_dir:   Path,
        video_dir:  Path,
        n_episodes: int,
        chunk_size: int = CHUNK_SIZE,
    ):
        import torch
        from torch.utils.data import Dataset
        import pyarrow.parquet as pq

        self._torch     = torch
        self._pq        = pq
        self.data_dir   = data_dir
        self.video_dir  = video_dir
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
        import cv2
        import numpy as np
        from PIL import Image

        ep, step, n_steps = self.samples[idx]

        # Kép
        vid  = self.video_dir / f"episode_{ep:06d}.mp4"
        cap  = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        image = Image.fromarray(frame)

        # State + action chunk
        tbl   = self._pq.read_table(self.data_dir / f"episode_{ep:06d}.parquet")
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
        action_chunk = np.stack(actions)

        return {
            "image":        image,
            "state":        self._torch.FloatTensor(state),         # (24,)
            "action_chunk": self._torch.FloatTensor(action_chunk),  # (10, 5)
            "language":     LANGUAGE,
        }


def _make_qwen_collate(processor):
    """Collate fn: PIL Image lista + language → Qwen2.5-VL inputs dict."""
    import torch

    def collate(batch):
        images    = [b["image"]    for b in batch]
        states    = torch.stack([b["state"]        for b in batch])  # (B, 24)
        actions   = torch.stack([b["action_chunk"] for b in batch])  # (B, 10, 5)
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


# =============================================================================
# Belső segédfüggvény: action_model réteg-csere
# =============================================================================

def _rebuild_action_model(action_model, old_action_dim, new_action_dim, new_state_dim, device):
    """
    Ha a fájl-patch nem fogta meg az összes konstanst (pl. dict-lookup stílus),
    az action_model-ben minden wrongly-sized Linear réteget cserél.

    Heurisztika a réteg szerepének meghatározásához:
      - "action" a névben  → new_action_dim (5)
      - "state" / "proprio" a névben → new_state_dim (24)
      - Mindkét szó vagy egyik sem → new_action_dim (5, biztonságos alapértelmezés)

    Csak a old_action_dim (23) méretű rétegeket érinti.
    """
    import torch
    import torch.nn as nn

    replaced = 0
    for name, layer in list(action_model.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue

        name_lower = name.lower()
        has_action = 'action' in name_lower
        has_state  = 'state' in name_lower or 'proprio' in name_lower

        # Csak a wrong-sized rétegeket kezeljük
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

        # Szülő modul + attr név meghatározása
        parts  = name.split('.')
        parent = action_model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]

        new_layer = nn.Linear(new_in, new_out, bias=(layer.bias is not None))
        new_layer = new_layer.to(device=device, dtype=torch.float32)
        setattr(parent, attr, new_layer)

        print(f"    {name}: in {layer.in_features}→{new_in}, out {layer.out_features}→{new_out}")
        replaced += 1

    # action_model egészét device-ra
    action_model = action_model.to(device)
    print(f"  {replaced} réteg cserélve")
    return action_model
