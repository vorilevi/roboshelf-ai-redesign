# %% [markdown]
# # Roboshelf VLA Fine-tune — kaggle_run.py
#
# **Előfeltételek Kaggle-n:**
# 1. `roboshelf-vla-v1` dataset hozzáadva input-ként
# 2. `roboshelf-vla-scripts` dataset hozzáadva input-ként *(tartalmazza: `kaggle_unifolm_vla_finetune.py`)*
# 3. GPU: T4 x1 bekapcsolva
# 4. Internet: ON (első futtatáshoz — VLM letöltés ~3-5 perc)
#
# **Futtatás:** Runtime → Run All

# %% [markdown]
# ## Cella 1 — Modul import

# %%
import sys
from pathlib import Path

# Auto-discovery: Kaggle átnevezi a dataset slug-ot (pl. "-1" suffix),
# ezért rglob-bal keressük meg a fájlt bárhol /kaggle/input/ alatt.
MODULE_NAME  = "kaggle_unifolm_vla_finetune.py"
KAGGLE_INPUT = Path("/kaggle/input")

module_dir = None
for candidate in KAGGLE_INPUT.rglob(MODULE_NAME):
    module_dir = candidate.parent
    print(f"Modul megtalálva: {candidate}")
    break

if module_dir is None:
    print("❌ A modul nem található /kaggle/input/ alatt!")
    print("\nElérhető input könyvtárak:")
    for d in sorted(KAGGLE_INPUT.iterdir()):
        print(f"  {d.name}/")
        for f in list(d.rglob("*.py"))[:5]:
            print(f"    └── {f.name}")
    print(
        "\nMegoldás:\n"
        "  1. 'Add data' → keresd a 'roboshelf-vla-scripts' datasetedet\n"
        "  2. Ha még nincs: hozz létre új Kaggle Datasetet, töltsd fel\n"
        "     a kaggle_unifolm_vla_finetune.py-t, majd add hozzá input-ként.\n"
        "  3. Runtime → Restart & Run All"
    )
    raise FileNotFoundError(f"{MODULE_NAME} nem található.")

if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

import kaggle_unifolm_vla_finetune as vla

print("✅ Modul importálva")
print(f"   ACTION_DIM={vla.ACTION_DIM}  STATE_DIM={vla.STATE_DIM}  CHUNK_SIZE={vla.CHUNK_SIZE}")

# %% [markdown]
# ## Cella 2 — Függőségek telepítése

# %%
vla.install_deps()

# %% [markdown]
# ## Cella 3 — Repo klónozás + fájl patchek
#
# ⚠️ **Kritikus:** Ez patcheli a `constants.py`-t (`ACTION_DIM=5`) és a
# `QWen2_5.py`-t (`eager` attention) — **mielőtt** bármilyen `unifolm_vla` import futna!

# %%
repo_dir = vla.clone_and_patch_repo()

# %% [markdown]
# ## Cella 4 — Dataset ellenőrzés

# %%
info = vla.check_dataset()

# %% [markdown]
# ## Cella 5 — Dataset összerakás

# %%
dataset = vla.build_dataset(info)

# %% [markdown]
# ## Cella 6 — Modell betöltés
#
# 8-bit VLM inject + dimenzió assert.
# Újrafuttatható: `_vlm` cache-elve van, nem tölt le újra.

# %%
model, processor = vla.load_model(info, repo_dir=repo_dir)

# %% [markdown]
# ## Cella 7 — LoRA + optimizer + DataLoader

# %%
model, dataloader, optimizer, scheduler = vla.setup_training(
    model,
    processor,
    dataset,
    batch_size=2,
    lr_action=1e-4,
    lr_lora=1e-5,
    max_steps=2000,
)

# %% [markdown]
# ## Cella 8 — Training (~1h)

# %%
final_loss = vla.train(
    model,
    dataloader,
    optimizer,
    scheduler,
    max_steps=2000,
    log_every=50,
    save_every=500,
)

# %% [markdown]
# ## Cella 9 — Checkpoint mentés

# %%
zip_path = vla.save_final(model, info, step=2000)
print(f"\nKész! Töltsd le a Kaggle Output panelből: {zip_path.name}")
