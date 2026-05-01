"""
ACT (Action Chunking Transformer) BC Training Script — F3d Phase 030.

Architektúra: ACT-style Transformer BC önálló CVAE-vel, proprioceptív inputra.
    - ObsEncoder:     MLP(24 → 256) + LayerNorm
    - VAE Encoder:    MLP(obs + action_chunk → z mean/logvar)  [csak tréning]
    - Transformer:    nn.Transformer encoder-decoder, action chunking
    - Action Head:    Linear(d_model → action_dim)

Platform: Mac M2 MPS (float32, opcionálisan bfloat16) vagy CPU fallback.
Config:   configs/bc/act_shelf_stock_v1.yaml
Dataset:  data/lerobot/scripted_v1/ (LeRobot v3.0 custom parquet, generálta lerobot_export.py)

Futtatás (repo gyökeréből):
    python3 tools/train_act.py \\
        --config  configs/bc/act_shelf_stock_v1.yaml \\
        --dataset data/lerobot/scripted_v1

M2 MPS várható idő: ~6-8h (100 epoch, 200 demo × ~150 lépés átlag = ~30k frame)
Kaggle T4 fallback: ~1-2h

Hivatkozások:
    ACT: Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
    Implementáció: önálló (nem lerobot wrapper) — lerobot.scripts.train Hydra config-ot vár,
                   ez a script a saját YAML sémánkat olvassa.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import pyarrow.parquet as pq
    _HAS_ARROW = True
except ImportError:
    _HAS_ARROW = False
    print("⚠️  pyarrow nem elérhető — pip install pyarrow --break-system-packages")

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent

OBS_DIM  = 24
ACT_DIM  = 5


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ACTDataset(Dataset):
    """
    Sliding-window dataset ACT tréninghez.
    Betölti a lerobot_export.py által generált parquet fájlokat.

    Minden minta: (obs_t [norm], actions[t:t+chunk] [norm], mask [float 0/1])
    - Rövid epizód végén: zero-padding, mask=0 a padding lépéseken.
    - Normalizáció: obs → z-score, action → minmax [-1, 1]
    """

    def __init__(self, dataset_dir: Path, chunk_size: int = 20, stats: dict | None = None):
        if not _HAS_ARROW:
            raise ImportError("pyarrow szükséges: pip install pyarrow --break-system-packages")

        self.chunk_size = chunk_size
        data_dir = Path(dataset_dir) / "data" / "chunk-000"
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Nincs parquet fájl: {data_dir}")

        obs_cols = [f"obs_{i}"    for i in range(OBS_DIM)]
        act_cols = [f"action_{i}" for i in range(ACT_DIM)]

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        obs_all: list[np.ndarray] = []
        act_all: list[np.ndarray] = []

        for pf in parquet_files:
            table = pq.read_table(pf)
            df    = table.to_pandas()
            T     = len(df)

            obs_ep = df[obs_cols].values.astype(np.float32)   # (T, 24)
            act_ep = df[act_cols].values.astype(np.float32)   # (T, 5)
            obs_all.append(obs_ep)
            act_all.append(act_ep)

            for t in range(T):
                end       = min(t + chunk_size, T)
                act_chunk = act_ep[t:end]                       # (≤chunk, 5)
                chunk_len = len(act_chunk)

                if chunk_len < chunk_size:
                    pad       = np.zeros((chunk_size - chunk_len, ACT_DIM), dtype=np.float32)
                    act_chunk = np.concatenate([act_chunk, pad], axis=0)
                    mask      = np.zeros(chunk_size, dtype=np.float32)
                    mask[:chunk_len] = 1.0
                else:
                    mask = np.ones(chunk_size, dtype=np.float32)

                self.samples.append((obs_ep[t], act_chunk, mask))

        # Normalizációs statisztikák
        if stats is not None:
            self.obs_mean = np.array(stats["obs"]["mean"], dtype=np.float32)
            self.obs_std  = np.array(stats["obs"]["std"],  dtype=np.float32) + 1e-8
            self.act_min  = np.array(stats["action"]["min"], dtype=np.float32)
            self.act_max  = np.array(stats["action"]["max"], dtype=np.float32)
        else:
            obs_cat       = np.concatenate(obs_all, axis=0)
            act_cat       = np.concatenate(act_all, axis=0)
            self.obs_mean = obs_cat.mean(0)
            self.obs_std  = obs_cat.std(0) + 1e-8
            self.act_min  = act_cat.min(0)
            self.act_max  = act_cat.max(0)

        print(f"Dataset: {len(parquet_files)} epizód → {len(self.samples)} minta "
              f"(chunk_size={chunk_size})")

    # --- normalizáció ---

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        rng = (self.act_max - self.act_min) + 1e-8
        return 2.0 * (action - self.act_min) / rng - 1.0

    def denormalize_action(self, action_norm: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.cpu().numpy()
        rng = (self.act_max - self.act_min) + 1e-8
        return (action_norm + 1.0) * 0.5 * rng + self.act_min

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        obs, act_chunk, mask = self.samples[idx]
        return (
            torch.from_numpy(self.normalize_obs(obs)),        # (24,)
            torch.from_numpy(self.normalize_action(act_chunk)),  # (chunk, 5)
            torch.from_numpy(mask),                            # (chunk,)
        )


# ─── Model ───────────────────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """MLP + LayerNorm obs encoder (ACT paper: linear projection for image tokens; itt MLP)."""

    def __init__(self, obs_dim: int = OBS_DIM, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # (B, d_model)


class ACTModel(nn.Module):
    """
    ACT-style BC model proprioceptív inputra.

    Tréning forward:
        obs → obs_feat                           (B, d_model)
        (obs, action_chunk) → VAE → z            (B, latent_dim)
        z, obs_feat → memory seq (len=2)         (2, B, d_model)
        chunk_size learned queries → Transformer → action_chunk_pred  (B, chunk, act_dim)
        Loss: MSE(pred, target) + kl_weight * KL

    Inference:
        z = 0 (prior mean)
        obs_feat → memory (len=2, z=0)
        queries → Transformer → action_chunk_pred
    """

    def __init__(self, cfg: dict):
        super().__init__()
        obs_dim    = cfg["dataset"]["obs_dim"]
        action_dim = cfg["model"]["action_dim"]
        chunk_size = cfg["model"]["chunk_size"]
        d_model    = cfg["model"]["transformer"]["d_model"]
        nhead      = cfg["model"]["transformer"]["nhead"]
        n_enc      = cfg["model"]["transformer"]["num_encoder_layers"]
        n_dec      = cfg["model"]["transformer"]["num_decoder_layers"]
        d_ff       = cfg["model"]["transformer"]["dim_feedforward"]
        dropout    = cfg["model"]["transformer"]["dropout"]
        latent_dim = cfg["model"]["vae"]["latent_dim"]

        self.chunk_size  = chunk_size
        self.latent_dim  = latent_dim
        self.kl_weight   = cfg["model"]["vae"]["kl_weight"]
        self.vae_enabled = cfg["model"]["vae"]["enabled"]

        # Obs encoder
        self.obs_encoder = ObsEncoder(obs_dim, d_model)

        # VAE encoder: (obs || flattened actions) → (mu, logvar)
        vae_in = obs_dim + chunk_size * action_dim
        self.vae_encoder = nn.Sequential(
            nn.Linear(vae_in, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2 * latent_dim),
        )

        # Latent → d_model projection
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Tanult pozicionális kódolás a memory-hoz (2 token: obs, z)
        self.pos_enc = nn.Parameter(torch.randn(2, 1, d_model) * 0.02)

        # Tanult action query embeddings
        self.query_embed = nn.Embedding(chunk_size, d_model)

        # Transformer (batch_first=False → (seq, B, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="relu", batch_first=False,
            norm_first=cfg["model"]["transformer"].get("normalize_before", True),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="relu", batch_first=False,
            norm_first=cfg["model"]["transformer"].get("normalize_before", True),
        )
        self.tf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc,
                                                norm=nn.LayerNorm(d_model))
        self.tf_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec,
                                                norm=nn.LayerNorm(d_model))

        # Action prediction head
        self.action_head = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "pos_enc" in name or "query_embed" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:     (B, obs_dim)
            actions: (B, chunk_size, action_dim) — tréningkor; None → inference

        Returns:
            actions_pred: (B, chunk_size, action_dim)
            kl:           scalar tensor (0 ha inference)
        """
        B = obs.shape[0]

        # 1. Obs feature
        obs_feat = self.obs_encoder(obs)   # (B, d_model)

        # 2. VAE
        if self.vae_enabled and actions is not None and self.training:
            act_flat = actions.reshape(B, -1)                      # (B, chunk*act)
            vae_in   = torch.cat([obs, act_flat], dim=-1)
            vae_out  = self.vae_encoder(vae_in)
            mu, logvar = vae_out.chunk(2, dim=-1)
            logvar   = logvar.clamp(-4.0, 4.0)
            eps      = torch.randn_like(mu)
            z        = mu + eps * torch.exp(0.5 * logvar)
            kl       = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        else:
            z  = obs.new_zeros(B, self.latent_dim)
            kl = obs.new_zeros(1).squeeze()

        z_feat = self.latent_proj(z)   # (B, d_model)

        # 3. Memory: (obs_feat, z_feat) → (2, B, d_model) + pozicionális kódolás
        memory = torch.stack([obs_feat, z_feat], dim=0) + self.pos_enc   # (2, B, d)

        # 4. Encoder
        memory_enc = self.tf_encoder(memory)   # (2, B, d_model)

        # 5. Decoder: tanult action queries → (chunk, B, d_model)
        tgt = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)   # (chunk, B, d)
        out = self.tf_decoder(tgt, memory_enc)                          # (chunk, B, d)

        # 6. Action head
        actions_pred = self.action_head(out)              # (chunk, B, act_dim)
        actions_pred = actions_pred.permute(1, 0, 2)      # (B, chunk, act_dim)

        return actions_pred, kl


# ─── Training ────────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    req = cfg["training"]["device"]
    if req == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req not in ("cpu",):
        print(f"⚠️  '{req}' nem elérhető → CPU fallback")
    return torch.device("cpu")


def train_one_epoch(
    model: ACTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    kl_weight: float,
    grad_clip: float,
) -> tuple[float, float, float]:
    """Egy epoch tréning. Visszaad: (avg_loss, avg_mse, avg_kl)."""
    model.train()
    total_loss = total_mse = total_kl = 0.0

    for obs, actions, mask in loader:
        obs     = obs.to(device)
        actions = actions.to(device)
        mask    = mask.to(device)

        actions_pred, kl = model(obs, actions)

        # MSE veszteség — padding maszkkal súlyozva
        mse_per_step = ((actions_pred - actions) ** 2).mean(-1)   # (B, chunk)
        mse = (mse_per_step * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        mse = mse.mean()

        loss = mse + kl_weight * kl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mse  += mse.item()
        total_kl   += kl.item() if isinstance(kl, torch.Tensor) else float(kl)

    n = len(loader)
    return total_loss / n, total_mse / n, total_kl / n


@torch.no_grad()
def validate(
    model: ACTModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for obs, actions, mask in loader:
        obs     = obs.to(device)
        actions = actions.to(device)
        mask    = mask.to(device)
        actions_pred, _ = model(obs, actions)
        mse = ((actions_pred - actions) ** 2).mean(-1)
        mse = (mse * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        total += mse.mean().item()
    return total / max(1, len(loader))


def train_act(cfg_path: str, dataset_override: str | None = None) -> ACTModel:
    """Fő tréning belépési pont."""
    with open(_REPO_ROOT / cfg_path) as f:
        cfg = yaml.safe_load(f)

    # --- Dataset ---
    ds_path = Path(_REPO_ROOT) / (dataset_override or cfg["dataset"]["path"])
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset nem található: {ds_path}\n"
            f"Futtasd: python3 tools/lerobot_export.py --in-dir data/demos/scripted_v1 "
            f"--out-dir data/lerobot/scripted_v1"
        )

    stats_path = ds_path / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None

    chunk_size = cfg["model"]["chunk_size"]
    dataset    = ACTDataset(ds_path, chunk_size=chunk_size, stats=stats)

    n_total = len(dataset)
    n_train = int(n_total * cfg["dataset"]["train_split"])
    n_val   = n_total - n_train

    g = torch.Generator().manual_seed(cfg["dataset"]["seed"])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Train: {n_train} minta | Val: {n_val} minta | Batch: {bs}")

    # --- Device & Model ---
    device = get_device(cfg)
    print(f"Device: {device}")

    model = ACTModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paraméterek: {n_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs   = cfg["training"]["num_epochs"]
    warmup_steps = cfg["training"]["warmup_steps"]
    total_steps  = num_epochs * max(1, len(train_loader))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Checkpoint dir ---
    ckpt_dir = Path(_REPO_ROOT) / cfg["training"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- TensorBoard ---
    writer = None
    if _HAS_TB:
        log_dir = Path(_REPO_ROOT) / cfg["training"]["tensorboard_log"]
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard: {log_dir}")

    # --- Tréning loop ---
    kl_weight  = cfg["model"]["vae"]["kl_weight"]
    grad_clip  = cfg["training"]["grad_clip"]
    save_every = cfg["training"]["save_every_n_epochs"]
    log_every  = cfg["training"]["log_every_n_steps"]

    best_val_loss = float("inf")
    global_step   = 0
    best_ckpt     = ckpt_dir / "best_model.pt"

    n_frames = len(dataset)
    print(f"Dataset: {n_frames} frame | train={n_train} val={n_val} | "
          f"~{n_train // max(1, bs)} batch/epoch")

    print(f"\n{'─'*65}")
    print(f"ACT BC tréning — {num_epochs} epoch | device={device} | "
          f"kl_w={kl_weight}")
    print(f"{'─'*65}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        ep_loss = ep_mse = ep_kl = 0.0
        for obs, actions, mask in train_loader:
            obs     = obs.to(device)
            actions = actions.to(device)
            mask    = mask.to(device)

            actions_pred, kl = model(obs, actions)
            mse_per = ((actions_pred - actions) ** 2).mean(-1)
            mse = (mse_per * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            mse = mse.mean()
            loss = mse + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_mse  += mse.item()
            kl_val   = kl.item() if isinstance(kl, torch.Tensor) else float(kl)
            ep_kl   += kl_val
            global_step += 1

            if writer and global_step % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/mse",  mse.item(),  global_step)
                writer.add_scalar("train/kl",   kl_val,      global_step)
                writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], global_step)

        nb = max(1, len(train_loader))
        avg_loss = ep_loss / nb
        avg_mse  = ep_mse  / nb
        avg_kl   = ep_kl   / nb

        # --- Val ---
        val_loss = validate(model, val_loader, device)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:4d}/{num_epochs} | "
              f"loss={avg_loss:.4f} mse={avg_mse:.4f} kl={avg_kl:.4f} | "
              f"val={val_loss:.4f} | {elapsed:.1f}s | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if writer:
            writer.add_scalar("epoch/train_loss", avg_loss, epoch)
            writer.add_scalar("epoch/val_loss",   val_loss, epoch)

        # --- Checkpoint ---
        if epoch % save_every == 0 or epoch == num_epochs:
            ckpt_path = ckpt_dir / f"act_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "cfg":         cfg,
            }, ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)
                print(f"  💾 Best model → {best_ckpt.name} (val={val_loss:.4f})")

        # --- Régi checkpointok törlése ---
        keep = cfg["training"].get("keep_last_n", 3)
        all_ckpts = sorted(ckpt_dir.glob("act_epoch_*.pt"),
                           key=lambda p: p.stat().st_mtime)
        for old in all_ckpts[:-keep]:
            old.unlink()

    print(f"\n{'─'*65}")
    print(f"✅ Tréning kész! Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: {ckpt_dir}")
    print(f"\nKövetkező: eval (50 ep, MuJoCo)")
    print(f"   python3 tools/eval_act.py --ckpt {ckpt_dir}")

    if writer:
        writer.close()

    return model


# ─── Inference helper (más scriptek importálhatják) ──────────────────────────

def load_policy(ckpt_dir: str | Path, device: str | None = None) -> tuple[ACTModel, dict]:
    """
    Betölti a legjobb checkpointot inference-hez.

    Returns:
        model: eval módban, a megadott device-on
        cfg:   az eredeti training config
    """
    ckpt_dir = Path(ckpt_dir)
    best_w   = ckpt_dir / "best_model.pt"

    # Config betöltése az utolsó epoch checkpointból
    ckpts = sorted(ckpt_dir.glob("act_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"Nincs checkpoint: {ckpt_dir}")

    meta = torch.load(ckpts[-1], map_location="cpu")
    cfg  = meta["cfg"]

    model = ACTModel(cfg)
    weights = torch.load(best_w, map_location="cpu") if best_w.exists() \
              else meta["model_state"]
    model.load_state_dict(weights)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    return model, cfg


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACT BC tréning — Phase 030 F3d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példa:
  python3 tools/train_act.py \\
      --config  configs/bc/act_shelf_stock_v1.yaml \\
      --dataset data/lerobot/scripted_v1
        """,
    )
    parser.add_argument("--config",   required=True,
                        help="YAML config fájl (configs/bc/act_shelf_stock_v1.yaml)")
    parser.add_argument("--dataset",  default=None,
                        help="Dataset könyvtár (felülírja a config dataset.path-t)")
    args = parser.parse_args()

    train_act(args.config, args.dataset)


if __name__ == "__main__":
    main()
