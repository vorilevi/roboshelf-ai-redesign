"""
PPO Fine-tune a BC (ACT v3) obs_encoder súlyaiból — F3d fázis.

Architektúra:
    obs_encoder  : BC-ből betöltve (MLP 24→64, LayerNorm, ReLU) — frozen első N lépésig
    policy_head  : Linear(64→5), Gaussian, log_std tanulható param
    value_head   : Linear(64→1)
    obs norm     : z-score (stats.json-ból, baked in)
    action space : [-1, 1]^5 (ScriptedExpert.step() inputja)

Miért PPO a BC-ből:
    - BC v3 val_mse=0.0045 (jó enkóder) DE SR=2% (demo lefedettség hiány)
    - Az enkóder hasznos feature-öket tanult
    - PPO a reward-ból fedezheti fel a teljes state-space-t
    - Encoder warm-start → sokkal kevesebb timestep kell mint scratch PPO

Futtatás (repo gyökeréből):
    python3 tools/train_ppo_from_bc.py \\
        --bc-ckpt results/bc_checkpoints_act_v3 \\
        --stats   data/lerobot/scripted_v1/meta/stats.json

Kimenet:
    results/ppo_from_bc_v1/ppo_stepXXX.pt   — checkpointok
    data/logs/ppo_bc_v1/                     — TensorBoard logok
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_TOOLS_DIR = _HERE.parent

if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import scripted_expert as _exp
from scripted_expert import (
    ScriptedExpert,
    MIN_SUCCESS_STEP,
)
from train_act import load_policy


# ─── Normalizáció ────────────────────────────────────────────────────────────

class ObsNorm(nn.Module):
    """Z-score normalizáció baked in, stats.json-ból."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.FloatTensor(mean))
        self.register_buffer("std",  torch.FloatTensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)


# ─── Policy ──────────────────────────────────────────────────────────────────

class PPOActorCritic(nn.Module):
    """
    Actor-Critic a BC obs_encoder tetején.

        obs → ObsNorm → obs_encoder (64) → policy_head → Gaussian(mean, std)
                                         → value_head  → V(s)
    """

    def __init__(self,
                 obs_encoder: nn.Module,
                 obs_norm: ObsNorm,
                 feat_dim: int = 64,
                 action_dim: int = 5,
                 log_std_init: float = -0.5):
        super().__init__()
        self.obs_norm    = obs_norm
        self.obs_encoder = obs_encoder

        self.policy_head = nn.Linear(feat_dim, action_dim)
        self.log_std     = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.value_head  = nn.Linear(feat_dim, 1)

        # Orthogonal init a policy/value fejeken
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _features(self, obs_t: torch.Tensor) -> torch.Tensor:
        x = self.obs_norm(obs_t)
        return self.obs_encoder(x)

    def forward(self, obs_t: torch.Tensor):
        feat = self._features(obs_t)
        mean = torch.tanh(self.policy_head(feat))          # [-1, 1]
        std  = self.log_std.exp().clamp(1e-4, 2.0)
        val  = self.value_head(feat).squeeze(-1)
        return mean, std, val

    @torch.no_grad()
    def act(self, obs_t: torch.Tensor, deterministic: bool = False):
        mean, std, val = self(obs_t)
        if deterministic:
            return mean.clamp(-1, 1), val, None
        dist   = Normal(mean, std)
        action = dist.sample().clamp(-1.0, 1.0)
        lp     = dist.log_prob(action).sum(-1)
        return action, val, lp

    def evaluate(self, obs_t: torch.Tensor, actions_t: torch.Tensor):
        mean, std, val = self(obs_t)
        dist     = Normal(mean, std)
        log_prob = dist.log_prob(actions_t).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, val


# ─── Dense reward (obs-ból számítható, nem kell env módosítás) ───────────────

def compute_dense_reward(obs_raw: np.ndarray, info: dict) -> float:
    """
    Phase-guided dense reward — az egyszerű dist-reward helyett.
    Komponensek:
      approach_rew : kar közel a 'stock mögötti' pozícióhoz (APPROACH fázis)
      dist_rew     : stock közel a targethez (PUSH fázis)
      contact_rew  : érintkezési bónusz
      success_rew  : siker bónusz
    """
    hand_pos   = obs_raw[0:3]
    stock_pos  = obs_raw[3:6]
    target_pos = obs_raw[6:9]
    contact_f  = obs_raw[23]

    push_vec = target_pos[:2] - stock_pos[:2]
    push_len = float(np.linalg.norm(push_vec)) + 1e-8
    push_dir = push_vec / push_len

    # APPROACH: kar a stock 'mögötti' ponthoz (push iránnyal ellentétes irány, 15cm)
    behind_xy      = stock_pos[:2] - push_dir * 0.15
    hand_to_behind = float(np.linalg.norm(hand_pos[:2] - behind_xy))
    approach_rew   = 1.5 * (1.0 - np.tanh(5.0 * hand_to_behind))

    # PUSH: stock→target közelség
    dist_rew    = 3.0 * (1.0 - np.tanh(5.0 * push_len))
    contact_rew = float(contact_f) * 1.5
    success_rew = 15.0 if info.get("success", False) else 0.0

    return approach_rew + dist_rew + contact_rew + success_rew


# ─── Rollout Buffer ───────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device):
        self.obs       = torch.zeros(n_steps, obs_dim,    device=device)
        self.actions   = torch.zeros(n_steps, action_dim, device=device)
        self.rewards   = torch.zeros(n_steps,             device=device)
        self.dones     = torch.zeros(n_steps,             device=device)
        self.values    = torch.zeros(n_steps,             device=device)
        self.log_probs = torch.zeros(n_steps,             device=device)
        self.ptr       = 0
        self.n_steps   = n_steps

    def add(self, obs, action, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor,
                    gamma: float, gae_lambda: float):
        adv      = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            nv       = last_value if t == self.n_steps - 1 else self.values[t + 1]
            nd       = self.dones[t]
            delta    = self.rewards[t] + gamma * nv * (1 - nd) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - nd) * last_gae
            adv[t]   = last_gae
        returns = adv + self.values
        return adv, returns

    def minibatches(self, batch_size: int, adv: torch.Tensor, ret: torch.Tensor):
        idx = torch.randperm(self.n_steps)
        for s in range(0, self.n_steps, batch_size):
            b = idx[s:s + batch_size]
            yield self.obs[b], self.actions[b], self.log_probs[b], adv[b], ret[b]


# ─── PPO update ──────────────────────────────────────────────────────────────

def ppo_update(policy: PPOActorCritic,
               optimizer: optim.Optimizer,
               buf: RolloutBuffer,
               last_val: torch.Tensor,
               gamma: float,
               gae_lambda: float,
               clip_range: float,
               n_epochs: int,
               batch_size: int,
               ent_coef: float,
               vf_coef: float,
               max_grad_norm: float):
    adv, ret = buf.compute_gae(last_val, gamma, gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    pg_losses, vf_losses, ent_vals = [], [], []
    for _ in range(n_epochs):
        for obs_b, act_b, old_lp_b, adv_b, ret_b in buf.minibatches(batch_size, adv, ret):
            lp, ent, val = policy.evaluate(obs_b, act_b)
            ratio        = torch.exp(lp - old_lp_b)
            pg_loss = -torch.min(
                ratio * adv_b,
                ratio.clamp(1 - clip_range, 1 + clip_range) * adv_b
            ).mean()
            vf_loss  = nn.functional.mse_loss(val, ret_b)
            loss     = pg_loss + vf_coef * vf_loss - ent_coef * ent.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ent_vals.append(ent.mean().item())

    return float(np.mean(pg_losses)), float(np.mean(vf_losses)), float(np.mean(ent_vals))


# ─── Eval ────────────────────────────────────────────────────────────────────

def _build_policy(args, device):
    """BC enkóder + PPO policy felépítése, opcionális checkpoint betöltéssel."""
    bc_model, _ = load_policy(str(args.bc_ckpt))
    obs_encoder  = bc_model.obs_encoder.to(device)

    stats_path = Path(args.stats) if Path(args.stats).is_absolute() \
                 else _REPO_ROOT / args.stats
    with open(stats_path) as f:
        stats = json.load(f)
    obs_mean = np.array(stats["obs"]["mean"], dtype=np.float32)
    obs_std  = np.array(stats["obs"]["std"],  dtype=np.float32) + 1e-8
    obs_norm = ObsNorm(obs_mean, obs_std).to(device)

    policy = PPOActorCritic(obs_encoder, obs_norm, feat_dim=64,
                            action_dim=5, log_std_init=args.log_std_init).to(device)
    start_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        start_steps = ckpt.get("total_steps", 0)
        print(f"Checkpoint betöltve: {args.resume} (lépés: {start_steps:,})")
    return policy, start_steps


def eval_policy(args):
    """50 epizód deterministikus eval, konzolra ír."""
    device = torch.device(args.device)
    policy, _ = _build_policy(args, device)
    policy.eval()

    env = ScriptedExpert(seed=123)
    results, dists = [], []

    print(f"\nEval: 50 epizód | deterministikus | MIN_SUCCESS_STEP={MIN_SUCCESS_STEP}")
    print("─" * 60)
    for ep in range(50):
        obs_np = env.reset()
        success, dist = False, float("inf")
        for step in range(300):
            obs_t  = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            act_t, _, _ = policy.act(obs_t, deterministic=True)
            obs_np, _, done, info = env.step(act_t.squeeze(0).cpu().numpy())
            dist    = info.get("place_dist", dist)
            success = info.get("success", False)
            if done:
                break
        results.append(success)
        dists.append(dist)
        sr_now = 100 * sum(results) / len(results)
        print(f"[{ep+1:2d}/50] {'✅' if success else '❌'}  "
              f"dist={dist:.3f}m  SR={sr_now:.1f}%")

    sr = 100 * sum(results) / 50
    print(f"\n{'═'*60}")
    print(f"EREDMÉNY: {sum(results)}/50  SR={sr:.1f}%  "
          f"átlag_dist={np.mean(dists):.3f}m")
    print(f"{'═'*60}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    # ── Policy (+ opcionális resume) ──
    policy, resumed_steps = _build_policy(args, device)
    total_steps = resumed_steps

    # Enkóder freeze logika (csak ha nem resume-oltunk az unfreeze pont után)
    _encoder_frozen = False
    if args.freeze_encoder_steps > 0 and total_steps < args.freeze_encoder_steps:
        for p in policy.obs_encoder.parameters():
            p.requires_grad_(False)
        _encoder_frozen = True
        print(f"Enkóder FROZEN az első {args.freeze_encoder_steps:,} lépésig")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr, eps=1e-5)

    # Optimizer state visszaállítás (ha van)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                print("Optimizer state visszaállítva ✅")
            except Exception as e:
                print(f"Optimizer state nem kompatibilis, friss init ({e})")

    # ── Env ──
    # Fix pozíció override: module-szintű konstansok felülírása
    if args.fixed_pos_x is not None:
        x = args.fixed_pos_x
        eps = 1e-6   # nulla szélességű range → mindig ugyanaz a pozíció
        _exp.STOCK_RESET_X_RANGE = (x - eps, x + eps)
        y_str = f"{args.fixed_pos_y:.3f}" if args.fixed_pos_y is not None else "free"
        print(f"Fix stock pozíció: x={x:.3f}, y={y_str}")
    if args.fixed_pos_y is not None:
        y = args.fixed_pos_y
        eps = 1e-6
        _exp.STOCK_RESET_Y_RANGE = (y - eps, y + eps)

    env = ScriptedExpert(seed=args.seed)

    # ── Buffer ──
    buf = RolloutBuffer(args.n_steps, 24, 5, device)

    # ── Logging ──
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.log_dir)
        print(f"TensorBoard: {args.log_dir}")
    except Exception:
        pass

    # ── Főciklus ──
    obs_np  = env.reset()
    obs_t   = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
    ep_rew  = 0.0
    ep_len  = 0
    ep_sr_buf: deque = deque(maxlen=100)
    ep_dist_buf: deque = deque(maxlen=100)
    recent_rews: list = []
    update_count  = 0
    t0 = time.time()

    print(f"\nPPO training — {args.total_timesteps:,} lépés | n_steps={args.n_steps} | "
          f"device={args.device}")
    print("─" * 70)

    while total_steps < args.total_timesteps:

        # ── Enkóder unfreeze ──
        if _encoder_frozen and total_steps >= args.freeze_encoder_steps:
            for p in policy.obs_encoder.parameters():
                p.requires_grad_(True)
            _encoder_frozen = False
            # Alacsonyabb LR az enkódernek
            optimizer = optim.Adam([
                {"params": policy.obs_encoder.parameters(), "lr": args.lr * 0.1},
                {"params": policy.policy_head.parameters()},
                {"params": policy.log_std},
                {"params": policy.value_head.parameters()},
            ], lr=args.lr, eps=1e-5)
            print(f"\n[{total_steps:,}] Enkóder UNFROZEN (encoder_lr={args.lr*0.1:.1e})")

        # ── Rollout gyűjtés ──
        buf.ptr = 0
        policy.eval()
        with torch.no_grad():
            for _ in range(args.n_steps):
                action_t, val_t, lp_t = policy.act(obs_t)
                action_np = action_t.squeeze(0).cpu().numpy()

                next_obs_np, reward, done, info = env.step(action_np)

                # Dense reward override (phase-guided, obs-ból számítható)
                if args.dense_reward:
                    reward = compute_dense_reward(next_obs_np, info)

                buf.add(
                    obs_t.squeeze(0),
                    action_t.squeeze(0),
                    torch.tensor(reward,       device=device),
                    torch.tensor(float(done),  device=device),
                    val_t.squeeze(0),
                    lp_t.squeeze(0),
                )

                ep_rew += reward
                ep_len += 1
                total_steps += 1

                if done:
                    ep_sr_buf.append(float(info.get("success", False)))
                    ep_dist_buf.append(info.get("place_dist", float("inf")))
                    recent_rews.append(ep_rew)
                    ep_rew, ep_len = 0.0, 0
                    obs_np = env.reset()
                else:
                    obs_np = next_obs_np

                obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(device)

            # Bootstrap value az utolsó obs-ra
            _, last_val, _ = policy.act(obs_t)
            last_val = last_val.squeeze(0)

        # ── PPO update ──
        policy.train()
        pg_l, vf_l, ent = ppo_update(
            policy, optimizer, buf, last_val,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )
        update_count += 1

        # ── Log ──
        if update_count % args.log_every == 0 and ep_sr_buf:
            sr    = float(np.mean(list(ep_sr_buf))) * 100
            dist  = float(np.mean(list(ep_dist_buf)))
            rew   = float(np.mean(recent_rews[-20:])) if recent_rews else 0.0
            elapsed = time.time() - t0
            fps   = total_steps / elapsed
            print(f"[{total_steps:7,}] SR={sr:5.1f}% | dist={dist:.3f}m | "
                  f"rew={rew:6.2f} | pg={pg_l:.4f} | vf={vf_l:.4f} | "
                  f"ent={ent:.3f} | {fps:.0f}fps")
            if writer:
                writer.add_scalar("ppo/sr_100ep",    sr,    total_steps)
                writer.add_scalar("ppo/dist_mean",   dist,  total_steps)
                writer.add_scalar("ppo/ep_rew_mean", rew,   total_steps)
                writer.add_scalar("ppo/pg_loss",     pg_l,  total_steps)
                writer.add_scalar("ppo/vf_loss",     vf_l,  total_steps)
                writer.add_scalar("ppo/entropy",     ent,   total_steps)

        # ── Checkpoint ──
        if update_count % args.save_every == 0:
            path = ckpt_dir / f"ppo_step{total_steps:07d}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
                "total_steps":       total_steps,
                "args":              vars(args),
            }, path)
            sr_now = float(np.mean(list(ep_sr_buf))) * 100 if ep_sr_buf else 0.0
            print(f"  💾 Checkpoint → {path.name}  (SR={sr_now:.1f}%)")

    # ── Final mentés ──
    final_path = ckpt_dir / "ppo_final.pt"
    torch.save({"policy_state_dict": policy.state_dict(),
                "total_steps": total_steps}, final_path)
    sr_final = float(np.mean(list(ep_sr_buf))) * 100 if ep_sr_buf else 0.0
    print(f"\n{'═'*70}")
    print(f"✅ PPO training kész! → {final_path}")
    print(f"   SR (utolsó 100 ep): {sr_final:.1f}%")
    print(f"   Total steps: {total_steps:,}")
    print(f"{'═'*70}")

    if writer:
        writer.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="PPO fine-tune BC enkóderből — F3d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Futtatás (repo gyökeréből):
  python3 tools/train_ppo_from_bc.py \\
      --bc-ckpt results/bc_checkpoints_act_v3 \\
      --stats   data/lerobot/scripted_v1/meta/stats.json
        """,
    )
    # Kötelező
    p.add_argument("--bc-ckpt", required=True,  help="BC checkpoint könyvtár")
    p.add_argument("--stats",   required=True,  help="stats.json útvonal")

    # Env / device
    p.add_argument("--device",  default="mps",  help="mps / cuda / cpu")
    p.add_argument("--seed",    type=int, default=42)

    # PPO hyperparaméterek
    p.add_argument("--total-timesteps",       type=int,   default=500_000)
    p.add_argument("--n-steps",               type=int,   default=2048,
                   help="Rollout hossza update előtt")
    p.add_argument("--batch-size",            type=int,   default=64)
    p.add_argument("--n-epochs",              type=int,   default=10)
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--gamma",                 type=float, default=0.99)
    p.add_argument("--gae-lambda",            type=float, default=0.95)
    p.add_argument("--clip-range",            type=float, default=0.2)
    p.add_argument("--ent-coef",              type=float, default=0.01)
    p.add_argument("--vf-coef",               type=float, default=0.5)
    p.add_argument("--max-grad-norm",         type=float, default=0.5)
    p.add_argument("--log-std-init",          type=float, default=-0.5)
    p.add_argument("--freeze-encoder-steps",  type=int,   default=50_000,
                   help="Enkóder frozen ennyi lépésig (0 = sosem frozen)")

    # Mentés / log
    p.add_argument("--ckpt-dir",    default="results/ppo_from_bc_v1")
    p.add_argument("--log-dir",     default="data/logs/ppo_bc_v1")
    p.add_argument("--save-every",  type=int, default=20,
                   help="Checkpoint mentés minden N update után")
    p.add_argument("--log-every",   type=int, default=5,
                   help="Log print minden N update után")

    p.add_argument("--resume", default=None,
                   help="Checkpoint útvonal a folytatáshoz (pl. results/ppo_from_bc_v1/ppo_final.pt)")
    p.add_argument("--eval-only", action="store_true",
                   help="Csak 50 epizód eval, nincs tréning")

    # Fix pozíció + dense reward (4 órás F3e teszt)
    p.add_argument("--fixed-pos-x", type=float, default=None,
                   help="Fix stock x pozíció (pl. 0.28) — mindig ugyanaz a reset")
    p.add_argument("--fixed-pos-y", type=float, default=None,
                   help="Fix stock y pozíció (pl. 0.0) — mindig ugyanaz a reset")
    p.add_argument("--dense-reward", action="store_true",
                   help="Phase-guided dense reward: approach+dist+contact+success")

    args = p.parse_args()

    if args.eval_only:
        eval_policy(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
