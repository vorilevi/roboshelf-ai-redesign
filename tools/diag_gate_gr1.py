"""
diag_gate_gr1.py — GR1T1 scripted expert paraméter-szonda.

Az 5. lépés kapujának három metrikáját méri egy paraméterkészletre, a repo
fájljainak módosítása NÉLKÜL (a konstansokat futásidőben írja felül).

Kapu-kritériumok:
    push-frame arány  >= 35%
    scripted expert SR >= 95%
    start-state lefedettség: a DEFAULT_ARM_POS-ból induló átmenet benne van

Futtatás:
    python3 tools/diag_gate_gr1.py --n 100 --push-gain 0.5 --max-dq 0.02
    python3 tools/diag_gate_gr1.py --n 100 --json   # gépi kimenet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import mujoco  # noqa: E402
from roboshelf_ai.mujoco.envs.manipulation.gr1_shelf_stock_env import (  # noqa: E402
    GR1ShelfStockEnv,
    ARM_QPOS_INDICES,
    _DEFAULT_ARM_POS,
    _JOINT_RANGES,
)

# ---------------------------------------------------------------------------
# A scripted expert újraimplementálva, paraméterezhetően.
# Logikailag azonos a tools/scripted_expert_gr1.py-vel (3d89f49 állapot).
# ---------------------------------------------------------------------------

_MID = (_JOINT_RANGES[:, 0] + _JOINT_RANGES[:, 1]) / 2.0
_HALF = (_JOINT_RANGES[:, 1] - _JOINT_RANGES[:, 0]) / 2.0

PUSH_ARM_SITE_XYZ = np.array([0.395, -0.264, 0.840], dtype=np.float64)
DEFAULT_ARM_NORM = np.clip(
    (_DEFAULT_ARM_POS - _MID) / (_HALF + 1e-6), -1.0, 1.0
).astype(np.float32)


def joint_to_norm(q: np.ndarray) -> np.ndarray:
    return np.clip((q - _MID) / (_HALF + 1e-6), -1.0, 1.0).astype(np.float32)


def jacobian(model, data, site_id) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, ARM_QPOS_INDICES]


PUSH_ARM_POS = np.array([-0.654, 0.111, 1.875, 0.500], dtype=np.float64)


class Controller:
    """SETTLE → PUSH.

    settle_mode="cartesian": J-transpose szervó PUSH_ARM_SITE_XYZ felé (3d89f49).
        Instabil — a kéz oszcillál/elsodródik, nem konvergál.
    settle_mode="joint": lineáris interpoláció a kiinduló ízületszögekből a
        PUSH_ARM_POS-ba settle_steps lépés alatt. Minden lépés más akció, de
        determinisztikus, és a végpont a kp=150 PD által stabilan tartott póz.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._step = 0
        self._q_start = None

    def reset(self):
        self._step = 0
        self._q_start = None

    def _jt(self, model, data, site_id, delta_xyz, gain, max_dq):
        J = jacobian(model, data, site_id)
        dq = np.clip(gain * J.T @ delta_xyz, -max_dq, max_dq)
        q = np.array([data.qpos[i] for i in ARM_QPOS_INDICES], dtype=np.float64)
        return np.clip(q + dq, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1]).astype(np.float32)

    def act(self, obs, model, data, site_id):
        self._step += 1
        c = self.cfg
        hand = obs[0:3].astype(np.float64)

        if self._step <= c["settle_steps"]:
            if c.get("settle_mode", "cartesian") == "const":
                # A 3d89f49 ELŐTTI viselkedés: konstans PUSH_ARM_POS parancs.
                # Kontroll-mérés — ennek a dokumentált 99,6% SR-t kell adnia.
                return joint_to_norm(PUSH_ARM_POS)
            if c.get("settle_mode", "cartesian") == "joint":
                if self._q_start is None:
                    self._q_start = np.array(
                        [data.qpos[i] for i in ARM_QPOS_INDICES], dtype=np.float64
                    )
                alpha = self._step / float(c["settle_steps"])
                q_cmd = self._q_start + alpha * (PUSH_ARM_POS - self._q_start)
                return joint_to_norm(
                    np.clip(q_cmd, _JOINT_RANGES[:, 0], _JOINT_RANGES[:, 1])
                )
            d = PUSH_ARM_SITE_XYZ - hand
            return joint_to_norm(
                self._jt(model, data, site_id, d, c["settle_gain"], c["settle_max_dq"])
            )

        stock = obs[3:6].astype(np.float64)
        target = obs[6:9].astype(np.float64)
        desired_y = target[1] + c["push_overshoot"]
        desired_z = stock[2]

        d = np.array([0.0, desired_y - hand[1], desired_z - hand[2]])
        # Push-lépéshossz korlátozása: a delta_xyz normalizálása a kívánt
        # lépésméretre, hogy a push ne "egy lépésben" történjen meg.
        if c["push_step_m"] is not None:
            n = np.linalg.norm(d)
            if n > c["push_step_m"]:
                d = d / n * c["push_step_m"]

        return joint_to_norm(
            self._jt(model, data, site_id, d, c["push_gain"], c["push_max_dq"])
        )


DEFAULTS = dict(
    settle_mode="cartesian",
    settle_steps=80,
    settle_gain=5.0,
    settle_max_dq=0.10,
    push_gain=5.0,
    push_max_dq=0.10,
    push_overshoot=0.03,
    push_step_m=None,   # None = a repo jelenlegi viselkedése (nincs korlát)
    settle_skip=0,
)


def run(cfg: dict, n_episodes: int, seed_base: int = 0, cap: int = 500) -> dict:
    """cap: harness-oldali lépéskorlát (a gyors feltérképezéshez).
    Nem az env-et módosítja — csak korábban vágja el a rollout-ot, és a
    levágott epizód kudarcnak számít, ahogy a valódi 500-as truncation is."""
    env = GR1ShelfStockEnv()
    ctrl = Controller(cfg)

    obs_eps, act_eps, succ, lens = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        ctrl.reset()
        o_list, a_list = [obs], []
        done = False
        info = {}
        while not done:
            a = ctrl.act(obs, env._model, env._data, env._hand_site_id)
            obs, r, term, trunc, info = env.step(a)
            o_list.append(obs)
            a_list.append(a)
            done = term or trunc or len(a_list) >= cap
        obs_eps.append(np.array(o_list[:-1], dtype=np.float32))
        act_eps.append(np.array(a_list, dtype=np.float32))
        succ.append(bool(info.get("placed", False)))
        lens.append(len(a_list))
    _CAP = cap

    env.close()

    succ = np.array(succ)
    lens = np.array(lens)
    ok = np.where(succ)[0]

    # ---- metrikák (csak a sikeres, exportált epizódokon) ----
    skip = cfg["settle_skip"]
    settle = cfg["settle_steps"]

    total_frames = 0
    push_frames = 0
    const_same = 0
    const_tot = 0
    start_err = []

    for i in ok:
        a = act_eps[i][skip:]
        T = len(a)
        if T == 0:
            continue
        total_frames += T
        # push frame = az eredeti epizódban a settle utáni lépések
        push_frames += max(0, lens[i] - max(settle, skip))
        if T > 1:
            dif = np.abs(np.diff(a, axis=0)).max(axis=1)
            const_same += int((dif < 1e-6).sum())
            const_tot += len(dif)
        # start-state: az első exportált frame kar-állapota mennyire van
        # a DEFAULT_ARM_POS-tól (obs[15:19] = normalizált joint pos)
        start_err.append(
            float(np.abs(obs_eps[i][skip][15:19] - DEFAULT_ARM_NORM).max())
        )

    sr = 100.0 * succ.mean()
    push_ratio = 100.0 * push_frames / max(total_frames, 1)
    const_ratio = 100.0 * const_same / max(const_tot, 1)
    start_cov = float(np.mean(start_err)) if start_err else float("nan")

    return dict(
        sr=sr,
        push_ratio=push_ratio,
        const_ratio=const_ratio,
        start_err=start_cov,
        n_success=int(succ.sum()),
        n_episodes=n_episodes,
        total_frames=int(total_frames),
        avg_len_success=float(lens[ok].mean()) if len(ok) else float("nan"),
        n_timeout=int((lens >= _CAP).sum()),
        gate_push=push_ratio >= 35.0,
        gate_sr=sr >= 95.0,
        gate_start=start_cov < 0.05,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle-mode", type=str, choices=["cartesian", "joint", "const"])
    p.add_argument("--settle-steps", type=int)
    p.add_argument("--settle-gain", type=float)
    p.add_argument("--settle-max-dq", type=float)
    p.add_argument("--push-gain", type=float)
    p.add_argument("--push-max-dq", type=float)
    p.add_argument("--push-overshoot", type=float)
    p.add_argument("--push-step-m", type=float)
    p.add_argument("--settle-skip", type=int)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--cap", type=int, default=500)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    cfg = dict(DEFAULTS)
    for k in list(cfg.keys()):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v

    t0 = time.time()
    m = run(cfg, a.n, a.seed, cap=a.cap)
    m["elapsed_s"] = round(time.time() - t0, 1)
    m["cfg"] = cfg
    m["label"] = a.label

    if a.json:
        print(json.dumps(m))
        return

    g = lambda b: "✅" if b else "❌"  # noqa: E731
    print(f"\n[{a.label or 'probe'}]  n={a.n}  ({m['elapsed_s']}s)")
    print(f"  cfg: {cfg}")
    print(f"  {g(m['gate_sr'])} expert SR      : {m['sr']:.1f}%   (cél ≥95%)")
    print(f"  {g(m['gate_push'])} push-arány     : {m['push_ratio']:.1f}%   (cél ≥35%)")
    print(f"  {g(m['gate_start'])} start-state err: {m['start_err']:.4f}  (cél <0.05)")
    print(f"     konstans akció : {m['const_ratio']:.1f}%")
    print(f"     avg_len (succ) : {m['avg_len_success']:.1f}   timeout: {m['n_timeout']}")
    print(f"     frame összesen : {m['total_frames']}")
    allg = m["gate_sr"] and m["gate_push"] and m["gate_start"]
    print(f"  KAPU: {'✅ ÁTMEGY' if allg else '❌ nem megy át'}")


if __name__ == "__main__":
    main()
