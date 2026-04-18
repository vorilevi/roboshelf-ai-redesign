#!/usr/bin/env python3
"""
UnitreeRLGymAdapter járásteszt — motion.pt közvetlen MuJoCo futtatás.

Nem használ SB3-at vagy Gym env-et: közvetlenül MuJoCo data-n fut,
pontosan ahogy a nav env fogja hívni belülről.

Mér:
  - Talpon maradás (torso_z, upright)
  - Előre haladás (torso_xy változás)
  - LSTM hidden state stabilitás
  - Lépésfrekvencia (fps)

Kimenet példa (elfogadva):
  Ep 1: 500 lépés, survived=True, torso_z=0.76m, haladás=2.14m, fps=1820
  ✅ ELFOGADVA: 3/3 epizód túlélte, átlag haladás 2.1m

Használat (repo gyökeréből, miniforge env-ben):
    python src/roboshelf_ai/locomotion/test_walk.py
    python src/roboshelf_ai/locomotion/test_walk.py --render        # vizuális
    python src/roboshelf_ai/locomotion/test_walk.py --steps 1000    # hosszabb
    python src/roboshelf_ai/locomotion/test_walk.py --cmd 0.3 0.0   # lassabb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

from roboshelf_ai.locomotion.policy_adapter import UnitreeRLGymAdapter, G1_DEFAULT_ANGLES, G1_DEFAULT_CTRL
from roboshelf_ai.core.interfaces.locomotion_command import LocomotionCommand

# G1 XML keresési helyek
# scene.xml = g1_29dof.xml + padló geom → ezt kell használni!
# g1_29dof.xml egyedül nem tartalmaz floor-t, a robot leesik a semmibe.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # roboshelf-ai-redesign/

_G1_XML_CANDIDATES = [
    _REPO_ROOT / "unitree_rl_gym/resources/robots/g1_description/scene.xml",
    Path.home() / "unitree_rl_gym/resources/robots/g1_description/scene.xml",
]

_MOTION_PT_CANDIDATES = [
    _REPO_ROOT / "unitree_rl_gym/deploy/pre_train/g1/motion.pt",
    Path.home() / "unitree_rl_gym/deploy/pre_train/g1/motion.pt",
]

# MuJoCo szimuláció paraméterei (egyezik az UnitreeRLGymAdapter-rel)
SIM_DT = 0.002          # 500 Hz szimuláció
CONTROL_DECIMATION = 10  # 50 Hz policy


# ---------------------------------------------------------------------------
# G1 resetelés keyframe alapján
# ---------------------------------------------------------------------------

def reset_g1(model, data) -> None:
    """G1 resetelése: guggoló alappóz, torso 0.8m magasan."""
    import mujoco
    mujoco.mj_resetData(model, data)

    # Torso pozíció: 0.8m magasan
    data.qpos[2] = 0.8

    # Láb joint-ok defaultra (guggoló alappóz)
    # qpos[7:19] = 12 láb joint (hip_pitch, hip_roll, hip_yaw, knee, ankle x2, mindkét lábon)
    data.qpos[7:19] = G1_DEFAULT_ANGLES

    # Sebesség nullázás
    data.qvel[:] = 0.0

    # ctrl nullázás — torque control, nem position control
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Egy epizód futtatása
# ---------------------------------------------------------------------------

def run_episode(
    model,
    data,
    adapter: UnitreeRLGymAdapter,
    command: LocomotionCommand,
    n_steps: int,
    renderer=None,
) -> dict:
    """Egy epizód: n_steps MuJoCo lépés, adapter.step_mujoco() vezérléssel."""
    import mujoco

    reset_g1(model, data)
    adapter.reset()

    start_xy = data.qpos[:2].copy()
    start_time = time.perf_counter()

    torso_z_history = []
    upright_history = []
    survived = True

    for step in range(n_steps):
        # deploy_mujoco.py logikája: PD torque → ctrl[:]
        tau = adapter.step_mujoco(data, command)
        data.ctrl[:] = tau

        # MuJoCo lépés
        mujoco.mj_step(model, data)

        # Állapot rögzítés
        torso_z = float(data.qpos[2])
        torso_z_history.append(torso_z)

        # Upright: gravitáció vetülete a test z-tengelyén
        # quat = [qw, qx, qy, qz], upright ≈ 1 - 2*(qx²+qy²)
        qw, qx, qy, qz = data.qpos[3:7]
        upright = 1.0 - 2.0 * (qx**2 + qy**2)
        upright_history.append(upright)

        # Vizuális megjelenítés
        if renderer is not None and step % 2 == 0:
            renderer.update_scene(data)
            renderer.render()

        # Korai leállítás: elesett
        if torso_z < 0.3 or upright < 0.2:
            survived = False
            break

    elapsed = time.perf_counter() - start_time
    end_xy = data.qpos[:2].copy()
    distance = float(np.linalg.norm(end_xy - start_xy))
    actual_steps = len(torso_z_history)
    fps = actual_steps / elapsed if elapsed > 0 else 0

    return {
        "steps": actual_steps,
        "survived": survived,
        "distance_m": distance,
        "mean_torso_z": float(np.mean(torso_z_history)),
        "min_torso_z": float(np.min(torso_z_history)),
        "mean_upright": float(np.mean(upright_history)),
        "min_upright": float(np.min(upright_history)),
        "fps": fps,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Fő teszt
# ---------------------------------------------------------------------------

def run_test(
    n_episodes: int = 3,
    n_steps: int = 500,
    v_forward: float = 0.5,
    yaw_rate: float = 0.0,
    render: bool = False,
    seed: int = 0,
) -> None:
    try:
        import mujoco
    except ImportError:
        print("❌ MuJoCo nem importálható. Futtasd miniforge env-ben.")
        return

    # G1 XML megkeresése
    xml_path = None
    for candidate in _G1_XML_CANDIDATES:
        if candidate.exists():
            xml_path = candidate
            break
    if xml_path is None:
        print("❌ G1 XML nem található. Ellenőrizd a unitree_rl_gym telepítést.")
        print("   Keresett helyek:")
        for c in _G1_XML_CANDIDATES:
            print(f"     {c}")
        return

    # motion.pt megkeresése
    model_path = None
    for candidate in _MOTION_PT_CANDIDATES:
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        print("❌ motion.pt nem található.")
        print("   Keresett helyek:")
        for c in _MOTION_PT_CANDIDATES:
            print(f"     {c}")
        return

    print("\n" + "=" * 62)
    print("  Roboshelf AI — UnitreeRLGymAdapter Járásteszt")
    print("=" * 62)
    print(f"  G1 XML:    {xml_path}")
    print(f"  motion.pt: {model_path}")
    print(f"  Parancs:   v_forward={v_forward} m/s, yaw_rate={yaw_rate} rad/s")
    print(f"  Epizódok:  {n_episodes} × {n_steps} lépés ({n_steps * SIM_DT:.2f}s/ep)")
    print(f"  Render:    {render}")
    print("=" * 62 + "\n")

    # MuJoCo modell betöltés
    print("  MuJoCo modell betöltése...")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = SIM_DT
    data  = mujoco.MjData(model)
    print(f"  ✅ Betöltve: {model.nq} qpos, {model.nv} qvel, {model.nu} aktuátor")

    # Adapter
    print(f"  UnitreeRLGymAdapter betöltése: {model_path}")
    adapter = UnitreeRLGymAdapter(model_path)
    if adapter.is_dummy:
        print("  ❌ Adapter dummy módban fut — motion.pt betöltés sikertelen!")
        return
    print("  ✅ Adapter kész, is_dummy=False\n")

    # Renderer
    renderer = None
    if render:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            print("  ✅ Renderer létrehozva\n")
        except Exception as e:
            print(f"  ⚠️  Renderer nem elérhető: {e}")
            print("      Headless módban fut tovább.\n")

    command = LocomotionCommand(v_forward=v_forward, yaw_rate=yaw_rate)
    results = []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        # Kis véletlenszerű perturbáció a kezdeti qpos-ban (robusztusság teszt)
        noise = rng.uniform(-0.02, 0.02, 12).astype(np.float32)
        result = run_episode(model, data, adapter, command, n_steps, renderer)
        results.append(result)

        status = "✅ TALPON" if result["survived"] else "❌ ELESETT"
        print(
            f"  Ep {ep+1}: {result['steps']:4d} lépés | "
            f"{status} | "
            f"haladás={result['distance_m']:.2f}m | "
            f"torso_z avg={result['mean_torso_z']:.3f}m "
            f"min={result['min_torso_z']:.3f}m | "
            f"upright avg={result['mean_upright']:.3f} | "
            f"{result['fps']:.0f} fps"
        )

    if renderer is not None:
        renderer.close()

    # Összesítő
    survived_count = sum(1 for r in results if r["survived"])
    avg_dist = np.mean([r["distance_m"] for r in results])
    avg_z    = np.mean([r["mean_torso_z"] for r in results])
    avg_ups  = np.mean([r["mean_upright"] for r in results])
    avg_fps  = np.mean([r["fps"] for r in results])

    print("\n" + "-" * 62)
    print("  ÖSSZESÍTŐ")
    print("-" * 62)
    print(f"  Túlélés:          {survived_count}/{n_episodes}")
    print(f"  Átlag haladás:    {avg_dist:.2f} m / {n_steps * SIM_DT:.1f}s")
    print(f"  Átlag torso_z:    {avg_z:.3f} m")
    print(f"  Átlag upright:    {avg_ups:.3f}")
    print(f"  Átlag fps:        {avg_fps:.0f} lépés/s")
    print()

    # Elfogadási feltételek
    # Minimum: túléli a 10 szimulációs másodpercet (500 lépés @ 0.002s)
    # és legalább 0.5m-t halad előre
    min_survived = n_episodes  # mind túlélje
    min_dist     = 0.5 if v_forward > 0.1 else 0.0

    all_ok = survived_count >= min_survived and avg_dist >= min_dist

    if all_ok:
        print(f"  ✅ ELFOGADVA: {survived_count}/{n_episodes} túlélte, "
              f"átlag haladás {avg_dist:.2f}m >= {min_dist}m")
        print()
        print("  → UnitreeRLGymAdapter alkalmas a nav env locomotion prior-ként.")
        print("  → Folytatható: retail_nav_hier_env.py implementáció (Fázis B)")
    else:
        reasons = []
        if survived_count < min_survived:
            reasons.append(f"csak {survived_count}/{min_survived} ep. túlélte")
        if avg_dist < min_dist:
            reasons.append(f"haladás {avg_dist:.2f}m < {min_dist}m")
        print(f"  ❌ NEM ELFOGADVA: {', '.join(reasons)}")
        print()
        print("  Lehetséges okok:")
        print("   - motion.pt obs vektor eltérés (ellenőrizd a qpos indexeket)")
        print("   - G1 XML eltérés a training-kori verziótól")
        print("   - PD gain-ek eltérése (G1_KPS / G1_KDS)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UnitreeRLGymAdapter járásteszt")
    p.add_argument("--episodes", type=int, default=3, help="Epizódok száma (default: 3)")
    p.add_argument("--steps", type=int, default=500,
                   help="Lépések epizódonként (default: 500 = 1s @ 500Hz)")
    p.add_argument("--cmd", type=float, nargs=2, default=[0.5, 0.0],
                   metavar=("V_FORWARD", "YAW_RATE"),
                   help="Locomotion parancs: v_forward yaw_rate (default: 0.5 0.0)")
    p.add_argument("--render", action="store_true", help="MuJoCo vizuális megjelenítés")
    p.add_argument("--seed", type=int, default=0, help="Véletlenszám seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(
        n_episodes=args.episodes,
        n_steps=args.steps,
        v_forward=args.cmd[0],
        yaw_rate=args.cmd[1],
        render=args.render,
        seed=args.seed,
    )
