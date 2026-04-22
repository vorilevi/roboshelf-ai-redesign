"""
F1 Sanity Check #1 — unitree_rl_mjlab G1 environment

Futtatás (miután a fork klónozva van):
    cd /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign
    python tools/f1_check_unitree_rl_mjlab.py \
        --repo /Users/vorilevi/roboshelf-ai-dev/unitree_rl_mjlab_roboshelf

Mit ellenőriz:
  1. A repo importálható-e
  2. A G1 MJCF betöltődik-e MuJoCo-ban
  3. 100 lépés fut-e crash nélkül
  4. Az observation dim HEIS-kompatibilis-e (27 proprioceptive + 12 command + 3 sensor)
  5. SHA pin mentése docs/dependencies/unitree_rl_mjlab_pinned_sha.txt-be

Elfogadási feltétel (F1):
  ✅ 100 lépés crash nélkül
  ✅ obs dim dokumentálva
  ✅ SHA pin mentve
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="unitree_rl_mjlab G1 sanity check")
    p.add_argument(
        "--repo",
        default="/Users/vorilevi/roboshelf-ai-dev/unitree_rl_mjlab_roboshelf",
        help="A fork lokális elérési útja",
    )
    p.add_argument("--steps", type=int, default=100, help="Futtatandó lépések száma")
    p.add_argument("--no-pin", action="store_true", help="SHA pin mentés kihagyása")
    return p.parse_args()


def save_sha_pin(repo_path: Path) -> str:
    """Aktuális HEAD SHA mentése docs/dependencies/-be."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H %ci"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    sha_line = result.stdout.strip()

    out_dir = Path(__file__).parent.parent / "docs" / "dependencies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "unitree_rl_mjlab_pinned_sha.txt"
    out_file.write_text(
        f"{sha_line}\n"
        f"repo: unitree-robotics/unitree_rl_mjlab\n"
        f"fork: vorilevi/unitree_rl_mjlab_roboshelf\n"
        f"pinned: {time.strftime('%Y-%m-%d')}\n"
    )
    print(f"  SHA pin mentve: {out_file}")
    return sha_line


def run_g1_sanity(repo_path: Path, steps: int) -> dict:
    """G1 environment betöltés és lépések futtatása."""
    sys.path.insert(0, str(repo_path))

    results = {
        "import_ok": False,
        "env_load_ok": False,
        "steps_ok": False,
        "obs_dim": None,
        "action_dim": None,
        "control_freq_hz": None,
        "crash": None,
    }

    # 1. Import teszt
    try:
        # A tényleges modul neve a repo struktúrájától függ — ezt F1-ben pontosítjuk
        # Lehetséges entry pointok (prioritás sorrendben):
        import_candidates = [
            ("legged_gym.envs.unitree.g1_env", "G1Env"),
            ("unitree_rl_gym.envs.g1", "G1Env"),
            ("envs.g1_env", "G1Env"),
        ]
        env_class = None
        for module_path, class_name in import_candidates:
            try:
                mod = __import__(module_path, fromlist=[class_name])
                env_class = getattr(mod, class_name)
                results["import_ok"] = True
                print(f"  Import OK: {module_path}.{class_name}")
                break
            except (ImportError, AttributeError):
                continue

        if env_class is None:
            print("  WARN: Nem találtunk ismert G1 env osztályt.")
            print("        Próbáld meg kézzel: python -c 'import <module>' a repo gyökeréből.")
            print("        Frissítsd ezt a scriptet az F1 sanity check után.")
            results["import_ok"] = False
            return results

    except Exception as e:
        results["crash"] = str(e)
        return results

    # 2. Env betöltés
    try:
        env = env_class()
        results["env_load_ok"] = True
        print(f"  Env betöltve: {env_class.__name__}")
    except Exception as e:
        print(f"  HIBA env betöltéskor: {e}")
        results["crash"] = str(e)
        return results

    # 3. Observation / action dim kiolvasása
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        results["obs_dim"] = obs_space.shape[0] if hasattr(obs_space, "shape") else None
        results["action_dim"] = act_space.shape[0] if hasattr(act_space, "shape") else None
        print(f"  obs_dim:    {results['obs_dim']}")
        print(f"  action_dim: {results['action_dim']}")
    except AttributeError:
        print("  WARN: observation_space / action_space nem érhető el — manuálisan ellenőrizd.")

    # 4. Lépések futtatása
    try:
        obs, _ = env.reset()
        t0 = time.perf_counter()
        for i in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        elapsed = time.perf_counter() - t0
        results["steps_ok"] = True
        print(f"  {steps} lépés OK — {elapsed:.2f}s ({steps/elapsed:.0f} lépés/s)")
    except Exception as e:
        print(f"  HIBA lépések futtatásakor: {e}")
        results["crash"] = str(e)

    return results


def heis_check(obs_dim: int | None) -> None:
    """HEIS obs dim kompatibilitás ellenőrzése."""
    # HEIS v1.0: 27 proprioceptive + 12 command + 3 sensor = 42 dim összesen
    # De a unitree_rl_mjlab saját dimenziókat használhat — dokumentáljuk, nem blokkoljuk
    heis_expected = 27 + 12 + 3
    if obs_dim is None:
        print("  HEIS check: obs_dim ismeretlen — manuálisan ellenőrizd")
    elif obs_dim == heis_expected:
        print(f"  HEIS check: ✅ obs_dim={obs_dim} pontosan HEIS-kompatibilis")
    else:
        print(f"  HEIS check: ⚠️  obs_dim={obs_dim} (HEIS elvárás: {heis_expected})")
        print("             → A heis_adapter.obs_to_heis() leképezést ennek megfelelően implementálni")


def main():
    args = parse_args()
    repo_path = Path(args.repo)

    print("=" * 60)
    print("F1 Sanity Check #1 — unitree_rl_mjlab G1 environment")
    print("=" * 60)
    print(f"Repo: {repo_path}")

    if not repo_path.exists():
        print(f"\n❌ HIBA: A repo nem található: {repo_path}")
        print("   Előbb fork + klónozás szükséges:")
        print("   git clone https://github.com/vorilevi/unitree_rl_mjlab_roboshelf.git \\")
        print(f"     {repo_path}")
        sys.exit(1)

    # SHA pin
    if not args.no_pin:
        print("\n[SHA pin]")
        sha = save_sha_pin(repo_path)
        print(f"  {sha}")

    # Sanity run
    print(f"\n[G1 env teszt — {args.steps} lépés]")
    results = run_g1_sanity(repo_path, args.steps)

    # HEIS check
    print("\n[HEIS kompatibilitás]")
    heis_check(results.get("obs_dim"))

    # Összesítő
    print("\n" + "=" * 60)
    print("ÖSSZESÍTŐ")
    print("=" * 60)
    ok = results["steps_ok"]
    print(f"  Import:      {'✅' if results['import_ok'] else '❌'}")
    print(f"  Env betöltés:{'✅' if results['env_load_ok'] else '❌'}")
    print(f"  {args.steps} lépés:    {'✅' if results['steps_ok'] else '❌'}")
    if results.get("crash"):
        print(f"  Hiba: {results['crash']}")

    if ok:
        print("\n✅ F1 #1 ELFOGADVA — unitree_rl_mjlab G1 sanity check sikeres")
        print("   Következő: tools/f1_check_wallx.py")
    else:
        print("\n❌ F1 #1 SIKERTELEN — lásd a hibát fent")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
