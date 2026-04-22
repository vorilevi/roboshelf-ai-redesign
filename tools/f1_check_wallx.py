"""
F1 Sanity Check #2 — WALL-OSS / WallX bfloat16 inference

Futtatás (miután a fork klónozva van):
    cd /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign
    python tools/f1_check_wallx.py \
        --repo /Users/vorilevi/roboshelf-ai-dev/wallx_roboshelf

Mit ellenőriz:
  1. A WallX repo fake_inference.py futtatható-e bfloat16-ban Mac M2-n
  2. Memóriahasználat < 16 GB
  3. Output action tensor dimenziója
  4. Inference latencia (ms/call)
  5. SHA pin mentése

Elfogadási feltétel (F1):
  ✅ fake_inference.py fut crash nélkül bfloat16-ban
  ✅ Memóriahasználat dokumentálva
  ✅ SHA pin mentve
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="WALL-OSS / WallX bfloat16 sanity check")
    p.add_argument(
        "--repo",
        default="/Users/vorilevi/roboshelf-ai-dev/wallx_roboshelf",
        help="A fork lokális elérési útja",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-pin", action="store_true")
    return p.parse_args()


def save_sha_pin(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H %ci"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    sha_line = result.stdout.strip()
    out_dir = Path(__file__).parent.parent / "docs" / "dependencies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "wallx_pinned_sha.txt"
    out_file.write_text(
        f"{sha_line}\n"
        f"repo: XSquareRobot/WallX\n"
        f"fork: vorilevi/wallx_roboshelf\n"
        f"pinned: {time.strftime('%Y-%m-%d')}\n"
    )
    print(f"  SHA pin mentve: {out_file}")
    return sha_line


def run_fake_inference(repo_path: Path, dtype: str) -> dict:
    """fake_inference.py futtatása subprocess-ként, hogy izolált legyen."""
    results = {
        "script_found": False,
        "run_ok": False,
        "action_dim": None,
        "latency_ms": None,
        "mem_gb": None,
        "output": "",
        "error": "",
    }

    # fake_inference.py keresése a repóban
    candidates = list(repo_path.rglob("fake_inference.py"))
    if not candidates:
        print("  WARN: fake_inference.py nem található a repóban.")
        print("        Elérhető Python fájlok:")
        for f in sorted(repo_path.rglob("*.py"))[:10]:
            print(f"          {f.relative_to(repo_path)}")
        return results

    script = candidates[0]
    results["script_found"] = True
    print(f"  Script: {script.relative_to(repo_path)}")

    # Futtatás
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script), "--dtype", dtype],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=300,  # 5 perc max (modell betöltés lassú lehet)
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results["output"] = proc.stdout
    results["error"] = proc.stderr
    results["latency_ms"] = elapsed_ms

    if proc.returncode == 0:
        results["run_ok"] = True
        print(f"  Futtatás OK — {elapsed_ms/1000:.1f}s")
        print(f"  stdout:\n{proc.stdout[:500]}")
    else:
        print(f"  HIBA (returncode={proc.returncode})")
        print(f"  stderr:\n{proc.stderr[:500]}")

    return results


def check_memory_usage(repo_path: Path, dtype: str) -> float | None:
    """Memóriahasználat mérése a fake_inference futása közben."""
    try:
        import resource
        # Mac-en: /usr/bin/time -l python fake_inference.py
        candidates = list(repo_path.rglob("fake_inference.py"))
        if not candidates:
            return None

        proc = subprocess.run(
            ["/usr/bin/time", "-l", sys.executable, str(candidates[0]), "--dtype", dtype],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Mac /usr/bin/time -l kimenetéből maximum resident set size kiolvasása
        for line in proc.stderr.splitlines():
            if "maximum resident set size" in line.lower():
                # bytes-ban adja vissza, GB-ra konvertálás
                parts = line.strip().split()
                if parts:
                    mem_bytes = int(parts[0])
                    mem_gb = mem_bytes / (1024 ** 3)
                    return mem_gb
    except Exception:
        pass
    return None


def main():
    args = parse_args()
    repo_path = Path(args.repo)

    print("=" * 60)
    print("F1 Sanity Check #2 — WALL-OSS / WallX bfloat16 inference")
    print("=" * 60)
    print(f"Repo:  {repo_path}")
    print(f"dtype: {args.dtype}")

    if not repo_path.exists():
        print(f"\n❌ HIBA: A repo nem található: {repo_path}")
        print("   Előbb fork + klónozás szükséges:")
        print("   git clone https://github.com/vorilevi/wallx_roboshelf.git \\")
        print(f"     {repo_path}")
        sys.exit(1)

    if not args.no_pin:
        print("\n[SHA pin]")
        sha = save_sha_pin(repo_path)
        print(f"  {sha}")

    print(f"\n[fake_inference.py — {args.dtype}]")
    results = run_fake_inference(repo_path, args.dtype)

    print("\n[Memóriahasználat]")
    mem_gb = check_memory_usage(repo_path, args.dtype)
    if mem_gb is not None:
        ok_mem = mem_gb < 16.0
        print(f"  Max RSS: {mem_gb:.2f} GB {'✅' if ok_mem else '⚠️  > 16 GB!'}")
    else:
        print("  Memória mérés nem elérhető — ellenőrizd Activity Monitor-ban manuálisan")

    # Eredmények mentése
    result_file = Path(__file__).parent.parent / "docs" / "dependencies" / "wallx_sanity_results.txt"
    result_file.write_text(
        f"WALL-OSS / WallX sanity check — {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"dtype: {args.dtype}\n"
        f"script_found: {results['script_found']}\n"
        f"run_ok: {results['run_ok']}\n"
        f"latency_total_ms: {results['latency_ms']:.0f}\n"
        f"mem_gb: {mem_gb}\n"
        f"stdout:\n{results['output']}\n"
    )
    print(f"\n  Részletes eredmény mentve: {result_file}")

    print("\n" + "=" * 60)
    print("ÖSSZESÍTŐ")
    print("=" * 60)
    ok = results["run_ok"]
    print(f"  Script megtalálva: {'✅' if results['script_found'] else '❌'}")
    print(f"  Futtatás OK:       {'✅' if results['run_ok'] else '❌'}")
    print(f"  Latencia:          {results['latency_ms']/1000:.1f}s (teljes betöltéssel)")
    if mem_gb:
        print(f"  Memória:           {mem_gb:.2f} GB")

    if ok:
        print("\n✅ F1 #2 ELFOGADVA — WALL-OSS bfloat16 sanity check sikeres")
        print("   Következő: tools/f1_check_unifolm.py")
    else:
        print("\n❌ F1 #2 SIKERTELEN — lásd a hibát fent")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
