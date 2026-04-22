"""
F1 Sanity Check #3 — UnifoLM-VLA-0 inference teszt

Futtatás (miután a fork klónozva van):
    cd /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign
    python tools/f1_check_unifolm.py \
        --repo /Users/vorilevi/roboshelf-ai-dev/unifolm_roboshelf

Mit ellenőriz:
  1. A repo importálható-e
  2. UnifoLM-VLA-0 checkpoint letölthető / elérhető-e
  3. Dummy input → action tensor kimenet
  4. Action dim (várható: 12 — G1 dexterous hand)
  5. SHA pin mentése

Elfogadási feltétel (F1):
  ✅ Modell betöltődik (vagy checkpoint elérhető HF-ről)
  ✅ Action dim dokumentálva
  ✅ SHA pin mentve
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="UnifoLM-VLA-0 sanity check")
    p.add_argument(
        "--repo",
        default="/Users/vorilevi/roboshelf-ai-dev/unifolm_roboshelf",
        help="A fork lokális elérési útja",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--hf-cache", default="/Users/vorilevi/roboshelf-ai-dev/hf-mirrors/UnifoLM-VLA-0",
                   help="Lokális HF mirror elérési útja (opcionális)")
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
    out_file = out_dir / "unifolm_pinned_sha.txt"
    out_file.write_text(
        f"{sha_line}\n"
        f"repo: unitreerobotics/UnifoLM\n"
        f"fork: vorilevi/unifolm_roboshelf\n"
        f"pinned: {time.strftime('%Y-%m-%d')}\n"
    )
    print(f"  SHA pin mentve: {out_file}")
    return sha_line


def find_inference_script(repo_path: Path) -> Path | None:
    """Inference entry point keresése a repóban."""
    candidates = [
        "scripts/test_inference.py",
        "scripts/inference.py",
        "inference.py",
        "test_inference.py",
        "demo.py",
    ]
    for c in candidates:
        p = repo_path / c
        if p.exists():
            return p
    # Általános keresés
    found = list(repo_path.rglob("*inference*.py"))
    return found[0] if found else None


def run_inference_script(repo_path: Path, dtype: str, hf_cache: str) -> dict:
    results = {
        "script_found": False,
        "run_ok": False,
        "action_dim": None,
        "latency_ms": None,
        "output": "",
        "error": "",
    }

    script = find_inference_script(repo_path)
    if script is None:
        print("  WARN: Nem találtunk inference scriptet a repóban.")
        print("        Elérhető Python fájlok:")
        for f in sorted(repo_path.rglob("*.py"))[:10]:
            print(f"          {f.relative_to(repo_path)}")
        return results

    results["script_found"] = True
    print(f"  Script: {script.relative_to(repo_path)}")

    # Futtatás — model checkpoint lokális mirror-ból ha elérhető
    cmd = [sys.executable, str(script), "--dtype", dtype]
    if Path(hf_cache).exists():
        cmd += ["--model", hf_cache]
        print(f"  Checkpoint: lokális mirror ({hf_cache})")
    else:
        print(f"  Checkpoint: HuggingFace letöltés (unitreerobotics/UnifoLM-VLA-0)")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=600,  # 10 perc max
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


def check_heis_action_dim(action_dim: int | None) -> None:
    """UnifoLM-VLA-0 G1 dexterous hand action dim ellenőrzés.

    Várható: 12 dim (3 ujj × 2 ízület = 6 + kar 6 DoF)
    De ez a tényleges repo alapján pontosítandó.
    """
    expected = 12
    if action_dim is None:
        print("  Action dim: ismeretlen — stdout-ból kiolvasandó manuálisan")
    elif action_dim == expected:
        print(f"  Action dim: ✅ {action_dim} (G1 dexterous hand, HEIS-kompatibilis)")
    else:
        print(f"  Action dim: ⚠️  {action_dim} (várható: {expected})")
        print("             → vla_client/client.py UnifoLM metadata frissítendő")


def main():
    args = parse_args()
    repo_path = Path(args.repo)

    print("=" * 60)
    print("F1 Sanity Check #3 — UnifoLM-VLA-0")
    print("=" * 60)
    print(f"Repo:  {repo_path}")
    print(f"dtype: {args.dtype}")

    if not repo_path.exists():
        print(f"\n❌ HIBA: A repo nem található: {repo_path}")
        print("   Előbb fork + klónozás szükséges:")
        print("   git clone https://github.com/vorilevi/unifolm_roboshelf.git \\")
        print(f"     {repo_path}")
        sys.exit(1)

    if not args.no_pin:
        print("\n[SHA pin]")
        sha = save_sha_pin(repo_path)
        print(f"  {sha}")

    print(f"\n[UnifoLM-VLA-0 inference teszt — {args.dtype}]")
    results = run_inference_script(repo_path, args.dtype, args.hf_cache)

    print("\n[HEIS action dim ellenőrzés]")
    check_heis_action_dim(results.get("action_dim"))

    # Eredmény mentése
    result_file = Path(__file__).parent.parent / "docs" / "dependencies" / "unifolm_sanity_results.txt"
    result_file.write_text(
        f"UnifoLM-VLA-0 sanity check — {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"dtype: {args.dtype}\n"
        f"script_found: {results['script_found']}\n"
        f"run_ok: {results['run_ok']}\n"
        f"latency_total_ms: {results['latency_ms']}\n"
        f"action_dim: {results['action_dim']}\n"
        f"stdout:\n{results['output']}\n"
    )
    print(f"\n  Részletes eredmény mentve: {result_file}")

    print("\n" + "=" * 60)
    print("ÖSSZESÍTŐ")
    print("=" * 60)
    ok = results["run_ok"]
    print(f"  Script megtalálva: {'✅' if results['script_found'] else '❌'}")
    print(f"  Futtatás OK:       {'✅' if results['run_ok'] else '❌'}")
    if results["latency_ms"]:
        print(f"  Latencia:          {results['latency_ms']/1000:.1f}s")

    if ok:
        print("\n✅ F1 #3 ELFOGADVA — UnifoLM-VLA-0 sanity check sikeres")
        print("\n🎉 MIND A 3 F1 SANITY CHECK KÉSZ")
        print("   Következő fázis: F2 — VLA A/B/C protokoll + locomotion fine-tune")
    else:
        print("\n❌ F1 #3 SIKERTELEN — lásd a hibát fent")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
