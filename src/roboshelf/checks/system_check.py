#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  ROBOSHELF AI — Fázis 2 rendszerellenőrzés                  ║
║  Futtasd: python3 src/roboshelf_phase2_check.py             ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
from pathlib import Path

OK = "  ✅"
WARN = "  ⚠️ "
ERR = "  ❌"

def check(label, ok, msg=""):
    status = OK if ok else ERR
    print(f"{status} {label}" + (f": {msg}" if msg else ""))
    return ok

def check_warn(label, ok, msg=""):
    status = OK if ok else WARN
    print(f"{status} {label}" + (f": {msg}" if msg else ""))
    return ok

errors = []

print("\n" + "="*60)
print("  ROBOSHELF AI — Fázis 2 rendszerellenőrzés")
print("="*60)

# ── Python ──────────────────────────────────────────────────
print("\n── Python ──")
v = sys.version_info
ok = v.major == 3 and v.minor >= 10
if not check(f"Python {v.major}.{v.minor}.{v.micro}", ok, "3.10+ szükséges"):
    errors.append("Python 3.10+ szükséges")

# ── Alapcsomagok ─────────────────────────────────────────────
print("\n── Alapcsomagok ──")

try:
    import mujoco
    check(f"MuJoCo {mujoco.__version__}", True)
    # 3.6+ ellenőrzés
    parts = list(map(int, mujoco.__version__.split(".")[:2]))
    if parts[0] < 3 or (parts[0] == 3 and parts[1] < 6):
        print(f"{WARN} MuJoCo 3.6+ ajánlott (van: {mujoco.__version__})")
except ImportError:
    check("MuJoCo", False, "pip install mujoco>=3.6.0")
    errors.append("mujoco")

try:
    import gymnasium
    check(f"Gymnasium {gymnasium.__version__}", True)
except ImportError:
    check("Gymnasium", False, "pip install gymnasium[mujoco]")
    errors.append("gymnasium[mujoco]")

try:
    import stable_baselines3 as sb3
    check(f"Stable-Baselines3 {sb3.__version__}", True)
except ImportError:
    check("Stable-Baselines3", False, "pip install stable-baselines3>=2.7.0")
    errors.append("stable-baselines3>=2.7.0")

try:
    import torch
    check(f"PyTorch {torch.__version__}", True)
    if torch.backends.mps.is_available():
        print(f"{OK} MPS (Apple Silicon GPU) elérhető 🍎")
    elif torch.cuda.is_available():
        print(f"{OK} CUDA elérhető")
    else:
        print(f"{WARN} Sem MPS, sem CUDA — CPU-n fut (lassabb)")
except ImportError:
    check("PyTorch", False, "pip install torch")
    errors.append("torch")

try:
    import numpy as np
    check(f"NumPy {np.__version__}", True)
except ImportError:
    check("NumPy", False, "pip install numpy")
    errors.append("numpy")

# ── G1 modell ────────────────────────────────────────────────
print("\n── Unitree G1 MJCF modell ──")

g1_found = None
g1_candidates = []

# site-packages alapú keresés
for p in sys.path:
    if "site-packages" in p:
        for subpath in [
            "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1",
            "mujoco_menagerie/unitree_g1",
        ]:
            candidate = Path(p) / subpath
            g1_candidates.append(candidate)
            if (candidate / "g1.xml").exists():
                g1_found = candidate
                break
    if g1_found:
        break

# Homebrew / home keresés
if not g1_found:
    extra = [
        Path("/opt/homebrew/share/mujoco_menagerie/unitree_g1"),
        Path.home() / "mujoco_menagerie" / "unitree_g1",
        Path.home() / "Documents" / "mujoco_menagerie" / "unitree_g1",
    ]
    for c in extra:
        g1_candidates.append(c)
        if (c / "g1.xml").exists():
            g1_found = c
            break

# find parancs fallback
if not g1_found:
    try:
        result = subprocess.run(
            ["find", str(Path.home()), "/opt", "-name", "g1.xml",
             "-path", "*/unitree_g1/*", "-maxdepth", "12"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().splitlines():
            if line.strip():
                g1_found = Path(line.strip()).parent
                break
    except Exception:
        pass

if g1_found:
    check(f"G1 modell: {g1_found}", True)
    # G1 mesh-ek lehetnek .obj, .stl, .msh formátumban
    mesh_dir = g1_found / "meshes"
    if mesh_dir.exists():
        meshes = list(mesh_dir.iterdir())
        check(f"G1 mesh fájlok ({len(meshes)} db, pl. {meshes[0].suffix if meshes else '?'})", len(meshes) > 0)
    else:
        # Néhány menagerie modell inline geom-okat használ mesh helyett — ez OK
        check_warn("G1 meshes/ mappa", False, "inline geom-ok lehetnek — MuJoCo teszt dönti el")

    # ── Kritikus teszt: tényleg betölthető-e a g1.xml? ──
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(g1_found / "g1.xml"))
        check(f"g1.xml betölthető MuJoCo-val ({model.nbody} body, {model.nu} aktuátor)", True)
    except Exception as e:
        check("g1.xml betöltés MuJoCo-val", False, str(e))
        errors.append("g1_xml_load")
else:
    check("G1 modell (g1.xml)", False,
          "pip install mujoco-playground  VAGY\n"
          "     git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie")
    errors.append("g1_model")

# ── Retail bolt XML ─────────────────────────────────────────
print("\n── Retail bolt környezet ──")

# Projekt gyökér meghatározása
script_dir = Path(__file__).resolve().parent
proj_root = script_dir.parent  # src/ szülője = roboshelf-ai/
store_xml = proj_root / "src" / "envs" / "assets" / "roboshelf_retail_store.xml"
g1_scene_xml = proj_root / "src" / "envs" / "assets" / "roboshelf_g1_scene.xml"
nav_env_py = proj_root / "src" / "envs" / "roboshelf_retail_nav_env.py"
train_py = proj_root / "src" / "training" / "roboshelf_phase2_train.py"

check("roboshelf_retail_store.xml", store_xml.exists(), str(store_xml) if not store_xml.exists() else "")
check_warn("roboshelf_g1_scene.xml", g1_scene_xml.exists(), "(opcionális)")
check("roboshelf_retail_nav_env.py", nav_env_py.exists())
check("roboshelf_phase2_train.py", train_py.exists())

# ── Gyors MuJoCo XML validáció ────────────────────────────
if store_xml.exists():
    print("\n── MJCF XML validáció ──")
    try:
        import mujoco
        # Csak a bolt XML-t töltjük be (G1 nélkül)
        model = mujoco.MjModel.from_xml_path(str(store_xml))
        check(f"Bolt XML betölthető ({model.nbody} body, {model.ngeom} geom)", True)
    except Exception as e:
        check("Bolt XML betöltés", False, str(e))
        errors.append("store_xml_invalid")

# ── Összefoglaló ─────────────────────────────────────────────
print("\n" + "="*60)
if not errors:
    print("  ✅ Minden rendben! Futtatás:")
    print()
    print("     python3 src/training/roboshelf_phase2_train.py --level teszt")
    print()
else:
    print(f"  ⚠️  {len(errors)} probléma találva. Megoldás:")
    print()
    if "mujoco" in errors or "gymnasium[mujoco]" in errors or "stable-baselines3>=2.7.0" in errors or "torch" in errors:
        pkgs = [e for e in errors if e not in ("g1_model", "store_xml_invalid")]
        if pkgs:
            print(f"     pip install {' '.join(pkgs)}")
    if "g1_model" in errors:
        print("     pip install mujoco-playground")
        print("     # HA ez nem működik:")
        print("     git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie")
    if "store_xml_invalid" in errors:
        print("     Ellenőrizd a src/envs/assets/roboshelf_retail_store.xml fájlt")
    print()
print("="*60 + "\n")
