# Ismert hibák és megoldások

_Minden visszatérő hiba ide kerül. Mindig olvasd el mielőtt parancsot adsz ki._

---

## 1. MuJoCo viewer — macOS

**Hiba:**
```
RuntimeError: `launch_passive` requires that the Python script be run under `mjpython` on macOS
```

**Megoldás:** `python3` helyett `mjpython` kell:
```bash
mjpython -c "..."
# vagy
mjpython script.py
```

---

## 2. unitree_rl_mjlab train.py — Mac M2 GPU hiba

**Hiba:**
```
IndexError: list index out of range
```
`select_gpus()` CUDA GPU-t keres, Mac M2-n nincs.

**Megoldás:** `--gpu-ids None` flag kötelező:
```bash
python scripts/train.py Unitree-G1-Flat --gpu-ids None ...
```

---

## 3. unitree_rl_mjlab — num_envs=1 nem konvergál

**Tünet:** `ep_len` csökken vagy stagnál, `fell_over=1.0` végig.

**Megoldás:** Mac M2-n mindig `--env.scene.num-envs 16`:
```bash
python scripts/train.py Unitree-G1-Flat --gpu-ids None --env.scene.num-envs 16 ...
```

---

## 4. roboshelf_common import hiba — kötőjel vs aláhúzás

**Hiba:**
```
ModuleNotFoundError: No module named 'roboshelf_common'
```
A mappa neve `roboshelf-common` (kötőjel), Python nem tudja importálni.

**Megoldás:** `setup.py`-ban `package_dir` fix, editable install:
```bash
pip install -e roboshelf-common/ --break-system-packages
```

---

## 5. MuJoCo QACC NaN/Inf — equality constraint

**Hiba:**
```
WARNING: Nan, Inf or huge value in QACC at DOF 12/29
```

**Root cause:** Equality constraint (`polycoef="0 0 0 0 0"`) fizikailag instabil.

**Megoldás:** Equality constraint blokkot eltávolítani az XML-ből. Passzív jointokat `damping=200 + armature=0.5`-tel rögzíteni. Timestep: `0.001` (1000 Hz).

---

## 6. Manip env — termék fizikailag elérhetetlen

**Tünet:** `dist=1.654m konstans`, `REACH 0%`, kar nem mozdul.

**Root cause:** Termék x pozíciója a Python reset kódban `1.2`-re volt állítva, miközben a kar max reach ≈ 0.58m.

**Megoldás:** `qpos[STOCK_QPOS_START] = 0.45` (nem 1.2). Kéz kezdőpozíció: `shoulder_pitch=0.5, elbow=0.3` (nem `pitch=1.0, elbow=1.0`).

---

## 7. SB3 training log — tee + ProgressBarCallback

**Tünet:** `tee`-vel a progress bar nem látható a terminálban. `tee` nélkül a log nem mentődik.

**Megoldás:** Két terminál tab — mindkettő kell:

- **1. tab:** training `tee`-vel → log megvan
  ```bash
  python3 -c "..." 2>&1 | tee results/LOGFAJL.log
  ```
- **2. tab:** progress követés `tail -f`-fel
  ```bash
  tail -f results/LOGFAJL.log
  ```

---

## 8. Háttérben indított training elvész

**Tünet:** `> log.txt 2>&1 &` után a process nem fut, log nem keletkezik.

**Megoldás:** Mindig előtérben futtasd a traininget. Checkpoint callback gondoskodik a mentésről.

---

## 9. vec_normalize.pkl — 0 byte

**Tünet:** `EOFError` betöltéskor.

**Root cause:** Training megszakadt mielőtt a `vec_env.save()` lefutott.

**Megoldás:** `CheckpointCallback(save_vecnormalize=True)` — minden N lépésnél automatikusan ment.
