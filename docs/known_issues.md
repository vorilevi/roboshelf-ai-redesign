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

---

## 10. Manip env — lineáris reward nem konvergál 5M lépésen

**Tünet:** `dist=0.932m konstans` 5M lépés után is, 0% success rate.

**Root cause:** `-w * dist` lineáris reward túl gyenge signal. 0.3m-nél a gradiens azonos mint 1m-nél — a PPO nem tanul. Ráadásul az obs-ból hiányzott a relatív `hand→stock` vektor.

**Megoldás (v7 env):**
- Reward: `1 - tanh(5 * dist)` — bounded [0,1], erős gradiens közel (DeepMind PandaPickCube minta)
- Obs: `hand→stock` és `stock→target` relatív vektorok hozzáadva (obs_dim 18 → 24)
- Grasp: contact force alapú flag az obs-ban (nem magassági heurisztika)
- Ref: `Obsidian: [[Panda-szerű reward-függvényt és obs-designt]]`

---

## 11. Manip env — DEFAULT_ARM_POS rossz, kéz nem éri el a terméket

**Tünet:** Reset után `hand→stock dist=0.39m`, 5M training után sem csökken.

**Root cause:** `shoulder_pitch=+0.5` a kart felfelé-hátrafelé viszi (z=0.685m), a termék z=0.870m → 0.185m vertikális különbség, amit a policy nem tud kompenzálni.

**Megoldás:** Grid search 12^4=20736 kombináción megtalálta az optimumot:
```python
_DEFAULT_ARM_POS = [-1.0, 0.2, -0.2, 1.2]
# → hand→stock dist reset után: 0.025m ✅ (volt 0.39m)
```

---

## 12. Manip env — target a robot mögött (x negatív)

**Tünet:** `stock_target_dist=0.976m konstans`, termék soha nem kerül a targethez.

**Root cause:** `target_shelf pos="-0.41 0 0.415"` — a robot mögött és derék alatt. A 4-DOF kar nem tud 180°-os forgást csinálni.

**Megoldás:** Target a robot előtt, elérhető magasságban:
```xml
<site name="target_shelf" pos="0.45 0.0 0.97" .../>
```
Debug igazolta: `hand→target dist = 0.025m` (elérhető).

---

## 13. train_shelf_stock.py — `--override` flag nem létezik

**Hiba:**
```
train_shelf_stock.py: error: unrecognized arguments: --override ppo.total_timesteps=500000
```

**Root cause:** A script nem támogat generikus `--override` flaget.

**Megoldás:** A `total_timesteps` felülírásához a dedikált flag kell:
```bash
python3 src/roboshelf_ai/tasks/manipulation/train_shelf_stock.py \
  --config configs/manipulation/shelf_stock_v7.yaml \
  --total-timesteps 500000
```

Elérhető flagek: `--config`, `--total-timesteps`, `--n-envs`, `--no-save`.

---

## 14. Manip env — policy collapse 5M lépésen (ent_coef túl alacsony)

**Tünet:** 500k smoke test után `GRASP 60%`, de 5M végén `dist=0.444m` (rosszabb), `LIFT 0%`.

**Root cause:** `ent_coef=0.01` → a policy túl hamar "magabiztos" lett a grasp stratégiában, exploit-olta a grasp reward-ot és soha nem próbálta ki a lift fázist. A lift reward (`w_lift=1.0`) elveszett a grasp zajában.

**Megoldás (v8 config):**
- `ent_coef: 0.01 → 0.05` — több exploráció a teljes training során
- `w_lift: 1.0 → 3.0` — erősebb lift signal
- `lift_trigger_threshold: 0.03m` — lift reward csak valódi emelkedésnél (nem rezgés)
- `w_grasp: 2.0` — marad (grasp már működik, ne erősítsük tovább)

```bash
python3 src/roboshelf_ai/tasks/manipulation/train_shelf_stock.py \
  --config configs/manipulation/shelf_stock_v8.yaml 2>&1 | tee results/manip_5m_v8.log
```
