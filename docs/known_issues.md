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

**Manip sandbox scene interaktív megjelenítése** (repo gyökeréből):
```bash
mjpython -c "
import mujoco, mujoco.viewer, time
m = mujoco.MjModel.from_xml_path('src/envs/assets/scene_manip_sandbox_v2.xml')
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        v.sync()
        elapsed = time.time() - step_start
        remaining = m.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)
"
```
✅ Működik — interaktív viewer nyílik, realtime sebességgel fut, körbe lehet járni a scene-t.

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

**Tapasztalat:** v8 ent_coef=0.05 is rossz volt (túl random). Az optimum: **ent_coef=0.03**.

---

## 15. Manip env — ujjak nem mozognak (passive_joint damping)

**Tünet:** `viz_gripper_test.py` futtatásakor az ujjak egyáltalán nem hajlanak, ctrl target ellenére.

**Root cause:** Az ujj jointek `class="passive_joint"` (damping=200, armature=0.5) — a position actuator ereje (kp=50, forcerange=±5N) nem győzi le a csillapítást. qpos alig változik.

**Megoldás:** A 7 jobb kéz ujj joint class-át `"finger_joint"`-ra kell változtatni (damping=0.1, armature=0.001):
```xml
<!-- ELŐTTE: -->
<joint name="right_hand_thumb_0_joint" class="passive_joint" .../>
<!-- UTÁNA: -->
<joint name="right_hand_thumb_0_joint" class="finger_joint" .../>
```
Érintett jointek: `right_hand_thumb_0/1/2`, `right_hand_middle_0/1`, `right_hand_index_0/1`.

**Ellenőrzés:**
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('src/envs/assets/scene_manip_sandbox_v2.xml')
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
d.ctrl[4:11] = [-0.8, 0.5, -1.2, 1.3, 1.5, 1.3, 1.5]
for _ in range(2000): mujoco.mj_step(m, d)
jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_hand_thumb_0_joint')
print('qpos:', d.qpos[m.jnt_qposadr[jid]])  # kell: ~-0.80
"
```

---

## 16. Manip env — _denorm_action shape mismatch (v9+)

**Hiba:**
```
ValueError: operands could not be broadcast together with shapes (5,) (4,)
```

**Root cause:** `_denorm_action(action)` az egész 5-dimenziós action vektort kapja, de `_JOINT_RANGES` csak 4 sort tartalmaz (csak a kar DOF-ok).

**Megoldás:** Mindig csak a kar slice-t adjuk át:
```python
# step()-ben és _compute_reward()-ban:
target_pos = self._denorm_action(action[:4])   # NEM action!
gripper_signal = float(np.clip(action[4], -1.0, 1.0))
```

---

## 17. Manip env — w_smooth túl erős → mozdulatlanság

**Tünet:** Training ep_rew_mean -120-ról indul (v9-nél -17 volt), policy REACH fázisban ragad, 0% siker.

**Root cause:** `w_smooth=-0.01` — minden lépésben tízszeres büntetés az akcióváltásra. A policy megtanulta, hogy a legkisebb büntetés = mozdulatlanság. A kéz nem közelít a dobozhoz.

**Megoldás:** `w_smooth` max értéke: **-0.001**. Ennél erősebb nem alkalmazható ebben a reward struktúrában.

---

## 18. Manip env — 1-lépéses sikerek reset véletlenből

**Tünet:** Eval-ban `lépés=1 | dist=0.044m | ✅ ELHELYEZVE` — a policy semmit nem csinált, a reset véletlenül jó pozícióba tette a dobozt.

**Root cause:** A reset nem ellenőrizte a kéz-doboz kezdeti távolságot.

**Megoldás:** Min 15cm távolság garantálása reset-kor (50 próbán belül mindig található érvényes pozíció):
```python
MIN_START_DIST = 0.15
for _ in range(50):
    stock_x = float(rng.uniform(0.35, 0.55))
    stock_y = float(rng.uniform(-0.15, 0.15))
    # ... pozíció beállítása ...
    mujoco.mj_forward(self._model, self._data)
    if np.linalg.norm(hand_pos - stock_pos) >= MIN_START_DIST:
        break
```
