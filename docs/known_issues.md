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

---

## 19. Manip env — target_shelf z=0.97 fizikailag elérhetetlen (Phase 030 F3b)

**Tünet:** Scripted expert 0/140 siker, PPO v9 policy 0% valódi sikerességi arány.
Az evaluations.npz "10% success" látszat: 1/10 ep_len=1 (lépés-1 terminálás lucky seed-del).

**Root cause:** A G1 kar PD kontrollere gravitáció ellen max z≈0.77-0.94m-t tud fenntartani.
- Stock kinematikailag z=0.870-re kerül, fizikai szimulációban z≈0.770-re süllyed
- Target z=0.97m, goal_radius=0.08m → kell stock_z > 0.89m
- Arm max stock z ≈ 0.84-0.90m → borderline elérhetetlen, a training nem konvergál

**Részleges megoldás (Phase 030 F3b):** Target magasság csökkentése 0.97 → 0.85m.

```xml
<!-- scene_manip_sandbox_v2.xml -->
<site name="target_shelf" pos="0.45 0.0 0.85" .../>
```

**Ellenőrzés eredménye:** Scripted expert z=0.85 targettel → **1% SR** (2.0% majd 1.0%).

**Miért nem működik z=0.85 sem:** A settled z=0.77 és target z=0.85 közt Δz=0.08m = goal_radius.
Ehhez sqrt(Δxy²+0.08²) < 0.08 → Δxy ≈ 0 szükséges. A reset range Δxy_max=0.18m → lehetetlen.
Még szűkített reset (±0.03x, ±0.04y) esetén is Δxy_max=0.05m → dist=0.094 > 0.08 → fail.

**Téves megközelítés (goal_radius=0.10 + szűk reset):** 100% SR, de 1-lépéses sikerek.
Oka: stock kinematikai z=0.870 > target z=0.85, és az 1. lépésben még z≈0.87-en van
→ dist < 0.10 → trivialis siker, arm semmit sem csinál.

**Status:** ❌ NYITOTT — lásd #20 (geometriai deadlock) és döntési opciók: known_issues.md#20

---

## 20. Manip env — F3b geometriai deadlock (scripted expert nem tud 500 demo-t gyűjteni)

**Dátum:** 2026-04-29  
**Status:** ❌ DÖNTÉSPONT — F3b blokkolt, három opció nyitott

**A probléma gyökere (kimért fizika):**

| Paraméter | Érték | Forrás |
|---|---|---|
| Stock kinematikai reset z | 0.870m | g1_shelf_stock_env.py line 282 |
| Stock fizikai settled z | ≈0.770m | Mérés: 10+ epizód átlaga |
| Asztal felszín z | 0.730m | scene_manip_sandbox_v2.xml |
| Arm max fenntartható stock z | ≈0.840-0.900m | Mérés: PD ctrl gravitáció ellen |
| Target z (csökkentett) | 0.850m | F3b fix: 0.97→0.85 |
| goal_radius | 0.080m | shelf_stock_v9.yaml |

**Geometriai lehetetlenség:**
- Settled stock z=0.770, target z=0.850 → Δz=0.080 = goal_radius
- Siker feltétele: sqrt(Δxy²+Δz²) < 0.080 → Δxy ≈ 0 szükséges (bármilyen laterális offszet kizár)
- Reset range: x∈[0.35,0.55], y∈[-0.15,0.15] → Δxy_max=0.18m >> Δxy_budget≈0

**Kísérlet-napló (F3b — 2026-04-29):**

| Kísérlet | Konfig | SR | Hibás ok |
|---|---|---|---|
| scripted_expert v1 | target z=0.97, GOAL_RADIUS=0.08 | 0/140 (0%) | stock z=0.415 bug, contact geom névtelen |
| scripted_expert v2 | target z=0.97, fix kontakt detekció | 0/10 (0%) | arm max z<0.89, fizikailag lehetetlen |
| policy_demo_collector | PPO v9 + v11 config (mismatch) | 0/50 (0%) | VecNormalize mismatch |
| policy_demo_collector | PPO v9 + v9 config | 0/50 (0%) | True PPO SR=0% (evaluations.npz artifact) |
| scripted_expert v3 | target z=0.85, GOAL_RADIUS=0.08 | 1/50 (2%) | geometriai deadlock |
| scripted_expert v4 | target z=0.85, GOAL_RADIUS=0.10, szűk reset | 10/10 (100%) | ❌ HAMIS: 1-lépéses trivial success |
| scripted_expert v5 | target z=0.85, GOAL_RADIUS=0.08, MIN_SUCCESS_STEP=25 | 1/100 (1%) | geometriai deadlock megmarad |

**PPO v9 "10% siker" analízis:**
- evaluations.npz shape: (100, 10) — 100 eval × 10 epizód
- ep_len=450.1 = (9×500 + 1×1)/10 → 1 epizód lépés-1-en terminált (lucky seed reset)
- Valódi PPO SR: **0%** — a "10%" nem tanult viselkedés, hanem mérési artifact

**Három nyitott opció:**

**A) Actuator gain fix** — kp értékek 3-4× növelése a MJCF-ben
- Megoldja a root cause-t: arm képes lesz stabilan tartani stock-ot z=0.85+ magasságban
- Fájl: `src/envs/assets/scene_manip_sandbox_v2.xml` → `<position kp="..." />` actuatorok
- Becsült hatás: scripted expert SR 20-40%
- Kockázat: instabilitás, egyéb viselkedés változás

**B) Push task redesign** — target z = 0.77m (settled szint), laterális pozicionálás
- Feladat: stock-ot tolja x,y pozícióba asztali magasságon (nem kell emelni)
- Nincs trivialis 1-lépéses siker: kinematikai z=0.870 → dist_z=0.10 > goal_radius=0.08
- Scripted expert: PUSH stratégia (arm stock magasságán tolja target irányba)
- Becsült SR: 20-40%
- Hátránya: gyengébb demonstráció (nem látványos pick-and-place)

**C) Ugrás F3e-re** — UnifoLM-VLA-0 LoRA fine-tune (Kaggle T4)
- Kihagyja F3b-d-t, VLA baseline direct
- Kockázat: VLA nem tanul a jelenlegi env-ben sem (ugyanaz a fizikai korlát érvényes)

---

## 21a. diag_kp_sweep — QACC mérési hiba (transient vs. steady-state)

**Dátum:** 2026-04-30  
**Status:** ✅ MEGOLDVA — diag_kp_sweep.py v2 javítva

**Tünet:** Első sweep futtatás (kp_sweep_20260430_2146.csv) — minden kp érték ❌, holott a kar stabilan működik.

**Root cause #1 — QACC_MAX=50 threshold téves:**
A `qacc_max` az egész settle+measure fázison futott. A settling tranziensen (step 0-250) QACC=300-500+ — normális PD válasz a nulláról induló mozgáshoz. Steady-state-ben (step>300) QACC=1-15. Az eredeti kód a transiens csúcsot mérte, nem a steady-state-et.

Mért QACC step-enként (pos#0 DEFAULT, kp=150):
- step 0: 393.8 → step 8: 35.8 (transiens) → step 300+: 1-15 (steady-state)

**Root cause #2 — Z_HAND_MIN=0.95 téves:**
A feladat valós igénye z≥0.87m (target z=0.85, kéz kell a fölé). pos#0 (DEFAULT kar pozíció) fizikailag sosem éri el 0.95m-t (max 0.88m bármilyen kp-nél). A 0.95m küszöb hibás volt.

**Root cause #3 — pos#3 `[-0.8, 0.3, -0.3, 1.4]` mindig hibás:**
z_max=0.78-0.81m (kp=10-500 között), geometriailag rossz konfiguráció. Eltávolítva a TEST_POSITIONS-ből.

**Javítás:** QACC mérés csak `measure_steps` fázisban + Z_HAND_MIN=0.87 + QACC_MAX=2000 (csak NaN/Inf).

**Helyesen mért eredmény (kp=150, javított script):**
- pos#0 DEFAULT:           z=0.8746m ✅ | osc=0.00006m ✅ | qacc_ss=14.65 ✅
- pos#1 pitch−:            z=1.0502m ✅ | osc=0.00010m ✅ | qacc_ss=10.22 ✅
- pos#2 pitch−− könyök ny: z=1.2096m ✅ | osc=0.00053m ✅ | qacc_ss=7.47 ✅
- pos#3 neutral roll/yaw:  z=1.0496m ✅ | osc=0.00321m ✅ | qacc_ss=27.47 ✅

**Alkalmazott fix:** `scene_manip_sandbox_v2.xml` arm_motor kp: 10 → 150 (2026-04-30).

---

## 21. Eval metrikák — aggregált ep_lengths.mean() elfedi az outliereket

**Dátum:** 2026-04-29  
**Status:** ✅ MEGOLDVA — `tools/eval_with_metrics.py` implementálva

**Tünet:** SB3 evaluations.npz `ep_lengths.mean()=450.1` → „10% siker" látszat, holott a valódi SR=0%.

**Root cause:** Az aggregált átlag elfedi, hogy 1 epizód step=1-en terminált (lucky reset artifact).
A standard SB3 eval loop nem ad per-epizód breakdown-t és nem szűri ki a rövid sikereket.

Konkrét eset (PPO v9):
- evaluations.npz: (100, 10) — 100 eval × 10 epizód
- ep_lengths[−1] = [500, 500, 500, 500, 500, 500, 500, 500, 500, 1] → mean=450.1
- 1 db step=1 epizód → 10% success látszat
- Valódi tanult SR: **0%**

**Megoldás:** `tools/eval_with_metrics.py` — per-epizód logging + MIN_SUCCESS_STEP guard:
```bash
python3 tools/eval_with_metrics.py --episodes 50
```
Kimenet: `sr_valid` = az igazi metrika (kizárja a step < 25 sikereket).

**Összefüggés:** #18 (lucky reset), #19 (F3b blokkolt)

---

## 22. G1 robot — pelvis rögzített, WBC nincs implementálva

**Dátum:** 2026-04-29  
**Status:** ℹ️ ISMERT KORLÁT — tudatos döntés Phase 030-ban

**Tünet:** A manipulációs env-ben a robot lábai/törzse nem mozog, csak a kar dolgozik.

**Root cause / döntés:** A teljes G1 robot 35 DOF-os (locomotion + kar + ujjak). A Phase 030
manipulációs sandbox-ban a pelvis pozíció és orientáció rögzített (`freejoint` nincs, az alap 
kinematikai fa a padlóhoz van fixálva). Whole-Body Control (WBC) — azaz a lábakkal való 
egyensúlyozás közbeni manipuláció — nincs implementálva.

**Hatás:**
- Előny: kisebb obs/action space, könnyebb convergencia, gyorsabb szimuláció
- Hátrány: nem reális; valós roboton a törzs instabilitása megzavarná a kar mozgását
- A 4-DOF kar workspace-korlátai súlyosabbak, mint ami WBC-vel lenne (test dőlés
  kompenzálhatná a magassági hiányt)

**Jövőbeli munka (Phase 04+):** Freejoint hozzáadása + WBC policy (locomotion + manipulation).
Előfeltétel: stabil manipulációs policy WBC nélkül.

---

## 23. F3b pick-and-place fizikailag lehetetlen — kar felülről nyomja le a stock-ot

**Dátum:** 2026-05-01  
**Status:** ❌ LEZÁRVA → F3c push task-ra pivot

**Tünet:** GRASP phase javítás után (arm → stock_xy, nem target_xy) az SR **0%-ra csökkent** (volt 1%).
Diagnosztika: step 1-ben contact=1.0, stock_z=0.842 (esik), step 20-nál stock_z=0.040 (padlón).

**Root cause — geometriai kényszer:**
A 4-DOF kar (shoulder_pitch/roll/yaw + elbow) konfigurációja miatt a kéz **mindig felülről** közelíti a stock-ot:
- Kéz z ≈ 0.87m (arm_motor kp=150, DEFAULT pos)
- Stock settled z = 0.770m (asztal 0.730 + félmagasság 0.040)
- Kéz z − stock z ≈ 0.10m → a tenyér a stock tetejére kerül

Amikor a kar stock_xy fölé megy, a tenyér a stock tetejét érinti és **lefelé nyomja** azt (nem oldalról fogja meg).
Felvétel sem lehetséges: a kar nem tud oldalsó fogáshoz elegendő laterális közelítést csinálni ezzel a kinematikával.

**Mért diagnosztikai trace (kp=150, GRASP fix):**
```
step  1: contact=1.00, stock_z=0.842 (esik)
step  5: contact=1.00, stock_z=0.780
step 10: contact=1.00, stock_z=0.540
step 15: contact=1.00, stock_z=0.220
step 20: contact=1.00, stock_z=0.040  ← padlón
```

**Az eredeti 1% SR magyarázata:**
A GRASP bug előtt az arm target_xy-ra ment (nem stock_xy-ra). A stock véletlenszerű reset-kor néha
a target közelében landolt, és az arm oldalirányú közelítése esetleg laterálisan érintette a stock-ot → 
véletlen push → 1% siker. Nem tanult viselkedés, hanem geometriai véletlen.

**Pivot döntés: F3c push task**

| Paraméter | F3b (pick&place) | F3c (push) |
|---|---|---|
| Target z | 0.85m | 0.77m (= stock settled z) |
| Szükséges emelés | Δz=0.08m | Nincs (asztali csúsztatás) |
| Scripted expert stratégia | REACH→GRASP→LIFT | APPROACH→PUSH |
| Approach height | — | 0.90m (stock felett, biztonságos) |
| Push height | — | 0.79m (stock középmagassága, laterális) |
| SR (Mac, kp=150) | 0% | **11% (10/91 epizód)** |

**F3c implementáció:**
- `scene_manip_sandbox_v2.xml`: target_shelf z: 0.85 → 0.77
- `tools/scripted_expert.py`: teljes átírás (APPROACH/PUSH/DONE fázisok)
- APPROACH_BEHIND_DIST=0.15m, APPROACH_HEIGHT=0.90m, PUSH_HEIGHT=0.79m, PUSH_THROUGH=0.06m
- Stock reset range kiterjesztve: x∈[0.25, 0.65], y∈[-0.15, 0.15]

**Összefüggés:** #19 (target magasság), #20 (geometriai deadlock döntési pontok), #21a (kp fix)
