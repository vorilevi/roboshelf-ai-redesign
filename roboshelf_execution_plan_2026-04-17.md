# Roboshelf AI Redesign — Részletes végrehajtási terv

_Létrehozva: 2026-04-17. Frissítendő minden jelentős döntés vagy lépés után._

---

## Alapelvek

Ez a dokumentum a `roboshelf_redesign_master_plan_2026-04-17.md` elveiből indul ki, és azokat konkrét, sorban végrehajtható lépésekre bontja le. A két dokumentum együtt olvasandó: a master plan adja az architektúrát és az indoklást, ez a fájl adja a konkrét parancsokat, fájlneveket és ellenőrző pontokat.

Minden lépésnél érvényes:
- A parancsok mindig a repo gyökeréről futnak: `cd /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign`
- Git push kizárólag a saját Mac terminálból, soha nem sandboxból
- `roboshelf-results/` tartalma nem kerül Gitbe
- Az ütemezési fájl (`roboshelf_schedule.md`) minden munkacsomag végén frissítendő

---

## 0. Munkacsomag — Előkészítés és repo-tisztítás

**Cél:** Rendezett kiindulóállapot, amelyből az új architektúra felépíthető.

### 0.1 Jelenlegi állapot felmérése

Ellenőrizd, hogy mely régi fájlok futtathatók még, és melyek töröttek:

```bash
# System check futtatása
python src/roboshelf/checks/system_check.py

# Vagy a régi belépési pont
python src/roboshelf_phase2_check.py
```

Várt eredmény: MuJoCo, Gymnasium, SB3, PyTorch verziók kinyomtatva, G1 model betöltve hiba nélkül.

### 0.2 Régi fájlok archiválása

A régi monolitikus fájlok ne törlődjenek — kerüljenek egy `src/legacy/` mappába.

```bash
mkdir -p src/legacy

# Régi train scriptek archiválása
cp src/training/roboshelf_phase2_train.py src/legacy/train_phase2_monolithic.py
cp src/training/roboshelf_phase2_finetune.py src/legacy/finetune_phase2_monolithic.py

# Régi env archiválása
cp src/envs/roboshelf_retail_nav_env.py src/legacy/retail_nav_env_monolithic.py
cp src/envs/roboshelf_manipulation_env.py src/legacy/manipulation_env_monolithic.py

# Megjegyzés fájl a legacy mappához
cat > src/legacy/README.md << 'EOF'
# Legacy fájlok

Ezek a fájlok a Phase 2 monolitikus rendszer archivált másolatai.
Ne töröld őket — baseline referenciának és regresszióteszthez kellenek.

Nem aktívan fejlesztett kód. Az aktív redesign a következő mappákban van:
- src/roboshelf_ai/ (fő csomag)
- src/roboshelf/ (segédmodulok)
EOF
```

### 0.3 Új mappastruktúra létrehozása

```bash
# Locomotion alrendszer
mkdir -p src/roboshelf_ai/locomotion
touch src/roboshelf_ai/locomotion/__init__.py

# Navigation task alrendszer
mkdir -p src/roboshelf_ai/tasks/navigation
touch src/roboshelf_ai/tasks/navigation/__init__.py

# Manipulation task alrendszer
mkdir -p src/roboshelf_ai/tasks/manipulation
touch src/roboshelf_ai/tasks/manipulation/__init__.py

# Interfészek (locomotion_command és társai)
mkdir -p src/roboshelf_ai/core/interfaces
touch src/roboshelf_ai/core/interfaces/__init__.py

# Reward komponensek
mkdir -p src/roboshelf_ai/core/rewards
touch src/roboshelf_ai/core/rewards/__init__.py

# Env-ek az új struktúrában
mkdir -p src/roboshelf_ai/mujoco/envs/locomotion
mkdir -p src/roboshelf_ai/mujoco/envs/navigation
mkdir -p src/roboshelf_ai/mujoco/envs/manipulation
touch src/roboshelf_ai/mujoco/envs/__init__.py
touch src/roboshelf_ai/mujoco/envs/locomotion/__init__.py
touch src/roboshelf_ai/mujoco/envs/navigation/__init__.py
touch src/roboshelf_ai/mujoco/envs/manipulation/__init__.py

# Config mappák
mkdir -p configs/locomotion configs/navigation configs/manipulation configs/demo

# Demo réteg
mkdir -p src/roboshelf_ai/demo
touch src/roboshelf_ai/demo/__init__.py

# Data mappa
mkdir -p data/demonstrations data/exports data/logs
```

### 0.4 .gitignore ellenőrzése

```bash
cat .gitignore
```

Kötelezően benne kell legyen (ha hiányzik, add hozzá):

```
roboshelf-results/
tmp/
data/demonstrations/
data/exports/
*.npz
*.zip
__pycache__/
*.pyc
.venv/
```

### 0.5 Git commit

```bash
git add -A
git commit -m "chore: archive legacy Phase2 files, scaffold redesign directory structure"
git push origin main
```

**Elfogadási feltétel:** `python src/roboshelf/checks/system_check.py` fut hiba nélkül, az új mappastruktúra létezik, a legacy fájlok archiválva vannak.

---

## 1. Munkacsomag — Locomotion interfész és command layer

**Cél:** Definiálni azt az interfészt, amelyen keresztül a high-level nav policy parancsol a low-level locomotion policy-nek. Ez az architektúra gerince.

### 1.1 LocomotionCommand dataclass

Fájl: `src/roboshelf_ai/core/interfaces/locomotion_command.py`

Ez a modul a hierarchikus rendszer kommunikációs szerződése. Tartalmazza:
- `LocomotionCommand` dataclass: `v_forward`, `yaw_rate`, `stance_width`, `step_height`, `duration`
- `LocomotionCommandSpace`: a parancs minimuma, maximuma, defaultja
- `validate_command()` helper: clip + type check

Implementáció szempontjai:
- Legyen numpy-független (csak Python stdlib + dataclasses)
- A field-ek SI-egységekben: m/s, rad/s
- Legyen bővíthető (pl. `v_lateral` a jövőben)

### 1.2 PolicyAdapter

Fájl: `src/roboshelf_ai/locomotion/policy_adapter.py`

Ez a wrapper egy betanított locomotion policy-t szolgáltatásként tesz elérhetővé.

Implementáció szempontjai:
- `LocomotionPolicyAdapter` osztály, konstruktorban `model_path` és opcionálisan `device`
- `step(obs, command: LocomotionCommand) -> np.ndarray` metódus — visszaadja az aktuátorvezérlést
- Belső state: `_model` (SB3 PPO), `_last_action`
- Ha `model_path` nem létezik, `DummyAdapter` módban fut (nulla akció) — fontos a fejlesztés korai fázisában

### 1.3 Import-teszt

```bash
python -c "from roboshelf_ai.core.interfaces.locomotion_command import LocomotionCommand; print('OK')"
python -c "from roboshelf_ai.locomotion.policy_adapter import LocomotionPolicyAdapter; print('OK')"
```

Várt eredmény: `OK` mindkettőnél, importhiba nélkül.

### 1.4 Git commit

```bash
git add src/roboshelf_ai/core/interfaces/ src/roboshelf_ai/locomotion/
git commit -m "feat: add LocomotionCommand interface and PolicyAdapter stub"
git push origin main
```

**Elfogadási feltétel:** Mindkét modul importálható, `DummyAdapter` visszaad nulla akciót, típushibák nélkül.

---

## 2. Munkacsomag — G1 Locomotion Command Env

**Cél:** Egy új MuJoCo env létrehozása, amelyben a G1 command-tracking locomotion policy-t tanul. Ez lesz a Fázis A tanítási env.

### 2.1 Env fájl

Fájl: `src/roboshelf_ai/mujoco/envs/locomotion/g1_locomotion_command_env.py`

Kötelező elemek:
- Gymnasium `Env` leszármazott
- **Observation space:** propriocepció (joint pozíciók, sebességek, IMU: roll/pitch/yaw-rate, talpnyomás szenzorok ha elérhetők) + parancs vektor (`v_forward`, `yaw_rate`)
- **Action space:** 29 DoF aktuátorparancs (ugyanaz mint a legacy env-ben)
- **Reset:** helyes G1 induló testtartás (a legacy env-ből átemelt keyframe), reset noise, `defaultctrl` beállítás
- **Fizika:** 2 sub-step (mint a legacy env-ben)
- **Reward:** `r_forward` (parancskövetés), `r_upright` (stabilitás), `r_alive` (él-e a robot), `r_smooth` (akció simaság), `r_energy` (energiahatékonyság)
- **Termination:** dőlési szög > küszöb, időlimit

Reward súlyok (kiindulópontként, kísérletezendő):
```python
REWARD_WEIGHTS = {
    "forward": 2.0,
    "upright": 1.5,
    "alive": 0.5,
    "smooth": -0.1,
    "energy": -0.05,
}
```

### 2.2 Config fájl

Fájl: `configs/locomotion/g1_command_v1.yaml`

Tartalmazza az env paramétereket, a PPO hyperparamétereket és az acceptance küszöbértékeket. Példa struktúra:

```yaml
env:
  sub_steps: 2
  max_episode_steps: 1000
  command_range:
    v_forward: [0.0, 1.5]
    yaw_rate: [-1.0, 1.0]

ppo:
  n_envs: 8
  n_steps: 2048
  batch_size: 256
  n_epochs: 10
  learning_rate: 3.0e-4
  total_timesteps: 5_000_000

acceptance:
  min_episode_length: 300
  forward_tracking_error_max: 0.3
```

### 2.3 Train script

Fájl: `src/roboshelf_ai/training/train_loco_v1.py`

Kötelező elemek:
- `SubprocVecEnv` wrapper (M2 CPU magok kihasználása)
- TensorBoard logging a `data/logs/loco_v1/` mappába
- Checkpoint mentés `roboshelf-results/loco/v1/` alá
- `EvalCallback` az acceptance kritériumok folyamatos mérésére
- `--config` argument a yaml config betöltéséhez

Futtatási parancs:
```bash
python src/roboshelf_ai/training/train_loco_v1.py --config configs/locomotion/g1_command_v1.yaml
```

### 2.4 Eval script

Fájl: `src/roboshelf_ai/locomotion/eval_loco.py`

Futtat egy policy-t render módban (MuJoCo viewer), kiírja az epizód-hosszt és a command-tracking hibát.

```bash
python src/roboshelf_ai/locomotion/eval_loco.py \
  --model roboshelf-results/loco/v1/best_model.zip \
  --config configs/locomotion/g1_command_v1.yaml \
  --episodes 5
```

### 2.5 Sanity run (rövidített tréning)

A teljes 5M lépéses tanítás előtt futtasd le 10 000 lépéssel, hogy az env nem crashel:

```bash
python src/roboshelf_ai/training/train_loco_v1.py \
  --config configs/locomotion/g1_command_v1.yaml \
  --total-timesteps 10000 \
  --no-save
```

**Elfogadási feltétel (Fázis A):** Az agent legalább 300 lépésen át talpon marad, a command-tracking hiba csökkenő trendet mutat az első 1M lépésen belül.

---

## 3. Munkacsomag — Hierarchikus navigációs env

**Cél:** A retail navigációs policy átírása high-level action space-re. A nav policy már nem 29 DoF-t vezérel, hanem parancsot ad a locomotion adapternek.

### 3.1 Előfeltétel

A 2. munkacsomag elfogadási feltétele teljesül: van egy elfogadható locomotion policy (`best_model.zip`).

### 3.2 Hierarchikus nav env

Fájl: `src/roboshelf_ai/mujoco/envs/navigation/retail_nav_hier_env.py`

Kötelező elemek:
- **Observation space:** robot pozíció, orientáció, célpont iránya és távolsága (polar coords), akadályok (raycast vagy proximity szenzorok), a locomotion adapter aktuális állapota
- **Action space:** `(v_forward, yaw_rate)` — 2D folytonos, nem 29D
- **Low-level execution:** minden nav step-ben `n_loco_steps` lépésen át a locomotion adapter fut a kiadott paranccsal
- **Scene betöltés:** a meglévő működő combined XML logika átemelve (`tmp/combined_g1_store.xml`)
- **Reward:** célközelítés, orientáció, ütközésmentesség — nincs stabilitási reward (azt a locomotion réteg kezeli)
- **Termination:** célpont elérve, ütközés, időlimit, locomotion összeomlás

### 3.3 Config fájl

Fájl: `configs/navigation/retail_nav_hier_v1.yaml`

```yaml
env:
  n_loco_steps: 10
  goal_radius: 0.5
  max_episode_steps: 500

locomotion:
  model_path: roboshelf-results/loco/v1/best_model.zip
  adapter: LocomotionPolicyAdapter

ppo:
  n_envs: 4
  total_timesteps: 3_000_000
```

### 3.4 Train script

Fájl: `src/roboshelf_ai/tasks/navigation/train_nav_hierarchical.py`

Futtatási parancs:
```bash
python src/roboshelf_ai/tasks/navigation/train_nav_hierarchical.py \
  --config configs/navigation/retail_nav_hier_v1.yaml
```

### 3.5 Eval script

Fájl: `src/roboshelf_ai/tasks/navigation/eval_nav.py`

Vizuálisan ellenőrizhető, hogy a robot céltudatosan halad-e, és a járása nem omlik össze navigálás közben.

**Elfogadási feltétel (Fázis B):** A robot legalább az esetek 50%-ában eléri a célpontot véletlenszerű startpozícióból, és a locomotion nem omlik össze navigálás közben.

---

## 4. Munkacsomag — Imitációs tanulás csatorna

**Cél:** Demonstrációs adatbevitel lehetőségének megnyitása, behavior cloning belépési pont.

### 4.1 Demonstráció formátum

Fájl: `src/roboshelf_ai/core/interfaces/demonstration.py`

Definiálja a `Demonstration` és `DemonstrationDataset` osztályokat. Mentési formátum: `.npz` (numpy), séma:
```
obs: (T, obs_dim)
actions: (T, action_dim)
rewards: (T,)
dones: (T,)
infos: list[dict]
metadata: dict (env_id, date, author, config_hash)
```

### 4.2 Scripted expert rollout

Fájl: `src/roboshelf_ai/scripts/collect_scripted_expert.py`

Egyszerű determinisztikus controller, amely a célpont felé fordítja és mozgatja a robotot — nem gépi tanulás, csak script. Erre BC-t lehet betanítani kezdeti locomotion policy-ként.

```bash
python src/roboshelf_ai/scripts/collect_scripted_expert.py \
  --episodes 100 \
  --output data/demonstrations/scripted_loco_v1.npz
```

### 4.3 Behavior cloning script

Fájl: `src/roboshelf_ai/locomotion/train_loco_bc.py`

BC előtanítás a demonstrációs adatokból, majd az így kapott policy inicializálja a RL tanítást.

```bash
python src/roboshelf_ai/locomotion/train_loco_bc.py \
  --data data/demonstrations/scripted_loco_v1.npz \
  --output roboshelf-results/loco/bc_init/policy.zip \
  --epochs 50
```

**Elfogadási feltétel:** A BC-vel inicializált policy gyorsabban tanul, mint a teljesen random inicializált (mért: lépések száma az első 300-lépéses stabil epizódig).

---

## 5. Munkacsomag — Manipulációs env (minimalizált sandbox)

**Cél:** Reach, grasp, lift, place komponensek külön env-ben, a teljes boltba való integrálás előtt.

### 5.1 Előfeltétel

A 3. munkacsomag elfogadási feltétele teljesül.

### 5.2 Manipulációs sandbox env

Fájl: `src/roboshelf_ai/mujoco/envs/manipulation/g1_pickplace_env.py`

Minimalizált scene: G1 + asztal + egy tárgy. Nincs bolt, nincs navigáció. Komponensek külön reward-dal mérhetők.

### 5.3 Config és train

Fájlok: `configs/manipulation/pickplace_v1.yaml`, `src/roboshelf_ai/tasks/manipulation/train_pickplace.py`

**Elfogadási feltétel (Fázis C):** Reach, grasp, lift, place szakaszok külön mérhetők és ismételhetők >70%-os sikerességgel.

---

## 6. Munkacsomag — Integráció és demo

**Cél:** Nav + manipuláció összekapcsolása, investor demo szkript.

### 6.1 Előfeltétel

5. munkacsomag elfogadási feltétele teljesül.

### 6.2 Task state machine

Fájl: `src/roboshelf_ai/demo/task_state_machine.py`

Állapotok: `NAVIGATE_TO_SHELF → PICK_ITEM → NAVIGATE_TO_DROP → PLACE_ITEM → DONE`

### 6.3 Demo script

Fájl: `src/roboshelf_ai/demo/investor_demo.py`

Reprodukálható demo: fix seed, fix kamera, KPI overlay (lépésszám, sikerességi arány, időtartam).

**Elfogadási feltétel (Fázis E):** A demo reprodukálhatóan lefut, vizuálisan prezentálható.

---

## Általános futtatási szabályok

### Környezet aktiválása

```bash
# Ha még nem aktív (miniforge/conda)
conda activate base   # vagy a projekt-specifikus env neve
```

### Repo gyökérből való futtatás ellenőrzése

```bash
pwd
# Elvárás: /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign
```

### TensorBoard indítása

```bash
tensorboard --logdir data/logs/ --port 6006
# Böngészőben: http://localhost:6006
```

### Checkpoint mentési konvenció

```
roboshelf-results/
├── loco/
│   ├── v1/           ← első tanítási run
│   │   ├── best_model.zip
│   │   ├── final_model.zip
│   │   └── evaluations.npz
│   └── bc_init/      ← BC előtanítás kimenete
├── nav/
│   └── hier_v1/
└── manipulation/
    └── pickplace_v1/
```

### Hibaelhárítás

| Hiba | Ok | Megoldás |
|---|---|---|
| `No such file or directory` | Nem a repo gyökeréből futott | `cd /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign` |
| `ModuleNotFoundError: roboshelf_ai` | PYTHONPATH hiányzik | `export PYTHONPATH=$PYTHONPATH:$(pwd)/src` |
| MPS hiba float64-gyel | M2 MPS + float64 inkompatibilitás | Hagyd CPU-n, a SubprocVecEnv kihasználja a magokat |
| XML betöltési hiba | combined XML hiányzik | Futtasd az XML merge scriptet előbb |

---

_Ez a dokumentum a master tervvel együtt kezelendő. Minden munkacsomag teljesítése után frissítsd a `roboshelf_schedule.md` fájlt is._
