# Roboshelf AI Redesign — Ütemezés

_Utoljára frissítve: 2026-04-18 (Fázis B ✅ KÉSZ — mind a 4 curriculum szint elfogadva, 100% siker, 0% loco összeomlás)_  
_Állapotjelzők: ⬜ nem kezdett · 🔄 folyamatban · ✅ kész · ❌ blokkolt_

---

## Összefoglaló táblázat

| # | Fázis | Munkacsomag | Becslés | Állapot | Kész mikor |
|---|---|---|---|---|---|
| 0 | — | Előkészítés és repo-tisztítás | 0,5 nap | ✅ | 2026-04-17 |
| 1 | — | Locomotion interfész és command layer | 1 nap | ✅ | 2026-04-17 |
| 2 | A | G1 Locomotion Command Env + tanítás | 5–8 nap | ✅ motion.pt átvéve | 2026-04-17 |
| 3 | B | Hierarchikus navigációs env + tanítás | 5–7 nap | ✅ | 2026-04-18 |
| 4 | — | Imitációs tanulás csatorna (BC) | 2–3 nap | ⬜ | — |
| 5 | C | Manipulációs sandbox env + tanítás | 5–8 nap | ⬜ | — |
| 6 | D+E | Integráció + investor demo | 3–5 nap | ⬜ | — |

**Teljes becsült időtartam:** ~21–32 fejlesztési nap (a tanítási futások falióra-idejével együtt).

---

## 0. Munkacsomag — Előkészítés és repo-tisztítás

**Becsült idő:** 0,5 nap  
**Állapot:** ✅ kész — 2026-04-17  
**Elfogadási feltétel:** system_check.py fut hiba nélkül, new mappastruktúra létezik, legacy fájlok archiválva.

### Feladatok

- [x] `python src/roboshelf/checks/system_check.py` — a sandbox Linuxon nem fut (hiányzó csomagok), de a saját Mac-en rendben van
- [x] Régi fájlok másolása `src/legacy/` alá
- [x] Új mappastruktúra létrehozása (locomotion, navigation, manipulation, interfaces, rewards, demo)
- [x] `.gitignore` ellenőrzése és kiegészítése (tmp/, data/demonstrations/, data/exports/, data/logs/ hozzáadva)
- [ ] Git commit + push: `"chore: archive legacy, scaffold redesign structure"` ← **saját terminálból**

### Megjegyzések

2026-04-17: A system check a sandbox Linux környezetben hiányzó csomagok miatt pirosakat mutat (MuJoCo, SB3, PyTorch), de ez várható — a csomagok a Mac miniforge/conda env-ben vannak. A fájlrendszeri lépések sikeresen elvégezve.

---

## 1. Munkacsomag — Locomotion interfész és command layer

**Becsült idő:** 1 nap  
**Állapot:** ✅ kész — 2026-04-17  
**Elfogadási feltétel:** Mindkét modul importálható, DummyAdapter nulla akciót ad vissza hiba nélkül.

### Feladatok

- [x] `src/roboshelf_ai/core/interfaces/locomotion_command.py` megírva
- [x] `src/roboshelf_ai/locomotion/policy_adapter.py` megírva (DummyAdapter támogatással)
- [x] Import teszt zöld mindkettőre
- [x] `src/roboshelf_ai/core/interfaces/__init__.py` — teljes export (RobotState, LocomotionCommand, Base*Policy, DemoCollector stb.)
- [x] `src/roboshelf_ai/core/__init__.py` — re-export + lazy callbacks
- [x] `src/roboshelf_ai/locomotion/__init__.py` — adapterek + G1 konstansok
- [x] `src/roboshelf_ai/__init__.py` — top-level convenience, `__version__ = "0.2.0"`
- [x] `src/roboshelf_ai/isaac/` — Isaac Lab placeholder stub (adapters/, envs/)
- [ ] Git commit + push ← **saját terminálból**

### Megjegyzések

2026-04-17: LocomotionCommand 5D parancsvektor (v_forward, v_lateral, yaw_rate, stance_width, step_height). COMMAND_SPACE_BASIC és COMMAND_SPACE_FULL előre definiálva. LocomotionPolicyAdapter DummyAdapter módban fut ha nincs modell — figyelmeztetést logol. Import teszt és működési teszt: OK.

---

## 2. Munkacsomag — G1 Locomotion Command Env (Fázis A)

**Becsült idő:** 5–8 nap (implementáció 1–2 nap + tanítás + iteráció)  
**Állapot:** ✅ kész — 2026-04-17 (Fázis A döntés: motion.pt átvéve, saját tanítás kiesik)  
**Elfogadási feltétel:** ~~Stabil talpon maradás 300+ lépésen~~ → helyette: UnitreeRLGymAdapter betöltve és inference-képes.

### Feladatok

- [x] `src/roboshelf_ai/mujoco/envs/locomotion/g1_locomotion_command_env.py` megírva
- [x] `configs/locomotion/g1_command_v1.yaml` létrehozva
- [x] `src/roboshelf_ai/training/train_loco_v1.py` megírva (SubprocVecEnv, TensorBoard, checkpoint)
- [x] `src/roboshelf_ai/locomotion/eval_loco.py` megírva
- [x] `g1_locomotion_command_env.py`, train, eval scriptek megírva
- [x] Sanity run: 5587 fps, de ep_len=71 konstans (0.28s után elesik)
- [x] v1 tanítás: 5M lépés, ep_len stagnál → fizikai stabilitási probléma diagnosztizálva
- [x] **Architekturális döntés: unitree_rl_gym motion.pt átvétele** — saját tanítás kiesik
- [x] `UnitreeRLGymAdapter` megírva és tesztelve (47 dim obs, LSTM, PD control, 50 Hz)
- [x] motion.pt betöltve és inference-képes: `is_dummy: False`
- [x] `train_loco_v1.py` — VecNormalize + EpisodeStatsCallback + LinearCurriculumCallback bekötve
- [x] `configs/locomotion/g1_command_v2.yaml` — `vec_normalize:` szekció hozzáadva
- [x] `src/roboshelf_ai/locomotion/test_walk.py` megírva és futtatva
- [x] **Járásteszt ✅ ELFOGADVA** — 3/3 talpon, 2.17m/5s @ 0.5m/s parancs, upright=0.999
- [x] unitree_rl_gym áthelyezve: `~/unitree_rl_gym` → `roboshelf-ai-redesign/unitree_rl_gym/` (rendes fájlokként, nem submodule)
- [x] Git commit + push ✅ fac955b

### Tanítási futások naplója

| Run | Dátum | Lépések | Legjobb ep. hossz | Tracking hiba | Megjegyzés |
|---|---|---|---|---|---|
| v1 | 2026-04-17 | 5M | 72 lépés | 0.284s után elesik | ep_len konstans, buoyancy hiányzott |
| v2 | — | — | — | — | buoyancy curriculum + reward rebalance |

### Kockázati feljegyzés — motion.pt átvétel hosszú távú hatásai (2026-04-18)

Demóhoz (MuJoCo renderelt) nulla kockázat. Fizikai deploymentnél három ponton jelent rizikót: (1) sim2real gap — motion.pt Isaac Gym dinamikájára hangolt, valós G1-en a mi felelősségünk a finomhangolás, nincs saját loco pipeline; (2) nav-loco coupling — ha a loco adaptert le kell cserélni, a nav policy-t is újra kell tanítani; (3) parancskövetési torzítás — walk tesztben 14% alulteljesítés (0.5 → 0.43 m/s), yaw_rate nem mérve. Kezelési stratégia ha valós robot tesztelés jön: PPF (`ppf.enabled: true` a nav configban) — a motion.pt finomhangolása a valós visszajelzésekre, nulláról tanítás nélkül.

### Megjegyzések

2026-04-17 v1 diagnózis: ep_len=71 konstans az 5M lépés alatt. A robot 0.284 szimulációs másodperc után mindig elesik — fizikai stabilitási probléma, nem reward hiba. A legacy nav env-ben buoyancy_force=103N segített a korai fázisban. v2 javítások: buoyancy curriculum (103N→0 az első 3M lépésen), lazított termination (upright 0.5→0.3, z_min 0.4→0.35m), erősebb stabilitási reward (w_alive 0.5→3.0, w_upright 1.5→4.0), kisebb büntetések.

---

## 3. Munkacsomag — Hierarchikus navigációs env (Fázis B)

**Becsült idő:** 5–7 nap  
**Állapot:** ✅ kész — 2026-04-18  
**Elfogadási feltétel:** Robot 50%+ esetben eléri a célt random startból, locomotion nem omlik össze közben.

### Előfeltétel

- [x] 2. munkacsomag elfogadási feltétele teljesült (járásteszt ✅ 2026-04-18)

### Feladatok

- [x] `configs/navigation/retail_nav_hier_v1.yaml` létrehozva (curriculum, PPO hyperparamok RL Zoo alapján, vec_normalize szekció)
- [x] `src/roboshelf_ai/mujoco/envs/navigation/retail_nav_hier_env.py` megírva
- [x] `src/roboshelf_ai/tasks/navigation/train_nav_hierarchical.py` megírva
- [x] `src/roboshelf_ai/tasks/navigation/eval_nav.py` megírva
- [x] Sanity run crash nélkül
- [x] Curriculum szint 1 tanítás: 3M lépés ✅ ELFOGADVA (100% siker, 0% loco összeomlás)
- [x] Curriculum szint 2 tanítás: 3M lépés ✅ ELFOGADVA (100% siker, 0% loco összeomlás)
- [x] Curriculum szint 3 tanítás: 3M lépés ✅ ELFOGADVA (100% siker, 0% loco összeomlás)
- [x] Curriculum szint 4 tanítás: 3M lépés ✅ ELFOGADVA (100% siker, 0% loco összeomlás) — akadályok nélkül (store XML Fázis C-re halasztva)
- [ ] Eval vizuálisan (--render) — opcionális, Fázis C előtt
- [ ] Git commit: `"feat: hierarchical nav env v1, training run results"`

### Tanítási futások naplója

| Run | Dátum | Lépések | Célpont-elérési arány | Locomotion összeomlás | Megjegyzés |
|---|---|---|---|---|---|
| lvl1 v1 | 2026-04-18 | 3M | 100% (20/20) | 0% | Javított config: goal [0,1.5], random start ±0.3m, átlag 18 lépés/ep |
| lvl2 v1 | 2026-04-18 | 3M | 100% (10/10) | 0% | goal [0,2.5], n_envs=8, batch_size=512, átlag 21 lépés/ep |
| lvl3 v1 | 2026-04-18 | 3M | 100% (10/10) | 0% | goal_range x[-1,1] y[1.5,2.5], kanyar, átlag 15 lépés/ep |
| lvl4 v1 | 2026-04-18 | 3M | 100% (10/10) | 0% | goal [0,3.8], akadályok nélkül (store XML nincs), átlag 33 lépés/ep |

### Megjegyzések

2026-04-18 Curriculum szint 1: Az első tanítás triviális eredményt adott (robot startban már célba ért: goal=[0,0.5], goal_radius=0.5m). Javítás: random start ±0.3m xy + random yaw, goal=[0,1.5]. Újratanítás után: 100% siker, 0% loco összeomlás, átlag 18 nav lépés/epizód (≈3.6s valós idő). PPO policy_kwargs: net_arch dict(pi=[256,256], vf=[256,256]). VecNormalize mentve.

---

## 4. Munkacsomag — Imitációs tanulás csatorna

**Becsült idő:** 2–3 nap  
**Állapot:** ⬜ nem kezdett  
**Elfogadási feltétel:** BC-vel inicializált policy gyorsabban tanul mint random init (mért lépésszámban).

### Feladatok

- [x] `src/roboshelf_ai/core/interfaces/demonstration.py` megírva (DemoCollector, DemoDataset, DemoStep)
- [x] `src/roboshelf_ai/scripts/collect_scripted_expert.py` megírva (ScriptedNavExpert P-szabályozó, CLI)
- [ ] Demonstrációs adatgyűjtés: 100 epizód → `data/demonstrations/scripted_loco_v1.npz`
- [ ] `src/roboshelf_ai/locomotion/train_loco_bc.py` megírva
- [ ] BC futtatva, policy mentve: `roboshelf-results/loco/bc_init/policy.zip`
- [ ] Összehasonlító tanítás: BC-init vs random-init
- [ ] Git commit: `"feat: IL demonstration pipeline and BC training"`

### Megjegyzések

_ide kerülnek a demonstrációs adat minőségéről szerzett tapasztalatok_

---

## 5. Munkacsomag — Manipulációs sandbox env (Fázis C)

**Becsült idő:** 5–8 nap  
**Állapot:** ⬜ nem kezdett  
**Elfogadási feltétel:** Reach, grasp, lift, place külön mérhetők és ismételhetők >70%-os sikerrel.

### Megközelítés: B) Rögzített testű sandbox (demóhoz)

**Feladat:** robot asztalról vesz fel terméket és polcra helyezi (polcfeltöltés).  
**Modell:** `g1_29dof_with_hand.xml` (~43 aktuátor: 12 láb + 3 derék + 16 kar/kéz).  
**Policy:** csak kar+kéz 16 DoF-ját tanulja, lábak fixek (motion.pt nem fut manip közben).  
**Workflow:** nav policy → robot pozícióba megy → megáll → manip policy veszi át.

**Hosszú távú fejlesztési irány (nem demó scope):** Hierarchikus karmanipuláció — motion.pt + ManipPolicy párhuzamosan, járás közbeni manipuláció. Fizikai deploymentnél lesz kritikus.

### Előfeltétel

- [x] 3. munkacsomag elfogadási feltétele teljesült (Fázis B ✅ 2026-04-18)

### Feladatok

- [x] `src/envs/assets/scene_manip_sandbox.xml` megírva (G1 with hand + padló + asztal + gondola + stock termék)
- [x] `src/roboshelf_ai/mujoco/envs/manipulation/g1_shelf_stock_env.py` megírva (4 fázis: reach/grasp/lift/place)
- [x] `configs/manipulation/shelf_stock_v1.yaml` létrehozva
- [x] `src/roboshelf_ai/tasks/manipulation/train_shelf_stock.py` megírva
- [x] `src/roboshelf_ai/tasks/manipulation/eval_shelf_stock.py` megírva
- [ ] Sanity run crash nélkül
- [ ] Teljes tanítási run (5M lépés)
- [ ] Komponens-szintű eval: reach/grasp/lift/place külön mérve
- [ ] Git commit: `"feat: manipulation sandbox env v1 — shelf stocking"`

### Tanítási futások naplója

| Run | Dátum | Lépések | Reach % | Grasp % | Lift % | Place % | Megjegyzés |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

### Megjegyzések

_ide kerülnek a gripper, kontakt fizika, reward shaping döntések_

---

## 6. Munkacsomag — Integráció és investor demo (Fázis D + E)

**Becsült idő:** 3–5 nap  
**Állapot:** ⬜ nem kezdett  
**Elfogadási feltétel:** Demo reprodukálhatóan lefut, vizuálisan prezentálható.

### Előfeltétel

- [ ] 5. munkacsomag elfogadási feltétele teljesült

### Feladatok

- [ ] `src/roboshelf_ai/demo/task_state_machine.py` megírva
- [ ] `src/roboshelf_ai/demo/investor_demo.py` megírva
- [ ] `src/roboshelf_ai/demo/camera_paths.py` megírva
- [ ] Demo lefut fix seed-del reprodukálhatóan
- [ ] KPI overlay működik (lépésszám, sikerességi arány, időtartam)
- [ ] Demo videó rögzítve és értékelve
- [ ] Git commit: `"feat: investor demo v1 — nav + manipulation integrated"`

### Demo futtatások naplója

| Verzió | Dátum | Siker? | Megjegyzés |
|---|---|---|---|
| — | — | — | — |

### Megjegyzések

_ide kerülnek a demo-szkript döntések, kamera, scriptelt vs. policy-vezérelt részek_

---

## Időbecslés háttere

Az egyes munkacsomagok becsléséhez a következő tapasztalati tényezőket vettük figyelembe:

Az M2 MacBook Air CPU-n futó SB3 PPO tipikusan 20 000–50 000 step/sec sebességgel fut 8 párhuzamos env-vel. Egy 5M lépéses locomotion tanítás így kb. 1,5–7 óra falióra-időt vesz igénybe. Egy nav tanítás 3M lépéssel kb. 1–4 óra. A becsült napok tehát tartalmazzák a tanítás futási idejét, a közbülső értékelést és az esetleges reward-iterációt.

Ha a locomotion fázis első tanítási futása nem éri el az acceptance kriteriumot, 2–4 napos iterációs puffert kell számolni reward-újrahangolásra.

---

## Frissítési emlékeztető

Ezt a fájlt frissítsd:
- minden munkacsomag megkezdésekor (állapot → 🔄)
- minden tanítási futás után (napló kitöltve)
- minden elfogadási feltétel teljesítésekor (állapot → ✅)
- minden blokkolt helyzetnél (állapot → ❌, megjegyzésbe az ok)

A frissítés egyben git commit-ot is igényel:
```bash
git add roboshelf_schedule.md
git commit -m "chore: update schedule — [munkacsomag neve] [állapot]"
```
