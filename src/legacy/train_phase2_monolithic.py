#!/usr/bin/env python3
"""
Roboshelf AI — Fázis 2: G1 Retail Navigáció PPO tanítás

Lokális tanítási script (M2 Mac).
A G1 humanoid megtanul járni a retail boltban a raktárig.

Használat:
  python src/training/roboshelf_phase2_train.py --level m2_20m_v22

# Kaggle/Colab használat (jelenleg nem aktív):
#   !python roboshelf_phase2_train.py --level teszt     # ~5 perc
#   !python roboshelf_phase2_train.py --level kozepes   # ~30 perc GPU-n
#   !python roboshelf_phase2_train.py --level teljes    # ~2-4 óra GPU-n
"""

import argparse
import os
import sys
import time
from pathlib import Path

# --- Import útvonal fix: envs mappa elérhetővé tétele ---
_THIS_DIR = Path(__file__).resolve().parent
_ENVS_DIR = _THIS_DIR.parent / "envs"
if str(_ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(_ENVS_DIR))

import numpy as np
import gymnasium as gym

# --- Output mappa (repo-relatív) ---
_REPO_ROOT = _THIS_DIR.parent.parent
OUTPUT_DIR = _REPO_ROOT / "roboshelf-results" / "phase2"

MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# --- Szintek ---
LEVELS = {
    # GPU/Kaggle szintek — KIKOMMENTÁLVA, M2-n nem releváns
    # "teszt":   { "total_timesteps": 100_000,    "n_envs": 4,  "description": "GPU teszt" },
    # "kozepes": { "total_timesteps": 2_000_000,  "n_envs": 8,  "description": "GPU közepes" },
    # "teljes":  { "total_timesteps": 10_000_000, "n_envs": 16, "description": "GPU teljes" },
    "m2_2ora": {
        # M2 CPU-ra optimalizált ~2 órás tanítás
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "M2 CPU ~2 óra (3M lépés, 4 env)",
    },
    "m2_6m": {
        # M2 CPU ~1 óra: 6M lépés, 4 env
        "total_timesteps": 6_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-5,
        "clip_range": 0.1,
        "description": "M2 CPU ~1 óra (6M lépés, 4 env)",
    },
    "m2_3m_fresh": {
        # M2 CPU fresh start, ~2 óra: 3M lépés, optimalizált reward shaping
        # w_forward=4.0, w_healthy=3.0, w_fall=-10.0, w_gait=0.18 (env-ben beégetve)
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~2 óra (3M lépés, új reward shaping)",
    },
    "m2_3m_nogait": {
        # M2 CPU fresh start, gait reward KIKAPCSOLVA (w_gait=0.0)
        # Tanulási sorrend: 1. járás megtanulása, 2. majd gait finom hangolás
        # w_forward=4.0, w_healthy=3.0, w_fall=-10.0, w_gait=0.0
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~2 óra (3M lépés, gait reward kikapcsolva)",
    },
    "m2_3m_v3": {
        # FIX: "stand and fall" probléma megoldása
        # w_forward=5.0 (domináns), w_healthy=0.5 (minimális), w_fall=-50.0 (erős)
        # Forrás: legged_gym + Gymnasium Humanoid-v4 tapasztalat
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~2 óra (stand-and-fall fix: w_healthy=0.5, w_fall=-50)",
    },
    "m2_3m_v4": {
        # FIX v4: w_healthy=0.0 (teljesen ki!), w_fall=-20 (mérsékelt), w_forward=5.0
        # v3 tanulság: w_fall=-50 túl agresszív → robot "befagyott" (ep hossz 35, reward -264)
        # v4: healthy nullán, fall mérsékelt → forward az egyetlen pozitív forrás
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v4: w_healthy=0.0, w_fall=-20, w_forward=5.0)",
    },
    "m2_3m_v5": {
        # FIX v5: helyes G1 kezdőpozíció! z=0.79 + kar joint szögek a keyframe alapján
        # Ez volt az igazi probléma: z=0.75 + karok rossz pozícióban → azonnal instabil
        # Reward: w_healthy=1.0 (mérsékelt), w_forward=5.0 (domináns), w_fall=-20
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v5: helyes G1 keyframe pozíció!)",
    },
    "m2_3m_v6": {
        # FIX v6: akció skálázás javítva!
        # Korábban: ctrl = ctrl_mean + action * ctrl_half (ctrl_mean rossz alap!)
        # Most: ctrl = default_ctrl + action * ctrl_half (keyframe az alap)
        # Nulla akció = egyensúlyi pozíció, nem random ctrl_mean
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v6: keyframe-alapú akció skálázás)",
    },
    "m2_3m_v7": {
        # FIX v7: sub-step 5→2, robot 100+ lépésen át stabil nulla akcióval
        # v6 tanulság: robot előre dőlt és 30 lépés alatt összecsuszik (5 sub-step túl gyors)
        # Most: 2 sub-step = lassabb fizika = több tanulási lehetőség
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~35 perc (v7: sub-step 5→2, stabil egyensúly)",
    },
    "m2_3m_v8": {
        # FIX v8: akció skála csökkentve (ACTION_SCALE=0.3 radian)
        # v7 tanulság: policy előre dőlt → 67 lépésnél terminál
        # Teljes ctrl_half helyett max ±0.3 radian eltérés az egyensúlytól → stabilabb mozgás
        "total_timesteps": 3_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU fresh start ~17 perc (v8: ACTION_SCALE=0.3, kisebb perturbáció)",
    },
    "m2_10m_v8": {
        # 10M lépés, egyébként ugyanaz mint v8
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU ~1 óra (10M lépés, v8 konfig: ACTION_SCALE=0.3)",
    },
    "m2_10m_v11": {
        # v11: reset zaj hozzáadva (noise_scale=0.01, Humanoid-v4 mintájára)
        # Eddig: determinisztikus reset → ±0.0 szórás eval-ban → policy befagyott
        # Most: kis véletlen zaj → policy tanul általánosítani → nem ragad lokális optimumba
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "M2 CPU ~1 óra (v11: reset noise, tracking reward, w_healthy=0.05)",
    },
    "m2_10m_v12": {
        # v12: SIKERTELEN — w_forward 8→4 csökkentés catastrophic forgetting-et okozott
        # reward: +133.6 → -121.3 összeomlás. Tanulság: scale shift finetune-nál végzetes.
        # ARCHIVÁLT, ne használd!
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "ARCHIVÁLT - v12 összeomlott (w_forward scale shift) → használd v12b-t!",
    },
    "m2_10m_v12b": {
        # v12b finetune: +94.8 reward (v11-hez képest jobb reward struktúra de 86 lépésnél még mindig esik)
        # Tanulság: finetune nem tud beégett mozgásmintán változtatni
        # ARCHIVÁLT (fresh start szükséges)
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "ARCHIVÁLT - v12b finetune: 94.8 reward, 86 lép (sub-step=1 fresh starthoz menj)",
    },
    "m2_10m_v13": {
        # v13: SIKERTELEN — sub-step=1 → cvel más skálán → tracking negatív (-2.04/lép)
        # reward=-330.9, ep=169 (hosszabb lett ✅ de navigáció nem tanul ❌)
        # ARCHIVÁLT
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "ARCHIVÁLT - v13: sub-step=1 sikertelen (tracking negatív lett)",
    },
    "m2_10m_v14": {
        # v14: Epizód végi dist bonus — az igazi navigációs jel
        # v13 tanulság: sub-step=1 → cvel negatív; sub-step=2 visszaállítva
        # v12b tanulság: per-lépés dist_shaping matematikailag gyenge (0.002m/lép)
        #
        # Megoldás: w_dist_final=200 × (start_dist - final_dist) EGYSZER az ep végén
        #   - Ha 86 lép alatt 0.5m közel: +100 bonus (= 1.16/lép ekvivalens) → DOMINÁNS
        #   - Ha helyben marad: +0 → nagy különbség a navigáló és nem-navigáló policy között
        #   - sub-step=2 visszaállítva (tracking működik 2 sub-stepnél)
        # Eredmény: 16M lépésnél reward=0.2, de robot 3.18m-nél maradt (12cm haladás!)
        # Diagnózis: lokális optimum — álló robot per-lépés reward +0.154 (pozitív!)
        #            stuck-detection hiánya → 85 lépésen át "biztonságos" az állás
        # → v15 megoldja ezt
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "ARCHIVÁLT - v14: ep-végi dist bonus, de lokális optimum (álló robot) → v15",
    },
    "m2_10m_v15": {
        # v15: Velocity tracking Gaussian + Stuck-detection + air_time fix
        # Eredmény: MINDEN ep 75 lépésnél végzett → stuck triggerelt, de lokális optimum maradt
        #   w_stuck=-15 < w_fall=-20 → álló stratégia még mindig kevésbé rossz
        #   w_air_time=1.0 túl gyenge → robot nem merte felemelni a lábát
        # → v16 megoldja: w_stuck=-20, stuck_window=40, w_air_time=3.0
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "ARCHIVÁLT - v15: stuck minden 75 lépésnél (w_stuck gyenge, air_time gyenge) → v16",
    },
    "m2_10m_v16": {
        # v16: Stuck büntetés = fall büntetés + rövidebb ablak + erős air_time
        # Eredmény: ep=40 MINDEN futásnál → az ablak méretét tanulja meg, nem a járást
        # Tanulság: stuck-detection a szimptómát bünteti, a PPO mindig megtalálja az ablakot
        # → v17 curriculum megközelítéssel oldja meg
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "ARCHIVÁLT - v16: ep=40 minden futásnál (ablak méretét tanulta) → v17",
    },
    "m2_10m_v17": {
        # v17: Curriculum tanítás — felhajtóerő + stuck-window annealing
        #
        # Diagnózis (v15/v16): stuck-detection alapú megközelítés zsákutca
        #   A PPO mindig az ablak méretére optimalizál (v15: 75, v16: 40)
        #   Alapprobléma: a robot fizikailag nem tud biztonságosan lépni → nem próbálja
        #
        # v17 curriculum (ETH Zürich ANYmal / Unitree pipeline mintájára):
        #   1. Felhajtóerő: ~206N Z-irányú erő a pelvis-en (gravitáció 60%-a)
        #      Robot effektív tömege: 35kg → ~14kg → könnyebben egyensúlyoz, mer lépni
        #      CurriculumCallback lineárisan nullázza 3M-7M lépés között
        #   2. Stuck-window annealing: 9999 (=ki) → 40 lépés (3M-7M között)
        #      Első 3M: nincs stuck-terminálás (robot tanul lépni)
        #      3M-7M: fokozatosan szigorodik
        #      7M-10M: teljes súly + 40 lépéses stuck (finomhangolás)
        #   3. Lineáris tracking visszaállítva (v11: w_forward=8.0 × forward_component)
        #      Gaussian (v15/v16) nem adott elég gradienst
        #   4. w_air_time=3.0 marad (v16-ból bevált)
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU ~1 óra (v17 FRESH: curriculum felhajtóerő + stuck annealing)",
        "curriculum": {
            "phase1_end":        3_000_000,   # eddig teljes felhajtóerő, stuck ki
            "phase2_end":        7_000_000,   # eddig lineáris csökkentés
            "max_buoyancy":      103.0,        # N (gravitáció 30%-a, konzervatív)
                                               # G1 URDF-ben lehet virtuális torzó link!
                                               # 206N × 2 torzó = 412N > súly → repülne
                                               # 103N biztonságos, ellenőrizd az első futásnál:
                                               # torso_z > 1.5m az első lépésnél? → csökkenteni kell
            "stuck_window_start": 9999,        # = kikapcsolva
            "stuck_window_end":   40,          # lépés
        },
    },
    "m2_20m_v17": {
        # v17 curriculum, 20M lépés
        # Eredmény: 8M-nál +199 reward (curriculum működött!), majd -15241 (kapálózás beégett)
        # Diagnózis: w_air_time feltétel nélkül → kapálózás optimum; 6M fázis1 túl hosszú
        # → v18 javítja: air_time feltételes + smoothness penalties + rövidebb fázis1
        "total_timesteps": 20_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "ARCHIVÁLT - v17 20M: +199@8M majd -15241 (kapálózás beégett) → v18",
        "curriculum": {
            "phase1_end":         6_000_000,
            "phase2_end":        14_000_000,
            "max_buoyancy":       103.0,
            "stuck_window_start": 9999,
            "stuck_window_end":   40,
        },
    },
    "m2_20m_v18": {
        # v18: Smoothness penalties + feltételes air_time + rövidebb curriculum + ent_coef=0.01
        #
        # v17 diagnózis: "kapálózás" lokális optimum
        #   - w_air_time=3.0 feltétel nélkül → helyben kapálózás kifizetődő a felhajtóerőben
        #   - 6M lépéses 1. fázis túl hosszú → entrópia elfogy, policy beég
        #   - Felhajtóerő eltűntével a kapálózás csillagászati ctrl/contact cost-ot okoz
        #
        # v18 három fix:
        #   1. air_time feltételes: csak v_forward > 0.1 m/s esetén jutalmaz
        #      Helyben kapálózás értéke: 0 → nem kifizetődő
        #   2. Smoothness penalties (Isaac Lab / ETH ANYmal alapján):
        #      action_rate=-0.01, dof_acc=-2.5e-7, dof_vel=-1e-3
        #      Nagy nyomatékú kapálózás azonnal büntetve
        #   3. Rövidebb curriculum: 1M fázis1, 1M-3M annealing (volt: 6M/14M)
        #      Kevesebb idő a könnyített fizikán → nem égeti be a kapálózást
        #   4. ent_coef=0.01 (volt: 0.001): entrópia megőrzése az annealing alatt
        "total_timesteps": 20_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "description": "M2 CPU ~2 óra (v18 FRESH 20M: smoothness + feltételes air_time + ent_coef=0.01)",
        "curriculum": {
            "phase1_end":         1_000_000,  # csak 1M felhajtóerő (volt: 6M)
            "phase2_end":         3_000_000,  # 1M-3M annealing (volt: 6M-14M)
            "max_buoyancy":       103.0,
            "stuck_window_start": 9999,
            "stuck_window_end":   40,
        },
    },
    "m2_20m_v19": {
        # v19: No-backward terminálás + orientációs büntetés + forward clip + hip lean + enyhébb smoothness
        #
        # v18 diagnózis: ep=166 (stabilabb!), de dist=3.37m (visszafelé megy!)
        #   Probléma 1: w_dof_vel=-1e-3 blokkolta az összes mozgást → robot megtanult minimálisan mozogni
        #   Probléma 2: w_forward × negatív forward_component → negatív critic célok → instabil
        #   Probléma 3: nincs orientációs jel → robot oldalaz/kifarol
        #   Probléma 4: hátrafelé menés büntetés nélkül → "hátrafelé menekülés" helyi optimum
        #
        # v19 négy fix:
        #   1. Forward hip lean (+0.1 rad hip_pitch reset-ben): gravitáció passzívan segít előre
        #   2. Orientációs büntetés: w_orientation=-2.0 × (1-cos(yaw_error))
        #      0 ha célra néz, -4.0 ha pontosan hátrafelé → folyamatos irányjelzés
        #   3. No-backward terminálás: ha avg v_forward < -0.2 m/s (30 lép ablak) → term + -20
        #      Hátrafelé menekülés mint az esés: egyformán büntetve
        #   4. Forward reward clipping: max(0, forward_component) → hátra = 0, nem negatív
        #      Stabilabb value becslés, tiszta gradiens jel
        #   5. Smoothness penalties enyhítve (v18 túl agresszív volt):
        #      w_dof_vel = 0.0 (ki), w_dof_acc = -1e-7 (-2.5e-7-ről), w_action_rate = -0.005 (-0.01-ről)
        "total_timesteps": 20_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "description": "M2 CPU ~2 óra (v19 FRESH 20M: no-backward + orientation + forward clip + hip lean)",
        "curriculum": {
            "phase1_end":         1_000_000,  # 1M teljes felhajtóerő (v18-cal egyező)
            "phase2_end":         3_000_000,  # 1M-3M annealing
            "max_buoyancy":       103.0,
            "stuck_window_start": 9999,
            "stuck_window_end":   40,
        },
    },
    "m2_20m_v20": {
        # v20: Grace period + Penalty curriculum + Hip lean eltávolítva + Contact clipping
        #
        # v19 diagnózis: ep=31 (v18: 166 volt!) — regresszió okai:
        #   1. Hip lean (+0.1 rad) destabilizálta az alappózt → elesik ~30 lépésen belül
        #   2. backward_window=30 grace period nélkül → korai terminálás (0.6s, robot nem stabilizált)
        #   3. contact_cost clippolás nélkül → -30 per step eséskor → tanulás blokkolva
        #   4. Büntetések teljes erővel kezdettől → felfedezés gátolva (forward grad = 0)
        #
        # v20 négy fix (env-ben implementálva):
        #   1. Hip lean ELTÁVOLÍTVA: visszatérés v18 alappózhoz
        #   2. Grace period = 150 lép (~3s): backward terminálás csak utána
        #   3. Penalty curriculum: orientation + action_rate + dof_acc → 0.0→1.0 skálázva
        #      Ugyanabban az 1M-3M ablakban ahol a buoyancy is nullára megy
        #   4. Contact clipping: cfrc_ext clip[-1, 1] → stabil contact_cost
        #
        # Curriculum összehangolás:
        #   0..1M:   teljes buoyancy (103N), penalty_scale=0.0 (szabad felfedezés)
        #   1M..3M:  buoyancy 103N→0N, penalty_scale 0.0→1.0 (párhuzamos annealing)
        #   3M..20M: buoyancy=0N, penalty_scale=1.0 (teljes tanulás)
        "total_timesteps": 20_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "description": "M2 CPU ~2 óra (v20: grace period + penalty curriculum + no hip lean + contact clip)",
        "curriculum": {
            "phase1_end":         1_000_000,  # 1M teljes buoyancy (v18/v19-cel egyező)
            "phase2_end":         3_000_000,  # 1M-3M: buoyancy + penalty párhuzamos annealing
            "max_buoyancy":       103.0,
            "stuck_window_start": 9999,
            "stuck_window_end":   40,
            "penalty_start":      1_000_000,  # [ÚJ v20] penalty curriculum start
            "penalty_end":        3_000_000,  # [ÚJ v20] penalty curriculum vége (= phase2_end)
        },
    },
    "m2_20m_v22": {
        # v22: FRISS TANÍTÁS (nem fine-tune!) — 3 fizikai fix + reward rebalance
        #
        # v21 diagnózis: ep=86, reward=-317, dist=3.18m (12cm haladás 3.3m-ből)
        #   Fő ok: fine-tune a v20-ból → beégett álló/forgó viselkedés nem tanulható ki
        #   Reward hiba: w_healthy=0.05 > w_dist×max_step=0.04 → állás jobban fizet!
        #   Fizikai hiba: egyenes lábból lábemelés → torzó billenés → esés (86. lépésnél)
        #
        # v22 négy változtatás (env v22-ben implementálva):
        #   A. Guggoló alappóz: hip_pitch=-0.1, knee=+0.3, ankle_pitch=-0.2 rad
        #      Forrás: Unitree unitree_rl_gym official config
        #   B. Lábcsúszás büntetés: w_feet_slip=-0.1 (talajon lévő láb ne csússzon)
        #   C. Lábak távolság büntetés: w_feet_distance=-1.0 (min 0.15m, ne keresztezzék)
        #   D. Reward rebalance: w_healthy=0.01, w_dist=8.0, w_orientation=-2.0
        #
        # Curriculum: azonos v20-zal (buoyancy + penalty annealing)
        #   0..1M:   teljes buoyancy (103N), penalty_scale=0.0
        #   1M..3M:  buoyancy 103N→0N, penalty_scale 0.0→1.0
        #   3M..20M: buoyancy=0N, penalty_scale=1.0
        "total_timesteps": 20_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "description": "M2 CPU ~2 óra (v22: guggoló póz + lábcsúszás/távolság + reward rebalance)",
        "curriculum": {
            "phase1_end":         1_000_000,
            "phase2_end":         3_000_000,
            "max_buoyancy":       103.0,
            "stuck_window_start": 9999,
            "stuck_window_end":   40,
            "penalty_start":      1_000_000,
            "penalty_end":        3_000_000,
        },
    },
    "m2_5m_v9": {
        # v9: tracking reward (sebesség × célirány), w_healthy=0.05, w_forward=8.0
        # Humanoid-v4 mintájára: sebesség-alapú forward reward folyamatos gradienst ad
        "total_timesteps": 5_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 1e-4,
        "clip_range": 0.15,
        "description": "M2 CPU ~30 perc (v9: tracking_reward, w_healthy=0.05, batch=512)",
    },
    "m2_10m_v10": {
        # v10: PONTOS Humanoid-v4 reward struktúra portolva G1-re
        # forward=1.25×velocity, healthy=5.0 (fix), ctrl=-0.1×action², fall=0
        # ACTION_SCALE=0.3 véd a stand-and-fall ellen (nem a healthy csökkentése)
        "total_timesteps": 10_000_000,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "description": "M2 CPU ~1 óra (v10: Humanoid-v4 reward portolva G1-re, 10M lépés)",
    },
}


def make_retail_env(n_envs=1, seed=42):
    """Retail nav env létrehozása VecNormalize-zel."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed

    # Importáljuk a retail nav env-et
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    def make_single(rank):
        def _init():
            env = RoboshelfRetailNavEnv()
            env.reset(seed=seed + rank)
            return env
        return _init

    set_random_seed(seed)
    if n_envs > 1:
        env = SubprocVecEnv([make_single(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_single(0)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env


def make_eval_env():
    """Eval env létrehozása.

    Megjegyzés: a VecNormalize statisztikát a tanítási env-ből szinkronizáljuk
    az EvalCallback sync_vec_normalize=True opcióval, ezért itt training=False.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    env = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    # norm_reward=False: eval-nál nem normalizáljuk a rewardot (valódi értékeket látunk)
    # training=False: a statisztika nem frissül eval közben
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    return env


def train(args):
    """Fő tanítási loop."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import torch

    cfg = LEVELS[args.level]
    timestamp = int(time.time())
    run_name = f"g1_retail_nav_{args.level}_{timestamp}"

    # Device meghatározás: CUDA > CPU
    # MPS (Apple Silicon) szándékosan kizárva: MlpPolicy float64-et használ,
    # amit az MPS framework nem támogat. CPU gyorsabb is MLP esetén.
    if torch.cuda.is_available():
        device = "cuda"
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU (M2 optimalizált)"

    print(f"\n{'='*60}")
    print(f"  ROBOSHELF AI — Fázis 2: G1 Retail Navigáció")
    print(f"  Szint: {args.level} ({cfg['description']})")
    print(f"  Timesteps: {cfg['total_timesteps']:,}")
    print(f"  Envs: {cfg['n_envs']}, Batch: {cfg['batch_size']}")
    print(f"  Device: {device_label}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Könyvtárak
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = MODELS_DIR / "best"
    best_dir.mkdir(exist_ok=True)

    # Környezet
    print("  Környezetek létrehozása...")
    env = make_retail_env(n_envs=cfg["n_envs"])
    eval_env = make_eval_env()

    print(f"  ✅ {cfg['n_envs']}× RoboshelfRetailNav env kész")
    print(f"  Obs space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # PPO modell
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=cfg["clip_range"],
        ent_coef=cfg.get("ent_coef", 0.001),
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(LOGS_DIR),
        verbose=1,
        seed=42,
        device=device,
    )

    # VecNormalize szinkronizáló callback (train → eval stats másolás)
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    class SyncVecNormalizeCallback(BaseCallback):
        """Eval előtt szinkronizálja a VecNormalize statisztikát train env-ből."""
        def __init__(self, train_env, eval_env_ref):
            super().__init__()
            self.train_env = train_env
            self.eval_env_ref = eval_env_ref

        def _on_step(self):
            return True

        def on_eval_start(self):
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env_ref, VecNormalize):
                self.eval_env_ref.obs_rms = self.train_env.obs_rms
                self.eval_env_ref.ret_rms = self.train_env.ret_rms

    sync_cb = SyncVecNormalizeCallback(env, eval_env)

    # Curriculum callback (v17: felhajtóerő + stuck-window annealing)
    curriculum_cfg = cfg.get("curriculum", None)

    class CurriculumCallback(BaseCallback):
        """
        Három párhuzamos annealing a curriculum tanításhoz:

        1. Felhajtóerő (buoyancy): gravitáció X%-át ellensúlyozza a pelvis-en
           - 0..phase1_end:          max_buoyancy (teljes segítség)
           - phase1_end..phase2_end: lineáris csökkentés max→0
           - phase2_end..:           0 (teljes súly)

        2. Stuck-window: beragadás detektálásának időablaka
           - 0..phase1_end:          stuck_window_start (pl. 9999 = kikapcsolva)
           - phase1_end..phase2_end: lineáris csökkentés start→end
           - phase2_end..:           stuck_window_end (pl. 40)

        3. Penalty scale [ÚJ v20]: orientációs + simasági büntetések skálázása
           - 0..penalty_start:       0.0 (nincs büntetés — szabad felfedezés)
           - penalty_start..penalty_end: lineáris növelés 0→1
           - penalty_end..:          1.0 (teljes büntetés — hardware-ready policy)
           Ha penalty_start/end nincs megadva a config-ban, penalty_scale fixen 1.0 marad.

        A callback set_attr()-ral írja az env példányok attribútumait,
        ami SubprocVecEnv esetén is működik (SB3 API).
        """
        def __init__(self, train_env, total_steps, ccfg, log_interval=50_000):
            super().__init__()
            self.train_env = train_env
            self.total_steps = total_steps
            self.ccfg = ccfg
            self.log_interval = log_interval
            self._last_log = 0

        def _on_step(self):
            t = self.num_timesteps
            c = self.ccfg
            phase1 = c["phase1_end"]
            phase2 = c["phase2_end"]

            # --- Felhajtóerő annealing ---
            max_b = c["max_buoyancy"]
            if t <= phase1:
                buoyancy = max_b
            elif t <= phase2:
                frac = (t - phase1) / (phase2 - phase1)
                buoyancy = max_b * (1.0 - frac)
            else:
                buoyancy = 0.0

            # --- Stuck-window annealing ---
            sw_start = c["stuck_window_start"]
            sw_end   = c["stuck_window_end"]
            if t <= phase1:
                stuck_window = sw_start
            elif t <= phase2:
                frac = (t - phase1) / (phase2 - phase1)
                stuck_window = int(sw_start + frac * (sw_end - sw_start))
            else:
                stuck_window = sw_end

            # --- Penalty scale annealing [ÚJ v20] ---
            # Ha a config tartalmaz penalty_start/penalty_end-et, akkor skálázódik.
            # Ha nem (régi konfig-ok visszafelé kompatibilitása), fixen 1.0 marad.
            p_start = c.get("penalty_start", None)
            p_end   = c.get("penalty_end", None)
            if p_start is None or p_end is None:
                penalty_scale = 1.0  # régi konfig: teljes büntetés (v17-v19 viselkedés)
            elif t <= p_start:
                penalty_scale = 0.0
            elif t <= p_end:
                penalty_scale = (t - p_start) / (p_end - p_start)
            else:
                penalty_scale = 1.0

            # Env attribútumok frissítése
            self.train_env.set_attr("buoyancy_force", buoyancy)
            self.train_env.set_attr("stuck_window", stuck_window)
            self.train_env.set_attr("penalty_scale", penalty_scale)

            # Logolás (ritkán, hogy ne lassítson)
            if t - self._last_log >= self.log_interval:
                self._last_log = t
                print(f"\n  [Curriculum] {t:,} lép | buoyancy={buoyancy:.0f}N "
                      f"({buoyancy/c['max_buoyancy']*100:.0f}%) | stuck_window={stuck_window} "
                      f"| penalty_scale={penalty_scale:.2f}")
            return True

    callbacks = [sync_cb]
    if curriculum_cfg is not None:
        curriculum_cb = CurriculumCallback(env, cfg["total_timesteps"], curriculum_cfg)
        callbacks.append(curriculum_cb)
        print(f"  🎓 Curriculum aktív: buoyancy {curriculum_cfg['max_buoyancy']:.0f}N→0, "
              f"stuck_window {curriculum_cfg['stuck_window_start']}→{curriculum_cfg['stuck_window_end']}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(LOGS_DIR),
        eval_freq=max(cfg["total_timesteps"] // 20, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg["total_timesteps"] // 10, 10000),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=run_name,
    )

    # Tanítás
    print(f"\n  🚀 Tanítás indítása...\n")
    start = time.time()

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks + [eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\n  ⏱️  Tanítás befejezve: {elapsed/60:.1f} perc ({elapsed/3600:.1f} óra)")

    # Mentés
    final_model = str(MODELS_DIR / f"{run_name}_final")
    model.save(f"{final_model}.zip")
    env.save(f"{final_model}_vecnormalize.pkl")
    env.save(str(best_dir / "best_model_vecnormalize.pkl"))
    print(f"  💾 Modell: {final_model}.zip")
    print(f"  💾 VecNormalize: {final_model}_vecnormalize.pkl")

    # Kiértékelés
    print(f"\n  📊 Kiértékelés...")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize as VN
    from roboshelf_retail_nav_env import RoboshelfRetailNavEnv

    ev = DummyVecEnv([lambda: RoboshelfRetailNavEnv()])
    ev = VN.load(f"{final_model}_vecnormalize.pkl", ev)
    ev.training = False
    ev.norm_reward = False

    rewards, lengths, dists = [], [], []
    for ep in range(10):
        obs = ev.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = ev.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        if 'dist_to_target' in info[0]:
            dists.append(info[0]['dist_to_target'])
        print(f"    Ep {ep+1}: reward={total_reward:.1f}, lépés={steps}, táv={dists[-1] if dists else '?':.2f}m")

    print(f"\n    📈 Átlag reward: {np.mean(rewards):.1f} (±{np.std(rewards):.1f})")
    print(f"    📏 Átlag hossz: {np.mean(lengths):.0f}")
    if dists:
        print(f"    📍 Átlag távolság céltól: {np.mean(dists):.2f}m (start: 3.3m)")

    env.close()
    eval_env.close()
    ev.close()

    print(f"\n  ✅ Fájlok elmentve: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboshelf AI — G1 Retail Nav PPO")
    parser.add_argument("--level", choices=list(LEVELS.keys()), default="teszt")
    args = parser.parse_args()
    train(args)
