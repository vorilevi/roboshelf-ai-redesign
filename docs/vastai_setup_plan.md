# Vast.ai Beállítási Terv — EAN / Isaac Lab Track

_Létrehozva: 2026-04-25 | Phase 030 | Bizalmas, belső használatra_

Ez a dokumentum a Vast.ai infrastruktúra teljes felépítési tervét rögzíti az EAN / Isaac Lab szál számára. A Phase 030 végrehajtási tervből ([`roboshelf_execution_plan_2026-04-22 pharse 030.md`](../roboshelf_execution_plan_2026-04-22 pharse 030.md)) következik: a Vast.ai az Isaac Lab Docker futtatás és a VLA inference (F4-től) kizárólagos platformja.

**Kapcsolódó dokumentumok:**
- `docs/vastai_schedule.md` — lépésről lépésre ütemezés parancsokkal
- `docs/vastai_known_issues.md` — Vast.ai hibák + AI hibák naplója
- `docs/known_issues.md` — általános projekt hibák

---

## Miért Vast.ai?

| Feladat | Platform | Indok |
|---|---|---|
| Isaac Lab Docker futtatás | **Vast.ai** | NVIDIA Container Toolkit szükséges, Mac M2-n nem fut |
| VLA inference (F4 — WALL-OSS, UnifoLM, GR00T) | **Vast.ai A100** | CUDA 12.x kötelező, 40GB+ VRAM |
| Retail VLA fine-tune (F5, ~500 epizód) | **Vast.ai A100** | Multi-epoch GPU training |
| MuJoCo + SB3 PPO (F2, F3) | **Mac M2** | CPU fallback elég, gyors iteráció |
| HEIS adapter, PIL DB, stub tesztek | **Mac M2** | Pure Python |

**Ökölszabály:** MuJoCo + PPO → Mac M2. Isaac Lab + VLA → Vast.ai (F3-tól).

---

## Architektúra áttekintés

```
Mac M2 (lokális fejlesztés)
  └─ git push → GitHub (vorilevi/roboshelf-ai-redesign)
                    └─ git pull → Vast.ai instance
                                    ├─ Isaac Lab Docker container
                                    │   └─ EAN RetailPickEnv
                                    └─ VLA inference server (F4-től)
                                        ├─ WALL-OSS
                                        ├─ UnifoLM-VLA-0
                                        └─ GR00T N1.6
```

**Checkpoint visszamentés:**
```
Vast.ai instance → roboshelf-results/ → rsync/scp → Mac M2 lokális
```
(A `roboshelf-results/` nem kerül Gitbe — manuális mentés szükséges!)

---

## 1. Fiók előkészítés

### 1.1 Billing feltöltés
- URL: https://vast.ai/billing
- Minimális feltöltés: **$20** (első Isaac Lab tesztekhez elegendő)
- Ajánlott: **$50** (EAN full setup + első VLA sanity check)
- Figyelem: Vast.ai prepaid — instance leáll ha a credit elfogy, de az adatok megmaradnak a volume-on

### 1.2 SSH kulcs feltöltés
- URL: https://vast.ai/account (Account → SSH Keys)
- Generálás (ha nincs még): `ssh-keygen -t ed25519 -C "vorilevi@gmail.com" -f ~/.ssh/vastai_key`
- Feltöltendő: `~/.ssh/vastai_key.pub` tartalma

### 1.3 API kulcs
- URL: https://vast.ai/account (Account → API Keys)
- A kulcsot mentsd: `~/.vastai/api_key` fájlba (nem Gitbe!)
- Vastai CLI telepítés: `pip install vastai --break-system-packages`
- Ellenőrzés: `vastai show instances`

---

## 2. Instance kiválasztás

### 2.1 EAN / Isaac Lab fejlesztési instance (F0–F3)

Isaac Lab futtatáshoz elegendő egy **RTX 3090 vagy RTX 4090** instance:

| Paraméter | Érték |
|---|---|
| GPU | RTX 3090 / RTX 4090 (24GB VRAM) |
| RAM | ≥ 32 GB |
| Disk | ≥ 50 GB |
| CUDA | 12.1+ |
| OS image | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| Becsült ár | $0.20–0.50/óra |

Keresési parancs (vastai CLI):
```bash
vastai search offers 'gpu_name=RTX_3090 cuda_vers>=12.1 disk_space>=50 ram>=32' --order dph_total
```

### 2.2 VLA inference instance (F4-től)

A/B/C teszt (WALL-OSS, UnifoLM, GR00T) és retail fine-tune:

| Paraméter | Érték |
|---|---|
| GPU | A100 80GB (ideális) / A100 40GB (minimális) |
| RAM | ≥ 64 GB |
| Disk | ≥ 100 GB |
| CUDA | 12.1+ |
| Becsült ár | $1.50–3.00/óra |

---

## 3. Docker image stratégia

### 3.1 EAN / Isaac Lab container

**Alap image:** `nvcr.io/nvidia/isaac-lab:latest` (Isaac Lab official)

Alternatív (ha az official nem elérhető):
```
nvcr.io/nvidia/pytorch:24.01-py3
```
majd manuális Isaac Lab telepítés.

**Docker run parancs (Vast.ai SSH-on belül):**
```bash
docker run --gpus all \
  -v /workspace/roboshelf:/workspace/roboshelf \
  -w /workspace/roboshelf \
  --rm -it nvcr.io/nvidia/isaac-lab:latest \
  bash
```

### 3.2 Container-en belüli setup
```bash
# Repo klónozás
cd /workspace
git clone https://github.com/vorilevi/roboshelf-ai-redesign.git
cd roboshelf-ai-redesign

# Roboshelf-common install
pip install -e roboshelf-common/

# EAN track dependency
pip install -r requirements.txt

# Sanity check
python -c "import isaaclab; print('Isaac Lab OK')"
```

---

## 4. Munkafolyamat — napi használat

### 4.1 Instance indítás (vastai CLI)
```bash
# Instance lista (rendelkezésre álló)
vastai search offers 'gpu_name=RTX_3090 cuda_vers>=12.1' --order dph_total

# Instance indítás (pl. offer_id=12345)
vastai create instance 12345 \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --disk 60 \
  --ssh

# Futó instance-ok listája
vastai show instances

# SSH csatlakozás
ssh -i ~/.ssh/vastai_key root@<ip> -p <port>
```

### 4.2 Kód szinkron (Mac → Vast.ai)
```bash
# Friss kód feltolás
rsync -avz --exclude='.git' --exclude='roboshelf-results/' \
  /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/ \
  root@<ip>:/workspace/roboshelf-ai-redesign/ -p <port>
```

### 4.3 Checkpoint visszamentés (Vast.ai → Mac)
```bash
rsync -avz \
  root@<ip>:/workspace/roboshelf-ai-redesign/roboshelf-results/ \
  /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/roboshelf-results/ -p <port>
```

### 4.4 Instance leállítás (cost control!)
```bash
# Instance leállítás (billing megáll, adatok megmaradnak a volume-on)
vastai stop instance <instance_id>

# Instance törlés (adatok törlődnek — csak ha checkpoint vissza van mentve!)
vastai destroy instance <instance_id>
```

**FONTOS:** Mindig mentsd vissza a checkpointokat rsync-kel mielőtt destroy-olod az instance-t!

---

## 5. Isaac Lab EAN track — első futtatás terve

### 5.1 Sanity check (F0 mérföldkő)
```bash
# Isaac Lab container-ben:
python -c "
import isaaclab
from isaaclab.envs import ManagerBasedRLEnv
print('Isaac Lab import OK')
"
```

### 5.2 RetailPickEnv első reset
```bash
# EAN repo (külön clone kell: vorilevi/roboshelf-ai-EAN)
git clone https://github.com/vorilevi/roboshelf-ai-EAN.git
cd roboshelf-ai-EAN
python train.py --task Roboshelf-RetailPick-Play-v0 --headless
```

Várt eredmény: env betölt, reset lefut, obs shape kiírva, 0-reward step loop fut crash nélkül.

### 5.3 Scene modul teszt
```bash
python -c "
from envs.roboshelf.scene.retail_scene_cfg import RetailSceneCfg
cfg = RetailSceneCfg()
print(f'Scene OK — num_envs: {cfg.num_envs}')
"
```

---

## 6. Kockázatok és mitigáció

| Kockázat | Valószínűség | Mitigáció |
|---|---|---|
| Isaac Lab image nem elérhető nvcr.io-n | Közepes | Manuális telepítés PyTorch image-ből |
| Credit elfogy futás közben | Közepes | Credit alert beállítás $5-nél, checkpoint callback |
| SSH timeout hosszú training alatt | Alacsony | `tmux` vagy `screen` session-ban futtatni |
| Checkpoint nem mentődött (destroy előtt) | Közepes | Mindig rsync checkpoint-ok ELŐTT destroy |
| Vast.ai instance nem talál megfelelő GPU-t | Alacsony | Rugalmas GPU lista (3090/4090/A100) |

---

## 7. Kapcsolódó dokumentumok

- `docs/vastai_schedule.md` — részletes lépéssor parancsokkal
- `docs/vastai_known_issues.md` — hibák és AI hibák naplója
- `docs/known_issues.md` — általános projekt hibák
- `docs/vla_abc_test_protocol.md` — VLA A/B/C teszt protokoll (F4)
- Obsidian: `[[roboshelf_vastai_setup_2026-04-25]]`
