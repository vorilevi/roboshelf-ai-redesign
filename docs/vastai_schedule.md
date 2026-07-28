# Vast.ai Beállítás — Lépésről lépésre ütemezés

_Létrehozva: 2026-04-25 | Phase 030_

Ez a fájl a `vastai_setup_plan.md` elveit bontja le konkrét, sorban végrehajtható lépésekre. Minden lépés egy terminálon vagy böngészőben végrehajtható atomi egység.

**Alapelv:** Egy lépés → ellenőrzés → következő lépés. Soha ne ugorj lépést át.

---

## Státusz tábla

| Lépés | Leírás | Státusz |
|---|---|---|
| S1 | Billing feltöltés | ⏳ Folyamatban |
| S2 | SSH kulcs generálás + feltöltés | ⏳ Vár |
| S3 | API kulcs + vastai CLI | ⏳ Vár |
| S4 | Instance keresés és indítás | ⏳ Vár |
| S5 | Docker + Isaac Lab sanity check | ⏳ Vár |
| S6 | EAN repo klónozás + első reset | ⏳ Vár |
| S7 | RetailPickEnv Scene teszt | ⏳ Vár |
| S8 | Checkpoint mentési workflow teszt | ⏳ Vár |

_Frissítsd ezt a táblát minden befejezett lépésnél!_

---

## S1 — Billing feltöltés

**Hol:** https://vast.ai/billing → „Add Credit"

**Összeg:** $20 minimum (ajánlott $50)

**Ellenőrzés:** A billing oldalon megjelenik a credit egyenleg.

**Elfogadási feltétel:** ✅ Legalább $20 credit látható a billing oldalon.

---

## S2 — SSH kulcs generálás és feltöltés

### S2.1 SSH kulcs generálás (Mac terminál)

```bash
# Ellenőrzés — van-e már kulcs?
ls ~/.ssh/

# Ha nincs ed25519 kulcs, generálj újat
ssh-keygen -t ed25519 -C "vorilevi@gmail.com" -f ~/.ssh/vastai_key
# Passphrase: hagyhatod üresen (Enter), vagy adj meg jelszót

# Public key tartalmának megjelenítése — ezt kell a Vast.ai-ba másolni
cat ~/.ssh/vastai_key.pub
```

### S2.2 SSH kulcs feltöltés Vast.ai-ba

**Hol:** https://vast.ai/account → „SSH Keys" → „Add SSH Key"

**Mit:** A `cat ~/.ssh/vastai_key.pub` kimenete (az egész sor, `ssh-ed25519`-től kezdve)

**Elfogadási feltétel:** ✅ A vastai.com account oldalán látható az SSH kulcs fingerprint.

---

## S3 — API kulcs és vastai CLI

### S3.1 API kulcs megszerzése

**Hol:** https://vast.ai/account → „API Keys" → „Generate API Key"

**Mentés (Mac terminál):**
```bash
mkdir -p ~/.vastai
# A következő sorba írd be a saját API kulcsodat:
echo "IDE_JON_AZ_API_KULCS" > ~/.vastai/api_key
chmod 600 ~/.vastai/api_key
```

### S3.2 Vastai CLI telepítés

```bash
pip install vastai --break-system-packages
```

### S3.3 CLI konfigurálás és ellenőrzés

```bash
vastai set api-key $(cat ~/.vastai/api_key)

# Ellenőrzés — futó instance-ok listája (egyelőre üres lesz)
vastai show instances
```

**Elfogadási feltétel:** ✅ `vastai show instances` hibátlanul fut (üres lista is OK).

---

## S4 — Instance keresés és indítás

### S4.1 Megfelelő instance keresése

```bash
# RTX 3090 instance-ok (Isaac Lab fejlesztéshez elég)
vastai search offers \
  'gpu_name=RTX_3090 cuda_vers>=12.1 disk_space>=50 ram>=32 num_gpus=1' \
  --order dph_total

# Ha 3090 nincs, 4090 is jó
vastai search offers \
  'gpu_name=RTX_4090 cuda_vers>=12.1 disk_space>=50 ram>=32 num_gpus=1' \
  --order dph_total
```

Jegyezd fel a kiválasztott offer `ID`-ját (első oszlop).

### S4.2 Instance indítás

```bash
# Csere: OFFER_ID = a keresésből kapott szám
vastai create instance OFFER_ID \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --disk 60 \
  --ssh \
  --env '-p 8888:8888'

# Instance ID lekérése
vastai show instances
```

Jegyezd fel az `INSTANCE_ID`-t és az SSH csatlakozási adatokat (`ssh_host`, `ssh_port`).

### S4.3 SSH csatlakozás tesztelés

```bash
# Csere: SSH_HOST és SSH_PORT a show instances kimenetéből
ssh -i ~/.ssh/vastai_key -p SSH_PORT root@SSH_HOST

# Ha sikerült:
nvidia-smi   # GPU ellenőrzés
nvcc --version   # CUDA verzió
```

**Elfogadási feltétel:** ✅ `nvidia-smi` kimenete látható, CUDA 12.x megerősítve.

---

## S5 — Docker + Isaac Lab sanity check

### S5.1 Isaac Lab image próba

```bash
# Az instance-on (SSH-on belül):

# 1. Isaac Lab official image próba
docker pull nvcr.io/nvidia/isaac-lab:latest

# Ha sikertelen (image nem elérhető), PyTorch base image + manuális telepítés
# (lásd vastai_known_issues.md #V1)
```

### S5.2 Isaac Lab container indítás

```bash
# Instance-on:
docker run --gpus all \
  -v /workspace:/workspace \
  -w /workspace \
  --rm -it nvcr.io/nvidia/isaac-lab:latest \
  bash

# Vagy PyTorch base image esetén:
docker run --gpus all \
  -v /workspace:/workspace \
  -w /workspace \
  --rm -it nvcr.io/nvidia/pytorch:24.01-py3 \
  bash
```

### S5.3 Alap sanity check (container-en belül)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import isaaclab; print('Isaac Lab OK')"   # csak official image esetén
```

**Elfogadási feltétel:** ✅ `torch.cuda.is_available()` → `True`, GPU neve kiírva.

---

## S6 — EAN repo klónozás és első reset

### S6.1 Repo klónozás (container-en belül)

```bash
# /workspace-ben:
git clone https://github.com/vorilevi/roboshelf-ai-EAN.git
cd roboshelf-ai-EAN

# roboshelf-common is kell (redesign repo-ból)
git clone https://github.com/vorilevi/roboshelf-ai-redesign.git
pip install -e roboshelf-ai-redesign/roboshelf-common/

# EAN requirements
pip install -r requirements.txt 2>/dev/null || echo "requirements.txt hiányzik vagy üres"
```

### S6.2 RetailPickEnv első reset

```bash
cd /workspace/roboshelf-ai-EAN

python train.py --task Roboshelf-RetailPick-Play-v0 --headless --num_envs 1
```

Várt kimenet:
```
Loading task: Roboshelf-RetailPick-Play-v0
Env reset OK — obs shape: (1, 4)
Step loop running...
```

**Elfogadási feltétel:** ✅ Env betölt, reset hibátlan, step loop fut 100 lépésen.

---

## S7 — RetailPickEnv Scene teszt

```bash
cd /workspace/roboshelf-ai-EAN

python -c "
from envs.roboshelf.scene.retail_scene_cfg import RetailSceneCfg
cfg = RetailSceneCfg()
print(f'Scene OK — num_envs: {cfg.num_envs}, spacing: {cfg.env_spacing}')
"

python -c "
from envs.roboshelf.configs.retail_pick_env_cfg import RetailPickEnvCfg
cfg = RetailPickEnvCfg()
print(f'Config OK — episode_length: {cfg.episode_length_s}s')
"
```

**Elfogadási feltétel:** ✅ Scene és config importálható, paraméterek megjelennek.

---

## S8 — Checkpoint mentési workflow teszt

### S8.1 Test checkpoint generálás

```bash
# Instance-on (rövid futtatás checkpoint generáláshoz):
cd /workspace/roboshelf-ai-EAN
python train.py --task Roboshelf-RetailPick-Play-v0 --headless --num_envs 4 \
  --max_iterations 100
```

### S8.2 Rsync visszamentés Mac-re

```bash
# Mac terminálban (NEM az instance-on):
rsync -avz \
  -e "ssh -i ~/.ssh/vastai_key -p SSH_PORT" \
  root@SSH_HOST:/workspace/roboshelf-ai-EAN/logs/ \
  /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/roboshelf-results/ean/
```

**Elfogadási feltétel:** ✅ Checkpoint fájlok megjelennek a Mac-en a `roboshelf-results/ean/` mappában.

---

## Napi használati cheat sheet

```bash
# Instance indítás
vastai start instance INSTANCE_ID

# SSH csatlakozás
ssh -i ~/.ssh/vastai_key -p PORT root@HOST

# Tmux session (hosszú futtatáshoz — ne veszítsd el SSH timeout miatt)
tmux new -s training
# Kilépés tmux-ból de folytatás: Ctrl+B, D
# Visszacsatlakozás: tmux attach -t training

# Instance megállítás (billing szünetel, adatok megmaradnak)
vastai stop instance INSTANCE_ID

# Instance törlés (CSAK checkpoint mentés UTÁN!)
vastai destroy instance INSTANCE_ID
```

---

## Becsült költségek

| Feladat | GPU | Idő | Becsült cost |
|---|---|---|---|
| S4–S8 setup + sanity check | RTX 3090 | 2 óra | ~$0.60 |
| RetailPickEnv Scene fejlesztés (F3) | RTX 3090 | 10 óra | ~$3.00 |
| VLA A/B/C teszt (F4, 3×50 epizód) | A100 40GB | 8 óra | ~$16.00 |
| Retail fine-tune (F5, 500 epizód) | A100 40GB | 20 óra | ~$40.00 |
| **Teljes Phase 030 Vast.ai cost** | | | **~$60–80** |
