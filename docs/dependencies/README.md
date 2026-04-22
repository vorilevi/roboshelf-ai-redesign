# Külső dependency SHA pinok — Phase 030

Ez a mappa tárolja a három kulcs külső repo commit SHA pinját.
A fájlok a fork létrehozásakor generálódnak (1. Fázis, ápr. 26–máj. 9).

## Miért szükséges a SHA pin?

A kínai open-source repók aktívan fejlesztés alatt állnak — breaking change bármikor
jöhet. A konkrét commit SHA rögzítésével biztosítjuk, hogy a saját fork mindig
egy ismert, tesztelt állapotra mutat vissza, és frissítés csak tudatos döntés után
történik.

## Fájlok (1. Fázis után léteznek)

| Fájl | Repo | Fork neve |
|---|---|---|
| `unitree_rl_mjlab_pinned_sha.txt` | `unitree-robotics/unitree_rl_mjlab` | `vorilevi/unitree_rl_mjlab_roboshelf` |
| `wallx_pinned_sha.txt` | `XSquareRobot/WallX` | `vorilevi/wallx_roboshelf` |
| `unifolm_pinned_sha.txt` | `unitreerobotics/UnifoLM` | `vorilevi/unifolm_roboshelf` |

## Frissítés menete

Ha egy upstream commitot be akarunk venni a saját forkba:

```bash
# Pl. unitree_rl_mjlab frissítés
cd /Users/vorilevi/roboshelf-ai-dev/unitree_rl_mjlab_roboshelf
git fetch upstream
git log upstream/main --oneline -10   # megnézzük a változásokat
# Ha rendben van:
git merge upstream/main
git log -1 --format="%H %ci" > /Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/docs/dependencies/unitree_rl_mjlab_pinned_sha.txt
git push origin main
```

## Lokális HuggingFace mirror (kínai licenc kockázat mitigáció)

A pretrained checkpointokat a fork létrehozásakor lokálisan tükrözzük:

```bash
# WALL-OSS checkpoint
huggingface-cli download XSquareRobot/wall-oss-flow-v0.1 \
  --local-dir /Users/vorilevi/roboshelf-ai-dev/hf-mirrors/wall-oss-flow-v0.1/

# UnifoLM-VLA-0
huggingface-cli download unitreerobotics/UnifoLM-VLA-0 \
  --local-dir /Users/vorilevi/roboshelf-ai-dev/hf-mirrors/UnifoLM-VLA-0/
```
