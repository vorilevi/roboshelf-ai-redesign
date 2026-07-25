# Vast.ai — Ismert hibák és AI hibák naplója

_Létrehozva: 2026-04-25 | Phase 030_

Ez a fájl két célt szolgál:
1. **Vast.ai specifikus hibák** — instance, Docker, SSH, Isaac Lab problémák
2. **AI-asszisztens hibák naplója** — rossz parancsok, fölösleges próbálkozások, félreértések, megismételt lépések

A második szekció különösen fontos: ha az AI ugyanazt a hibát kétszer csinálja, az rögzítendő ide. Ez a projekt visszajelzési mechanizmusa az AI asszisztens számára.

---

# I. VAST.AI ISMERT HIBÁK

## V1. Isaac Lab official Docker image nem elérhető nvcr.io-n

**Tünet:**
```
docker pull nvcr.io/nvidia/isaac-lab:latest
Error response from daemon: manifest unknown
```

**Root cause:** Az Isaac Lab official container image neve változhat, vagy NGC regisztráció szükséges.

**Megoldás:**
1. NGC token regisztráció: https://ngc.nvidia.com/setup/api-key
2. Docker login: `docker login nvcr.io -u '$oauthtoken' -p NGC_API_KEY`
3. Alternatíva: `nvcr.io/nvidia/pytorch:24.01-py3` + manuális Isaac Lab telepítés:
   ```bash
   pip install isaaclab isaacsim-rl --extra-index-url https://pypi.nvidia.com
   ```

---

## V2. SSH csatlakozás timeout hosszú training alatt

**Tünet:** SSH kapcsolat megszakad, training folyamat elveszik.

**Megoldás:** Mindig `tmux` session-ban futtatni:
```bash
tmux new -s training
python train.py ...
# Ctrl+B, D → kilépés de a session fut
# Visszacsatlakozás: tmux attach -t training
```

---

## V3. Vastai CLI — „unauthorized" hiba

**Tünet:**
```
vast: error: unauthorized
```

**Megoldás:**
```bash
vastai set api-key $(cat ~/.vastai/api_key)
```
Ha az API kulcs fájl nem létezik: lekérni a https://vast.ai/account oldalról, majd `echo "KULCS" > ~/.vastai/api_key`.

---

## V4. Instance credit elfogy futás közben — folyamat elvész

**Tünet:** Instance hirtelen leáll, checkpoint nem mentődött.

**Megelőzés:**
- Credit alert beállítás: https://vast.ai/billing → Alerts → Alert at $5
- `CheckpointCallback(save_freq=50000, save_vecnormalize=True)` minden training scriptben
- Rsync checkpoint-ok naponta legalább egyszer

---

## V5. Docker container-on belül nincs internet (roboshelf-common telepítés)

**Tünet:**
```
pip install -e roboshelf-common/  →  Network unreachable
```

**Megoldás:** A Vast.ai instance-on (container kívül) előre fel kell telepíteni, vagy a repo-t bind mounttal belülre vinni:
```bash
# Container kívül (instance shell-ben):
pip install -e /workspace/roboshelf-ai-redesign/roboshelf-common/
```

---

## V6. `rsync` port megadása — hibás szintaxis

**Helytelen:**
```bash
rsync -avz root@HOST:/path/ /local/ -p PORT   # ← a -p az rsync-nek mást jelent!
```

**Helyes:**
```bash
rsync -avz -e "ssh -i ~/.ssh/vastai_key -p PORT" root@HOST:/path/ /local/
```

---

# II. AI ASSZISZTENS HIBÁK NAPLÓJA

_Ez a szekció az AI asszisztens (Claude) saját hibáit, rossz parancsait, fölösleges próbálkozásait rögzíti. Célja a jövőbeli ismétlések elkerülése._

**Formátum:**
```
## AI-N. [Dátum] Rövid leírás
Kontextus: mi volt a feladat
Hiba: mit csinált rosszul az AI
Következmény: mi lett belőle
Tanulság: hogyan kell helyesen
```

---

## AI-1. [2026-04-25] Vast.ai beállítás dokumentáció nélkül indult

**Kontextus:** Felhasználó kérte a Vast.ai beállítását. Az AI azonnal a billing oldalra irányított, dokumentáció és ütemezés nélkül.

**Hiba:** Kihagyta a tervezési és dokumentálási fázist. Ugrott az azonnali végrehajtásra anélkül, hogy a projekt konvenciói (Phase 030 stílusú tervdokumentum, Obsidian kereszthivatkozás, known_issues fájl) elkészültek volna.

**Következmény:** Felhasználónak kellett visszaküldenie, hogy előbb a dokumentáció készüljön el.

**Tanulság:** Minden új Vast.ai / infrastruktúra feladatnál először: terv + ütemezés + known_issues fájl, aztán a tényleges végrehajtás. Ez az összes Phase 030 konvenciójával megegyező sorrend.

---

## AI-2. [2026-04-25] Vast.ai API key webes beállítás lépés kihagyva

**Kontextus:** S3 lépés — vastai CLI telepítés + API key beállítás.

**Hiba:** Az ütemezésben (vastai_schedule.md S3) nem szerepelt, hogy a vast.ai webes felületen is be kell állítani az API key-t: Account → API Keys → létrehozás. Az AI feltételezte, hogy az SSH kulcs hozzáadása után az API key is automatikusan létezik.

**Következmény:** `vastai set api-key` parancs 404-es hibát adott, mert a webes felületen még nem volt API key generálva.

**Tanulság:** A Vast.ai beállítás két külön webes lépést igényel: (1) SSH key hozzáadása, (2) API key generálása — ez utóbbit az ütemezésbe explicit lépésként kell felvenni.

---

## AI-3. [2026-04-25] Isaac Lab pip install — rossz csomagnév feltételezve

**Kontextus:** S7 lépés — Isaac Lab telepítés pytorch container-ben.

**Hiba:** Az AI `pip install isaaclab` parancsot adott, amit ellenőrzés nélkül javasolt. Az Isaac Lab nem érhető el egyszerű pip csomagként — a helyes módszer az official NGC Docker image (`nvcr.io/nvidia/isaac-lab:2.3.2`) használata, nem egy általános pytorch container + pip telepítés.

**Következmény:** Felesleges hibaüzenet, időveszteség, a futó instance közben számlázódott.

**Tanulság:** Isaac Lab telepítési módját mindig a hivatalos dokumentációból kell ellenőrizni. A helyes út: Vast.ai instance-t közvetlenül az `nvcr.io/nvidia/isaac-lab:X.Y.Z` image-gel kell létrehozni, nem pytorch base image-gel.

---

## AI-4. [2026-04-25] SSH kulcs regisztráció lépés hiányzott az ütemezésből

**Kontextus:** S4-S5 — instance létrehozás és SSH csatlakozás.

**Hiba:** Az S2 lépésben SSH kulcsot generáltunk, de nem szerepelt az ütemezésben, hogy azt a Vast.ai webes felületen is regisztrálni kell (cloud.vast.ai/manage-keys/). Így az első instance-nál SSH connection refused hibát kaptunk.

**Következmény:** Extra `vastai attach ssh` parancs kellett, plusz időveszteség és bizonytalanság.

**Tanulság:** Az S2 lépést ki kell egészíteni: `ssh-keygen` után azonnal: cloud.vast.ai/manage-keys/ → Add SSH Key → publikus kulcs beillesztése. Csak ezután szabad instance-t létrehozni.

---

## AI-5. [2026-04-25] Több instance indult egyszerre — felesleges költség

**Kontextus:** Isaac Lab image tesztelése során.

**Hiba:** Az első `create instance` `success: False`-t adott vissza, de az instance mégis elindult (`loading` státuszban). Az AI nem ellenőrizte az instance státuszt a következő `create instance` parancs kiadása előtt, így 3 instance futott egyszerre.

**Következmény:** 3x $0.13/hr számlázódott egyszerre, két instance-t kellett manuálisan törölni.

**Tanulság:** `create instance` után MINDIG: `vastai show instances-v1` — ellenőrizni hogy hány instance fut, mielőtt újat indítunk. `success: False` nem jelenti, hogy nem indult el.

---

## AI-9. [2026-07-24] Checkpoint full model state_dict → disk full Kaggle-en

**Kontextus:** T1 UnifoLM-VLA-0 fine-tune, step 2000 checkpoint mentése.

**Hiba:** A `train()` függvény `model.state_dict()`-et mentett checkpointba. A 7B VLM + 588M DiT action model teljes state dictje ~14GB, ami kétszer mentve (~28GB) túllépte a Kaggle `/kaggle/working` ~20GB limitet. A step_01000.pt már ott volt, step_02000.pt mentésekor `RuntimeError: iostream error` keletkezett.

**Következmény:** Training crash step 2000-nél, az összes addigi munka elveszett.

**Tanulság:** Checkpoint mentésekor SOHA ne `model.state_dict()` — csak trainable paraméterek (`requires_grad=True`), azaz LoRA rétegek + action model (~200MB). Előző checkpointot mindig törölni kell mentés előtt. A `save_final()` menthet teljes modelt, mert az csak egyszer fut le.

---

## AI-10. [2026-07-25] Kaggle notebook névtelenség — nem tudtuk melyik a "jó"

**Kontextus:** T1 eval notebook futtatása Kaggle-en. Több notebookot hoztunk létre (kaggle_vla_eval_t1 javítások közben), de a nevek auto-generáltak maradtak (pl. `notebook7b97304d37`).

**Hiba:** Nem adtunk egyedi, leíró nevet a notebookoknak létrehozáskor, és nem dokumentáltuk melyik verzió volt sikeres. Ennek következtében nem volt egyértelmű melyik notebook az aktuális, működő verzió.

**Következmény:** Időpocsékolás: keresés, összehasonlítás, félreértések hogy "melyiket futtatjuk most".

**Tanulság:**
- Kaggle notebookot létrehozásakor AZONNAL adjunk egyedi nevet: pl. `roboshelf-t1-eval-v1`, `roboshelf-t1-finetune-v2`
- Tartsuk nyilván itt (vastai_known_issues vagy Index) melyik notebook mire való és mi az aktuális "production" verzió
- Ha egy notebook crashel és újat kell csinálni, a régit nevezzük át `-broken` vagy `-archived` utótaggal, ne hagyjuk névtelenül

**Aktuális notebook registry:**

| Notebook neve (Kaggle) | Cél | Státusz |
|---|---|---|
| `notebook7b97304d37` | T1 UnifoLM-VLA-0 fine-tune (10k lépés) | ✅ KÉSZ — step_10000.pt output |
| `roboshelf-t1-eval-v1` | T1 50-ep eval | ✅ KÉSZ — 86% SR (43/50) · Script: `notebooks/kaggle_vla_eval_t1_v2.py` |

---

## AI-11. [2026-07-25] mujoco_menagerie --filter=blob:none → STL fájlok üresek

**Kontextus:** T1 eval notebook — mujoco_menagerie sparse clone a booster_t1 mesh fájlokhoz.

**Hiba:** A sparse clone parancsban `--filter=blob:none` flag szerepelt. Ez azt jelenti, hogy a git csak a fa-struktúrát tölti le, a tényleges fájl-tartalmak (blob-ok) nem kerülnek lemezre. A MuJoCo közvetlenül a fájlrendszerről olvassa az STL fájlokat, nem git protokollon keresztül, ezért üres/hiányzó fájlokat talált.

**Következmény:** `ValueError: Error opening file .../booster_t1/assets/Trunk.stl` (és más STL fájlok). Többszöri sikertelen futás.

**Tanulság:** mujoco_menagerie sparse clone-nál SOHA ne használj `--filter=blob:none`-t. A helyes parancs:
```bash
git clone --depth=1 --sparse https://github.com/google-deepmind/mujoco_menagerie.git MENAGERIE_DIR
git -C MENAGERIE_DIR sparse-checkout set booster_t1
```
Ellenőrzés: `len(list(MENAGERIE_DIR.glob("booster_t1/assets/*.stl"))) > 0`

---

## AI-12. [2026-07-25] Unifolm_VLA.forward() vs predict_action() — inference API tévesztés

**Kontextus:** T1 VLA eval inference loop — `model(qwen_inputs=inp)` hívás.

**Hiba:** A `forward()` metódus training-only: az `action` kulcsot ground truth-ként várja, és `{"action_loss": ...}` dict-et ad vissza. Inference-hez a `predict_action()` metódust kell hívni, ami `{"normalized_actions": numpy_array}` dict-et ad vissza. Az AI először `model(...)` hívással próbálkozott, majd `inp["action"] = zeros` hozzáadásával, ami tovább rontott (loss-t számolt a zérus targetra).

**Következmény:** Három sikertelen futás (`KeyError: 'action'`, `{'action_loss': ...}`, `{'normalized_actions': ...}` hibák sorban).

**Tanulság:** Unifolm_VLA inference = `model.predict_action(qwen_inputs=inp)`, visszatérési érték: `out["normalized_actions"][0]` → `(CHUNK_SIZE, ACTION_DIM)` numpy array. Forrás: `unifolm_vla/model/framework/unifolm_vla.py`. Soha ne hívd a `forward()`-ot inference-kor.

_Ide kerül minden jövőbeli AI hiba is. A szekció növekszik a projekt előrehaladásával._

---

## AI-6. [2026-07-24] Kaggle dataset path formátum figyelmen kívül hagyva

**Kontextus:** T1 fine-tune Kaggle notebook elkészítése. A G1 notebookban a helyes path már dokumentálva volt.

**Hiba:** A T1 notebookban `DEFAULT_DS_ROOT = Path("/kaggle/input/roboshelf-t1-push-v1/t1_push_v1")` path-et írtam, holott a Kaggle a saját adatseteket `/kaggle/input/datasets/{username}/{dataset-slug}/` formátumban mountolja. A helyes path: `/kaggle/input/datasets/leventevrss/roboshelf-t1-push-v1/t1_push_v1`. Ugyanez vonatkozott a scripts dataset path-re is.

**Következmény:** `ModuleNotFoundError` — a notebook nem találta a scriptet. A felhasználónak kellett `os.walk`-kal megkeresni a helyes path-et.

**Tanulság:** Kaggle notebook írásakor MINDIG a meglévő G1 notebook path-jeit kell mintának venni: `/kaggle/input/datasets/leventevrss/{dataset-slug}/`. Soha ne feltételezzük a rövidebb `/kaggle/input/{slug}` formátumot.

---

## AI-7. [2026-07-24] bitsandbytes import install_deps() előtt

**Kontextus:** T1 fine-tune notebook OOM javítása — 8-bit Adam bevezetése.

**Hiba:** A javított cellában `import bitsandbytes as bnb` a `vla.install_deps()` hívás ELŐTT szerepelt. Friss Kaggle kernelben a csomag még nincs telepítve, ezért `ModuleNotFoundError` keletkezett "Save & Run All" módban (míg interaktív futtatásnál az előző session-ből örökölt telepítés elfedte a hibát).

**Következmény:** A committed notebook azonnal crashelt.

**Tanulság:** Minden külső csomag importja (`bitsandbytes`, `peft`, stb.) csak `vla.install_deps()` UTÁN kerülhet a cellába.

---

## AI-8. [2026-07-24] Gradient checkpointing és 8-bit Adam nem volt alapértelmezett

**Kontextus:** T1 UnifoLM-VLA-0 fine-tune T4 GPU-n.

**Hiba:** A notebook első verziója nem tartalmazott gradient checkpointingot és standard AdamW-t használt. A T4 16GB-on a 7B VLM backward pass + 588M DiT action model optimizer state-jei OOM-ot okoztak.

**Következmény:** 2 sikertelen futtatás (OOM optimizer.step()-nél, majd OOM loss.backward()-nál).

**Tanulság:** UnifoLM-VLA-0 T4-en csak ezekkel fut stabilan: (1) `gradient_checkpointing_enable()` a VLM-re, (2) `bnb.optim.AdamW8bit`, (3) `batch_size=1`. Ezeket a következő T1/T2/stb. notebookban rögtön be kell építeni.
