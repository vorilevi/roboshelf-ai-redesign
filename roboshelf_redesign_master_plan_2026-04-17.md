# Roboshelf AI redesign master terv

Ez a dokumentum a 2026-04-17-ig közösen átbeszélt döntések, tanulságok és futtatási környezet alapján rögzíti a Roboshelf AI teljes redesign tervét. A cél nem a jelenlegi monolitikus Phase 2 PPO rendszer további foltozása, hanem egy új, moduláris, pretrain + imitációs tanulás + hierarchikus RL architektúra felépítése, amelyből később investor demo készülhet.

## Kiinduló helyzet

A mostani projekt történetileg egy egyetlen env-re épülő PPO tanítási vonalból nőtt ki, ahol a Unitree G1 közvetlenül 29 aktuátorra ad akciót a retail bolt szimulációban. A jelenlegi env a 
`src/envs/roboshelf_retail_nav_env.py`, a bolt MJCF a `src/envs/assets/roboshelf_retail_store.xml`, a jelenlegi train script pedig a `src/training/roboshelf_phase2_train.py`.

A beszélgetés alapján ez a felépítés már túl sok felelősséget tol egyetlen policy-re: egyszerre kellene lokomóciót, stabilitást, iránytartást és célfeladatot megtanulnia. Ezért született meg az a döntés, hogy a projektet rétegekre bontjuk: külön locomotion prior/policy, külön navigation task policy, később külön manipulation policy, végül demo orchestrator.

## Redesign cél

A redesign elsődleges célja egy olyan rendszer felépítése, ahol a járás nem minden tasknál nulláról újratanulandó alacsony szintű motorvezérlés, hanem egy külön modul vagy policy szolgáltatás. A retail navigációs policy így már nem 29 DoF közvetlen aktuátorparancsot ad, hanem magasabb szintű mozgási szándékot, például célirányt, sebességet vagy gait-paramétert.

A várt eredmény az, hogy gyorsabban lehessen új taskokat tanítani, kisebb legyen a reward engineering kockázata, és a lokomóciós tudás újrahasznosítható legyen a navigáció és a későbbi manipuláció alatt is. A végső üzleti cél továbbra is egy befektetői demó, amelyben a G1 egy retail vagy hypermarket digitális twin környezetben hitelesen végrehajt feladatokat.

## Már rögzített tanulságok

A session-kontekstus szerint a jelenlegi Phase 2 vonalon több kritikus technikai tanulság már megszületett, ezeket a redesign során meg kell tartani. Ilyen a helyes G1 induló testtartás, a keyframe-alapú `defaultctrl`, a 2 sub-step fizika, a reset noise szerepe, valamint az a tapasztalat, hogy egy beégett rossz policy finomhangolással gyakran nem menthető, ezért architekturális váltásnál fresh start kell.

A mostani nav env működő combined scene-t tölt be a G1 mappájába írt ideiglenes XML-en keresztül, így a menagerie mesh útvonalak helyesen maradnak feloldva. Ez fontos örökség: a redesign első iterációja építhet erre a működő szimulációs alapra, de a policy-szintű logikát már szét kell bontani.

## Futtatási környezet

A jelenlegi fejlesztési főgép egy MacBook Air M2, miniforge/conda Python környezettel, ahol a terminálban a környezet jellemzően már aktív, külön aktiválás nem szükséges. A rendszerellenőrzés alapján a lokális stack tartalmazza a MuJoCo 3.6.0-t, Gymnasium 1.2.3-at, Stable-Baselines3 2.7.1-et, PyTorch 2.10.0-t, NumPy 2.4.3-at, és MPS elérhető ugyan, de a jelenlegi SB3 MLP alapú tanításnál a gyakorlatban CPU használat van érvényben a float64/MPS problémák miatt. Mivel az MPS nem támogatja optimálisan a float64-et, a Stable-Baselines3 SubprocVecEnv wrapperét megpróbáljuk, kihasználva az M2 mind a 8 CPU magját.

A Unitree G1 modell lokálisan a Miniforge site-packages alatt található, a check script szerint ezen az útvonalon:  
`/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1`.  
A G1 `g1.xml` onnan sikeresen betölthető MuJoCo-val.

## Lokális fájlhelyek a saját gépen

A session-kontekstus szerint a korábbi repo helye Mac-en eredetileg `roboshelf-ai-dev/roboshelf-ai` volt. A mostani munkamenetben azonban már az új repo-struktúra alatt dolgoztatok, ahol a parancsok innen futottak:  
`/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign`.

A most visszaellenőrzött fontos fájlok a jelenlegi redesign repóban a következők:

| Szerep | Lokális útvonal | Megjegyzés |
|---|---|---|
| Rendszerellenőrző script | `/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/src/roboshelf_phase2_check.py` | Létezik és futott  |
| Retail nav env | `/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/src/envs/roboshelf_retail_nav_env.py` | A jelenlegi működő env  |
| Retail store XML | `/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/src/envs/assets/roboshelf_retail_store.xml` | A bolt MJCF  |
| Phase 2 train | `/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/src/training/roboshelf_phase2_train.py` | Régi monolitikus train script  |
| Phase 2 finetune | `/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/src/training/roboshelf_phase2_finetune.py` | Archiválandó, nem redesign fókusz  |

## Git repók

A korábbi session-kontekstus szerint a meglévő GitHub repo:  
`https://github.com/vorilevi/roboshelf-ai`.

A redesign alatt ténylegesen egy külön helyi repo vagy repo-változat alatt dolgoztok a `roboshelf-ai-redesign` mappában, amely a mostani munkamenetben az aktív lokális gyökérkönyvtár volt.

A gyakorlati szabály az marad, hogy a sandboxból nem történik Git push, mert nincs hitelesítés; minden commit és push a saját Mac terminálból megy. Ez a redesign végrehajtására is igaz.

## Mit tartunk meg a régi rendszerből

Nem mindent kell kidobni. A működő, bizonyítottan hasznos elemeket át kell emelni az új architektúrába.

Megtartandó elemek:
- a MuJoCo + G1 + store kombinált scene betöltési logika, mert már működik stabilan.
- a bolt MJCF és a jelenlegi asset-struktúra, mint elsődleges Phase B navigációs sandbox.
- a korábbi hibákból leszűrt fizikai stabilitási fixek: helyes induló póz, `defaultctrl`, 2 sub-step, reset noise, combined XML a G1 dir-ben.
- a rendszerellenőrző script mint lokális sanity-check belépési pont.

Nem megtartandó, hanem fokozatosan leváltandó elemek:
- a monolitikus `RoboshelfRetailNavEnv`, ahol a task és a locomotion össze van keverve.
- a régi `m2_*` level-ekre épülő kísérleti train nomenklatúra mint fő fejlesztési irány.
- a finetune-centrikus gondolkodás, ha a policy viselkedésileg rossz lokális optimumba ragadt.

## Új architektúra

A redesign négy logikai rétegre épül.

### Locomotion réteg

Ez a réteg felel a G1 stabil állásáért, alapjárásáért, irányváltásáért és mozgáskövetéséért.  
A policy bemenete propriocepció és lokomóciós célparancs, a kimenete pedig alacsony szintű aktuátorvezérlés vagy target joint command.

### Navigation task réteg

Ez a policy már nem közvetlen 29 DoF akciót ad, hanem magasabb szintű parancsot a locomotion rétegnek, például kívánt előrehaladási sebességet, yaw-rate-et, vagy lokális waypointot. Ez lesz a retail Phase B belépési pont.

### Manipulation réteg

A későbbi pick-and-place feladatok külön policy-t kapnak, amely kezdetben elszeparált minimal env-ben tanul, nem a teljes boltban. Ezt csak a navigációs réteg stabilizálása után érdemes bekapcsolni.

### Demo orchestration réteg

Ez a legfelső réteg scriptelt vagy félig scriptelt jeleneteket, kamerákat, KPI-ket és feladat-sorrendet kezel az investor demo számára. Itt a cél nem az end-to-end tanítás, hanem a külön policy-k összehangolt futtatása.

## Javasolt új repo-struktúra

A redesign repóban a következő szerkezet javasolt.

```text
roboshelf-ai-redesign/
├── README.md
├── CONTEXT.md
├── configs/
│   ├── locomotion/
│   ├── navigation/
│   ├── manipulation/
│   └── demo/
├── src/
│   ├── core/
│   │   ├── callbacks/
│   │   ├── wrappers/
│   │   ├── rewards/
│   │   ├── utils/
│   │   └── interfaces/
│   ├── envs/
│   │   ├── assets/
│   │   ├── locomotion/
│   │   ├── navigation/
│   │   └── manipulation/
│   ├── locomotion/
│   │   ├── train_loco_bc.py
│   │   ├── train_loco_rl.py
│   │   ├── eval_loco.py
│   │   └── policy_adapter.py
│   ├── tasks/
│   │   ├── navigation/
│   │   │   ├── retail_nav_hier_env.py
│   │   │   ├── train_nav_hierarchical.py
│   │   │   └── eval_nav.py
│   │   └── manipulation/
│   │       ├── retail_pickplace_env.py
│   │       ├── train_pickplace.py
│   │       └── eval_pickplace.py
│   ├── demo/
│   │   ├── investor_demo.py
│   │   ├── camera_paths.py
│   │   └── metrics_overlay.py
│   └── roboshelf_phase2_check.py
├── data/
│   ├── demonstrations/
│   ├── exports/
│   └── logs/
└── roboshelf-results/
```

Ez a struktúra úgy választja szét a felelősségeket, hogy a lokomóciós és task tanítás külön-külön is futtatható legyen. Az `envs/` alatt csak a környezetek maradnak, a train script-ek pedig funkcionális domének szerint külön mappába kerülnek.

## Fájl-migrációs terv

### Azonnal áthelyezendő vagy lemásolandó fájlok

Az alábbi fájlok tartalma megtartandó, de új strukturális helyre kell őket tenni.

| Jelenlegi fájl | Új hely | Művelet |
|---|---|---|
| `src/envs/roboshelf_retail_nav_env.py` | `src/envs/navigation/retail_nav_lowlevel_env.py` | Másolat + átnevezés, majd fokozatos tisztítás  |
| `src/training/roboshelf_phase2_train.py` | `src/tasks/navigation/train_nav_monolithic_legacy.py` | Archiváló másolat, ne vesszen el  |
| `src/training/roboshelf_phase2_finetune.py` | `src/tasks/navigation/train_nav_finetune_legacy.py` | Archiváló másolat  |
| `src/roboshelf_phase2_check.py` | `src/core/utils/system_check.py` vagy marad ideiglenesen | Később tisztítható  |
| `src/envs/assets/roboshelf_retail_store.xml` | marad `src/envs/assets/` alatt | Nem mozgatandó most  |

### Újonnan létrehozandó első fájlok

A redesign első munkacsomagjában ezeket a fájlokat kell létrehozni.

1. `src/core/interfaces/locomotion_command.py` — egységes parancsstruktúra a high-level és low-level réteg között.  
2. `src/locomotion/policy_adapter.py` — wrapper, amely egy locomotion policy-t meghívható szolgáltatásként tesz elérhetővé.  
3. `src/envs/locomotion/g1_locomotion_command_env.py` — új env, ahol a cél egy parancskövető lokomóciós policy tanítása.  
4. `src/tasks/navigation/retail_nav_hier_env.py` — új env, amelynek akciótere nem 29 DoF, hanem pl. `(v_forward, yaw_rate)` vagy `(local_dx, local_dy, local_dyaw)`.  
5. `src/tasks/navigation/train_nav_hierarchical.py` — a hierarchikus nav train script.  

## Mit másolunk fel és mit nem

### Amit version control alá kell tenni

A redesign során minden forráskódot, konfigurációt, könnyű súlyú leíró állományt és asset-definíciót fel kell tenni Gitbe. Ide tartoznak:
- `src/` teljes forráskódja.
- `configs/` összes YAML/JSON/Python configja.
- `CONTEXT.md`, `README.md`, architektúra dokumentumok.
- `src/envs/assets/*.xml` és minden kis méretű kézi asset-definíció.

### Amit nem kell felmásolni Gitbe

A nagy súlyú vagy futás közben keletkező állományok nem kerüljenek a repo-ba.

Nem feltöltendő:
- `roboshelf-results/` teljes tartalma: modellek, checkpointok, TensorBoard logok, `evaluations.npz`.
- ideiglenes combined XML fájlok a G1 mappában vagy tmp helyeken.
- nagy demonstrációs nyersfájlok, ha lesznek teleop logok, kivéve ha külön dataset release készül.
- helyi Python környezet, cache-ek, `.venv`, `__pycache__`, notebook checkpointok.

### Amit opcionálisan külön storage-ba kell tenni

A jövőbeli demo- vagy IL-adatokat valószínűleg külön tárhelyre kell tenni, nem a fő repóba. Ide tartoznak a mozgásdemók, teleoperation logok, render videók és modell snapshotok.

## Tanítási stratégia

### Fázis A — locomotion prior

Az első redesign-tanítás nem retail task lesz, hanem egy G1 command-following locomotion policy. A cél az, hogy a policy stabilan kövesse a kívánt előrehaladási sebességet és yaw-rate-et különböző reset zajok, enyhe perturbációk és stance variációk mellett.

Elvárt eredmények:
- a robot legalább több száz szimulációs lépésen át talpon maradjon.
- a parancskövetési hiba trendszerűen csökkenjen.
- a policy ne „állj és dőlj el” vagy „kapálózz” lokális optimumba tanuljon.

Tanítási módszer:
- első körben RL command-tracking env-ben;
- később vagy párhuzamosan teleop/imitációs adatokból behavior cloning előtanítás, ha lesz demonstrációs pipeline.

### Fázis B — hierarchikus navigáció

Miután a locomotion policy elfogadható, a retail nav env-et újra kell fogalmazni high-level env-ként. A high-level policy itt már csak mozgási célparancsot ad, a low-level locomotion policy végzi a végrehajtást.

Elvárt eredmények:
- a policy értelmesen haladjon a boltban a cél felé.
- a haladás ne a járás újratanulásából, hanem az útvonal- és orientációválasztásból következzen.
- a reward kevesebb kényes egyensúlyozási komponensből álljon, mint a mai monolitikus env-ben.

### Fázis C — manipuláció

A manipulációs env már létezik vázlatként a kontextus szerint, de a tanítása még nem kezdődött el. A redesign szerint ezt külön, minimalizált manipulációs sandboxban kell elkezdeni, nem teljes boltban.

Elvárt eredmények:
- reach, grasp, lift, place komponensek külön mérhetők legyenek.
- a manipuláció és navigáció külön debugolható maradjon.

### Fázis D — integráció

A navigációs és manipulációs policy-ket csak azután szabad összekötni, hogy külön-külön teljesítenek stabil acceptance kriteriumokat. Az integrált feladat egy felső szintű task controllerből vagy state machine-ből induljon, ne rögtön end-to-end RL-ből.[web:381

### Fázis E — investor demo

A demo első verziója lehet részben scriptelt, részben policy-vezérelt. Nem kell teljesen nyílt világú általános intelligenciát demonstrálnia; a lényeg a vizuálisan hiteles és üzletileg érthető end-to-end jelenet.

## Konkrét végrehajtási terv

### 1. munkacsomag — architektúra-szétválasztás

Cél: a mostani monolitikus nav rendszert logikailag kettévágni locomotion és task rétegre.

Feladatok:
- létrehozni az új mappastruktúrát;
- archiválni a legacy train script-eket;
- kivezetni a locomotion command interfészt;
- elkészíteni az első új locomotion env vázat.

Elvárt eredmény:
- új fájlszerkezet a repóban, törés nélkül;
- a régi env még futtatható marad archivált baseline-ként;
- az új locomotion modul üres vagy kezdeti stub formában futtatható importhibák nélkül.

### 2. munkacsomag — locomotion training v1

Cél: első command-tracking locomotion policy betanítása.

Feladatok:
- új reward struktúra a lokomócióhoz;
- high-level command sampling;
- stabil resetek és perturbációk;
- train/eval script írása és lokális futtatása CPU-n.

Elvárt eredmény:
- több száz lépéses stabil járás vagy legalább stabil command-following trend;
- replay videóval igazolható mozgásminőség.

### 3. munkacsomag — hierarchical nav v1

Cél: a retail nav policy átállítása high-level action space-re.

Feladatok:
- új `retail_nav_hier_env.py`;
- a locomotion adapter bekötése;
- egyszerűsített reward (waypoint/cél távolság, orientáció, ütközés);
- rövid training sanity runok.

Elvárt eredmény:
- az agent láthatóan céltudatosabban mozog, mint a monolitikus PPO baseline;
- a járásminőség romlása nélkül tud a task reward javulni.

### 4. munkacsomag — imitation learning csatorna

Cél: demonstrációs adatbevitel lehetőségének megnyitása.

Feladatok:
- teleop vagy scripted expert rollout formátum definiálása;
- adatformátum és mentési konvenció;
- behavior cloning belépési pont létrehozása a locomotion vagy navigation policy-hez.

Elvárt eredmény:
- legalább egy egyszerű BC pipeline, amely képes meglévő rolloutokból inicializált policy-t előállítani.

## Acceptance criteriumok

Az egyes fázisok akkor tekinthetők késznek, ha mérhető kimenetet adnak.

| Fázis | Minimum elvárás |
|---|---|
| A — locomotion prior | stabil talpon maradás és command-tracking több száz lépésen át  |
| B — hierarchical nav | célirányú haladás a boltban járás-összeomlás nélkül  |
| C — manipulation | reach/grasp/place szakaszok külön mérhetők és ismételhetők  |
| D — integráció | navigáció + manipuláció közti váltás stabil state machine-nel  |
| E — demo | reprodukálható, vizuálisan prezentálható end-to-end jelenet  |

## Kockázatok

A legnagyobb kockázat az, hogy a redesign közben visszacsúszik a projekt a legacy Phase 2 gondolkodásba, és újra mindent egy policy-vel akar megoldani. Ezt szerkezeti fegyelemmel kell megelőzni: külön env, külön train script, külön interface.

További kockázat, hogy a locomotion prior minősége gyenge lesz, és akkor a hierarchikus nav csak zajos alaprendszerre épül. Ezért a locomotion fázist nem szabad elkapkodni, még ha üzletileg csábító is rögtön a retail taskra ugrani.

## Ajánlott következő konkrét lépés

A redesign végrehajtásának leghelyesebb első gyakorlati lépése nem újabb Phase 2 tréning futtatása, hanem az új repóstruktúra és az első három fájl létrehozása: `locomotion_command.py`, `policy_adapter.py`, `g1_locomotion_command_env.py`. Ez teremti meg azt az interfészt, amelyre a hierarchical navigation ténylegesen ráépíthető.

## Operatív szabályok

Minden lokális futtatási parancsot a repo gyökeréből kell indítani, mert a korábbi kontextus szerint a rossz munkakönyvtár rendszeresen `No such file or directory` típusú hibákat okozott. Git push továbbra is csak a saját Mac terminálból történjen, nem sandboxból.

A `roboshelf-results/` mappa maradjon a repo mellett vagy a repo alatt lokális outputként, de ne váljon a redesign forráskód-rétegének részévé. A redesign során a kód, config és dokumentáció legyen verziózva; a modellek, checkpointok és logok maradjanak lokális artifactok.

## Folyamatos karbantartás

Ezt a tervdokumentumot rendszeresen frissíteni kell minden jelentősebb döntés, új fájl, új tanítási eredmény vagy architekturális fordulópont után. Minden frissítés célja az, hogy a redesign ne csússzon vissza a régi monolitikus Phase 2 logikába, és a projekt mindig a hierarchikus, pretrain + IL + task policy irányt kövesse.

Ajánlott frissítési ritmus:
- minden sikeres vagy sikertelen tanítási futás után, különösen ha új reward, curriculum vagy env-változat születik.
- minden új fájl létrehozásakor vagy fájlátszervezéskor.
- minden olyan ponton, amikor az env, az action space vagy az interfész megváltozik.
- minden demo- vagy investor-célú mérföldkő után.

## Emlékeztető szabály

A dokumentumot minden új beszélgetés elején és minden nagyobb végrehajtási lépés előtt újra át kell nézni, hogy ne felejtődjön el a kijelölt cél: a redesign megtervezése, a lokomóció és task szétválasztása, majd ezek tanítása. Ha a beszélgetés elkalandozik a régi Phase 2 baseline felé, a dokumentum legyen a referencia, amely visszahúz a redesign tervhez.