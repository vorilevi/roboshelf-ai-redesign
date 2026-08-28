# Shelf Life — ténycsomag

**Készült:** 2026-08-05 · **Tárgy:** a Shelf Life alprojekt M0–M2 szakasza
**Célja:** hogy a beszámolót más is megírhassa, és minden szám visszakereshető legyen.

---

## 0. Hogyan kell ezt olvasni

A dokumentum három dolgot **szigorúan szétválaszt**:

| jel | mit jelent |
|---|---|
| **T** | **Tény.** Van hozzá fájl a lemezen, és meg van adva, melyik szkript melyik futása állította elő. |
| **T?** | **Állítás artefaktum nélkül.** Szerepel a projektjegyzetben, de nincs mögötte se napló, se olyan kód, ami előállította volna. |
| **É** | **Értelmezés.** Következtetés, nem mérés. Oda van írva, mely tényekre támaszkodik. |

A 2. és 3. szakasz a lényeg: ott van összeszedve, mi **nem** áll szilárd talajon.

Két korlát, amit az olvasónak tudnia kell:

1. Ezt a csomagot **ugyanaz az LLM állította össze, amelyik a méréseket végezte és a hibákat elkövette.** A „T" jelölés ellenőrizhető (a fájl vagy megvan, vagy nincs); a szelekció — hogy mi került be és mi maradt ki — nem az.
2. A mérések **kizárólag MuJoCo-szimulációban** készültek. Valódi hardveren egyetlen szám sem lett ellenőrizve.

---

## 1. TÉNYEK

### 1.1 A jelenet és a termék

| érték | forrás |
|---|---|
| SKU: `alpro_barista_coconut_1l` — Alpro Barista Coconut 1 L, tetra gable-top | `src/envs/assets/shelflife_sku_private/…json` |
| háló: 17 825 csúcs, 35 179 lap; iPhone 17 Pro / Scaniverse fotogrammetria, 2026-08-03 | ua., `geometry` |
| befoglaló méret 7,86 × 20,42 × 7,96 cm | ua., `geometry.bbox_m` |
| ütközési fél-méret [0,0393 · 0,0398 · 0,1021] m → **7,9 × 8,0 × 20,4 cm** | `shelflife_scene_gene01_sku_v1.meta.json` |
| tömeg **1,03 kg** → súly 10,1 N (g = 9,81) | ua., `mass_kg` (a súly számítva) |
| letelepedési elcsúszás 2,07 mm | ua., `settle_drift_mm` |
| dátum típusa **best_before**, helye a gable-top gerinc, formátum `%d.%m.%y` | SKU-json, `date_field` |
| a valódi dobozon leolvasott dátum + tételkód rögzítve *(konkrét érték: privát SKU-bejegyzés, ide szándékosan nem másolva)* | ua., `observed_value` |
| nyomtatás: tintasugaras 5×7 mátrix, karaktermagasság **3,6 mm**, blokkszélesség 47 mm | ua. |

> A dátum **nem a textúrába van festve**, hanem külön matricaként kerül a modellre — hogy az eval epizódonként mást tehessen ugyanoda. (Forrás: a `placement` mező kommentje.)

### 1.2 Törzs — M0

**Ma újramérve** (`tools/shelflife_torso_torque.py`, 2026-08-05, log: `results/shelflife_verify/torso_torque.log`).
Próbapóz: pre-grasp IK-póz, 2,0 mm / 1,4°. 2000 lépés, csillapítás 100, armatúra 0,5.

| kp | csúcsnyomaték | tartónyomaték | yaw-hiba | tenyér-hiba |
|---|---|---|---|---|
| 2 000 *(eredeti)* | 1 096 Nm | 874 Nm | 0,4372 rad | 317,3 mm |
| 10 000 | 5 484 Nm | 2 627 Nm | 0,2626 rad | 279,9 mm |
| 30 000 | 16 469 Nm | 2 388 Nm | 0,0795 rad | 27,1 mm |
| **100 000** *(élő)* | **54 950 Nm** | **765 Nm** | **0,0031 rad** | **1,8 mm** |
| 300 000 | 164 847 Nm | 1 358 Nm | 0,0018 rad | 1,6 mm |

Élő beállítás a jelenetben: `kp=100000`, csillapítás 100, armatúra 0,5 — külön `gene_torso` osztályban.
Forrás: `shelflife_scene_gene01_sku_v1.xml` 550–551. sor, `shelflife_build_scene_sku.py` 178. sor.

M0 kilépési feltétele (`shelflife_torso_tune.py` docstringje): yaw-hiba < 0,010 rad **és** tenyér-hiba < 5 mm 2000 lépésnél. **kp = 100 000 teljesíti** (0,0031 rad / 1,8 mm).

> A csúcsnyomaték nagyjából `kp × kezdeti szöghiba` — 100 000 × 0,55 rad ≈ 55 000 Nm —, tehát az első lépés tranziense, nem tartós terhelés. A **tartónyomaték** a fizikailag értelmes szám.

### 1.3 A primitív-szótár — M1

Tanúsítvány: `results/shelflife_api/spec.json` (2026-08-04 21:01), napló: `results/shelflife_api/m1.log`.

| primitív | mért érték |
|---|---|
| `reset_home()` | ismételhetőség 0,00 mm |
| `preset('pre_grasp')` | IK 1,3 mm / 1,26° · ízülettartalék 0,202 rad |
| `preset('grasp')` | IK 1,43 mm / 1,34° · tartalék 0,345 rad |
| `preset('inspect')` | IK 1,96 mm / 2,3° · tartalék 0,264 rad |
| `preset('lift')` | IK 1,31 mm / 1,47° · tartalék 0,334 rad |
| `preset('shelf_out')` | IK 2,51 mm / 1,54° · tartalék 0,185 rad |
| `approach_until(goal)` | hiba 4,5 mm · 4,7 s · termék 0,00008 mm |
| `approach_until(contact)` | `guard` · 0,3 s · termék 31,6 mm |
| `close_until(grip)` | `guard` · 3 kontaktus [little, middle, thumb] · **9,9 N** |
| `can_see_date()` a polcon | **nem látszik** — „kilóg: vízsz +30,1°, függ +23,6°, fél-FOV 22,5°, táv 0,39 m" |
| `view()` | `RuntimeError` — nincs OpenGL a sandboxban |

⚠️ **Ez a tanúsítvány elavult.** Lásd a 3.1 pontot.

**Kódméret** (`wc -l`, 2026-08-05):
`shelflife_api.py` 411 · `shelflife_motion.py` 331 · `shelflife_grasp.py` 823 · `shelflife_program_v0.py` 185 (141 nem üres, nem komment) · `shelflife_jaw.py` 238 · `shelflife_api_measure.py` 220.

### 1.4 Fogás — M2

#### a) A teljes lánc mai futása

`python3 tools/shelflife_program_v0.py --coverage`, 2026-08-05, log: `results/shelflife_verify/program_rerun.log`

```
reset_home                   goal       alaphelyzet
→ pre_grasp                  goal       508 mm megtéve, termék 0.0 mm
→ grasp                      goal       138 mm megtéve, termék 0.0 mm
close_until(grip)            grip       3 kontaktus [index, middle, thumb], 30.3 N
→ lift                       guard      1 mm megtéve, termék 4.5 mm
→ shelf_out                  guard      18 mm megtéve, termék 4.3 mm
→ inspect                    guard      12 mm megtéve, termék 10.5 mm
forgatás +0° right           guard      4 mm megtéve, termék 6.2 mm
dátum látszik?               NEM
DÖNTÉS                       NEM_OLVASHATO
→ aside fölé                 guard      3 mm megtéve, termék 7.1 mm
place_until(support)         support    1 mm megtéve, termék 4.3 mm
open_hand                    goal       kéz nyitva
→ shelf_out                  goal       108 mm megtéve, termék 0.1 mm
```

Amit ez rögzít:

- **A teljes feladat végigfut a szótárból**, hét szakaszon, minden ige jelent.
- **A fogás bezár** (tripod: hüvelyk + mutató + középső, 30,3 N), és a termék a fogásig **0,0 mm-t mozdul**.
- **A fogás nem tart.** Az `lift` 1 mm után őrfeltételre áll meg, a termék 4,5 mm-t csúszik a kézhez képest. A lánc vége: `nem fogja`, a termék a polcon.
- A `NEM_OLVASHATO` döntés a jelenlegi kamerahelyzet következménye, nem VLM-hiba — VLM még nincs bekötve.

#### b) Az állkapocs-szerkesztés

`python3 tools/shelflife_jaw.py`, 2026-08-05, log: `results/shelflife_verify/jaw_rerun.log`

| érték | |
|---|---|
| állkapocs nyílása (medial + distal fogófelületek) | **117,9 mm** |
| a karton legkisebb szélessége | 79 mm |
| tartalék | 39 mm |
| szerkesztett `tweak_cm` javaslat | [2,1 · −1,8 · 3,0] |

Záráspróba a szerkesztett ponton (kontaktszám és a normálisok eredőjének aránya az összerőhöz):

| zárás | kontakt | eredő/össz | ujjak |
|---|---|---|---|
| 0,00 | **11** | 0,89 | index, little, middle, ring |
| 0,15 | 13 | 0,67 | mind az 5 |
| 0,25 | 16 | 0,57 | mind az 5 |
| **0,35** | 19 | **0,47** | mind az 5 |
| 0,55 | 21 | 0,32 | mind az 5 |
| 0,65 | 23 | 0,32 | mind az 5 |
| 0,80 | 21 | 0,38 | mind az 5 |

Nyitott kézzel (0,00 szinten) **11 kontaktpont van** — a szkript maga is figyelmeztet rá.

#### c) A rács-alapú illesztés (elvetve)

`results/shelflife_api/fit3.log` (2026-08-04 22:37): 480 megfelelő helyzet; nyertes `tweak_cm = [0,5 · −1,0 · 4,0]`, rések 6,2 / 7,1 mm, oppozíció 160°, zárás 0,35-nél mind az 5 ujj, **eredő/össz 0,14**.

Ez az eredmény **be lett írva a tervfájlba, majd visszavonva**. Az indoklás a tervfájlban áll:

> „A 2026-08-04-i illesztés eredményét visszavontuk: az csak a NYITOTT kéz réseit nézte, a zárás oda már nem ért el (nulla kontaktus). A helyes kritérium a kettő EGYÜTT."

#### d) Az élő fogási terv

`models/shelflife_sku/alpro_barista_coconut_1l/grasp_plan.json`

| mező | érték |
|---|---|
| `tweak_cm` | [−4,0 · 6,0 · 0,0] |
| `close_amount` | 0,35 |
| `reach_err_mm` / `_deg` | 1,11 / 1,27 |
| `path_worst_mm` | 1,53 |
| `joint_margin_rad` | 0,327 |
| útvonal | 15 waypoint, 14 cm-től 0-ig, cm-enként |

A fájl saját státuszmezője (`_m2_status`) kimondja, hogy M2 nincs lezárva.

### 1.5 Amit a jelenet a dátumról tud

`spec.json` → `can_see_date_on_shelf`: **nem látszik**, mert a dátummező kilóg a kamera látóteréből (vízszintesen +30,1°, függőlegesen +23,6°, fél-FOV 22,5°, távolság 0,39 m).

**É** — Ez nem hiba, hanem a feladat magja: a polcon álló terméknél a dátum eleve nem látható, ezért kell kivenni és megforgatni.

### 1.6 A fogáspróba hitelesítése és a mérés ismételhetősége

**Forrás:** `tools/shelflife_grip_test.py --selftest`, 2026-08-05.

| hitelesítési eset | elvárt | mért | |
|---|---|---|---|
| a tárgy végig rögzítve | ~0 mm csúszás | 0,00 mm | ✅ |
| a tárgy 30 cm-re, elengedve | nagy csúszás, 0 ujj | 430 mm, 0 ujj | ✅ |
| leesett tárgy padló-ütközése | 0 N a kéz felől | 0,0 N | ✅ |

**T** — A mérőeszköz a három hitelesítési esetet helyesen adja vissza. Ez az első eszköz a projektben, amelyik ismert jó ÉS ismert rossz esetre is le van futtatva.

**T** — Báziseset a megközelítéses protokollal (`run_approach()`): 2 ujj érintkezik · 79,8 N · a kéz 29,2 mm-t emelkedik, a termék −5,0 mm-t, azaz **követés −17%**. A robot nem emeli fel a dobozt.

**T — a mérés NEM ismételhető.** Ugyanaz a paraméterkészlet (ujj-erőkorlát ±1,5 Nm), csak a beállási lépésszám változik:

| settle | ujj | középperec | erő | követés |
|---|---|---|---|---|
| 30 | 2 | 0 | 17,5 N | +32% |
| 60 | 2 | 0 | 61,9 N | −106% |
| 100 | 3 | 1 | 53,2 N | +9% |
| 150 | 3 | 1 | 102,3 N | +1% |
| 200 | 3 | 1 | 100,8 N | −26% |

**T** — Az ok azonosítva: a záró ciklus erőküszöbre (15 N) áll meg, ezért hosszabb beállásnál **más zárási szinten** fejeződik be, más kontaktus-elrendezéssel. A `settle` nem a pontosságot változtatja, hanem azt, hogy melyik fogást mérjük.

**Következmény, kimondva:** a 2026-08-05-én futtatott **egyfutásos paraméter-összehasonlítások — beleértve a Menagerie-referenciaértékek mérését — nem értelmezhetők**, mert a lényegtelen paraméterből származó szórás nagyobb, mint a változatok közti különbség. Ezekre a számokra semmilyen állítást nem szabad építeni.

**Ami nincs megvizsgálva:** a KAR aktuátorai kp=3000-rel és erőkorlát nélkül futnak. Az ujjak 1,5 Nm-es nyomatéka önmagában nem tud 100 N kontaktuserőt előállítani, tehát a kar a valószínű forrás — ez még nincs megmérve.

### 1.7 A fogás geometriája és a 86 mm-es eltérés (2026-08-06)

**Forrás:** `tools/shelflife_hand_span.py`, `shelflife_grasp_redesign.py`,
`shelflife_gravcomp.py`, `shelflife_two_finger.py`, `shelflife_gripper_test.py`.

**T** — Hol ér a kéz a dobozhoz (doboz: Ø58 × 145 mm):

| ujj | szög a tengely körül | magasság a talptól |
|---|---|---|
| hüvelyk | 178° | 144 mm (99%) |
| mutató | −75° | 124 mm (85%) |
| középső | −68° | 86 mm (59%) |
| gyűrűs | −82° | 55 mm (38%) |
| kisujj | −89° | 31 mm (21%) |

Oppozíciós szög **104°** (180° = szemben). Magasságszórás 113 mm = 78%.

**T** — Átmérő-söprés 45–95 mm: az oppozíció végig **51–56°**. A tárgy
mérete nem befolyásolja.

**T** — A kéz mérete: ujjak 82 mm, hüvelyk 122 mm, tenyér ~69 × 79 mm.

**T** — A kar pozíciópontossága: az inverz kinematika 1,1 mm, a számolt
fogáspont 2,2 mm-re a céltól, a TÉNYLEGES mozgás után **86 mm**. Beállási
idő szerint: 92,9 / 86,9 / 86,0 / 86,3 mm (n=14…80, settle=60…800).

**T** — Az ok NEM gravitáció. Nehézségi nyomaték a karon ≈ 5 Nm,
kényszererő 300–1200 Nm. A kéz **51,2 mm-rel benne van a polcban**, és 24
ponton önmagával is ütközik. A gravitációkompenzáció mérve **semmit nem
javított** (86,9 → 86,8 mm), és a saját hitelesítése buktatta meg.

**T** — Kétujjas STRATÉGIA a meglévő ötujjas kézzel: az oppozíció
**104° → 175°**. A süllyedés viszont **17,6 mm** után megáll: a behajlított
kisujj és a tenyér alja a polclapnak ütközik.

**T** — A kéz **alaphelyzetben, érintetlenül 13 ponton ütközik önmagával.**
Ez a publikált GENE.01 modell tulajdonsága, nem a póz következménye.

**T** — Valódi kétujjas fogó (Robotiq 2F85, Menagerie) ugyanazon a polcon,
ugyanazzal a dobozzal: a nyílás **93,1 mm nyitva / 8,8 mm zárva**. A
mérőeszköz hitelesítése 3/3. Fogás **nincs**: minden magasságon 0 kontaktus,
a fogó leereszkedés közben lelöki a dobozt. **Az ok nincs azonosítva.**

**T** — A pozíciószabályzó KÉT független rendszerben sem ér oda: humanoid
kar 86 mm, a fogó szánja **190 mm** (0,39 helyett 0,20). Ezt eddig egyik
mérésünk sem ellenőrizte.

**É** — A hat kritérium-hiba közös mintája: *a kapu csak akkor ér valamit,
ha a MÉRT alapállapothoz van kalibrálva, nem egy elképzelt ideálishoz.*

### 1.8 A kétujjas fogó — a javított mérés (2026-08-06, késő)

**Forrás:** `tools/shelflife_gripper_test.py`. Hitelesítés 3/3.

**T** — A „0 kontaktus minden magasságon" eredmény **mérési műtermék volt**,
hét egymást követő hibából. Ezek közül a legsúlyosabbak: a kontaktusszűrő
geomNÉVRE szűrt, miközben a fogó 28 geomjából csak 4-nek van neve (24 geom
láthatatlan volt); záráskor nem tartottam a szánt, ezért a szervók
visszarángatták a világ origójába; és a nyílást a párnák KÖZÉPPONTJA közt
mértem, nem a felületük közt (~10 mm eltérés).

**T** — A javítások után, 95 mm-es fogásmagasságnál:

| | érték |
|---|---|
| a termék elmozdulása a közelítés alatt | **0,9 mm** |
| kontaktusok záráskor | **14** |
| szorítóerő | **302 N** |
| a zárás megáll | **58,1 mm-en** (a doboz 58,1 mm) |

**T** — A záróparancs a termék átmérőjéből számolódik
(`close_to_width(58,1 − 2,0)`), nem beégetett érték.

**T — DE az emelés száma NEM használható.** A fogó 40 mm-t emelkedik, a
termék 0,9 mm-t, és végig 5 ponton érintkezik a polccal. Közben a fogás
stabil (58 mm-es rés, 12–18 kontaktus, ~100 N). 100 N × 0,7 = 70 N
tartóképesség egy 3,4 N súlyú dobozon — **húszszoros tartalék mellett nem
csúszhat meg**, tehát a szám nem hihető.

**Ami nincs megmérve:** a szabadlevegős fogáspróba (polc nélkül), ami
eldöntené, hogy a polc vagy a kinematikus vezetés a hiba forrása.

---

## 2. ÁLLÍTÁSOK ARTEFAKTUM NÉLKÜL

Ezek a projektjegyzetben mért adatként szerepelnek, de nincs mögöttük napló a lemezen.

| # | az állítás | státusz |
|---|---|---|
| **1** | **Nyomaték: kp=30 000 → 257 Nm · 100 000 → 219 Nm · 300 000 → 460 Nm** | **CÁFOLVA.** Lásd lent. |
| 2 | „Ellenőrzés az újraépített jeleneten: yaw-hiba 0,0031 rad · tenyér 1,8 mm" | **MEGERŐSÍTVE** — ma pontosan ez jött ki. |
| 3 | Ujjankénti érkezési szintek: hüvelyk 0,13 · mutató 0,28 · középső 0,31 · gyűrűs 0,28 · kisujj 0,28 | nincs napló |
| 4 | Kontaktus-normálisok a fogás pillanatában: mutató 2,9 N · középső 14,2 N · hüvelyk 13,2 N · eredő **14,1 N** · nyomaték **1,03 Nm** | nincs napló; szó szerint csak a `shelflife_grasp_film.py` docstringjében |
| 5 | Rések a zárás indulásakor: 46 / 70 / 73 / 73 / 69 mm | **részben**: a fit-naplók a hüvelyk 46 mm-t és az ujjak 69 mm-t tartalmazzák, a másik hármat nem |
| 6 | Hüvelyk–mutató nyílás nyitott kézzel **18,2 cm** | nincs napló; csak forráskód-komment |
| 7 | „minden geom 57,1 mm · fogófelületek 117,7 mm · ujjbegyek 169,8 mm" | nincs napló; csak forráskód-komment. A 117,7-ből ma **117,9** lett |
| 8 | Ujj-`kp` kísérletek: „100 fölött nulla kontaktus, 1000 fölött NaN-instabilitás" | nincs napló |
| 9 | „a pozíció-aktuátorok 51,6 N-t fejtettek ki és 780 mm-re kilőtték a kartont" | nincs napló |
| 10 | „a termék 0,55-ig 0,0 mm-t mozdul, a kontaktus 0,60-nál jön létre 1,3 mm-rel" | nincs napló |

### 2.1 Az 1. tétel részletesen — ezt ma megmértük

A jegyzet M0-táblázata tartalmaz egy nyomaték-oszlopot, és a `kp = 100 000` választását ezzel indokolja:

> „A 300 000-es változat ugyanazt tudja **kétszeres nyomatékkal**, ezért nem az nyert; a 219 Nm reális egy humanoid derékízületére."

Két dolog derült ki:

1. **A `shelflife_torso_tune.py` nem mér nyomatékot.** Egyetlen sor sincs benne, ami `actuator_force`-ot vagy `qfrc_actuator`-t olvasna. A `results/shelflife_grasp/m0_torso.log` naplóban sincs ilyen oszlop.
2. **A tényleges értékek nagyságrendekkel nagyobbak** (mai mérés):

| kp | jegyzet | mért csúcs | mért tartó |
|---|---|---|---|
| 30 000 | 257 Nm | 16 469 Nm | 2 388 Nm |
| 100 000 | 219 Nm | 54 950 Nm | **765 Nm** |
| 300 000 | 460 Nm | 164 847 Nm | **1 358 Nm** |

**Ami a következtetésből megmarad:** a tartónyomaték `kp = 100 000`-nél tényleg kisebb (765 Nm), mint 300 000-nél (1358 Nm) — a *döntés iránya* helyes marad.
**Ami megdől:** a konkrét számok, és az az indoklás, hogy „219 Nm reális egy humanoid derékízületére". 765 Nm nem az; egy valódi humanoid derékhajtás nagyságrendje ennek töredéke. **É** — ez azt jelenti, hogy a szimulált törzs a mai beállítással irreálisan erős hajtást feltételez, és ez a szám valódi hardver felé kommunikálva félrevezető lenne.

---

## 3. ÜTKÖZÉSEK (a mai ellenőrzés találta)

### 3.1 Az M1 kilépési tanúsítvány elavult

| | |
|---|---|
| `spec.json` készült | 2026-08-04 **21:01** |
| `shelflife_motion.py` módosítva | 22:05 |
| `shelflife_api.py` módosítva | 22:53 |
| `shelflife_grasp_plan.py` módosítva | 23:02 |

A tanúsítvány szerint `close_until(grip)` → `guard`, 3 kontaktus [little, middle, thumb], **9,9 N**.
A ma futtatott kód szerint → `grip`, 3 kontaktus [index, **middle**, thumb], **30,3 N**.

**Más a leállási ok, más az egyik ujj, háromszoros az erő.** A D1 „befagyasztás" ki lett mondva, a réteg utána mégis változott — ezt a projektjegyzet **jelzi is** („Két interfész-változás a D1 befagyasztás óta, mindkettő jelentve"). Ami hiányzik: a tanúsítványt nem generáltuk újra, így a lemezen lévő M1-bizonyíték nem a jelenlegi kódot írja le.

### 3.2 A lemezen lévő fogás-futás ellentmond a jegyzetnek

`results/shelflife_grasp/run_approach_until.json` (2026-08-04 10:09): **1 kontaktus**, csak a mutatóujj, **3,7 N**, `held: false`, három szakasz őrfeltételre állt meg.

A jegyzet ezzel szemben „tripod, 30,3 N, tiszta érkezés 0,0 mm"-t ír. **Ma a 30,3 N reprodukálódott** — tehát nem a jegyzet téved, hanem a JSON régi (délelőtti állapot). Dátum nélkül olvasva viszont félrevezető: ez az egyetlen strukturált fogás-eredmény a lemezen.

### 3.3 A „0,47 → 0,32" olvasata

A jegyzet az állkapocs-áttörésnél így ír: *„normálisok eredője / összerő: 0,47 → 0,32"*, ami javulásként olvasható.

A mai mérés szerint a **választott** zárási szinten (0,35) az érték **0,47**; a 0,32 csak 0,55-ös zárásnál áll elő. Összevetve:

| | eredő/összerő a választott szinten |
|---|---|
| rács-illesztés (elvetve) | **0,14** |
| állkapocs-szerkesztés („áttörés") | **0,47** |

**É** — Ezen az egyetlen mérőszámon a szerkesztett pont **rosszabb**, mint amit korábban elvetettünk. Ez nem érvényteleníti a szerkesztéses megközelítést (a rács-illesztést az ÚT miatt vetettük el, nem az erőzárás miatt), de a jegyzet jelenlegi megfogalmazása erősebbnek mutatja az eredményt, mint amilyen.

### 3.4 Apró eltérések

- `approach_until(goal)` ideje: jegyzet 4,8 s · `spec.json` 4,7 s.
- Állkapocs nyílása: jegyzet 117,7 mm · mai futás 117,9 mm.
- `m0_torso.log` saját ajánlása `TORSO_KP = 300000`; a jegyzet és a kód 100 000. Az eltérés oka a naplóban nincs dokumentálva. (Magyarázat: a szkript pontozása tenyér-hiba, majd `qacc` — nyomaték nélkül; a 300 000 ott 1,5 mm-rel győzött az 1,6 ellen.)

---

## 4. ÉRTELMEZÉSEK

Ezek **nem mérések**. Mindegyiknél oda van írva, mire támaszkodik.

**É1 — A törzs bevonása volt az M0 tényleges eredménye, nem a kp-hangolás.**
Alap: 1.2 (tenyér-hiba 317 mm → 1,8 mm) és a `grasp_plan.json` `joint_margin_rad = 0,327`. A kar önmagában 0,05–0,14 rad ízülettartalékkal dolgozott, a törzzsel 0,33-mal. A kp-hangolás ezt tette használhatóvá, de a nyereség a plusz szabadságfokból jött.

**É2 — A szótár kifejezőereje elég, a fogás vezérlési probléma.**
Alap: 1.4 a). A teljes feladat hét szakaszban, 141 érdemi kódsorban megírható, és minden ige lefut és jelent. Ami blokkol, az egyetlen dolog: a fogás nem tart. Ez a D2 döntési pont bemenete.

**É3 — A kontaktusszám rossz mérőszám volt a fogásra.**
Alap: 1.4 a) — 3 kontaktus és 30,3 N mellett is elenged terhelés alatt. A helyes mérőszám az emelési próba, illetve a normálisok eredőjének aránya.

**É4 — Az M2 jelenlegi állapota részeredmény, nem megoldás.**
Alap: 1.4 a) `lift → guard, termék 4,5 mm` és 1.4 b) 11 kontaktus nyitott kézzel. Se az élő terv, se a szerkesztett pont nem ad megtartó fogást.

**É5 — Az emberi fizikai intuíció kétszer állított meg hamis következtetést.**
Alap: a projektjegyzet két idézete (a kézméretnél és az állkapocsnál). ⚠️ **n = 2, egy projekt, és ezt a megállapítást az az LLM írja, amelyik a hibákat elkövette.** Ez hipotézis a D4-hez, nem eredmény. A mai ellenőrzés ehhez annyit tesz hozzá, hogy a hibák egy része (2. szakasz) **nem is intuícióval, hanem egyszerű forrásellenőrzéssel** kideríthető lett volna.

**É6 — A vizuális csatorna gyorsabb volt a numerikusnál.**
Alap: a jegyzet leírása arról, hogy a film megmutatta a rés-problémát. ⚠️ Erre **nincs mérés** — nem naplóztuk, hány kör ment el a számokra. Anekdota.

---

## 5. AMI NINCS MEGMÉRVE

Hogy a beszámoló ne állítson többet a valóságnál:

- **Nincs VLM bekötve.** A `NEM_OLVASHATO` döntés geometriai láthatóságból jön, nem képolvasásból. Az M4 (VLM-híd, N=30) nem kezdődött el.
- ~~**Nincs ügynök.** Az M5 (code-as-policy tényleges tesztje) nem kezdődött el. A `program_v0.py`-t ember írta; ez a baseline.~~
  **2026-08-16 JAVÍTVA — ez a mondat téves volt.** A `program_v0.py`-t **nem ember írta, hanem egy LLM** (Claude), ahogy a teljes `tools/shelflife_*` állományt is: 48 fájl, ~13 900 sor. A code-as-policy hurok tehát **hetek óta fut** — szótár → program → futtatás → napló és render → újratervezés a hibából.
  Amit pontosítani kell helyette:
  - **Emberrel a hurokban.** Az irányt beszélgetésben a felhasználó adja (pl. „nézzük meg videón", „váltsunk kétujjas fogóra"). Ez nem önjáró ügynökhurok — a Waddle sem az, de a megfogalmazásnak pontosnak kell lennie.
  - **Nincs sikerarány.** Az ág soha nem futott 50 epizódon. Ez a tényleges hiány, nem az ügynök.
- **Egyetlen SKU.** A transzfer-állításhoz (M6) nincs adat.
- **Nincs eval-harness.** Nincs 50 epizódos futás, nincs szórás, nincs sikerarány. Minden szám **egyetlen futásból** származik.
- ~~**Nincs ismételhetőségi mérés** a fogásra.~~ **2026-08-05: megmérve, és a mérés NEM ismételhető** — l. 1.6. Amíg a záró protokoll erőküszöbre áll meg, a fogásmérésre nem szabad rangsort építeni.
- **A dátum-matrica realizmusa nyitott** (M3). A jelenlegi matrica leragasztott címkének néz ki, és a `product_960px.png` alapján a zöld peremhez illeszkedik, nem a fehér panelhez. Ez az evalt optimista irányba torzítaná.
- **Nincs hardveres validáció.** Semmi.

---

## 6. REPRODUKÁLÁS

```bash
cd ~/roboshelf-ai-dev/roboshelf-ai-redesign

# 1.2 — a törzs nyomatéka és követési hibája
python3 -u tools/shelflife_torso_torque.py

# 1.4 a) — a teljes lánc
python3 -u tools/shelflife_program_v0.py --coverage

# 1.4 b) — az állkapocs-szerkesztés (nem ír fájlt --write nélkül)
python3 -u tools/shelflife_jaw.py

# 1.3 — az M1 tanúsítvány újragenerálása
#   ⚠️ FELÜLÍRJA a results/shelflife_api/spec.json fájlt
python3 -u tools/shelflife_api_measure.py

# 1.6 — a fogáspróba HITELESÍTÉSE (ismert jó + ismert rossz eset)
python3 -u tools/shelflife_grip_test.py --selftest

# 1.6 — a báziseset a megközelítéses protokollal
python3 -u tools/shelflife_grip_test.py --approach

# 1.6 — a Menagerie-változatok
#   ⚠️ AZ EREDMÉNYE JELENLEG NEM ÉRTELMEZHETŐ (l. az ismételhetőséget)
python3 -u tools/shelflife_hand_params.py
```

A rendereléshez OpenGL kell. Fejlesztői sandboxban `tools/shelflife_render_env.py` indít Xvfb-t; macOS-en nincs rá szükség.

**Fájlok, amikre a fenti számok támaszkodnak:**

```
results/shelflife_grasp/m0_torso.log             M0 kp-sweep (nyomaték NÉLKÜL)
results/shelflife_grasp/run_approach_until.json  régi fogás-futás (08-04 10:09)
results/shelflife_api/spec.json                  M1 tanúsítvány (ELAVULT, 21:01)
results/shelflife_api/m1.log                     M1 napló
results/shelflife_api/program_v0.log             a lánc 21:41-es futása
results/shelflife_api/program_cov.log            lefedettségi futás 21:42
results/shelflife_api/fit.log fit2.log fit3.log  rács-illesztés
results/shelflife_api/plan_fc.log                0 bájt — a puffereletlen kimenet hiánya
results/shelflife_verify/torso_torque.log        MAI mérés — a törzs nyomatéka
results/shelflife_verify/program_rerun.log       MAI mérés — a teljes lánc
results/shelflife_verify/jaw_rerun.log           MAI mérés — az állkapocs
models/shelflife_sku/…/grasp_plan.json           az élő fogási terv
src/envs/assets/…_sku_v1.meta.json               a termék fizikai adatai
src/envs/assets/shelflife_sku_private/…json      SKU-bejegyzés (PRIVÁT, nem publikálható)
```

---

## 7. BRIEF A FÜGGETLEN ÍRÓNAK

Ha a beszámolót más modell vagy ember írja, ez a feladata — **nem** összefoglaló, hanem **kritika**:

1. **Hol állít a jegyzet többet, mint amit az adat alátámaszt?** A 2. és 3. szakasz egy induló lista, nem a teljes. Keress továbbiakat.
2. **Melyik szám mögött nincs artefaktum?** Ellenőrizd a 2. szakasz tételeit, és nézd meg, találsz-e olyat, amit kihagytam.
3. **Hol van a keretezés túl kedvező vagy túl szigorú?** Külön figyelj a „áttörés", „megvan a fő ok", „a szótár elég" típusú fordulatokra — mindegyik értelmezés, nem tény.
4. **Mi hiányzik?** Az 5. szakasz sorolja, amiről tudom, hogy nincs megmérve. Ami ott nincs, de hiányzik, az érdekes.
5. **A D4-es megfigyelés** (ember vs. LLM munkamegosztás) az a pont, ahol a beszámoló írója **maga a vizsgálat tárgya**. Ott a leginkább indokolt a független olvasat.

**Amit az írónak tudnia kell a korlátairól:** a méréseket nem tudja újrafuttatni, hacsak nincs hozzáférése a repóhoz és MuJoCóhoz. Minden szám, amit nem tud fájlból ellenőrizni, ebből a csomagból örökölt — és ezt a beszámolóban jelölnie kell.

---

## 8. FORRÁSJEGYZETEK

- `shelflife_kiserleti_terv` — a feladat, a mérőszámok, a publikus/privát határ
- `shelflife_waddle_interfesz_terv` — M0–M9 mérföldkövek, D1–D6 döntési pontok, a hét mérési hiba
- `shelflife_publikacios_terv` — publikációs és megosztási terv
- Waddle Labs: <https://www.waddlelabs.ai/research/introducing-waddle>

> **Privát adat.** A `*_private*` könyvtárak (SKU-adatbázis, szkennelt hálók, referencia-fotók) `.gitignore` alatt vannak, és **nem kerülhetnek publikus repóba**. Commit előtt: `git status --short | grep -i private`
