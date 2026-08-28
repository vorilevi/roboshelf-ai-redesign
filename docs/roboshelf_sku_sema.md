# Roboshelf — egységes SKU-séma (Product Intelligence Layer)

**Verzió:** 0.1 · **Készült:** 2026-08-05
**Egyesíti:** `roboshelf_strategic_confidential` öt mezőcsoportját · az `EAN projekt` négy pontját · a 2026-08-05-i kiegészítést (erőküszöbök) · és a súrlódást, ami eddig egyik listán sem szerepelt.

> **Ez a dokumentum a séma EGYETLEN forrása.** Ahol a termékadatokról kell írni — kísérleti terv, publikációs terv, befektetői anyag, kód —, oda ne másoljuk át a mezőlistát, hanem erre hivatkozzunk.

---

## 0. Két alapelv

### 0.1 Minden fizikai mező mellé odakerül, HONNAN VAN

Ez nem formalitás. 2026-08-05-én kiderült, hogy a termék súrlódása (`0.9 / 0.02 / 0.002`) egy szkriptbe beírt szám, az ujjbegyeké pedig a MuJoCo alapértelmezése — miközben a fogás pont a csúszáson bukik. **Adatnak látszott, feltételezés volt.**

Ezért minden mérhető mező kötelező kísérője:

```json
"mass_kg": { "value": 1.03, "source": "manufacturer", "confidence": "high" }
```

| `source` | mit jelent |
|---|---|
| `measured` | mi mértük, van hozzá naplófájl (`evidence` mezőben) |
| `manufacturer` | gyártói adat vagy a csomagoláson szerepel |
| `derived` | másik mezőből számolva (a képlet a `note`-ban) |
| `estimated` | szakértői becslés, nincs mögötte mérés |
| `default` | a szimulátor alapértelmezése — **ez nem adat** |

A `default` és az `estimated` **külön színnel jelenik meg** minden riportban. Ha egy eval eredménye `default` mezőn múlik, az eredmény nem publikálható.

### 0.2 Egy adat, egy forrás

A séma három olyan mezőt is örököl, ami ma **két helyen** él:

| adat | korábban hol | most |
|---|---|---|
| tömeg (1,03 kg) | a jelenetépítő szkriptben + a generált `meta.json`-ban; az SKU-bejegyzésben **nem is szerepelt** | ✅ **megoldva** — `physics.mass_kg`, a jelenet onnan generálódik |
| súrlódás | a jelenetépítő szkriptben; ujjbegyen alapértelmezés | ✅ **megoldva a forrás oldalán** — `contact.mujoco_friction`, `source: default`-ként jelölve. Az ÉRTÉK továbbra sincs megmérve |
| fogástechnika | `grasp.recommended: null` az SKU-ban **és** külön `grasp_plan.json` | ✅ **megoldva** — `grasp.grasp_plan_ref` hivatkozik, nem másol |

> **Elvégezve 2026-08-05.** A jelenetépítő (`shelflife_sku_import.py`) mostantól az SKU-bejegyzésből olvassa a tömeget, a súrlódást és a `condim`-et, és minden mezőnél kiírja, mért adat-e:
>
> ```
>    physics.mass_kg              1.03                  [manufacturer / high]
> ⚠️ contact.mujoco_friction      [0.9, 0.02, 0.002]    [default / none]
> ⚠️ contact.condim               4                     [estimated / medium]
> ```
>
> Az újragenerált jelenet **bitre azonos** a korábbival, és a fogás viselkedése változatlan (tripod, 30,3 N, termék 0,0 mm) — tehát ez tiszta áthelyezés, nem viselkedésváltozás.
>
> **A bejegyzés jelenlegi állapota: 23 forrásjelölt mező, ebből 14 (61%) `default` vagy `estimated`.** Ez a szám önmagában is riport: a termékadatbázisunk kétharmada még feltételezés.

---

## 1. A séma

Nyolc réteg. A ✅/⚠️/❌ a **jelenlegi** állapot az 1. SKU-nál (Alpro Barista Coconut 1 L).

### 1.1 Azonosítás — `identity`

| mező | típus | forrás | most |
|---|---|---|---|
| `sku_id` | string | belső | ✅ |
| `ean_gtin` | string (EAN-13) | manufacturer | ❌ |
| `brand`, `product_name`, `category` | string | manufacturer | ✅ |
| `package_type` | enum (`tetra_gable_top`, `pet_bottle`, `can`, `pouch`, `carton_box`, …) | manufacturer | ✅ |
| `volume_ml` / `net_weight_g` | number | manufacturer | ✅ (térfogat) |

> Az `ean_gtin` a planogram-párosítás kulcsa és az Open Food Facts kapcsolódási pontja. Enélkül a külső adatforrás nem használható.

### 1.2 Vizuális réteg — `visual`

| mező | típus | forrás | most |
|---|---|---|---|
| `mesh` | útvonal (OpenUSD) | measured (3D scan) | ✅ 17 825 csúcs / 35 179 lap |
| `texture` | útvonal + felbontás | measured | ✅ 8192 px |
| `scan_provenance` | eszköz, szoftver, dátum | — | ✅ |
| `texture_quality` | enum + megjegyzés | estimated | ✅ (elmosódott, rögzítve) |
| `reference_photos[]` | többnézetes fotók | — | ⚠️ egy darab |

### 1.3 Geometria — `geometry`

| mező | típus | forrás | most |
|---|---|---|---|
| `bbox_m` | [x,y,z] | derived (a hálóból) | ✅ |
| `long_axis` | enum | derived | ✅ |
| `collision_proxy` | típus + méret + eltolás | derived | ✅ box, fél-méret [0,0393 · 0,0398 · 0,1021] |
| `collision_fidelity` | enum (`box`, `convex_hull`, `sdf`, `mesh`) | — | ⚠️ box, nincs jelölve |

> A `collision_fidelity` azért kell, mert a doboz-proxy a gable-top **tetejét levágja** — épp ott, ahol a dátum van. A fogás szempontjából ez rendben, a döntögetés szempontjából nem.

### 1.4 Fizikai réteg — `physics`

| mező | egység | forrás | most |
|---|---|---|---|
| `mass_kg` | kg | manufacturer | ✅ **1,03** — a bejegyzésben, a jelenet onnan generálódik |
| `com_m` | [x,y,z] a modell frame-jében | estimated | ⚠️ **`null`, `estimated`/`low`** — a feltételezés most már ki van mondva |
| `inertia` | tenzor vagy `auto` | derived | ✅ `auto`, jelölve |
| `fill_state` | enum (`full`, `empty`, `partial`) | estimated | ✅ `full` |
| `liquid_sloshing` | bool | estimated | ✅ `false` — **kimondott egyszerűsítés**, a döntésnél hibát okozhat |

> **A `fill_state` nem kozmetika.** Egy tele 1 literes karton súlypontja mozog, ha döntjük — a dátum leolvasásához pedig épp döntenünk kell. Ma merev testként modellezzük, és ez a feltételezés sehol nincs leírva.

### 1.5 Érintkezés — `contact` *(EZ HIÁNYZOTT MINDKÉT KORÁBBI LISTÁRÓL)*

| mező | egység | forrás | most |
|---|---|---|---|
| `friction_static` | — | measured | ⚠️ **`null`, `default`/`none`** — a hiány most már látszik |
| `friction_dynamic` | — | measured | ⚠️ `0.9`, de `default`-ként jelölve |
| `restitution` | — | estimated | ⚠️ `null`, jelölve |
| `surface_finish` | enum (`matte_carton`, `glossy_pet`, `metal`, `shrink_film`) | manufacturer | ✅ `matte_carton` |

> **Miért ez a kulcs.** A „milyen erővel kell megfogni, hogy ne csússzon" mező **nem független adat, hanem következmény**:
>
> ```
> F_min  ≥  m · g / (n · μ_static)          n = a szembefogó felületek száma
> ```
>
> Súrlódás nélkül ez nem tölthető ki — SKU-nként újra kellene mérni, kézformánként külön. Súrlódással egyszer mérjük a felületet, és minden robotra számolható.
>
> A mai adatokkal: 1,03 kg · 9,81 = 10,1 N; μ = 0,9, n = 2 → **F_min ≈ 5,6 N**. A mért fogás **30,3 N** — ötszöröse —, és mégis csúszik. Ez önmagában bizonyítja, hogy **nem az erő kevés, hanem a geometria rossz**: az erők nem szemben hatnak.
>
> ⚠️ **És a 0,9 valójában nem is érvényesül.** Mérve (2026-08-05):
>
> | | csúszó súrlódás |
> |---|---|
> | `product_0_col` (beírva a jelenetépítőben) | 0,90 |
> | ujjbegyek (`medial` + `distal`, 6 geom) | **1,00 — a MuJoCo alapértelmezése** |
> | **tényleges** (MuJoCo elemenkénti maximumot vegyít) | **1,00** |
>
> Vagyis a gondosan beírt terméksúrlódást **a kéz alapértelmezése felülírja**. A jelenlegi szimuláció súrlódási viselkedése egyetlen mért számot sem tartalmaz — ez a `default`-jelölés létjogosultságának mintapéldája. A `contact` réteg ezért **párban** tárolandó: a termék felülete az SKU-bejegyzésben, a kéz felülete a robotprofilban, és a párosítás szabálya (max / geometriai átlag / mért pár) kimondva.

### 1.6 Csomagolás-mechanika — `packaging` *(A 2026-08-05-I KIEGÉSZÍTÉS)*

| mező | egység | forrás | most |
|---|---|---|---|
| `material` | enum (`carton_laminate`, `pet`, `hdpe`, `aluminium`, `glass`, `film`) | manufacturer | ✅ `carton_laminate` |
| `rigidity` | enum (`rigid`, `semi_rigid`, `compliant`, `flexible`) | estimated | ✅ `semi_rigid` — a korábbi „merev” csak a TELE állapotra igaz |
| `wall_thickness_mm` | mm | manufacturer | ❌ |
| `grip_force_min_N` | N | **derived** a súrlódásból | ❌ |
| `grip_force_max_N` | N | **measured** vagy manufacturer | ❌ |
| `deformation_onset_N` | N — ahol maradandó alakváltozás kezdődik | measured | ❌ |
| `crush_risk` | enum (`none`, `cosmetic`, `product_loss`) | estimated | ✅ `product_loss` — szivárgás, nem csak kozmetikai |

> **A két erőküszöb közti sáv a fogás valódi mozgástere.** Ha `grip_force_min > grip_force_max`, a terméket ezzel a kézzel nem lehet biztonságosan megfogni — és ezt **előre tudni** kell, nem a boltban kideríteni.
>
> Gable-top kartonnál a felső küszöb nem a szakítószilárdság, hanem a **behorpadás**: az oldalfal jóval előbb enged, mint hogy bármi elszakadna, és a behorpadt doboz eladhatatlan.

### 1.7 Manipuláció — `manipulation`

| mező | típus | forrás | most |
|---|---|---|---|
| `grasp_zones[]` | felület vagy térfogat a modell frame-jében | measured | ❌ |
| `forbidden_zones[]` | ua. (kupak, dátummező, vonalkód) | estimated | ✅ `[kupak, date_field]` |
| `preferred_axis` | enum | estimated | ✅ `side` |
| `grasp_plan_ref` | **hivatkozás** a robotspecifikus tervfájlra | derived | ✅ hivatkozás |
| `upright_required` | bool | estimated | ✅ `true` |

> **A `grasp_zones` termékadat, a `grasp_plan` robotadat.** A doboz oldalfala ott van, akármelyik robot nyúl érte; hogy a GENE.01 melyik ízületkonfigurációval éri el, az a robothoz tartozik. Ezért a kettő külön él, és az SKU-bejegyzés csak **hivatkozik** a tervre: `grasp_plans/{robot_id}/{sku_id}.json`.
>
> A `forbidden_zones` a dátummezőt is tartalmazza — ha az ujj rajta van, nem olvasható el.

### 1.8 Lejárat — `expiry`

A séma **legérettebb** része; ez már ma is teljes.

| mező | forrás | most |
|---|---|---|
| `type` (`use_by` \| `best_before`) + `type_source_hu` | manufacturer | ✅ |
| `location_human` + `location_source_hu` | manufacturer | ✅ |
| `format` (`%d.%m.%y`) + `layout_lines` | measured | ✅ |
| `print_style`, `char_height_mm`, `block_width_mm` | measured | ✅ |
| `placement` (sík, normális, uv-tengelyek, középpont, méret, pontosság) | measured | ✅ |
| `ocr_difficulty` | estimated | ❌ |
| `decision_rules` (`not_expired` / `expired` / `unreadable`) | belső szabály | ✅ |

| `storage_conditions` | manufacturer | ❌ — `use_by`-nál **kötelező** a címkén |
| `shelf_life_class` | derived | ❌ |
| `date_components` (`d_m_y` \| `m_y` \| `y`) | derived | ⚠️ a `format`-ból |

#### A jogi alap — ez teszi a mezőt KÖVETKEZTETHETŐVÉ, nem kitalálandóvá

Az **1169/2011/EU rendelet X. melléklete** szerint a dátumjelölésnél két megengedett forma van, harmadik nincs:

> A `best before` / `best before end` (illetve `use by`) szavakat **vagy maga a dátum követi, vagy egy utalás arra, hogy a címkén hol található a dátum.**

**Következmény a felvételi folyamatra:** EU-s élelmiszernél a `location` mező **soha nem hiányzik a csomagolásról** — vagy a dátum van ott, vagy az odavezető utalás. Nem kell megkeresni: ki kell olvasni. (Ugyanez vonatkozik a fagyasztási dátumra: „Fagyasztás ideje…" + dátum **vagy** utalás.)

**A formátum is jogilag kötött**, nem szabad szöveg: nap/hónap/év, ebben a sorrendben, kódolatlanul — és hogy melyik komponens szerepel, az az eltarthatóságból következik:

| eltarthatóság | kötelező komponensek | `shelf_life_class` |
|---|---|---|
| < 3 hónap | nap + hónap | `short` |
| 3–18 hónap | hónap + év elég | `medium` |
| > 18 hónap | év elég | `long` |

Ez **ellenőrzésre használható**: ha az OCR olyan dátumot ad vissza, ami nem illik a termék eltarthatósági osztályához, az olvasási hiba — nem kell elhinni. Ingyen kapott konzisztencia-teszt az M4-hez.

#### A döntési szabályaink jogi megfelelése

A `type` **jogi következménnyel jár**, és a szabályaink illeszkednek:

| eset | jogi helyzet | a mi szabályunk |
|---|---|---|
| `use_by` lejárt | a rendelet 24. cikk (1) szerint az élelmiszer **nem biztonságos**, forgalomba hozni tilos | **KIVONNI** ✅ |
| `best_before` lejárt | eladható, ha még kifogástalan állapotú, de **erősen ajánlott jelezni** a fogyasztónak | **JELÖLNI** ✅ |
| olvashatatlan | — | **NEM_OLVASHATO** (emberhez) |

⚠️ **Egy korlát a JELÖLNI művelet tervezéséhez:** a rendelet 8. cikk (4) szerint az élelmiszer-vállalkozó **nem módosíthatja** a terméket kísérő információt, beleértve a dátumot. A robot tehát *jelölhet* (matrica, polccímke, rendszerbejegyzés), de **a dátumhoz nem nyúlhat**, és nem takarhatja el.

> Ez a mező nem képből jön, hanem a bejegyzésből — ez a védőárok magja. A **kép csak a dátum ÉRTÉKÉT** adja; hogy az érték mit jelent és mi a teendő, az az SKU-bejegyzésből és a jogszabályból.

### 1.9 Kereskedelem — `retail`

| mező | forrás | most |
|---|---|---|
| `planogram_position` | belső | ❌ |
| `facing_count`, `restock_priority` | belső | ❌ |
| `shelf_zone` (hűtött / száraz / fagyasztott) | belső | ❌ |

### 1.10 Származás — `provenance`

| mező | most |
|---|---|
| `created`, `created_by`, `updated` | ✅ |
| `verified_by` mezőnként | ⚠️ csak a dátumtípusnál |
| `evidence[]` — mérési naplók útvonala | ❌ |

---

## 2. OpenUSD — mi kerül bele és mi nem

**A javaslat: rétegzett, egy belépési ponttal.** Az USD a mérvadó **geometria + fizika** hordozó, a szemantikus réteg mellette él, de **az USD-ből hivatkozva** — így egyetlen fájlt kell megnyitni ahhoz, hogy minden meglegyen.

### 2.1 Ami natívan USD

| séma-réteg | USD-megfelelő |
|---|---|
| `visual.mesh`, `texture` | `UsdGeomMesh` + `UsdShade` |
| `geometry.bbox`, `collision_proxy` | `UsdPhysicsCollisionAPI`, `UsdPhysicsMeshCollisionAPI` (`approximation`) |
| `physics.mass_kg`, `com_m`, `inertia` | `UsdPhysicsMassAPI` (`mass`, `centerOfMass`, `diagonalInertia`) |
| `contact.friction_*`, `restitution` | `UsdPhysicsMaterialAPI` (`staticFriction`, `dynamicFriction`, `restitution`) |
| `manipulation.grasp_zones` | `UsdGeomSubset` a hálón — **a felület része, nem külön koordinátalista** |
| `expiry.placement` | `UsdGeomXform` a dátumsíkra — a jelenlegi sík+normális+uv helyett |

> Ez a lista önmagában érv az USD mellett: a fizikai réteg fele **már ma is szabványos USD-mező**. Amit most szabad szövegben és külön JSON-ban tartunk, annak van kanonikus helye.

### 2.2 A dátum mint VariantSet — ez a valódi nyeremény

A dátummatricát ma külön geomként tesszük a modellre, hogy epizódonként más legyen. USD-ben ez pontosan egy **`VariantSet`**:

```
def "Product" {
    variantSet "expiry_date" = {
        "2027-03-22" { over "DateDecal" { ... } }
        "2026-01-05" { over "DateDecal" { ... } }
        "unreadable" { over "DateDecal" { ... } }
    }
}
```

Az eval epizódonként variánst vált, a 8K textúrához hozzányúlás nélkül — **beleértve a szándékosan olvashatatlan eseteket is**, amikre az M4-nél szükségünk lesz.

### 2.3 Ami NEM való USD-be

| mi | miért |
|---|---|
| `expiry.type`, `decision_rules` | üzleti szabály, nem jelenet; és ez a **védőárok** — külön kell kezelni, mert nem minden telepítéshez adjuk oda |
| `retail.*` | a planogram naponta változik, a geometria nem |
| `provenance`, `source`/`confidence` mezőnként | az USD metaadatokban technikailag megy, de olvashatatlan és nehezen diffelhető |
| `grasp_plan` | robotfüggő; egy termék-USD-hez több robot terve tartozhat |

### 2.4 A javasolt elrendezés

```
skus/<sku_id>/
  product.usda            ← belépési pont: geometria + fizika + variánsok
  product.payload.usdc    ← a nehéz háló, payload-ként (lusta betöltés)
  semantics.json          ← expiry, decision_rules, packaging, provenance
  textures/
grasp_plans/<robot_id>/<sku_id>.json
index.parquet             ← generált, lekérdezhető: EAN → útvonal + kulcsmezők
```

A `product.usda` `customData`-ban hivatkozza a `semantics.json`-t, tehát **egy fájlból minden elérhető**. Az `index.parquet` azért kell, mert az USD **nem lekérdezhető adatbázis**: „mutasd az összes olyan SKU-t, ahol a dátum a tetején van" kérdésre ezer fájlt kellene megnyitni.

---

## 3. A válasz arra, hogy „egységesítsük és hivatkozzuk be"

**Igen — és ez ugyanaz az elv, amit a kódra már felírtunk: *egy adat, egy forrás*.** Csak eggyel lejjebb, az adatrétegen.

Amit hozzátennék:

1. **A jelenet generált, nem írott.** A `shelflife_build_scene_sku.py` ma beírja a tömeget és a súrlódást. Ezeknek az SKU-bejegyzésből kell jönniük, különben a duplikáció visszatér. Ez a konkrét, mai lépés.
2. **Az USD-átállás ne blokkolja az M2-t.** A séma **most** rögzíthető, a tárolási forma később követi. A séma az érték; az USD a hordozó.
3. **A `source`/`confidence` mezők nélkül a séma nem véd meg semmitől.** Egy hiánytalanul kitöltött adatbázis, ami tele van `default` értékekkel, rosszabb az üresnél, mert magabiztossá tesz — pontosan ez történt a súrlódással.

---

## 4. Következő lépések

| # | lépés | mikor |
|---|---|---|
| ~~1~~ | ~~`source`/`confidence` bevezetése~~ | ✅ **kész 2026-08-05** — 23 mező |
| ~~2~~ | ~~tömeg + súrlódás az SKU-bejegyzésbe~~ | ✅ **kész 2026-08-05** — a jelenet bitre azonos maradt |
| **3** | **ujjbegy–karton súrlódás megmérése** (lejtő-teszt) | **KÖVETKEZŐ** — az M2 része, mert a csúszás ezen múlik |
| 4 | `grip_force_min` számítása, `grip_force_max` megmérése | M2 után |
| 5 | `ean_gtin` felvétele | a 2. SKU-nál (M6) |
| 5b | `storage_conditions` + `shelf_life_class` felvétele; az OCR-eredmény ellenőrzése a jogilag kötött formátum ellen | M4 (VLM-híd) |
| 6 | USD-export a meglévő bejegyzésből | M6 után, amikor két SKU-n látszik, mi ismétlődik |

---

## 5. Kapcsolódó

- `roboshelf_strategic_confidential` — az eredeti öt mezőcsoport (BIZALMAS)
- `EAN projekt_ Robotika és kiskereskedelem` — a Product Intelligence Layer szerepe a stratégiában
- `shelflife_kiserleti_terv` — a publikus/privát határ
- `shelflife_waddle_interfesz_terv` — M2, a fogás és a csúszás

> **Privát adat.** A kitöltött SKU-bejegyzések `*_private*` könyvtárban élnek és `.gitignore` alatt vannak. **Ez a séma-dokumentum publikálható; a kitöltött bejegyzések nem.**
