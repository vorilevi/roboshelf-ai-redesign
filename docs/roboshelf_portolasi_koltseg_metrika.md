# A portolási költség metrikája

**Készült:** 2026-08-16 · **Státusz:** definíció, mérés ELŐTT rögzítve

---

## Miért létezik

A projekt két állítást tesz, és csak az egyik van mérve:

| állítás | mérve? |
|---|---|
| **hordozhatóság** — ugyanaz a recept három gyártó humanoidján ≥70% | ✅ sikerarány, 50 epizód, robotonként |
| **fejlesztési sebesség** — az adaptáció költsége összeomlik, ha olcsó a visszajelzési hurok | ❌ csak anekdota: „napok", „5 döntés", „25 perc" |

Ez ugyanaz a hiány, ami a sikerarányoknál volt 2026 nyaráig, csak a másik tengelyen. Amíg a költség nincs mérve, a „fejlesztési sebesség" nem állítás, hanem érzés.

> **A metrikát a mérés ELŐTT kell kimondani.** 2026-08-16-án egyetlen nap alatt nyolc mérési hiba jött abból, hogy a kritériumot utólag fogalmaztam meg az eredményhez.

---

## A mértékegység

**Egy portolás** = egy új robot eljuttatása a nyers modellfájltól a **≥70% sikerarányig** a szabvány tolási feladaton, 50 epizódon mérve.

**Kezdet:** az első commit, ami az új robot állományát érinti.
**Vég:** az az eval-futás, amelyik először jelent ≥70%-ot.

---

## Három szakasz — és miért nem mindegy

A portolás nem homogén. A módszerek közti különbség **csak az egyik szakaszban** jelenik meg:

| | szakasz | mit tartalmaz |
|---|---|---|
| **P0** | alapozás | jelenet, env, diagnosztika, kartartás keresése |
| **P1** | első teljes kör | szakértő → export → finomhangolás → eval |
| **P2** | **hibakeresés** | az első bukott evaltól a küszöbig |

**A P2 az érdekes.** A P0 és a P1 nagyjából mechanikus; a P2-ben derül ki, hogy mennyibe kerül megérteni, mi a baj.

---

## ⚠️ AZ ÖSSZEHASONLÍTÁS FŐ VESZÉLYE

**A robotok nem egyformán nehezek.** A GR1T1-ben egy rejtett fizikai hiba volt (a passzív ízületeknek volt csillapításuk, de rugóerejük nem); a T1-ben nem. Ezért:

> A T1 (kézi) és a GR1T1 (ügynök-asszisztált) portolás **nem** összemérhető közvetlenül. A T1 gyors volt, mert könnyebb volt.

**Az egyetlen tiszta összehasonlítás ugyanazon a roboton belül van:**

    GR1T1 · P2 · ember-vezérelt   2026-07-27 → 07-28   → 20% és az alatt
    GR1T1 · P2 · ügynök-asszisztált  2026-08-01 → 08-02   → 94%

Ugyanaz a robot, ugyanaz a hiba, ugyanaz a feladat. **Ez a mérés.** Minden más kontextus.

---

## Amit mérünk

### A. GÉPI KÖLTSÉG — `measured`, a git-ből és a naplókból

| # | mező | forrás |
|---|---|---|
| A1 | naptári átfutás (óra) | commit-időbélyegek |
| A2 | commitok száma | git |
| A3 | új sorok / módosított sorok | `git numstat` |
| A4 | érintett fájlok, új vs. módosított | git |
| A5 | teljes betanítási körök száma | naplók + jegyzetek |
| A6 | GPU-perc — betanításra | naplók |
| A7 | GPU-perc — diagnózisra | naplók |

### B. EMBERI KÖLTSÉG — a valódi célváltozó

| # | mező | forrás |
|---|---|---|
| B1 | emberi döntések száma | jegyzet + beszélgetésnapló · `derived` |
| B2 | emberi hibakeresési idő (perc) | `estimated` — visszamenőleg csak becsülhető |

**Mi számít emberi döntésnek** (előre rögzítve, hogy ne lehessen utólag hozzáigazítani):

> Döntés az, ahol az ember **két vagy több technikailag életképes irány közül választ**, hatáskört ad vagy von meg, vagy kritériumot mond ki.
>
> **NEM döntés:** nyugtázás, „mehet", „folytasd", egy magától értetődő következő lépés jóváhagyása.

### C. ITERÁCIÓS SZERKEZET — ez a MAGYARÁZÓ változó

| # | mező | forrás |
|---|---|---|
| C1 | mérési iterációk száma a P2-ben | naplók |
| C2 | egy iteráció medián késleltetése (s) | naplók |
| C3 | elvetett hipotézisek száma a sikeres előtt | jegyzetek |

**A tézis, amit ez tesztel:**

> Nem az intelligencia a különbség, hanem az **iteráció ára**. Ha egy kör 20 másodperc, megengedhető tíz „buta" kérdést megmérni. Ha két és fél óra, egyet sem.

Ez ellenőrizhető: ha a tézis igaz, a **C2 fordítottan arányos** a P2 hosszával, és a C3 magasabb az olcsóbb hurkú ágban (több hipotézist lehet elvetni).

### D. HORDOZHATÓSÁGI HÁNYADOS

| # | mező | képlet |
|---|---|---|
| D1 | változatlanul újrahasznált fájlok | git |
| D2 | **hordozhatósági hányad** | `újrahasznált / (újrahasznált + új + módosított)` |

Ez teszi számszerűvé a hordozhatóság állítást: **mekkora része a stacknek megy át érintetlenül.**

---

## Forrásjelölés

Ugyanaz a rend, mint a termékadatoknál (`roboshelf_sku_sema`):

    measured   — a git vagy a napló adja
    derived    — másik mért adatból számolt
    estimated  — visszamenőleg becsült (⚠️ csak a B2 ilyen)
    unknown    — nincs adat

⚠️ **Egy visszamenőleges rekonstrukció nem ugyanaz, mint egy élesben futó mérés.** A B2 és részben a C1–C3 a kézi ágra csak becsülhető. Ezt minden kimenetben jelölni kell, és **a becsült mezőkre nem szabad állítást építeni** külső kommunikációban.

---

## Amit ez NEM mér

- **A minőséget.** A gyorsabb portolás nem feltétlenül jobb policy-t ad. A sikerarány külön mérce.
- **A tanulási görbét.** A második robot azért is gyorsabb, mert az elsőn tanultunk. Ez a metrikából nem választható szét — csak több portolásból.
- **A kockázatot.** Az ügynök `rm -rf`-fel törölt egy exportot (2026-08-02, helyreállt). A gyorsaságnak ára is van.

---

## Elfogadási feltétel a metrikára magára

A metrika akkor használható, ha a **GR1T1 két P2 szakaszára** kitölthető úgy, hogy minden mező vagy `measured`, vagy jelölten `estimated` — és a két oszlop egymás mellé tehető.

Ha egy mező egyik ágon sem tölthető ki, **ki kell venni a metrikából**, nem kitalálni hozzá adatot.
