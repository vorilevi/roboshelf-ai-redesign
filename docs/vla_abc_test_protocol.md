# VLA A/B/C Teszt Protokoll

_Létrehozva: 2026-04-23 | Phase 030 F2 | Belső dokumentum_

## Célja

Három nyílt forráskódú VLA modell összehasonlítása retail pick-place feladaton,
EIbench-kompatibilis metrikák alapján. A nyertes modell kerül retail-specifikus
fine-tune-ra (F5).

## Modellek

| Jelölés | Modell | Repo | Inferencia platform |
|---|---|---|---|
| A | WALL-OSS | `X-Square-Robot/wall-x` | Vast.ai A100 (CUDA 12.x) |
| B | UnifoLM-VLA-0 | `unitreerobotics/unifolm-vla` | Vast.ai A100 (CUDA 12.4) |
| C | GR00T N1.6 | `nvidia/groot-n1.6` | Vast.ai A100 (GPU kötelező) |

## Environment

- **Framework:** unitree_rl_mjlab (mjlab==1.2.0), G1 pick-place scene
- **Robot:** Unitree G1, 29 DoF, 50 Hz control
- **Feladat:** "place product on shelf" — 1 termék, 1 polcpozíció
- **Epizódok:** 50/modell (valós F4-ben), 5/modell (stub smoke test)

## Metrikák (EIbench-kompatibilis)

| Metrika | Leírás | Egység |
|---|---|---|
| `success_rate` | Sikeres epizódok aránya | 0.0–1.0 |
| `completion_time_s` | Átlagos feladatvégzési idő | másodperc |
| `energy_efficiency` | torque × displacement integrál | (alacsonyabb = jobb) |
| `inference_latency_ms` | VLA forward pass időtartama | ms |
| `min_human_distance_m` | Legközelebbi ember-távolság | méter |

## Composite Score

```
composite_score = success_rate / (avg_inference_latency_ms / 100.0)
```

Normalizálás: 100 ms = 1.0 referencia latencia.

## Döntési szabály

1. Legmagasabb `composite_score` nyer.
2. Ha a top-2 között **< 5% különbség** → **WALL-OSS preferált** (geopolitikai diverzifikáció).
3. Döntés dokumentálva: `docs/vla_abc_results.md` (F4 után automatikusan generálva).

## Futtatás

### Stub smoke test (Mac M2, bármikor)
```bash
cd ~/roboshelf-ai-dev/roboshelf-ai-redesign
python scripts/eval_vla_abc.py --model all --episodes 5 --stub
```

### Valós A/B/C teszt (F4, Vast.ai A100)
```bash
python scripts/eval_vla_abc.py --model all --episodes 50 --no-stub
```

### Egy modell külön
```bash
python scripts/eval_vla_abc.py --model wall-oss --episodes 50 --no-stub
```

## Kimeneti fájlok

```
results/vla_abc/
  wall_oss/
    metrics.json     — EIbench-kompatibilis összesítő
    episodes.csv     — epizód-szintű adatok
  unifolm_vla_0/
    metrics.json
    episodes.csv
  groot_n1.6/
    metrics.json
    episodes.csv
  summary.csv        — összesítő táblázat, composite score
```

## Stub smoke test eredmény (2026-04-23, 5 epizód)

| Modell | Success% | Lat(ms) | Composite | Helyezés |
|---|---|---|---|---|
| WALL-OSS | 80.0% | 41.0 | 1.953 | 🥇 1. |
| UnifoLM-VLA-0 | 60.0% | 56.8 | 1.057 | 3. |
| GR00T N1.6 | 80.0% | 76.7 | 1.042 | 2. |

_Megjegyzés: stub értékek, nem valós inference. Valós eredmény F4-ben (Vast.ai)._

## Kapcsolódó fájlok

- `scripts/eval_vla_abc.py` — kiértékelő script
- `roboshelf-common/vla_client/client.py` — VLAClient, VLAModel enum
- `roboshelf-common/heis_adapter/adapter.py` — EIBenchMetrics export
- Obsidian: `roboshelf_execution_plan_2026-04-22 pharse 030` — F4 részletek
