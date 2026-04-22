# roboshelf-common

Közös komponensek a Roboshelf AI két trackjéhez (Phase 030).

## Modulok

| Modul | Leírás |
|---|---|
| `lerobot_pipeline/` | LeRobotDataset v3.0 konverzió és betöltés. Bármilyen demonstrációs forrást (Unitree G1 BrainCo, OpenHE, saját rollout) egységes Parquet+MP4 formátumra hoz. |
| `product_intelligence_layer/` | PIL termék-metaadat séma és SQLite-alapú adatbázis. Mindkét track (humanoid MJCF + EAN Isaac Lab) ugyanebből olvas. |
| `heis_adapter/` | HEIS 2026 Q1 v1.0 szabvány szerinti obs/action konverzió + EIbench metrika export. |
| `vla_client/` | Egységes inference kliens: WALL-OSS, UnifoLM-VLA-0, GR00T N1.6. Modell-swap egyetlen config flag. |

## Használat

Mindkét track `pyproject.toml`-jában (vagy `setup.py`-ban):

```toml
# pyproject.toml
[tool.poetry.dependencies]
roboshelf-common = { path = "../../roboshelf-common", develop = true }
```

Vagy PYTHONPATH-on keresztül fejlesztés során:

```bash
export PYTHONPATH=$PYTHONPATH:/Users/vorilevi/roboshelf-ai-dev/roboshelf-ai-redesign/roboshelf-common
```

## Szerzők

Roboshelf AI fejlesztőcsapat — Phase 030, 2026 Q2
