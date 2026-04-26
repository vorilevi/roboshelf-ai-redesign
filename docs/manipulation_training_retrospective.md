# Manipulation Policy Training — Retrospektív

> **Ez a dokumentum az Obsidian-ban él.**
>
> Helye: `01 Projects/Roboshelf AI/AI betanitas/manipulation_training_retrospective.md`
>
> A repóban csak hivatkozásként szerepel — a teljes tartalom, verzióhistória és jövőbeli tervek ott találhatók.

## Gyors összefoglaló (2026-04-27)

- **Cél:** G1 robot megfogja a dobozt és felrakja a polcra — elfogadás: ≥ 70% sikerességi arány
- **Legjobb eredmény:** 10% siker (v9, 500k és 5M futás)
- **Jelenlegi futás:** v11 — 3M timestep, folyamatban
- **Legfontosabb fix:** ujj jointek `class="passive_joint" → "finger_joint"` (damping 200→0.1)
- **Hyperparaméter korlátok:** ent_coef=0.03, w_smooth max -0.001, w_lift=5.0

## Kapcsolódó fájlok

| Fájl | Leírás |
|---|---|
| `src/envs/assets/scene_manip_sandbox_v2.xml` | MuJoCo scene |
| `src/roboshelf_ai/mujoco/envs/manipulation/g1_shelf_stock_env.py` | Env implementáció |
| `configs/manipulation/shelf_stock_v*.yaml` | Konfigurációk |
| `docs/known_issues.md` | Ismert hibák (#15–18) |
| `tools/viz_gripper_test.py` | Gripper vizualizáció |
| `tools/viz_manip_policy.py` | Policy lejátszás |
