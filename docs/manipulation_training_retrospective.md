# Manipulation Policy Training — Retrospektív

> **Ez a dokumentum az Obsidian-ban él.**
>
> Helye: `01 Projects/Roboshelf AI/AI betanitas/manipulation_training_retrospective.md`
>
> A repóban csak hivatkozásként szerepel — a teljes tartalom, verzióhistória és elemzések ott találhatók.

## Gyors összefoglaló (2026-04-29 — F3b BLOKKOLT)

- **Cél:** G1 robot megfogja a dobozt és felrakja a polcra — elfogadás: ≥ 70% sikerességi arány
- **PPO sandbox (F3a):** 12 verzió, ~49h compute, $0 cost → max 10% (valójában 0%, mérési artifact) → **LEZÁRVA**
- **F3b scripted expert:** 5 iteráció → max 2% SR → **BLOKKOLT** (geometriai deadlock)
- **Gyökér ok:** G1 4-DOF PD arm gravitáció ellen max z≈0.84-0.90m; stock settled z≈0.77m; target z=0.85m → Δz=0.08m = goal_radius → geometriailag szinte lehetetlen siker
- **Döntési pont nyitott:** A) kp gain fix | B) push task (target z=0.77) | C) F3e (VLA)

## Kapcsolódó technikai fájlok (repo)

| Fájl | Leírás |
|---|---|
| `src/envs/assets/scene_manip_sandbox_v2.xml` | MuJoCo scene (target_shelf z=0.85m) |
| `src/roboshelf_ai/mujoco/envs/manipulation/g1_shelf_stock_env.py` | Env implementáció |
| `configs/manipulation/shelf_stock_v9.yaml` | Legjobb PPO config (igaz SR: 0%) |
| `configs/manipulation/shelf_stock_v12_final.yaml` | Utolsó PPO config |
| `tools/scripted_expert.py` | Scripted expert (F3b, blokkolt) |
| `tools/policy_demo_collector.py` | PPO rollout gyűjtő |
| `docs/known_issues.md` | Ismert hibák (#15–20; #19-20 = F3b deadlock) |
| `roboshelf_schedule.md` | F3b állapot: ❌ blokkolt |
