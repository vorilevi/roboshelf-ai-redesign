# Legacy fájlok

Ezek a fájlok a Phase 2 monolitikus rendszer archivált másolatai.
Ne töröld őket — baseline referenciának és regresszióteszthez kellenek.

Nem aktívan fejlesztett kód. Az aktív redesign a következő mappákban van:
- `src/roboshelf_ai/` — fő csomag (locomotion, navigation, manipulation, demo)
- `src/roboshelf/` — segédmodulok (checks, replay)

| Fájl | Forrás | Megjegyzés |
|---|---|---|
| `train_phase2_monolithic.py` | `src/training/roboshelf_phase2_train.py` | Legacy PPO train script |
| `finetune_phase2_monolithic.py` | `src/training/roboshelf_phase2_finetune.py` | Legacy finetune script |
| `retail_nav_env_monolithic.py` | `src/envs/roboshelf_retail_nav_env.py` | Monolitikus nav env (loco + task együtt) |
| `manipulation_env_monolithic.py` | `src/envs/roboshelf_manipulation_env.py` | Monolitikus manipulációs env |
