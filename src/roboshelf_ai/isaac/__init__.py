"""
roboshelf_ai.isaac
==================
Isaac Lab integráció — PLACEHOLDER (Fázis F+)

Ez a csomag akkor töltődik be teljesen, amikor a projekt átkerül
NVIDIA Isaac Lab-ra GPU-s tanításhoz. Addig üres stub.

Tervezett struktúra:
    isaac/
    ├── adapters/
    │   └── robot_state_adapter.py   # IsaacEnv → RobotState konverter
    └── envs/
        ├── nav_isaac_env.py         # Isaac Lab nav env (BaseNavPolicy-kompatibilis)
        └── manip_isaac_env.py       # Isaac Lab manip env

Átálláskor a MuJoCo env-ek lecserélhetők Isaac Lab megfelelőikre,
de az összes policy (UnitreeRLGymAdapter, NavPPOPolicy stb.)
változatlanul marad — a RobotState absztrakció miatt.

Referenciák:
    https://isaac-sim.github.io/IsaacLab/
    unitree_rl_gym Isaac Lab branch (ha elérhető)
"""

# Jelenleg nincs exportálnivaló.
# Ha Isaac Lab elérhető, ide kerülnek a from isaac.adapters import ... sorok.
