"""
roboshelf_ai.isaac.envs — PLACEHOLDER
======================================
Isaac Lab-alapú tanítási env-ek.

Tervezett fájlok (Fázis F+):

    nav_isaac_env.py
        RetailNavIsaacEnv(DirectRLEnv)
        - Ugyanaz a reward struktúra mint retail_nav_hier_env.py
        - GPU-párhuzamosítás: n_envs=4096+
        - Observation space: azonos a MuJoCo nav env-vel (9 dim)
        - UnitreeRLGymAdapter.step(RobotState, command) hívás változatlan

    manip_isaac_env.py
        G1PickPlaceIsaacEnv(DirectRLEnv)
        - Pick & place GPU env

Portolási stratégia:
    1. MuJoCo env-ek acceptance feltételeit teljesíteni (Fázis B–C)
    2. Ugyanazokat a reward függvényeket átírni Isaac Lab tensor API-ra
    3. Policy súlyok MuJoCo-ban tanítva → Isaac Lab fine-tune
"""
