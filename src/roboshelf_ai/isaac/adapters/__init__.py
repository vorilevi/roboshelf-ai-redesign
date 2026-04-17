"""
roboshelf_ai.isaac.adapters — PLACEHOLDER
==========================================
Isaac Lab → RobotState konverter és egyéb adapterek.

Implementálni kell (Fázis F+):

    class IsaacRobotStateAdapter:
        \"\"\"Isaac Lab ArticulationView → RobotState konverzió.\"\"\"

        @staticmethod
        def from_isaac(
            articulation,       # isaac_lab.envs.mdp.ArticulationView
            env_idx: int = 0,
        ) -> RobotState:
            # qpos, qvel, quat, omega kinyerése Isaac tensor API-ból
            # majd RobotState(qpos=..., qvel=..., ...) visszaadása
            raise NotImplementedError

Amikor ez kész, a RobotState.from_isaac() placeholder is implementálható.
"""
