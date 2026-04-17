"""Phase 2 training entrypoint skeleton."""

from roboshelf_ai.configs.phase2 import PHASE2_NAME
from roboshelf_ai.mujoco.envs.retail_nav import RetailNavConfig, RetailNavEnv

def main():
    config = RetailNavConfig(name=PHASE2_NAME)
    env = RetailNavEnv(config=config)
    print(f"Loaded {env.config.name} for {env.config.robot_name}")

if __name__ == "__main__":
    main()
