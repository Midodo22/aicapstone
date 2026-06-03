import gymnasium as gym


gym.register(
    id="HCIS-OpenFridge-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.open_fridge_env_cfg:OpenFridgeEnvCfg",
    },
)
