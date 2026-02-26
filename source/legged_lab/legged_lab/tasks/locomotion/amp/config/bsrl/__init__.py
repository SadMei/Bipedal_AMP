from .bsrl_amp_env_cfg import BSRLAmpEnvCfg, BSRLAmpEnvCfgPlay
import gym

gym.register(
    id="Isaac-BSRL-Amp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_amp_env_cfg:BSRLAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg:G1AmpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BSRL-Amp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_amp_env_cfg:BSRLAmpEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg:G1AmpPPORunnerCfg",
    },
)
