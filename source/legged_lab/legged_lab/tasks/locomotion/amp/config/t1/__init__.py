"""Register Men T1 AMP training and playback tasks."""

import gymnasium as gym

from legged_lab.envs import ManagerBasedAmpEnv

from . import agents


gym.register(
    id="LeggedLab-Isaac-AMP-T1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_amp_env_cfg:T1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-T1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_amp_env_cfg:T1AmpEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1RslRlOnPolicyRunnerAmpCfg",
    },
)
