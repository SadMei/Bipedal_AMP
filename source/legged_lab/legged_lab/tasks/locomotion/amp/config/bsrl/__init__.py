import gymnasium as gym

from .bsrl_amp_env_cfg import BSRLAmpEnvCfg, BSRLAmpEnvCfgPlay
from . import agents

from legged_lab.envs import ManagerBasedAmpEnv

gym.register(
    id="LeggedLab-Isaac-AMP-BSRL-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_amp_env_cfg:BSRLAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BSRLRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-BSRL-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_amp_env_cfg:BSRLAmpEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BSRLRslRlOnPolicyRunnerAmpCfg",
    },
)
