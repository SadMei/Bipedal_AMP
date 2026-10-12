"""Physics-informed residual world-model API for Legged Lab.

The core is dimensioned by :class:`WorldModelStateSpec`; T1 and BSRL adapters
share the same replay/model/trainer contracts. Isaac extraction remains
strictly opt-in.
"""

from .config import ResidualWorldModelCfg, WorldModelStateSpec
from .collector import CollectionMetrics, TransitionCollector
from .dynamics import PhysicsInformedWorldModel, WorldModelStep
from .interfaces import DataCollectionEnv, EnvStepResult, PipelineState
from .isaac_data_env import IsaacLabDataCollectionEnv, concatenate_observation_groups
from .amp_adapter import (
    AmpWorldModelContractAudit,
    BSRL_12_JOINT_NAMES,
    amp_reference_joint_state,
    amp_world_model_frame,
    bsrl_estimator_actor_indices,
    bsrl_world_model_state_spec,
    validate_amp_world_model_contract,
)
from .mjx_backend import (
    BSRL_12_FRAME_JOINT_NAMES,
    ControlMode,
    IntegratorMode,
    MJXBackendCfg,
    MJXNominalPhysics,
    bsrl_r2_mjx_cfg,
    bsrl_serial_mjx_cfg,
    t1_09_mjx_cfg,
    t1_run39_mjx_cfg,
    t1_wbtest_mjx_cfg,
    wbtest_mjx_cfg,
)
from .models import (
    VARIANCE_PARAMETERIZATION,
    EnsemblePrediction,
    ResidualWorldModel,
    load_residual_world_model,
    variance_parameterization,
)
from .motion_reference import MotionReferenceBatch, MotionReferenceProvider
from .initialization import default_standing_frame
from .ppo_finetune import PPOWorldModelFineTuneCfg, PPOWorldModelFineTuner, wbtest_observation_builders
from .replay import ReplayBuffer, TransitionBatch, sample_mixed
from .sac import SACAgent, SACCfg
from .state import (
    MODEL_ORIENTATION_INPUT,
    FrameHistory,
    context_state,
    dynamic_state,
    model_input,
    pack_frame,
    yaw_invariant_tilt_quaternion,
)
from .trainer import WorldModelTrainer, WorldModelTrainerCfg

__all__ = [
    "EnsemblePrediction",
    "CollectionMetrics",
    "ControlMode",
    "DataCollectionEnv",
    "EnvStepResult",
    "FrameHistory",
    "AmpWorldModelContractAudit",
    "BSRL_12_FRAME_JOINT_NAMES",
    "BSRL_12_JOINT_NAMES",
    "IsaacLabDataCollectionEnv",
    "MJXBackendCfg",
    "IntegratorMode",
    "MJXNominalPhysics",
    "MODEL_ORIENTATION_INPUT",
    "MotionReferenceBatch",
    "MotionReferenceProvider",
    "default_standing_frame",
    "PPOWorldModelFineTuneCfg",
    "PPOWorldModelFineTuner",
    "wbtest_observation_builders",
    "PhysicsInformedWorldModel",
    "PipelineState",
    "ReplayBuffer",
    "ResidualWorldModel",
    "ResidualWorldModelCfg",
    "SACAgent",
    "SACCfg",
    "load_residual_world_model",
    "TransitionBatch",
    "VARIANCE_PARAMETERIZATION",
    "variance_parameterization",
    "TransitionCollector",
    "WorldModelStateSpec",
    "WorldModelStep",
    "WorldModelTrainer",
    "WorldModelTrainerCfg",
    "context_state",
    "concatenate_observation_groups",
    "amp_reference_joint_state",
    "amp_world_model_frame",
    "bsrl_r2_mjx_cfg",
    "bsrl_estimator_actor_indices",
    "bsrl_serial_mjx_cfg",
    "bsrl_world_model_state_spec",
    "wbtest_mjx_cfg",
    "t1_wbtest_mjx_cfg",
    "t1_09_mjx_cfg",
    "t1_run39_mjx_cfg",
    "dynamic_state",
    "model_input",
    "pack_frame",
    "sample_mixed",
    "yaw_invariant_tilt_quaternion",
    "validate_amp_world_model_contract",
]
