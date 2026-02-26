import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs import ManagerBasedAmpEnvCfg
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm

@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # robot animation (for reference)
    robot_anim: ArticulationCfg = None
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # 每 10 秒重新采样一次指令
        rel_standing_envs=0.02, # 2% 的时间让机器人站立不动
        rel_heading_envs=1.0,   # 100% 的时间控制朝向
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg():
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # # observation terms (order preserved)
        # # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # actions = ObsTerm(func=mdp.last_action)
        
    # ---------------------------------------------------------------------------------------------
    # 策略网络的观测 (Policy Observations) - 机器人"大脑"能看到的数据
    # ---------------------------------------------------------------------------------------------
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 1. 基座角速度 (Base Angular Velocity) - 对应 IMU 陀螺仪数据
        # noise: 添加噪声模拟真实传感器的误差
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        # 2. 根节点切向/法向向量 (Root Local Rotation Tangent/Normal) - 帮助机器人理解身体朝向
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm, noise=Unoise(n_min=-0.05, n_max=0.05))
        
        # 3. 速度指令 (Velocity Commands) - 机器人接收到的目标速度命令 (例如: 前进 1m/s)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # 4. 关节位置 (Joint Positions) - 电机编码器数据
        # 注意: 这里的 joint_pos 通常是归一化后的或者相对于默认姿态的偏差
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        
        # 5. 关节速度 (Joint Velocities) - 电机速度
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        
        # 6. 上一次动作 (Last Action) - 网络上一步输出的动作，用于平滑控制
        actions = ObsTerm(func=mdp.last_action)
        
        # 7. 关键部位位置 (Key Body Positions) - AMP 算法特有，用于模仿学习
        # 比如：脚相对于身体的位置。这能帮助机器人学在特定动作下脚该放哪里。
        # [重要] 如果你的机器人没有手，记得去 g1_amp_env_cfg.py 里修改 KEY_BODY_NAMES
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING, # 在子类 (如 g1_amp_env_cfg.py) 中具体指定
            noise=Unoise(n_min=-0.08, n_max=0.08),
        )
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()
    
    @configclass
    class DiscriminatorCfg(ObsGroup):
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False
            
    disc: DiscriminatorCfg = DiscriminatorCfg()
            
    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp.ref_root_local_rot_tan_norm,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
    
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()
        


@configclass
class EventCfg:
    """Configuration for events."""

    # ---------------------------------------------------------------------------------------------
    # 物理属性随机化 (Domain Randomization) - 增强 Sim2Real 鲁棒性
    # ---------------------------------------------------------------------------------------------
    
    # 1. 摩擦力随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),  # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.0), # 动摩擦系数范围
            "restitution_range": (0.0, 0.0),      # 弹性系数 (0 = 不反弹)
            "num_buckets": 64,
        },
    )

    # 2. 质量随机化 (Mass Randomization)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING), # 在子类中指定部位
            "mass_distribution_params": (-1.0, 3.0), # 质量变化范围 (kg)
            "operation": "add",
        },
    )

    # reset
    # 3. 外力干扰 (External Force Disturbance) - 提高抗干扰能力
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "force_range": (0.0, 0.0),   # 施加的力范围 (N)
            "torque_range": (-0.0, 0.0), # 施加的力矩范围 (Nm)
        },
    )

    reset_from_ref = EventTerm(
        func=mdp.reset_from_ref, 
        mode="reset",
        params=MISSING
    )

    # interval
    # 4. 推机器人 (Push Robot) - 间隔性推一下
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0), # 每 5 秒推一次
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, # 瞬间改变速度范围
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- 惩罚项 (Penalties) - 负权重
    # 如果机器人想通过奇怪的抖动来骗分，这些惩罚项会阻止它
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) # 惩罚垂直方向的速度 (不想让它上下乱跳)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # 惩罚非偏航角速度 (不想让它前后左右摇晃)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5) # 惩罚力矩 (节能)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # 惩罚关节加速度 (平滑动作)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # 惩罚动作变化率 (避免高频抖动)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # 2. 姿态异常终止
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(60.0), # 如果倾斜超过 60 度，判负并重置
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class MotionDataCfg:
    """Motion data settings for the MDP."""
    motion_dataset = MotionDataTerm(
        motion_data_dir="", 
        motion_data_weights={},
    )
    
@configclass
class AnimationCfg:
    """Animation settings for the MDP."""
    animation = AnimTerm(
        motion_data_term="motion_dataset",
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ], 
        num_steps_to_use=10, 
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )


##
# Environment configuration
##


@configclass
class LocomotionAmpEnvCfg(ManagerBasedAmpEnvCfg):
    """Configuration for the AMP locomotion environment."""

    # scene
    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Motion data
    motion_data: MotionDataCfg = MotionDataCfg()
    # Animation
    animation: AnimationCfg = AnimationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

