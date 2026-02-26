from legged_lab.tasks.locomotion.amp.amp_env_cfg import AmpEnvCfg

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from legged_lab.assets.bsrl import BSRL_CFG # 引入你的机器人本体配置

@configclass
class BSRLAmpEnvCfg(AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 将场景中的机器人替换为你的 BSRL 配置
        self.scene.robot = BSRL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- 以下是基础的配置调整，你需要根据你的机器人进行修改 ---

        # 1. 调整控制步长 (默认可能是控制 50Hz, 仿真 200Hz)
        self.decimation = 4 

        # 2. 初始状态 (你可以在资产里配，也可以在这里覆盖)
        # self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)

        # 3. 动作控制
        self.actions.joint_pos.scale = 0.25 # 将神经网络 -1~1 的输出映射为实际的角度范围

        # 4. 观测空间 (AMP 核心)
        # 你的机器人只有下肢，所以关键部位(Key Body)只有脚踝（或脚底）。
        # 这些名字必须和你的 URDF/USD 里的 link 名字完全匹配！
        KEY_BODY_NAMES = [
            "link_right_ankle_roll",
            "link_left_ankle_roll",
        ]
        self.observations.policy.key_body_pos_b.params["asset_cfg"] = mdp.SceneEntityCfg(
            "robot", body_names=KEY_BODY_NAMES
        )
        self.observations.critic.key_body_pos_b.params["asset_cfg"] = mdp.SceneEntityCfg(
            "robot", body_names=KEY_BODY_NAMES
        )
        self.observations.disc.key_body_pos_b.params["asset_cfg"] = mdp.SceneEntityCfg(
            "robot", body_names=KEY_BODY_NAMES
        )
        self.observations.disc_demo.ref_key_body_pos_b.params["asset_cfg"] = mdp.SceneEntityCfg(
            "robot", body_names=KEY_BODY_NAMES
        )
        self.animation.animation.motion_data_components[6] = "key_body_pos_b"


        # 4. 观测空间 (可以暂时不管，使用 AmpEnvCfg 基础配置)
        
        # 5. 奖励函数 (可以暂时不管，使用 AmpEnvCfg 基础配置)
        
        # 6. 终止条件
        # 如果机器人根节点高度低于 0.3 米认为跌倒 (需根据你的机器人腿长调整)
        self.terminations.base_height.params["minimum_height"] = 0.3
        # 身体基础如果碰到地面，认为跌倒
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class BSRLAmpEnvCfgPlay(BSRLAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 播放模式下，往往环境数量为 1
        # self.viewer.eye = (2.0, 2.0, 1.0)
        pass

