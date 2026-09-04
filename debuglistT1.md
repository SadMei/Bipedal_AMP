1. mode collapse 时 立刻停训，修改参数重新训练
2. AMP 风格参数 课程 （后期task占主导）
3.
# T1 AMP Debug List

按实验发生时间记录 T1 从迁移、数据检查、奖励修正到前进专项训练的改动、效果与结论。

本文以每个 run 目录内的 `params/env.yaml`、`params/agent.yaml` 和 TensorBoard 事件为配置与数值依据；代码在训练期间多次继续修改，因此不能用当前源码反推历史 run。视频中的肉眼结论单独记录，不用单一总奖励替代实际播放判断。

## 实验时间线

### 2026-08-21：T1 初次迁移到 AMP

- 新增 25 DoF T1 AMP 任务，最初使用 `Men_T1_0722_new_shoe/urdf/men_t1_path_25dofs_collision_feet_points.urdf`。
- 关节顺序固定为左右腿、腰、躯干和双臂共 25 个关节；动作是关节位置目标，scale 全部为 `0.25`。
- Actor 使用 5 帧历史：基座角速度、根姿态 6D、速度指令、关节位置、关节速度、上一动作和 6 个关键刚体位置，共 `525` 维；不直接观察基座线速度。
- Critic 使用对应无噪声特权观测并额外包含基座线速度，共 `540` 维。
- 初始 AMP 判别器没有基座线速度，也没有按指令选择参考类别。18 个 PKL 最初按文件等概率采样，因此短转向片段会被相对过采样。
- 使用工程内独立的 `external/rsl_rl` AMP runner，没有替换系统或其他仓库的 RSL-RL；普通 AMP/BSRL 路径保持兼容。
- 多次 `19/39/98 iteration` smoke test 验证任务注册、参考数据读取、AMP rollout 和 checkpoint 保存。

正式 run：`2026-08-21_16-24-37_formal_run2_8192`

- 配置：`8192 env / 50000 iteration`，style scale `5.0`，task/style 混合系数 `0.4`，entropy `0.01`，无速度课程。
- 实际在 `23505` 轮停止并保留 checkpoint；最后落盘 checkpoint 为 `model_23500.pt`。
- 最后日志：episode length `500/500`，time-out `99.26%`，XY 速度误差约 `0.608 m/s`，yaw 误差约 `0.323 rad/s`。
- 录像：`videos/t1_amp_model_23000_mixed-step-0.mp4`、`t1_amp_model_23000_large_commands-step-0.mp4`、`t1_amp_model_23000_fast_forward-step-0.mp4`。
- **现象与结论**：大指令下前进仍不明显。判别器没有平移速度时，原地做出类似步态也可能得到较高风格分，不足以约束“动作像且确实向前移动”。

### 2026-08-22：速度跟踪 reward fix

正式 run：`2026-08-22_15-26-48_rewardfix_formal1_8192`

- 所有速度跟踪统一到 base frame，避免世界系速度与机体系指令混用。
- 线速度/偏航跟踪权重从 `1/1` 提高到 `3/2`。
- 新增 XY 超额速度误差惩罚：超过 `0.25 m/s` 后平方惩罚，权重 `-1.0`，最大 excess `2.0 m/s`。
- 加入三阶段课程：边界 `3000/9000`；`vx` 最终 `[-0.7, 1.1] m/s`，`vy` 最终 `[-0.55, 0.55] m/s`，yaw 最终 `[-2, 2] rad/s`；跟踪 std 为 `0.5/0.4/0.3`。
- AMP task/style 混合改为 `0.7/0.3`，优先逼迫策略响应指令。
- 参考文件权重改为帧数权重。18 个文件共 `14463` 帧，因此每个参考帧近似等概率，不再让多个短转向文件压过较长的前进动作。
- 实际停止于约 `22225` 轮，最后 checkpoint 为 `model_22000.pt`。
- 最后日志：episode length 约 `496.7/500`，time-out `98.89%`，XY 速度误差约 `0.251 m/s`，yaw 误差约 `0.360 rad/s`，noise std 约 `3.06`。
- 录像：`videos/t1_amp_rewardfix_model_22000_forward_1ms-step-0.mp4`。
- **现象与结论**：数值上的速度误差明显下降，但录像出现关节接近限位、姿态扭曲的问题。速度指标正常不能排除动作/URDF/参考姿态链路错误，因此进入完整对齐检查。

### 2026-08-23：参考数据与动作链路对齐

- 使用用户重新上传的正确源 PKL，通过转换脚本重新生成 AMP 数据。
- 对所有参考文件检查：关节名与策略顺序一致、6 个关键刚体顺序一致、数组 shape 正确、无 NaN/Inf、根四元数单位化。
- 增加 FK/参考姿态诊断，复核仿真关节状态与参考关键点位置。
- 给关节位置目标加入从当前 URDF 自动生成的 5% 内缩限位，防止高方差策略直接把 PD 目标压在机械停止位置。
- AMP 判别器两侧同时增加 base-frame 根线速度：agent 与 demonstration 维度保持严格一致，使平移成为风格判别的一部分。
- reward 回调到较温和配置：线速度/偏航权重 `2/1`，超额速度误差 `-0.25`，task/style 混合恢复 `0.4/0.6`，entropy `0.005`。
- 课程跟踪 std 改为 `0.4/0.3/0.25`。
- `alignmentfix_smoke_8192` 完成 499 轮并生成录像；后续正式任务启动后很快被下一组并行对照替代。
- 录像：`videos/t1_amp_alignmentfix_model_499_forward_1ms-step-0.mp4`。
- **结论**：关节、关键点和参考数据契约通过检查；后续出现的弱前进或风格不持久，不能再简单归因于关节顺序错位。

### 2026-08-24：课程与风格权重并行对照

调优 run：`2026-08-24_10-26-08_aligned_curriculum_style_formal1_8192`

- 课程边界从 `3000/9000` 延长到 `5000/12000`。
- 三阶段速度跟踪 std 全部固定为 `0.4`。
- style scale 从 `5.0` 提高到 `5.5`，task/style 混合保持 `0.4/0.6`。
- 完成 `50000` 轮并生成 `model_49999.pt`。
- 最后日志：episode length `500/500`，time-out `99.91%`，XY 速度误差约 `0.197 m/s`，yaw 误差约 `0.569 rad/s`，noise std 约 `1.61`。
- 录像：`videos/t1_aligned_tuned_model_4500_forward_1ms-step-0.mp4`、`t1_amp_aligned_tuned_model_25000_forward_1ms-step-0.mp4`。

基准 run：`2026-08-24_10-30-59_aligned_baseline_formal1_8192`

- 与调优 run 并行训练，保留课程边界 `3000/9000`、std `0.4/0.3/0.25`、style scale `5.0`。
- 因肉眼风格明显较差，按要求在约 `26037` 轮停止；最后 checkpoint 为 `model_26000.pt`。
- 最后日志：episode length `500/500`，time-out `99.70%`，XY 速度误差约 `0.184 m/s`，yaw 误差约 `0.715 rad/s`。
- 录像：`videos/t1_aligned_baseline_model_4500_forward_1ms-step-0.mp4`、`t1_amp_aligned_baseline_model_7000_forward_1ms-step-0.mp4`、`t1_amp_aligned_baseline_model_25000_forward_1ms-step-0.mp4`。
- **结论**：较低的 XY 误差或较高的训练期 style reward 不等价于肉眼模特步更好；必须固定指令、关闭评估扰动后比较视频。

### 2026-08-25：参考速度范围、0809 URDF 与指令匹配参考

共同配置调整：

- 根据参考动作实测速率收窄课程范围，而不是沿用过大的通用人形机器人指令：
  - 第一阶段：`vx=[-0.2, 0.4]`、`vy=[-0.25, 0.2]`、yaw `[-0.85, 0.8]`；
  - 第二阶段：`vx=[-0.55, 0.6]`、`vy=[-0.45, 0.35]`、yaw `[-1.15, 1.2]`；
  - 第三阶段：`vx=[-0.7, 0.8]`、`vy=[-0.6, 0.5]`、yaw `[-2.05, 1.9]`。
- 三阶段 std 均为 `0.4`。
- 删除 `feet_air_time`，避免长单支撑奖励与参考步频冲突。
- 删除 `arm_energy`，避免手臂摆动被额外压制。
- style scale 先提高到 `6.0`，task/style 混合为 `0.4/0.6`；2 轮 smoke test 通过。

0809 对照 run：`2026-08-25_11-26-09_0809_samecfg_formal1_8192`

- 唯一主要资产变化是改用 `Men_T1_0809_new_shoe` 四点足底碰撞 URDF，其余训练条件沿用上述配置。
- 截至本文记录时运行到约 `30524/50000` 轮，最新 checkpoint 至少为 `model_30000.pt`。
- `30477` 轮日志：episode length `500/500`，time-out `99.58%`，XY 速度误差约 `0.210 m/s`，yaw 误差约 `0.347 rad/s`，noise std 约 `0.121`。

前进指令匹配 run：`2026-08-25_11-45-04_forward_command_matched_formal1_8192`

- AMP 不再从 18 个动作中无条件抽样，而是根据当前速度指令选择对应类别：`forward`、`backward`、`lateral_left/right`、`rotate_left/right`。
- 类别内部仍按帧数加权，因此所有被选中类别中的参考帧近似等概率。
- 前进专用第一阶段延长：课程边界 `8000/16000`；第一阶段只采样 `vx=[0.2, 0.6] m/s`，`vy=0`、yaw `=0`，并禁用站立指令。
- 第二、三阶段再逐步引入后退、横移和旋转。
- 判别器同时观察速度指令，当前/参考两侧均为 `4 × 83`，避免当前指令与参考类别互相矛盾。
- 该 run 的落盘配置仍是 **0722 URDF**，style scale `6.0`、task/style `0.4/0.6`。
- 截至本文记录时运行到约 `30089/50000` 轮，`30055` 轮日志：episode length `500/500`，time-out `99.75%`，XY 速度误差约 `0.263 m/s`，yaw 误差约 `0.378 rad/s`，noise std 约 `0.088`。
- 录像：`t1_forward_command_matched_model_4500_training_range-step-0.mp4`、`model_20000_training_range-step-0.mp4`、`model_26000_training_range-step-0.mp4`、`model_26000_forward_06-step-0.mp4`、`model_26000_forward_075-step-0.mp4`。
- **现象**：中后期能看到一至数步模特步特征，但持续性不足，曾表现为原地踏步、踉跄或摔倒。说明指令匹配有效改善了风格方向，但没有单独解决长时稳定性。

### 2026-08-26：降低风格强度、移除 Actor 关键点观测

run：`2026-08-26_14-30-07_forward_style05_scale5_formal1_8192`

- 根据前一轮“风格存在但不稳定”的结果，将 style scale 从 `6.0` 降到 `5.0`，task/style 改为 `0.5/0.5`。
- 其余保持指令匹配与前进专用课程不变。
- 该进程启动时仍使用 **0722 URDF**，Actor 仍包含关键刚体位置。
- 截至本文记录时运行到约 `2988/50000` 轮；`2948` 轮 XY 速度误差约 `0.134 m/s`、time-out `98.80%`。尚处第一阶段，不能据此判断完整课程最终效果。

run：`2026-08-26_16-17-17_forward_no_actor_keybody_style05_scale5_formal1_8192`

- 在上一 run 基础上，只从 Actor 注释掉 6 个关键刚体的基座系位置历史。
- Actor 外部观测由 `525` 降为 `435` 维；Critic `540` 维及 AMP 判别器关键点观测保持不变。
- 该进程启动时仍使用 **0722 URDF**，与含关键点版本并行训练。
- 截至本文记录时运行到约 `1350/50000` 轮；`1306` 轮 XY 速度误差约 `0.158 m/s`、time-out `98.34%`。仍处第一阶段，暂不能形成消融结论。

播放链路修复：

- 早期部分 AMP 录像直接使用训练配置，可能保留随机参考帧 reset、域随机化、观测噪声、执行器延迟和 `2.5–5 s` 随机推力。
- 这些因素会造成录像中的突然跳变、重置和抖动，不能全部归因于策略训练失败。
- `play.py --nominal_eval` 现会关闭训练扰动、从默认状态复位并关闭延迟；固定前进指令还会禁止课程在 reset 时覆盖指令。
- 当前较可信的长时录像是 `videos/t1_forward_command_matched_model_26000_forward_06_nominal_30s-step-0.mp4`。

### 2026-08-28：Estimator 与 Critic 特权观测补齐

- Estimator 使用 435 维可部署 Actor 历史观测，监督输出为 8 维：基座高度 1、基座系线速度 3、双脚接触概率 2、双脚世界高度 2；输出与原始观测拼接后，Actor MLP 输入为 443 维。
- 梯度检查确认：PPO Actor 梯度不会经估计值反传至 Estimator；Estimator 仅由仿真真值 MSE 监督更新。监督损失不会误更新 Actor MLP。
- 原实现的 Critic 只有真值基座线速度和关键刚体位置，没有显式包含基座高度、双脚接触与双脚世界高度，未完整复现 `wbt_est` 的非对称 Critic。
- 新增仅在 `--use_estimator` 时启用的 `state_est_critic` 特权组，补入缺少的 5 个真值量。原 Critic 的基座线速度 3 维加上该组后覆盖完整 8 维目标；5 帧历史使 Critic 从 540 维增至 565 维。
- 特权组不加入默认 T1 配置，因此无 Estimator 的旧 checkpoint 仍保持 540 维 Critic 合同，不受影响；AMP 判别器仍为 332 维，未混入 Estimator 输出或无参考侧配对的标签。
- 首次直接修改公共 Critic 的 smoke 在检查到旧 checkpoint 兼容性风险后立即停止并删除；随后改为条件启用并重新完成 100 轮 smoke。
- 条件版本 smoke 第 99 轮：value loss `0.0183`，Estimator 总损失 `0.0872`（高度 `0.0003`、线速度 `0.0141`、接触 `0.0728`、足高 `0.0001`），AMP style reward `2.63`，action-rate `-0.0247`，关节限位 `-0.0021`；无 NaN、OOM 或维度错误。smoke 日志和 run 已删除。
- 旧的无完整 Critic 特权正式训练停止于约 `908/50000`，最后完整 checkpoint 为 `model_500.pt`。
- 新正式训练使用同一任务流水线启动：`forward_no_keybody_estimator_criticpriv8_noheading_linrew3_style06_lerp04_formal2_8192`，`8192 env × 50000`，物理 GPU 2。

### 2026-08-28：关闭随机 heading 与直行约束

run：`2026-08-28_15-27-58_forward_no_keybody_estimator_criticpriv8_noheading_yawpen2_headinghold05_formal1_8192`

- 保持 `heading_command=False`，不使用 Isaac Lab 默认的随机世界航向机制。
- 线速度跟踪权重提高到 `3.0`，偏航速度跟踪为 `1.0`；两项 std 当时均为 `0.5`。
- 对 `|yaw_command| < 0.1 rad/s` 的环境增加实际 yaw rate 平方惩罚，权重 `-2.0`。
- 同一零转向子集增加相对 episode 初始朝向的指数保持奖励，权重 `0.5`、`std=0.15 rad`。
- Estimator 和完整 8 维 Critic 特权观测均启用；默认资产为 0809 四点足底 URDF。
- 完成 `50000` 轮并生成 `model_49999.pt`。固定名义条件录像显示：`model_15000` 风格弱，`model_28500` 已出现较明显模特步，`model_49000/49999` 后期反而丢失部分风格。
- **结论**：Estimator 可补偿未直接观测的状态，但它不会自动提供“目标朝向”；单靠状态估计不能解决侧向弯行。后期继续训练还存在风格遗忘，因此选用 `model_28500.pt` 作为后续短程微调起点，而不是从头再训 50000 轮。

### 2026-08-30：从 model_28500 进行两组航向微调

两组实验均从上一 run 的 `model_28500.pt` resume，实际再训练 `10000` 轮，结束于 `model_38499.pt`：

- entropy `0.005 → 0.001`，减少后期动作探索。
- 判别器学习率 `1e-5 → 2e-6`，降低判别器过快饱和的风险。
- 启用按 AMP style 自动保存最佳 checkpoint。
- 固定 `50%` 环境采样直行指令：`vy=0`、`yaw_rate=0`，前进速度范围为 `0.2–0.8 m/s`；另外 `50%` 保留完整方向训练。
- Actor 与 Critic 均增加 2 维航向误差编码；5 帧历史使外部观测分别由 `435/540` 增至 `445/550` 维。Estimator 仍输出 8 维并拼入 Actor，因此 Actor MLP 实际输入为 `453` 维；Critic 加 25 维历史特权组后实际输入为 `575` 维。

零 yaw 锁定版：`2026-08-30_23-38-09_forward_estimator_headingobs_straight50_ampstable_finetune10k_8192`

- 航向观测为 `sin/cos(initial_yaw-current_yaw)`，只在零转向指令下提供有效误差；转向时输出中性值 `[0, 1]`。
- 奖励仍是只对零转向指令生效的初始航向保持奖励。
- 最终 `model_38499`：episode length `500/500`、time-out `99.48%`、XY 速度误差 `0.161 m/s`、yaw 速度误差 `0.262 rad/s`、style reward `0.315`、Estimator loss `0.0150`。
- `model_best_style.pt` 对应 iteration `32700`。

积分航向版：`2026-08-30_23-47-53_forward_estimator_integratedheading_straight50_ampstable_finetune10k_8192`

- reset 时目标航向取当前 yaw，之后每步按 `yaw_rate_command × dt` 积分；Actor 与 Critic观察 `sin/cos(target_yaw-current_yaw)`。
- 航向奖励改为 `exp(-(yaw-heading_target)^2 / 0.15^2)`，对零转向和非零转向指令都生效。配置字段名仍叫 `zero_command_heading_hold_exp`，但字段名不代表实际门控逻辑。
- 最终 `model_38499`：episode length `500/500`、time-out `99.46%`、XY 速度误差 `0.177 m/s`、yaw 速度误差 `0.249 rad/s`、style reward `0.360`、Estimator loss `0.0153`。
- `model_best_style.pt` 对应 iteration `33900`。
- 统一用 0809 URDF、`0.8 m/s`、单环境和 `--nominal_eval` 录制 30 秒视频：`t1_headingobs_zero_lock_model38499_forward08_nominal_30s-step-0.mp4` 与 `t1_integrated_heading_model38499_forward08_nominal_30s-step-0.mp4`。
- **肉眼结论**：两版比旧策略走得更直，但仍有踉跄；较紧的航向核 `std=0.15 rad` 和较强 `yaw rate L2=-2.0` 可能促使策略通过频繁横向落脚快速纠偏。下一次优先做小幅放松（例如航向 `std=0.25–0.30`），不要同时改 AMP 风格参数。

### 2026-09-01：Estimator 缩减为 4D

- T1 与 BSRL 共用的 Estimator 合同从 8D 改为 4D，只预测 `base_height(1) + base_lin_vel_b(3)`；足接触概率和双脚世界高度不再进入监督标签、Actor 或 Critic 特权组。
- T1 Policy 历史仍为 `445` 维，Actor MLP 输入由 `453` 降为 `449`；Critic 主观测仍为 `550` 维，额外特权历史由 `25` 降为 `5`，最终输入由 `575` 降为 `555`。
- 足接触和足高观测函数以注释/可复用函数形式保留，未删除，以便未来显式相位任务复用；AMP 当前不做逐帧相位跟踪，因此不再让离散接触标签主导 Estimator loss。
- 纯模型梯度与维度检查已经通过，BSRL 4D Isaac 正式训练也验证了公共实现。T1 4D 正式 run `2026-09-01_11-28-06_estimator4d_formal1_8192` 已从头启动，配置为 `8192 env × 30000` 轮，tmux 为 `t1_estimator4d_formal1`，使用物理 GPU 2。
- T1 环境运行时再次确认 Policy `445`、Estimator 标签/输出 `4`、Actor MLP `449`、Critic `550+5=555`、AMP 判别器 `332` 维；`model_0.pt` 已生成并进入 PPO。历史 T1 8D checkpoint 与当前 4D 网络结构不直接兼容。
- 后续训练已同时关闭 `zero_command_yaw_rate_l2` 和 `zero_command_heading_hold_exp`：前者对直行改善不明显且可能过度限制自然纠偏，后者的积分目标航向在实际表现中不够准确。两项定义均以注释形式保留；正在运行的 `estimator4d_formal1_8192` 已完成环境实例化，仍按启动时的 `-2.0` 与 `+0.5/std=0.15` 配置训练，不受源码修改影响。

## 当前参考数据合同

- 目录：`data/MotionData/t1_25dof/amp/walk_and_run_amp_ready`。
- 共 18 个 PKL、`14463` 帧：前进 3 个、后退 3 个、左横移 3 个、右横移 3 个、左旋转 3 个、右旋转 3 个。
- 每个文件必须包含与 `T1_JOINT_NAMES` 完全一致的 25 关节顺序，以及固定的 6 个关键刚体顺序：左右脚踝 pitch、左右 wrist yaw、左右 shoulder roll。
- 加载时检查 `root_pos/root_rot/dof_pos/key_body_pos` shape、有限值和单位四元数。
- 非指令匹配任务按总帧数采样全部动作；指令匹配任务先按命令选类别，再在类别内按帧数采样。
- `random_fetch=True` 只控制判别器随机抽取参考片段，不会直接瞬移真实机器人；录像跳变来自 episode 终止后的 reset 或仍开启的训练扰动。

## 当前默认训练配置（2026-09-01）

### 资产与控制

- 默认 T1 资产为 `Men_T1_0809_new_shoe/urdf/men_t1_path_25dofs_collision.urdf`，即 0809 新鞋、普通足底碰撞版本，不使用四点球形足底碰撞。
- `men_t1_path_25dofs_collision_feet_points.urdf` 等其他原始 0809 URDF 仍保存在同一目录，仅供显式对照，不再作为默认资产。
- 只保留一个 T1 注册任务：`LeggedLab-Isaac-AMP-T1-v0`。原 AlignedBaseline、0809、ForwardCommandMatched 和 Play task id 均已删除；播放统一用同一个任务配合 `play.py --nominal_eval`。
- 平地、`8192 env`、episode `10 s`；物理频率 `200 Hz`（`dt=0.005 s`），策略频率 `50 Hz`（decimation `4`）。
- 25 维关节位置动作，scale `0.25`；目标按 URDF 硬限位向内收缩 5%。
- 执行器延迟 `0–4` 个物理步，即 `0–20 ms`。

### Actor、Critic 与 AMP

- Actor 当前使用 5 帧带噪历史：基座角速度 `3`、根姿态 6D `6`、速度指令 `3`、关节位置 `25`、关节速度 `25`、上一动作 `25`、航向误差编码 `2`，合计 `89 × 5 = 445` 维。
- 6 个关键刚体位置已从 Actor 移除；Critic 和判别器仍保留它们。
- Critic 外部观测为 `550` 维；启用 Estimator 时再拼接 5 维基座高度历史特权状态，Critic MLP 输入为 `555` 维。
- Estimator 默认启用，输出的 4 维状态在 Actor 内部拼接，使 Actor MLP 输入为 `449` 维。
- 指令匹配判别器为 4 帧、每帧 `83` 维；其中包含 base-frame 根线速度和速度指令。
- 网络为 `[512, 256, 128]` ELU，PPO learning rate `1e-4` adaptive、clip `0.2`、gamma `0.99`、lambda `0.95`；最近微调使用 entropy `0.001`。
- T1 后续新训练使用 style scale `5.0`、task/style `0.4/0.6`，与 G1/BSRL 对齐；判别器学习率 `2e-6`。当前运行的 `estimator4d_formal1_8192` 启动时已加载 `6.0`，不受本次源码修改影响。

### 奖励、课程与随机化

- 线速度/偏航跟踪权重 `3/1`，当前 std 分别为 `0.4/0.5`。
- 50% 环境固定采样 `vy=0`、`yaw_rate=0` 的直行指令，`vx=0.2–0.8 m/s`。
- 当前有零转向 yaw rate L2 `-2.0`；积分航向跟踪奖励权重 `0.5`、`std=0.15 rad`，该奖励实际对所有 yaw 指令生效。
- XY 超额速度误差权重 `-0.25`，阈值 `0.25 m/s`，最大 excess `2.0 m/s`。
- 主要惩罚：平地姿态 `-1`、非偏航角速度 `-0.05`、力矩 `-2e-6`、关节加速度 `-1e-7`、action rate `-0.005`、关节限位 `-1`、足滑 `-0.1`、非足部接触 `-0.1`、终止 `-200`。
- `feet_air_time` 与 `arm_energy` 已删除。
- 材料摩擦 `0.5–1.5`、恢复系数 `0–0.2`；默认关节位置 `±0.05 rad`；base COM 为 x `±0.025 m`、y/z `±0.05 m`；Kp/Kd 和 armature `×0.8–1.2`；质量 `×0.9–1.1`；关节摩擦 `0–0.05`。
- reset 时施加 base 外力 `±15 N`、力矩 `±4 Nm`；每 `2.5–5 s` 注入六维根速度扰动，各分量范围 `±0.5`，z 为 `±0.2`。
- 训练从随机参考帧复位；终止条件为 10 秒超时、pelvis 接触、根高度低于 `0.5 m` 或倾角超过 `60°`。

### 当前各模块观测表

| 模块 | 当前输入 | 历史与维度 | 备注 |
|---|---|---:|---|
| Policy / Estimator 输入 | base 角速度 `3`、root 旋转 6D `6`、速度指令 `3`、25 关节位置 `25`、关节速度 `25`、上一动作 `25`、积分航向误差 sin/cos `2` | `89 × 5 = 445` | 有训练噪声；Actor 不含 6 个关键刚体位置，也不直接观察 base 线速度 |
| Estimator 输出 | base 高度 `1`、base-frame 线速度 `3` | 当前帧 `4` | MLP `[256,128]`；监督损失系数 `1.0`，Actor 梯度不反传到 Estimator |
| Actor MLP | Policy 历史 `445` + Estimator 预测 `4` | `449` | `[512,256,128]` ELU |
| Critic 主观测 | Policy 的无噪声版本 + base 线速度 `3` + 6 个关键刚体相对 base 位置 `18` | `110 × 5 = 550` | 不使用 Estimator 预测值 |
| Critic Estimator 特权 | base 高度 `1` | `1 × 5 = 5` | base 线速度真值已在 Critic 主观测中 |
| Critic 奖励特权 | 双足接触 `2`、双足世界系 XY 速度 `4`、非允许接触计数 `1`、pelvis 接触标志 `1` | `8 × 5 = 40` | 精确覆盖 feet slide、undesired contact 和 base-contact termination；Critic 总输入 `595` |
| AMP agent 判别器 | root 旋转 6D `6`、base 角速度 `3`、关节位置/速度 `25+25`、6 个关键刚体位置 `18`、base 线速度 `3`、速度指令 `3` | `83 × 4 = 332` | 参考动作类别会按当前速度指令匹配 |
| AMP demo 判别器 | 对应的参考 root 旋转/角速度、关节状态、关键点、root 线速度及参考速度指令 | `83 × 4 = 332` | 与 agent 侧严格同维同序 |

### 当前奖励表

| 奖励项 | 权重 | 当前含义/参数 |
|---|---:|---|
| `track_lin_vel_xy_exp` | `+3.0` | base-frame XY 速度跟踪；课程三阶段 std 均为 `0.4` |
| `zero_yaw_initial_heading_lin_vel_xy_exp` | `+1.0` | 仅精确零 yaw 指令启用；在 episode 初始 yaw 坐标系跟踪 XY 速度，std `0.4`，防止转动身体后仍获得完整前进奖励 |
| `track_ang_vel_z_exp` | `+1.0` | base-frame yaw-rate 跟踪，std `0.5` |
<!-- | `zero_command_yaw_rate_l2` | 关闭（原 `-2.0`） | 对直行改善不明显且可能限制自然纠偏；当前正在运行的 4D run 仍保留启动时配置 |
| `zero_command_heading_hold_exp` | 关闭（原 `+0.5`） | 积分目标航向精度不足；当前正在运行的 4D run 仍保留启动时配置 | -->
| `lin_vel_xy_error_excess_l2` | `-0.25` | XY 误差超过 `0.25 m/s` 后平方惩罚，excess 上限 `2.0 m/s` |
| `flat_orientation_l2` / `ang_vel_xy_l2` | `-1.0 / -0.05` | 抑制机身倾斜与 roll/pitch 角速度 |
| `dof_torques_l2` / `dof_acc_l2` | `-2e-6 / -1e-7` | 力矩和关节加速度正则 |
| `action_rate_l2` / `dof_pos_limits` | `-0.005 / -1.0` | 动作平滑与关节限位惩罚 |
| `feet_slide` / `undesired_contacts` | `-0.1 / -0.1` | 足底滑动与非允许刚体接触惩罚 |
| `termination_penalty` | `-200` | 非超时终止惩罚 |
| 当前关闭/删除项 | `0` 或不存在 | `lin_vel_z_l2=0`；无 `feet_air_time`、无 `arm_energy` |
| AMP style reward | scale `5.0` | LSGAN；最终 task/style 混合比例 `0.4/0.6`；当前在训 4D run 仍为启动时的 `6.0` |

### 当前训练参数表

| 类别 | 当前值 |
|---|---|
| 环境与控制 | `8192 env`；物理 `200 Hz`（`dt=0.005`）；策略 `50 Hz`（decimation `4`）；episode `10 s`；25 维位置动作，scale `0.25`，目标裁到 URDF 限位内侧 5%；执行器延迟 `0–4` 物理步 |
| 指令与课程 | 每 `10 s` 重采样，无站立指令，`heading_command=False`；固定 `50%` 为 `vy=wz=0` 的直行样本；其余样本只激活后退/左横移/右横移/左转/右转中的一个分量，避免无对应参考的复合指令；第 `0/8000/16000` 轮依次使用 `vx=(0.2,0.6)/(-0.55,0.6)/(-0.7,0.8)`、`vy=0/(-0.45,0.35)/(-0.6,0.5)`、yaw `=0/(-1.15,1.2)/(-2.05,1.9)`，实际采样再裁到对应动作类别的实测速率覆盖范围 |
| 参考动作 | 18 个 PKL；从 PKL root 位姿自动计算每个 clip 的 body-frame 平移速度或 yaw rate，先按命令方向选择类别，再在类别内选择速度最接近的 clip；AMP 连续 `4` 帧 |
| Rollout 与保存 | 每环境每轮 `24` 步；默认 `30000` 轮；每 `500` 轮保存；日志根 `men_t1_amp` |
| PPO | epochs `5`、mini-batches `16`、LR `1e-4 adaptive`、clip `0.2`、gamma `0.99`、lambda `0.95`、desired KL `0.01`、max grad norm `1.0`、entropy `0.001` |
| 网络 | Actor/Critic `[512,256,128]` ELU；初始动作 std `1.0`，采用 log-std 参数化；Actor 增广观测归一化关闭，Estimator 输入使用运行均值/方差归一化；Estimator `[256,128]` ELU |
| AMP | 判别器 `[1024,512]` ELU；LR `2e-6`；replay buffer `1000`；gradient penalty `10`；trunk/linear weight decay `1e-4/1e-2`；LSGAN；每 `100` 轮检查最佳 style checkpoint（warm-up `250` 轮） |
| 随机化 | 材料摩擦 `0.5–1.5`、质量 `×0.9–1.1`、Kp/Kd 和 armature `×0.8–1.2`、零位 `±0.05 rad`、COM x `±0.025 m`/y-z `±0.05 m`、关节摩擦 `0–0.05`；reset 外力 `±15 N`/力矩 `±4 Nm`；每 `2.5–5 s` 六维速度推扰 |
| Reset / 终止 | 从匹配的参考动作随机帧复位；10 秒超时、pelvis 接触、base 高度 `<0.5 m` 或倾角 `>60°` 终止 |

## Estimator 迁移状态（已完成训练验证）

- 已将 `wbt_est` 的监督状态估计结构移入 AMP；Estimator、标签和 Critic 特权组现在是主任务默认配置，不再为它注册或切换独立任务。
- Estimator 输入当前 445 维可部署 Actor 历史观测，MLP 为 `[256, 128]`，现输出 4 维：`base_height(1) + base_lin_vel_b(3)`。旧版足接触和足高函数仍保留，但不再进入 AMP Estimator 合同。
- Actor 内部输入为 `445 + 4 = 449` 维；Critic 输入为主观测 `550`、基座高度历史 `5` 和奖励相关接触特权 `40`，共 `595` 维。PPO Actor 梯度默认不进入 estimator，estimator 由仿真真值监督损失训练。
- 2026-09-01 补齐 Critic 的奖励特权：接触标志与足部真实 XY 速度对应 `feet_slide`，非允许接触计数对应 `undesired_contacts`，pelvis 接触对应终止惩罚。T1 没有 `feet_air_time` 奖励，因此未加入无用的足部计时观测。
- CPU 合成 rollout 已确认 Actor、Critic、AMP 判别器和 estimator 能在同一次 update 中正常更新；TorchScript/ONNX 导出通过。
- 旧 8D 版本已完成 Isaac 仿真 100 轮 smoke 与多次正式训练；两组 `model_38499` 的 Estimator 总损失约 `0.015`，无 NaN、维度错配或训练中断。这些数值只属于历史 8D checkpoint，不作为当前 4D 结果。
- 2026-09-01 已将 BSRL/T1 共用 Estimator 缩减为 4D；模型梯度、维度和 BSRL Isaac 正式训练启动均已验证。历史 8D checkpoint 与新结构不直接兼容，后续 4D 训练从头开始。

## 训练目录速查

| Run | URDF | 状态/最后记录 | 主要变量 |
|---|---|---:|---|
| `formal_run2_8192` | 0722 | 停止于 23505 | 初始迁移基准 |
| `rewardfix_formal1_8192` | 0722 | 停止于约 22225 | base-frame 速度、强 task reward、课程 |
| `alignmentfix_smoke_8192` | 0722 | 完成 499 | 数据/FK/限位/判别器线速度检查 |
| `aligned_curriculum_style_formal1_8192` | 0722 | 完成 49999 | 长课程、std 0.4、style 5.5 |
| `aligned_baseline_formal1_8192` | 0722 | 停止于约 26037 | 旧课程和 style 5.0 对照，肉眼风格差 |
| `0809_samecfg_formal1_8192` | 0809 | 完成 49999 | 0809 四点足底 URDF 对照 |
| `forward_command_matched_formal1_8192` | 0722 | 完成 49999 | 指令选参考类别、前进课程、style 6 |
| `forward_style05_scale5_formal1_8192` | 0722 | 停止于 32500 | task/style 0.5/0.5、style 5 |
| `forward_no_actor_keybody_style05_scale5_formal1_8192` | 0722 | 停止于 30500 | 上一项再移除 Actor 关键点 |
| `forward_no_keybody_estimator_criticpriv8_noheading_linrew3_style06_lerp04_formal2_8192` | 0809 | 停止于 44000 | Estimator、完整 Critic 特权、无 heading |
| `forward_no_keybody_estimator_criticpriv8_noheading_yawpen2_headinghold05_formal1_8192` | 0809 | 完成 49999 | 零 yaw 惩罚与初始航向保持；28500 风格较好 |
| `forward_estimator_headingobs_straight50_ampstable_finetune10k_8192` | 0809 | 完成 38499 | 从 28500 微调；零 yaw 航向观测、50% 直行 |
| `forward_estimator_integratedheading_straight50_ampstable_finetune10k_8192` | 0809 | 完成 38499 | 从 28500 微调；积分航向观测/奖励、50% 直行 |
| `rewardpriv_estimator4d_formal1_8192` | 0809 | 新航向速度奖励加入前停止 | 4D Estimator；Critic 新增足接触/足速、非法接触和 pelvis 接触特权，总输入 595D |
| `rewardpriv_initialheadingvel_estimator4d_formal1_8192` | 0809 | 停止于 218 | 增加零 yaw 时的 episode 初始航向坐标系 XY 速度奖励 `+1.0`；reset 仍在新指令采样前选参考帧，停止不用 |
| `resetmatch_rewardpriv_initialheadingvel_estimator4d_formal1_8192` | 0809 | 2026-09-01 启动，30000 轮 | 修正顺序：先采样新指令，再按类别选择参考 clip/随机帧并 reset；其余配置与上一项一致 |
| `speedmatched_singleaxis_rewardpriv_estimator4d_formal1_8192` | 0809 | 2026-09-01 启动，30000 轮 | 保留上一 run 并行对照；新增同类别内最近速度 clip 匹配，后两阶段改为 50% 直行 + 50% 单轴非前进类别，不再生成无参考的复合指令 |
| `speedmatched_no_policy_heading_estimator4d_formal1_8192` | 0809 | 2026-09-03 启动，30000 轮 | 基于速度匹配版本，仅从 Policy 移除 2D `integrated_heading_error`；Critic 保留该特权量，4D Estimator 和其余配置不变 |
| `speedmatched_no_policy_heading_estimator4d_feetpoints_formal1_8192` | 0809 四点足底 | 2026-09-03 启动，30000 轮 | 与上一 run 并行的接触拓扑单变量对照；Policy 同为 435D、Estimator 同为 4D，其余训练参数不变。启动时临时加载 `men_t1_path_25dofs_collision_feet_points.urdf`，环境创建后源码默认已恢复为普通足底 URDF，不影响现有进程 |
| `resetmatch_rewardpriv_initialheadingvel_estimator4d_feetpoints_formal1_8192` | 0809 四点足底 | 2026-09-03 启动，30000 轮 | 以效果较好的 `2026-09-01_16-15-31_resetmatch...` 为严格基线，仅替换足底接触拓扑；类别内按帧权重随机选参考 clip，不做速度最近邻；Policy 恢复 2D integrated heading，输入 445D |

> 注意：已经启动的进程会继续使用各自日志 YAML 中记录的资产，不会因当前源码默认切换为 0809 而热更新。之后新启动的 T1 训练才默认使用 0809。

### 2026-09-03：T1 基线选择与四点足底对照

- 后续 T1 调参以 `2026-09-01_16-15-31_resetmatch_rewardpriv_initialheadingvel_estimator4d_formal1_8192` 为 baseline，不再以 speed-matched run 为 baseline。
- baseline 的参考采样为“先按指令匹配动作类别，再在类别内按帧数权重随机选 clip”，不匹配指令速度大小，也不强制非直行指令为单轴。
- 四点足底对照 run 的冻结 `env.yaml` 与 baseline 相比，除资产路径外只多出用于显式记录旧语义的 `single_axis_commands: false`、`motion_speed_animation: null` 和 `match_command_speed: false`；训练网络、奖励和其余环境参数一致。
- 四点足底环境创建完成后，源码资产默认已恢复为普通 `men_t1_path_25dofs_collision.urdf`；已启动进程仍使用冻结的四点 URDF。

历史 T1 run 已统一移动到 `logs/rsl_rl/men_t1_amp/`。run 之间的 URDF、观测、奖励和训练参数差异以本表、时间线及各 run 自带的 `params/*.yaml` 区分，不再通过多个 experiment 目录或 task id 区分。

## 已确认问题与跨实验结论

- **已修正：参考采样不均衡。** 由每文件等权改为按帧数加权，避免短转向片段过采样。
- **已修正：reset 动作与新指令错配（2026-09-01）。** 原顺序先从全数据集随机参考帧复位、再采样新 episode 指令，导致阶段 1 约 `57.8%` reset 来自非前进动作。现仅对启用 command-conditioned animation 的 T1 改为“更新课程 → 采样新指令 → 按指令选择动作类别/clip/随机帧 → `reset_from_ref`”；reset 后只同步 episode 初始 yaw，不重复采样指令。BSRL 保持原 reset 路径。512 env × 100 轮 smoke 已通过且日志已清理。
- **已修正：类别内速度错配与复合指令无参考（2026-09-01）。** 每个 PKL 的代表速度由 root 位姿差分后转换到 body frame 自动计算，参考 clip 改为同方向内按速度最近邻选择；指令采样范围限制在该方向现有动作的速度覆盖内。阶段 2/3 的非直行部分只采一个运动轴，固定约 50% 前进，其余五类各约 10%。阶段 3 的 512 环境审计得到 `253/60/42/52/52/53`（前进/后退/左横移/右横移/左转/右转），没有复合指令；同时修复了高级索引 `uniform_()` 未写回导致“直行”子集混入负向 vx 的旧问题。
- **已修正：速度奖励坐标系。** 当前指令和速度奖励均使用 base frame。
- **已修正：判别器缺少平移信息。** 当前 T1 判别器包含根线速度；指令匹配版本进一步包含速度指令。
- **已修正：动作目标撞限位。** 关节目标使用当前 URDF 生成的 5% 内缩限位。
- **已修正：录像配置会误导判断。** 正式评价应使用 `--nominal_eval`，关闭训练随机化、推力、噪声、延迟和参考帧 reset。
- **有效但未完全解决：按指令选择参考类别。** 已能看到前进模特步特征，但连续性和抗踉跄能力仍不足。
- **训练指标不能代替视觉和部署评价。** 多个 run 都有接近 100% time-out 和较低 XY 误差，但步态风格、关节余量与持续稳定性差异很大。
- **判别器容易饱和。** 多个后期 run 的 agent/demo score 接近 `-1/+1`、disc loss 很小；此时提高 style scale 不一定提供更有效的学习方向，反而可能加重 task/style 冲突。
- **前进课程后期会重新混入全方向。** `16000` 轮之后训练覆盖后退、横移和旋转；如果目标是长期保持模特步前进，应分别评估第一阶段 checkpoint 与全课程 checkpoint，不能只看最终轮次。
- **0722 与 0809 必须按 run 的 YAML 区分。** 当前默认虽已是 0809，但 8 月 26 日两个前进新 run 均在切换默认资产前启动，实际仍是 0722。
- **后续只改参数时不再新增任务。** 沿用同一 task，执行“改代码 → 新 run → 再改代码 → 新 run”；只有观测合同或需要长期兼容的结构变化才考虑新 task id。
- **2026-08-31 起取消“结构变化就注册新任务”的旧做法。** T1 始终使用唯一 task id 和 `men_t1_amp` 日志根目录；即使调整观测合同，也优先依靠不同 `run_name`、checkpoint 与 YAML 留档区分。
- **Policy 真机可用性是硬约束。** 真机无法直接获取、依赖仿真真值或无法可靠估计的观测，未经用户明确允许不得加入 Policy；此类信息默认只允许用于 Critic 特权观测、Estimator 监督标签或训练奖励，并需保证部署输入合同不依赖它。
- **航向观测语义必须随 checkpoint 保存并在回放时恢复。** 零 yaw 锁定版和积分航向版维度完全相同，但输入含义不同；直接用当前默认配置播放旧版不会报 shape 错，却会让策略收到错误语义。`play.py --heading_observation_mode zero_lock|integrated` 用于显式匹配。
- **航向变直不等于步态更稳定。** 最新两组航向微调均改善直行，但存在踉跄；`std=0.15 rad` 的指数奖励在约 `17.2°` 误差时只剩最大值的约 `1.8%`，可能诱导急促纠偏。

## 下一步建议

1. 下一轮优先从表现较好的 checkpoint 短程微调，先只将积分航向奖励 `std: 0.15 → 0.25–0.30`；保持 AMP scale、混合比例和其他奖励不变，验证踉跄是否减少。
2. 若仅放宽 std 仍频繁横向补步，再单独将零 yaw rate L2 从 `-2.0` 降到约 `-1.2`，不要两项同时改动，以便归因。
3. 对比固定使用 `vx=0.8 m/s`、默认起始状态、匹配 checkpoint 的航向观测模式和 `--nominal_eval`，同时记录航向误差、横向位移、足宽与首次失稳时间。
4. 使用默认 0809 URDF 继续实验时，不要 resume 0722 checkpoint；所有结构和语义变化均以 run 内 YAML 为准。
