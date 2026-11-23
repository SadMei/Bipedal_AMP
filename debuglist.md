1. 站不起来：位置误差过大/elmo驱动的力矩限太小/PD太小/质量不对/站立姿态不支持/关节摩擦太大/静止帧太少/速度限幅小
2. 抖动：D过大/解算误差(knee ankle)
3. 滤波过猛，统一100/200hz即可
<!-- 4. 嵌入式策略端存在问题 可能 -->
<!-- 5. 整理仓库，合并 rosbag 分析工具 以及 闭链计算工具 -->
6. sim2sim 闭链参数同步，踝部误差域随机化，
7. 力矩时滞-4ms 原因分析
<!-- 7. 动力学事件目前？ -->
<!-- 8. MIT 模式？ -->
# BSRL AMP Debug List

按实验发生时间记录“改动 + 效果 + 结论”。实验条目只保留在时间线中，跨实验得到的通用经验汇总在末尾。

## 实验时间线

### 2026-07-22

- **第一版成功训练**：使用旧 URDF、原生 `ImplicitActuator`、默认膝角 `-0.6`、指令范围 `x/y=±1.0 m/s`、yaw `±0.4 rad/s`，`joint_pos_penalty=-1.0`。**效果**：机器人可以稳定行走，作为后续实验的最早基准。

### 2026-07-30

- **新坐标模型**：统一新 URDF/USD 的关节坐标轴、限位以及 GMR/AMP 关节符号。**效果**：模型与参考动作坐标约定一致，但策略相较第一版明显更保守。

### 2026-08-02

- **贴地默认姿态**：采用 GMR 行走第一帧整理出的对称姿态，默认根高设为 `0.8665 m`、hip pitch `-0.2 rad`、knee pitch `+0.4 rad`、ankle pitch `-0.2 rad`。**效果**：默认站姿下脚底接触正常。
- **恢复第一版动力学与事件**：恢复第一版 PD、armature、限位及随机事件；保留 delayed actuator 框架，但延迟设为 `0`。**效果**：用于隔离新 URDF/坐标变化，避免把动力学差异混入判断。
- **新 URDF + `joint_pos_penalty=-1.0`**：保持上述动力学和贴地姿态训练。**效果**：可以稳定移动，但关节幅度偏小；末段风格奖励约 `0.069`、腾空奖励约 `0.076`。
- **降低姿态惩罚至 `-0.3`**：只将 `joint_pos_penalty` 从 `-1.0` 降到 `-0.3`。**效果**：动作幅度明显增大，末段风格奖励约 `0.080`、腾空奖励约 `0.093`、平均 episode 长度约 `979/1000`，mixed 播放稳定。

### 2026-08-03

- **加入 `0～4` 步延迟和高速指令**：执行器每次 reset 随机采样 `0～4` 个物理步延迟；速度范围扩大为 `vx=±2.0 m/s`、`vy=±1.2 m/s`、yaw `±0.6 rad/s`，其余保持 `joint_pos_penalty=-0.3` 版本不变。**效果**：完成 10000 轮，平均 episode 约 `976/1000`、最终 XY 速度误差约 `0.71 m/s`；能执行高速指令，但小前向指令响应偏弱。由于延迟和指令范围同时改变，无法单独判断各自影响。

### 2026-08-04

- **加入机身 COM 和 PD 增益随机化**：在上一版基础上增加 `base_link` COM 偏移（x `±0.025 m`、y/z `±0.05 m`）以及全关节 Kp/Kd `×0.8～1.2`。**效果**：平均 episode 约 `953/1000`，但 XY 速度误差恶化到约 `2.26 m/s`，线速度跟踪奖励由约 `2.12` 降至 `0.29`；默认姿态播放 6 秒净位移仅约 `5 mm`，策略退化为站立和转向。
- **去除全关节默认姿态惩罚**：仅将 `joint_pos_penalty=-0.3` 改为 `0.0`，保留 hip yaw/roll 偏离惩罚、关节限位、AMP 风格和动作平滑约束。**效果**：XY 速度误差恢复到约 `0.84 m/s`，线速度跟踪奖励约 `1.91`、腾空奖励约 `0.114`；`vx=0.6 m/s` 时基本站立，`vx=1.0 m/s` 后才以较大步幅启动。**结论**：取消全关节姿态惩罚释放了动作幅度，但形成约 `0.6～1.0 m/s` 的低速启动死区。
- **mixed 播放指令时序修复**：原 `play.py` 在取得首帧观测后才写入 mixed 指令，并可能被 CommandManager 的周期重采样短暂覆盖。现改为环境创建前配置首段指令，关闭 mixed 模式周期重采样，并在下一帧观测生成前切换后续指令。**效果**：mixed 指令与 actor 观测严格同步；该 bug 只影响切换时序，不能解释机器人连续数秒后退。
- **完整域随机化 + `std=0.3`**：保持高速指令、`joint_pos_penalty=0` 和 `0～4` 步延迟；加入全身摩擦 `0.5～1.5`、恢复系数 `0～0.2`、关节零位 `±0.05 rad`、COM、Kp/Kd、全身质量及惯量 `×0.8～1.2`、armature `×0.8～1.2`、关节摩擦 `0～0.05`，并将速度奖励 std 从 `0.5` 收紧到 `0.3`。**效果**：完成 `8192 env / 30000` 轮，末 500 轮 episode 约 `961/1000`、风格奖励约 `0.231`，但 XY 速度误差约 `2.36 m/s`、线速度奖励仅约 `0.121`；播放基本站立。**结论**：策略收敛到偏重存活和 AMP 风格的局部最优。
- **默认姿态起步复核**：较早的 `grounded_delayed_first` 和 `joint_pos_penalty=-0.3` 策略都能从默认关节姿态、零根速度立即起步，其中前者在强前向指令下 3 秒前进约 `2.09 m`。**结论**：默认静止姿态本身不是起步退化的根因。

### 2026-08-05

- **速度课程 + 跟踪奖励 `3→5` + 大误差惩罚**：按 `0～5k / 5k～12k / 12k～20k` 三阶段将 `vx` 从 `±1.0` 扩到 `±1.5/±2.0 m/s`、`vy` 从 `±0.5` 扩到 `±0.8/±1.2 m/s`、yaw 从 `±0.3` 扩到 `±0.45/±0.6 rad/s`，同时将 std 从 `0.5` 收紧到 `0.4/0.3`；新增阈值 `0.25 m/s`、权重 `-1.0`、最大 excess `2.0 m/s` 的超额速度误差惩罚。run：`veltrack5_curriculum_full_dr_delay04_8192_20k`。
- **中途效果与发散**：`model_11000.pt` 的 XY 误差约 `0.58 m/s`、线速度奖励约 `3.36`，但风格奖励仅约 `0.036`，播放出现明显八字腿。训练于第 `11988` 轮发散：action-rate 惩罚扩大到约 `-1.398e9`，平均奖励约 `-6.7e10`，value loss 变为 `inf`，没有生成 `model_12000.pt`。停止前 time-out 约 `95.4%`，因此不是 episode 提前终止导致进程停止。**结论**：动作输出偶发失控，随后 action-rate 与 value target 爆炸。

### 2026-08-06

- **速度奖励权重单变量回退 `5→3`**：保留速度课程、超额误差惩罚、完整随机化、`0～4` 步延迟和 `joint_pos_penalty=0`。`model_7000.pt` 的 XY 误差约 `0.99 m/s`、线速度奖励约 `0.66`、episode 约 `741/1000`；播放时髋关节很快撞限位，随后两腿张开并跪下。训练在 `7565～7567` 轮发散：value loss 从 `1.83e4` 升至 `1.11e32` 后变为 `inf`，adaptive learning rate 随后升至 `2.92e-3/6.67e-3`。**结论**：本轮由 critic 先发散，降低速度奖励既没有恢复 AMP 风格，也削弱了速度任务信号。

### 2026-08-07 至 2026-08-08

- **非足部接触惩罚 + 基座高度终止 `0.5 m`**：将 undesired contact 权重设为 `-0.1`，正确排除实际带碰撞的足部 link，并将基座高度终止阈值从 `0.4 m` 提高到 `0.5 m`。run：`contactpenalty_height05_veltrack3_8192_20k`，实际停在约第 `2459` 轮。停止前 episode 约 `673/1000`、XY 误差约 `0.592 m/s`、time-out `49.3%`、基座过低终止 `47.3%`。后续发现速度奖励坐标系错误，因此该轮不能用于评价接触惩罚本身。
- **速度奖励统一到 base frame**：线速度跟踪、超额速度误差和偏航跟踪分别改用 `root_lin_vel_b[:, :2]` 与 `root_ang_vel_b[:, 2]`，与 base-frame 指令保持一致。run：`baseframe_veltrack_8192_20k`。**效果**：完成 20000 轮，末段 episode 约 `991/1000`、XY 误差约 `0.616 m/s`、线速度奖励约 `1.520`、风格奖励约 `0.042`、std 约 `3.13`；前后、侧向和转向指令均能响应。由于自碰撞关闭，侧向行走仍会穿腿。**结论**：命令和奖励坐标系不一致是此前重要的真实 bug。

### 2026-08-09

- **启用机器人自碰撞**：仅将 `enabled_self_collisions` 从 `False` 改为 `True`。run：`baseframe_selfcollision_8192_20k`。**效果**：完成 20000 轮，末段 episode 约 `855/1000`、XY 误差约 `0.847 m/s`、线速度奖励约 `0.911`、风格奖励约 `0.045`、std 约 `3.38`；可以行走且不再直接穿腿，但开局弹跳并岔腿。诊断发现 raw action 从首帧约 `+4/-5` 经 `last_action` 历史反馈增长到约 `±40`，髋 yaw/roll 长时间卡机械限位。
- **raw action `±8` + 90% 软限位 + 20% 默认姿态复位（失败）**：为抑制上述动作增长，在 wrapper 裁 raw action、将 PD 目标缩到 URDF 90% 软限位，并让 `20%` episode 从默认静止姿态复位。run：`actionclip8_softlimits_defaultreset20_selfcollision_8192_20k`。**效果**：虽完成 20000 轮，但最终 episode 仅约 `35/1000`，约 `99%` 因基座过低终止，播放直接向后倒；std 从 `1.0` 增长到 `8504`。**结论**：环境看不到裁剪外动作，而熵奖励仍鼓励增大方差；软限位还无法覆盖参考数据中的贴限位姿态。

### 2026-08-10

- **取消熵奖励（主动停止）**：保留 raw action `±8` 和 `20%` 默认姿态复位，将 `entropy_coef` 从 `0.01` 改为 `0`，PD 目标恢复到 URDF 硬限位。**效果**：约 10 轮后主动停止，不作为性能结论；原因是完全去掉熵可能再次收敛到原地站立。
- **有界方差 + PD 目标硬限位（失败）**：恢复 `entropy_coef=0.01`，新增 `BoundedStdActorCritic`，将 std 限制在 `0.05～2.0`；继续保留 raw action `±8`、`20%` 默认姿态复位和硬限位 PD 目标。run：`actionclip8_hardlimits_boundedstd2_entropy001_defaultreset20_selfcollision_8192_20k`。**效果**：第 `2070` 轮停止，`model_2000.pt` 基本向后倒；episode 约 `41.48/1000`、基座过低终止 `99.69%`、time-out `0%`、风格奖励约 `0.0104`，std 稳定在 `1.94`。**结论**：方差边界生效，但没有解决无法站立的问题。
- **有界方差 + 宽松 PD 目标（主动停止）**：只把处理后的 PD 目标恢复为 `(-100, 100)`，其余保留上一版设置。run：`actionclip8_unboundedtargets_boundedstd2_entropy001_defaultreset20_selfcollision_8192_20k`。**效果**：为横向核查宇树配置，在第 `16` 轮主动停止，不作为效果结论。参考数据中大量 hip roll/yaw 帧贴机械限位，宽松目标可能提供限位预载力矩，但该假设尚未完成独立消融验证。
- **恢复宇树训练核心**：删除 `BoundedStdActorCritic`，恢复宇树 G1 使用的原生 `RslRlPpoActorCriticCfg` 和 scalar std；删除 raw action `±8` 裁剪，使 `clip_actions=None`；恢复 `100%` AMP 参考帧复位。保留宽松 PD 目标、完整随机化、新增奖励、新 URDF、贴地默认姿态、自碰撞和 `0～4` 步延迟。运行时 YAML 确认配置生效，AMP 终止观测专项测试 `4/4` 通过。run：`unitree_actorcritic_noclip_refreset100_full_dr_delay04_selfcollision_8192_30k`。

### 2026-08-11

- **宇树核心版本完成 30000 轮**：生成最终 checkpoint `model_29999.pt`，总训练时间约 `19 h`。末轮 episode 长度约 `885/1000`、time-out `87.85%`、基座过低终止 `9.36%`、姿态异常终止 `3.07%`；XY 速度误差约 `0.823 m/s`、yaw 误差约 `0.848 rad/s`、线速度奖励约 `0.997/3.0`、偏航奖励约 `1.042/2.0`、风格奖励约 `0.040`、std 约 `3.35`。训练全程未出现 NaN/Inf，`model_24000.pt` mixed 视频确认机器人并非完全静止。**结论**：恢复宇树核心后重新获得可训练且数值稳定的策略，但最终大范围速度跟踪和 AMP 风格仍属一般。

### 2026-08-18

- **当前训练链路复核**：重新检查 BSRL 资产、动作、观测、AMP 数据、PPO、奖励、事件、课程和终止条件。关节顺序、左右脚顺序、足部碰撞 link、base-frame 速度奖励、AMP 当前/参考观测维度以及 `0～4` 步延迟实现均一致，未发现新的坐标或排列错误。
- **材料摩擦范围收窄**：将训练期全身静摩擦和动摩擦由 `0.5～1.5` 改为 `0.8～1.2`，减少极端摩擦对步态风格和训练稳定性的干扰；没有改变 URDF/USD 的关节硬限位。
- **稳定行走速度档位**：将三阶段最高指令收窄为 `vx=±1.2 m/s`、`vy=±0.8 m/s`、yaw `±0.5 rad/s`，对应课程阶段为 `0.6/0.9/1.2`、`0.4/0.6/0.8`、`0.25/0.375/0.5`；每 5 秒根节点 x/y 速度扰动由 `±0.5` 降为 `±0.2 m/s`。目的为优先获得能稳定走起来的策略，不追求高速性能。
- **air-time 与质量随机化收敛到保守配置**：双足 air-time 直接采用宇树 G1 AMP 的权重 `0.5`、阈值 `0.4 s`；关闭 base 质量额外增加 `-1～+3 kg`，仅保留全身质量与惯量 `×0.8～1.2`。目的为减少长单支撑偏好和复合质量随机化。
- **GMR 姿态/足宽正则完成**：在 BSRL IK 中加入以当前屈膝默认姿态为中心的关节正则，hip yaw/roll cost 设为 `10`，并把人根坐标系中的脚目标横向间距限制到 `0.30 m`。全量数据仍为 `30 Hz`、共 `14975` 帧（`499.17 s / 8.32 min`），格式、坐标约定、左右脚与关节顺序、有限值、单位四元数和 URDF 关节限位检查均通过；人工确认 walk/run 视频正常后已覆盖正式 AMP 数据。
- **正则化数据效果**：跑步 root-frame 足宽 p90/max 由 `0.357/0.805 m` 降至 `0.300/0.376 m`，左右 hip yaw 贴限位比例由 `18.4%/12.1%` 降至 `1.8%/1.8%`、hip roll 由 `39.9%/37.4%` 降至 `9.2%/9.9%`；走路足宽 p90/max 由 `0.361/0.662 m` 降至 `0.298/0.361 m`，hip yaw 由 `11.5%/12.5%` 降至 `0.9%/1.2%`、hip roll 由 `27.9%/28.1%` 降至 `3.3%/4.6%`。**结论**：参考动作中的宽腿、长期 yaw 偏置和贴髋限位污染已显著降低。
- **步频复核**：从左右膝峰值粗略估计，走路每条腿约 `1.5 Hz`、跑步每条腿约 `2.3 Hz`，对应总落脚节奏约 `3.0/4.6 step/s`。**结论**：参考数据并不缺高步频，当前低频大步更可能来自宽指令、长单支撑奖励、参考动作大关节幅度以及鲁棒性随机化的共同偏置。
- **稳定行走新一轮训练启动**：正式 run 为 `2026-08-18_22-21-04_stable_walk_gmr_regularized_airtime05_massscale_commands12_delay04_8192_30k`，使用 `8192 env / 30000` 轮、每 `1000` 轮保存 checkpoint。已确认进入 PPO 循环并生成 TensorBoard 事件及 `model_0.pt`；第 20 轮吞吐约 `7.9 万 step/s`，课程处于第一阶段。启动前处理了当前 IsaacLab `2.3.1`/Isaac Sim `5.1` 安装缺少版本函数以及失效本地代理阻断默认地面 USD 下载的问题，未修改 IsaacLab 底层库。

### 2026-08-28

- **当前最佳 BSRL run**：`2026-08-28_01-43-09_no_actor_feet_noheading_estimator_criticpriv8_formal1_8192`，完成 `30000` 轮并生成 `model_29999.pt`。
- Actor 移除左右足端基座系 FK 位置；Critic 与 AMP 判别器仍保留足端关键点，避免把仿真可得但部署不方便的足端位置直接交给策略。
- `heading_command=False`、`rel_heading_envs=0`，训练和部署都直接使用 `vx/vy/yaw_rate`，不再用随机世界航向替换 yaw-rate 指令。
- 默认启用 8 维 Estimator：基座高度、基座系线速度、双脚接触概率和双脚世界高度；Critic 同时获得对应真值特权观测。
- 最终 iteration：episode length `1000/1000`、time-out `99.29%`、XY 速度误差 `0.284 m/s`、yaw 误差 `0.426 rad/s`、style reward `0.290`、Estimator loss `0.0130`。
- **结论**：相较直接把足端 FK 交给 Actor 的版本，该配置播放与部署表现更好，因此后续 BSRL 主任务以此为基准。

### 2026-08-31：任务和日志收敛

- BSRL 只保留 `LeggedLab-Isaac-AMP-BSRL-v0` 一个注册任务；NoActorFeet 与 Play 变体均并入主配置或由 `play.py --nominal_eval` 处理。
- 所有 BSRL run 已统一移动到 `logs/rsl_rl/bsrl_amp/`，agent 的 `experiment_name` 固定为 `bsrl_amp`。
- 后续奖励、观测或超参数调整不再注册新任务：直接修改主配置并用唯一 `run_name` 开新一轮；差异写入本文和 run 自带的 `params/*.yaml`。

### 2026-09-01：Estimator 缩减为 4D

- BSRL 与 T1 共用的 Estimator 输出由 8 维缩减为 4 维，只保留 `base_height(1) + base_lin_vel_b(3)`；足接触和足部世界高度的观测函数保留在代码中，但不再作为 Estimator 标签、Actor 输入或 Critic 特权。
- 原因：AMP 不做逐帧相位跟踪，足接触标签离散且容易在触地边界闪烁，足高又能由本体状态近似推导；旧训练中接触损失长期占主要比例，可能诱发落脚抖动。
- 新 BSRL 正式 run：`2026-09-01_11-19-28_lafan1_newurdf_estimator4d_formal1_8192`，从头训练 `8192 env × 30000` 轮；tmux 为 `bsrl_lafan1_newurdf_estimator4d_formal1`，使用物理 GPU 0。
- 实际 `model_0.pt` 权重合同已核对：Estimator 输出层 `(4,128)`、Actor 首层 `(512,244)`、Critic首层 `(512,290)`。第 148 轮 Estimator loss `0.0046`（高度 `0.0001`、线速度 `0.0045`），AMP 判别器 loss `0.1732`，episode length `603/1000`；无 NaN、OOM 或维度错误，正式训练继续运行。
- 不带 Estimator 的对照 run `2026-08-31_21-52-19_lafan1_newurdf_noestimator_formal1_8192` 仍在物理 GPU 1 运行；第 24355 轮 episode length `982/1000`、time-out `97.87%`、XY/yaw 误差 `0.326/0.470`，两组训练互不覆盖。

### 2026-09-01：提高腿部 PD 的对照实验

- 保留此前名义 PD 作为对照：hip pitch/roll/yaw `100/2`、knee pitch `150/4`、ankle roll/pitch `40/2`。
- 新名义 PD：hip pitch `150/3`、hip roll `100/2`、hip yaw `100/2`、knee pitch `200/4`、ankle roll/pitch `100/3`。
- Kp/Kd 启动随机化仍为各自独立乘以 `0.8～1.2`；armature、力矩/速度/功率限制及 `0～4` 物理步控制延迟均保持不变。
- 本轮只改变上述 PD，不停止此前训练；使用主任务和 4D Estimator。正式 run：`2026-09-01_17-34-09_pd_hip150_knee200_ankle100_estimator4d_formal1_8192`，`8192 env × 30000` 轮，tmux：`bsrl_pd_hip150_knee200_ankle100_estimator4d_formal1`，物理 GPU 3。
- 后续实验遵循并行对照原则：除非用户明确要求终止，否则新实验不得停止仍在运行的旧训练。
- 新增不带 Estimator 的单变量对照：保持新 PD、URDF、奖励、随机化、AMP/PPO 参数和训练规模全部一致，仅移除 4D Estimator 及其 Actor 估计输入；此前带 Estimator 的新 PD 训练继续运行。正式 run：`2026-09-02_11-10-01_pd_hip150_knee200_ankle100_noestimator_formal1_8192`，`8192 env × 30000` 轮，tmux：`bsrl_pd_hip150_knee200_ankle100_noestimator_formal1`，物理 GPU 0。

## 历史有效训练配置（2026-08-18）

### 仿真与控制

- 平地、`8192 env`、episode `20 s`；物理频率 `200 Hz`（`dt=0.005 s`），策略频率 `50 Hz`（`decimation=4`）。
- 12 维关节位置动作，以默认关节位置为 offset；所有关节 action scale 均为 `0.25`，处理后的 PD 目标只做 `[-100, 100]` 数值保护，raw action 不裁剪。实际关节状态仍受 URDF 生成的 USD 硬限位约束，但 PD 目标没有逐关节裁到 URDF 限位。
- 默认姿态：根高 `0.8665 m`，hip yaw/roll `0`、hip pitch `-0.2 rad`、knee pitch `+0.4 rad`、ankle pitch `-0.2 rad`、ankle roll `0`。
- 自碰撞开启；执行器每个 episode 随机延迟 `0～4` 个物理步，即 `0～20 ms`。
- PD：hip `Kp/Kd=100/2`、knee `150/4`、ankle `40/2`；armature `0.01`，仿真力矩上限 `1000`，速度上限 `10 rad/s`。

### 指令与课程

- 每 `10 s` 重采样一次；`2%` 环境为站立指令。线速度在基座坐标系中均匀采样。
- `0～5000` 轮：`vx=±0.6 m/s`、`vy=±0.4 m/s`、yaw 上限 `±0.25 rad/s`、线速度奖励 std `0.5`。
- `5000～12000` 轮：`vx=±0.9 m/s`、`vy=±0.6 m/s`、yaw 上限 `±0.375 rad/s`、std `0.4`。
- `12000` 轮以后：`vx=±1.2 m/s`、`vy=±0.8 m/s`、yaw 上限 `±0.5 rad/s`、std `0.3`。
- `heading_command=True` 且 `rel_heading_envs=1.0`：实际 yaw 指令由随机目标朝向与当前朝向的误差生成，再裁到上述 yaw 范围；它不是所有环境都直接均匀采样 yaw 速度。

### 观测、PPO 与 AMP

- Actor 使用 5 帧带噪历史：基座角速度、根旋转 6D、速度指令、12 关节位置/速度、上一步动作、左右脚相对基座位置；不直接输入基座线速度。
- Critic 使用同样的 5 帧结构，额外输入无噪声基座线速度，且其余观测不加噪。
- AMP 判别器使用 4 帧：根旋转、基座角速度、关节位置/速度、左右脚相对基座位置；参考数据同维同序。
- 数据为 `walk1_subject1.pkl` 和 `run1_subject2.pkl`，各自权重 `1.0`，训练时按动作文件等概率选取，再在文件内随机取连续 4 帧。
- 使用 RSL-RL 原生 ActorCritic、scalar std、`[512, 256, 128]` ELU；entropy `0.01`、学习率 `1e-4` adaptive、PPO clip `0.2`、gamma `0.99`、lambda `0.95`。
- AMP style scale `5.0`，`task_style_lerp=0.4`，即组合时约 `40%` task、`60%` style；未启用 symmetry loss。

### 当前奖励

- 线速度跟踪 `+3.0`，base frame，std 随课程为 `0.5/0.4/0.3`；yaw 跟踪 `+2.0`，std `0.5`。
- XY 速度超额误差 `-1.0`：误差超过 `0.25 m/s` 后平方惩罚，excess 最大裁到 `2.0 m/s`。
- 非偏航角速度 `-0.1`、平地姿态 `-0.2`、力矩 `-1.5e-7`、关节加速度 `-1.25e-7`、action rate `-0.005`。
- 双足单支撑 air-time `+0.5`，上限时间 `0.4 s`；足部滑动 `-0.6`，非足部接触 `-0.1`。
- 关节位置限位 `-0.5`、hip yaw/roll 偏离默认位姿 `-0.2`、终止 `-200`。
- 全关节默认位姿惩罚 `joint_pos_penalty=0`；垂向线速度、基座高度、关节速度等若干已配置项当前权重为 `0`。

### 当前事件与终止

- 每个 episode 随机化全身材料：静/动摩擦 `0.5～1.5`、恢复系数 `0～0.2`。
- 启动时按环境随机化：全身质量及惯量 `×0.8～1.2`、base COM（x `±0.025 m`、y/z `±0.05 m`）、默认关节位置 `±0.05 rad`、Kp/Kd `×0.8～1.2`、armature `×0.8～1.2`、关节摩擦 `0～0.05`；不再额外增加 base 质量。
- 每 `5 s` 给根节点 x/y 速度各扰动 `±0.2 m/s`；持续外力/力矩和六维根速度扰动当前均关闭。
- 训练 `100%` 从 walk/run 参考动作随机帧复位；播放则从默认静止姿态复位，并关闭随机化和 actuator 延迟。
- 终止：20 秒超时、base 接触、base 高度低于 `0.5 m`、机身倾斜超过 `60 deg`。

## 当前主配置（2026-09-01）

- 当前正式 run：`rewardpriv_estimator4d_formal1_8192`，2026-09-01 15:52 重启，8192 env、30000 轮、带 4D Estimator；Critic 奖励特权补齐后的第一轮。
- 原 `lafan1_newurdf_estimator4d_formal1_8192` 已从 `model_7000.pt` 恢复到独立 GPU，继续跑至总计 30000 轮；不带 Estimator 的同名对照已完成 30000 轮。
- BSRL 资产改为直接加载 `data/Robots/BSRL_urdf/urdf/export.urdf`，Isaac Lab 启动时生成缓存 USD；仓库内旧 USD 不再是训练资产源，也不需要随 URDF 改动单独维护。
- 当前硬限位：左右 hip roll 均为 `[-0.3, 0.3] rad`，左右 knee pitch 均为 `[-0.1, 2.2] rad`。结合 `soft_joint_pos_limit_factor=0.9`，运行时软限位分别为 `[-0.27, 0.27]` 和 `[0.015, 2.085] rad`。
- 环境、控制、奖励、随机化和课程沿用上述 8 月 18 日稳定行走配置，但 `heading_command` 已关闭，yaw-rate 指令不再被随机目标航向替换。
- Actor 不包含左右足端 FK：每帧 `48` 维、5 帧共 `240` 维；Estimator 预测 4 维后，Actor MLP 实际输入 `244` 维。
- Critic 主观测每帧 `57` 维、5 帧共 `285` 维；其中已经包含真值基座线速度。基座高度特权为 `5` 维，奖励相关接触特权为 `60` 维，Critic MLP 实际输入 `350` 维；这些特权无论 Actor 是否启用 Estimator 都保留。
- AMP 判别器仍使用根姿态 6D、基座角速度、12 关节位置/速度和左右足端关键点，单帧 `39` 维、4 帧共 `156` 维。
- BSRL 基准默认保持普通 ActorCritic；同一个任务加 `--use_estimator` 时启用 4D Estimator、`state_est_label` 和 `state_est_critic`。不为这项对照注册新任务，也不切换日志根目录。
- PPO 默认 `30000` 轮，entropy `0.01`、学习率 `1e-4` adaptive；AMP style scale `5.0`、task/style `0.4/0.6`。
- 唯一日志根目录为 `logs/rsl_rl/bsrl_amp/`，唯一任务为 `LeggedLab-Isaac-AMP-BSRL-v0`。

### 当前各模块观测表

| 模块 | 当前输入 | 历史与维度 | 备注 |
|---|---|---:|---|
| Policy / Estimator 输入 | base 角速度 `3`、root 旋转 6D `6`、速度指令 `3`、12 关节位置 `12`、关节速度 `12`、上一动作 `12` | `48 × 5 = 240` | 有训练噪声；Actor 不含左右足端 FK，也不直接观察 base 线速度 |
| Estimator 输出 | base 高度 `1`、base-frame 线速度 `3` | 当前帧 `4` | MLP `[256,128]`；监督损失系数 `1.0`，Actor 梯度不反传到 Estimator |
| Actor MLP | Policy 历史 `240` + Estimator 预测 `4` | `244` | `[512,256,128]` ELU |
| Critic 主观测 | Policy 的无噪声版本 + base 线速度 `3` + 左右足相对 base 位置 `6` | `57 × 5 = 285` | 不使用 Estimator 预测值 |
| Critic 高度特权 | base 高度 `1` | `1 × 5 = 5` | 覆盖 base-height termination；base 线速度真值已在主观测中 |
| Critic 奖励特权 | 双足接触 `2`、双足 air/contact time `2+2`、双足世界系 XY 速度 `4`、非足部接触计数 `1`、base 接触标志 `1` | `12 × 5 = 60` | 覆盖 feet air-time、feet slide、undesired contact 与 base-contact termination；Critic 总输入 `350` |
| AMP agent 判别器 | root 旋转 6D `6`、base 角速度 `3`、关节位置/速度 `12+12`、左右足相对 base 位置 `6` | `39 × 4 = 156` | 不含 base 线速度和速度指令 |
| AMP demo 判别器 | 参考 root 旋转、角速度、关节位置/速度和左右足关键点 | `39 × 4 = 156` | 与 agent 侧严格同维同序 |

### 当前奖励表

下表是当前源码默认值；2026-09-14 的无裁剪/限位 -5/mimic 平滑组合尚未启动 BSRL 新训练，旧 run 继续使用冻结 YAML。

| 奖励项 | 权重 | 当前含义/参数 |
|---|---:|---|
| `track_lin_vel_xy_exp` | `+3.0` | base-frame XY 速度跟踪；std 按课程为 `0.5 → 0.4 → 0.3` |
| `track_ang_vel_z_exp` | `+2.0` | base-frame yaw-rate 跟踪，std `0.5` |
| `lin_vel_xy_error_excess_l2` | `-1.0` | XY 误差超过 `0.25 m/s` 后平方惩罚，excess 上限 `2.0 m/s` |
| `ang_vel_xy_l2` / `flat_orientation_l2` | `-0.1 / -0.2` | 抑制 roll/pitch 角速度与机身倾斜 |
| `dof_torques_l2` / `dof_acc_l2` | `-1.5e-7 / -1e-6` | 力矩惩罚不变；实际关节加速度平方惩罚对齐 mimic（原 `-1.25e-7`） |
| `action_rate_l2` / `dof_pos_limits` | `-0.1 / -5.0` | 一阶 raw-action 变化平方惩罚（原 `-0.005`）；12 个实际关节角越过 90% 软限位后的线性惩罚 |
| `action_smoothness` | `-0.001` | 新增二阶 raw-action 差分平方和；各环境独立历史、episode reset 清零，不裁剪目标或直接惩罚动作幅度 |
| `feet_air_time` | `+0.5` | 双足正向 air-time，阈值 `0.4 s` |
| `feet_slide` / `undesired_contacts` | `-0.6 / -0.1` | 足底滑动与非足部接触惩罚 |
| `joint_deviation_hip_l1` | `-0.2` | hip yaw/roll 偏离默认姿态 |
| `termination_penalty` | `-200` | 非超时终止惩罚 |
| 当前关闭项 | `0` | `lin_vel_z`、base 高度、body 线加速度、关节速度/速度限位、接触力、全关节默认姿态偏差 |
| AMP style reward | scale `5.0` | LSGAN；最终 task/style 混合比例 `0.4/0.6` |

### 2026-09-07 无 Estimator + 镜像损失对照

- 正式 run：`2026-09-07_14-49-15_noestimator_mirror01_friction0515_formal1_8192`，tmux `bsrl_noestimator_mirror01_friction0515_formal1`，GPU 3，`8192 env × 30000` 轮。
- Actor/Critic 观测归一化均关闭；Estimator 关闭。AMP 判别器仍按框架固定逻辑归一化自己的 156D 输入，这与 Actor/Critic 开关相互独立。
- 启用左右镜像策略损失，系数 `0.1`；未启用镜像数据增广。材料静/动摩擦范围扩大为 `0.5～1.5`。
- Estimator helper 的新训练约定为：Estimator 的 240D 输入、Estimator 增广后的 244D Actor 输入及 Critic 输入分别归一化。播放历史 Estimator checkpoint 时，`play.py` 从该 run 的 `params/agent.yaml` 恢复 Actor/Critic 的历史开关，避免当前 helper 默认值造成 normalizer `state_dict` 键不匹配。
- 已用历史 `2026-09-01_15-52-09_rewardpriv_estimator4d_formal1_8192/model_29999.pt` 实际验证：成功恢复 `actor_obs_normalization=True`、`critic_obs_normalization=False` 并严格加载完整 BSRL Estimator 权重。

### 2026-09-14：髋部 Kp 150、膝部 Kd 2、踝部 Kd 1 的 PD 对照

- 本轮仅修改 BSRL 名义 PD，左右腿使用相同参数；保持当前无 Estimator、镜像损失 `0.1`、Actor/Critic 不归一化、材料摩擦 `0.5–1.5` 的配置，从头训练，不加载旧 checkpoint。
- 上一组 PD 对照策略：`2026-09-07_14-49-15_noestimator_mirror01_friction0515_formal1_8192/model_29999.pt`。旧参数保留如下，历史 run/ONNX 不修改：

| 关节 | 旧 Kp / Kd | 本轮 Kp / Kd |
|---|---:|---:|
| Hip yaw | 100 / 2 | 150 / 2 |
| Hip roll | 100 / 2 | 150 / 2 |
| Hip pitch | 150 / 3 | 150 / 2 |
| Knee pitch | 200 / 4 | 200 / 2 |
| Ankle pitch | 100 / 3 | 100 / 1 |
| Ankle roll | 100 / 3 | 100 / 1 |

- 用户在正式启动前将踝部 Kd 从 `0` 修订为 `1`，没有启动过本次 Kd=0 训练。保留启动时 Kp/Kd `×0.8–1.2` 随机化；踝部实际 Kd 范围为 `0.8–1.2`，其余关节 Kd 为 `1.6–2.4`。
- URDF、动作/观测、参考 PKL、奖励、课程、PPO/AMP、armature、执行器延迟和力矩/速度/功率限制均不修改。本轮仍为关节空间 `DelayedPowerLimitedPDActuator`，尚未加入并联踝电机侧联合限制；不把其效果与 PD 改动混合。
- 正式规模 `8192 env × 30000` 轮，seed `42`；沿用唯一主任务 `LeggedLab-Isaac-AMP-BSRL-v0`、日志根 `logs/rsl_rl/bsrl_amp`，不注册新任务，不做 smoke，不停止已有训练或修改 T1。
- 正式 run：`2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192`，完整目录为 `logs/rsl_rl/bsrl_amp/2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192`。
- tmux：`bsrl_pd_hip150_knee200_kd2_ankle100kd1_formal1`；物理 GPU `2`；启动日志 `logs/launches/bsrl_pd_hip150_knee200_kd2_ankle100kd1_formal1_20260914.log`。
- **启动验收**：已运行到第 `11` 轮，`model_0.pt`（约 16 MB）与 TensorBoard 日志均已生成；前期奖励/loss 标量全部有限，未发现 NaN/Inf、OOM 或 traceback。初始化阶段有 headless GLFW 和未选中 GPU 枚举警告，但 GPU 2 的物理仿真及 PPO 已正常运行。这里只确认启动与数值正常，不据此判断最终步态效果。
- **对照配置核验**：与 9 月 7 日上述 run 的冻结 YAML 递归比较，`agent.yaml` 仅 run 名称不同；`env.yaml` 除日志路径外仅 6 处 PD 数值变化（髋 yaw/roll Kp、髋 pitch/膝 Kd、左右踝轴 Kd 配置）。URDF SHA256 仍为 `74a2c67b35f8181d40a7eead313d4893c1f6a7b6d123034c711cd9270df01c6a`。两份 walk/run PKL 均成功加载；既有 `t1_41`、`t1_42` 训练进程保持运行。

### 2026-09-14：初版 MuJoCo 并联踝闭链回放（后续发现安装反向，保留追溯）

- **勘误**：下列初版闭链漏了相对 URDF 的 `Rz(pi)` 安装变换，连杆落在鞋尖侧，长短支链方位也反向。6 项测试只证明数学自洽，不能支持“真实安装已对齐”的结论；旧力矩/24 Nm 结果及串联等效电机换算只作历史记录。修正与重新验收见下一条，旧文件不覆盖。

- 目录：`/root/mjw/parallel_ankle`，完整参数、验证与复现见该目录 `README.md`。原上传示意模型保存在 `parallel_ankle_approximate.py/.xml`，不再把它的 68 mm 摇臂及 10°参考角混入真实机构。
- 以 `bsrl_ankle_core_flat/bsrl_bridge_bringup.yaml` 的左右机构几何和原 `AnkleKinematics.cpp` 为依据：摇臂 70 mm、长/短杆 245/165 mm、电机中心高度 245/165.298360850917 mm、侧向偏置 32.9 mm；连接点严格遵循原解算器的 `radius * normalize(d0)`。
- **零位原则**：URDF 的 `q=0` 是标准关节零位，不重新标定，不修改既有编码器零偏或 URDF 关节原点。初始化只调整新增杆件的装配姿态，保证标准零位和默认站姿直接闭合。
- 完整机器人保留原本体、碰撞、质量惯量和 12 维 pitch/roll 等关节位置动作。4 个踝部直接执行器替换为 4 个真实机构电机驱动，pitch/roll 被动；力矩只施加到电机，经 `connect` 闭环约束传递，不再额外施加 `J^T tau`。
- 测试旧策略：`2026-09-07_14-49-15_noestimator_mirror01_friction0515_formal1_8192/model_29999.pt` 对应原导出 ONNX，Actor 240D、无 Estimator。从该 run 冻结 YAML 恢复旧 PD（yaw/roll 100/2、hip pitch 150/3、knee 200/4、ankle 100/3），**没有使用当前新训练的 ankle 100/1**。
- 保留既有 MuJoCo 1 ms 物理步和 50 Hz 策略、接触设置。串联/闭链均用同一显式 PD 和同一名义条件，无随机推力/延迟/噪声，不自动 reset；这不是对实机控制频率的修改。

| 30 s 全程指令 `[0.6,0,0]` | 摔倒 / 数值失败 | 平均 body vx | 最终 yaw 偏移 | 踝电机最大力矩 |
|---|---|---:|---:|---:|
| 串联独立关节限幅 | 无 / 无 | 0.700 m/s | 84.1° | 32.26 Nm（等效换算值） |
| 初版安装反向闭链，电机峰值 112 Nm | 无 / 无 | 0.699 m/s | 82.3° | 32.51 Nm |
| 同一闭链，仅踝电机限幅 24 Nm | 无 / 无 | 0.615 m/s | 51.3° | 24.00 Nm |

- **效果**：旧策略可在显式闭链中行走，串联/闭链总体接近；仍存在明显累积偏航，不能宣称闭链解决走弯。24 Nm 对照左长支链触限约 14.75%，仍完成全程；其他关节限值未改，不能把这组视为完整复现实机保守配置或实机限流建议。
- **验收**：6 项 CPU 测试通过，覆盖左右共 882 个全范围姿态；用 MuJoCo 约束雅可比独立核对原 C++ J（误差 <2e-8）；初始端点误差 <1e-12 m。30 s 闭链行走的最大端点误差约 0.135 mm，J 条件数最大 2.70；3 段各 30000 个物理步全部有限，无 MuJoCo 数值警告。
- **动力学近似**：缺少独立杆件实测质量惯量。原机器人质量惯量保留，新增数值辅助体每个 1 g（总增 8 g），无附加电机 armature，原关节 armature 0.01 保留。每体 0.5–2 g 的 10 s 敏感性检查均无摔倒，平均 vx 0.645–0.655 m/s。闭环数值柔度不等于真实传动柔性；间隙、热限制、驱动响应仍未建模。
- 视频（1280×720、50 fps、30 s）已存入 `videos/`：
  - `bsrl_model29999_realgeometry_closedankle_forward06_30s_20260914.mp4`
  - `bsrl_model29999_serial_control_forward06_30s_20260914.mp4`
  - `bsrl_model29999_closedankle_motor24_forward06_30s_20260914.mp4`
- 原始状态、力矩、冻结模型和校验摘要：`parallel_ankle/results/20260914_closed_30s_peak112`、`20260914_serial_30s`、`20260914_closed_30s_commission24`。当前 BSRL 新 PD 训练与 `t1_41/t1_42` 均保持运行；本次没有改 T1、硬件配置或既有 ONNX。

### 2026-09-14：修正闭链相对鞋子装反 180°（仅独立仿真）

- **根因**：不是 yaw 指令符号问题。URDF 原 CAD 鞋尖为 +X，足侧球铰 X 约 `-67.027 mm`，右长杆在 -Y、左长杆在 +Y；初版把解算器的 `+66.695 mm` 端点及相反长短方位直接装入 URDF。此前自洽测试未做物理安装验证。
- **修改范围**：只改 `parallel_ankle` 仿真适配层及生成的专用 XML：机构统一乘 `Rz(pi)`；`q_core=-q_urdf`，`J_urdf=J_core(-q)@diag(-1,-1)`，同步正逆解、角度/速度/力矩映射和球铰初始化。策略仍输出 pitch/roll，URDF 标准零位、鞋子、原 sim2sim XML、硬件 C++/YAML、编码器标定均不改；这不是实机驱动符号调整。
- **独立验证**：新增 `check_mounting.py` 从原 mesh 提取球铰点，验证前后方向及左右长短方位；旧错误安装会被测试拒绝。8 项 CPU 回归通过，左右合计 882 个全范围姿态闭合，MuJoCo 约束 J 与转换后 J 一致。右 PR / 左 AB 各 18 s，FK 最大误差 <9e-5 rad、闭合误差 <3 um。
- **重新回放**：仍用 9 月 7 日的 `model_29999` 原 ONNX/旧 PD，固定 `[0.6,0,0]`，完成 30 s；1500 个策略步、30000 个物理步，无摔倒/重置/数值失败，平均 body vx `0.701 m/s`，电机峰值 `31.92 Nm`，最大闭合误差 `0.146 mm`，最大 J 条件数 `2.484`。最终 yaw 仍偏 `83.77°`，不宣称走直已解决。
- **尚存尺寸差异**：CAD 足侧连接点 Y 约 ±22 mm，上传解算器约 ±27.298 mm；CAD 电机中心 X 约 -10 mm，解算器为 0。修正方向后足侧端点距 CAD 约 5.45–5.47 mm，摇臂端点约 13.73–13.79 mm；本轮保留用户指定的解算器尺寸，未擅自按 CAD 改参数。杆件惯量、柔性、间隙等近似仍然存在，不能称为完整实机动力学验证。
- 新录像：`videos/bsrl_model29999_closedankle_mounting180_fixed_forward06_30s_20260914.mp4`；数据：`parallel_ankle/results/20260914_closed_mounting_rz180_30s_peak112`；CAD 对照图：`parallel_ankle/results/20260914_mounting_rz180_cad.png`。旧结果保留并标注错误安装，不覆盖历史文件，不停止或修改正在运行的 BSRL / T1 训练。

### 2026-09-14：按确认的 CAD 零位连接点修正 XML 闭链与杆件显示

- **用户确认**：原 CAD 零位连接点正确。专用 `parallel_ankle` 以原球铰中心、电机轴线为基准，不改 URDF 零位或编码器标定。摇臂约 `68 mm`、长/短杆 `245.2/165.3 mm`、足侧 Y=±22 mm、轴线 X≈-10.02 mm；保留左右原有约 0.1 mm 的坐标差异。这替代了上一版与 CAD 有 5.5/14 mm 偏差的几何。
- **解算一致性**：原硬件 C++ 模型不能表达这个 X 轴线偏置。仿真 `Geometry` 改为按同一定长约束求解析解/J，速度与力矩使用同一新 J；旧 C++ 调用保留在 `RobotGeometry` 做历史回归。没有修改硬件 C++/YAML/电机零偏，也没有把旧 J 混用于新装配。
- **视觉修正**：原 shank mesh 无损拆成静止小腿/外壳、运动摇臂、运动杆件、足侧球铰内圈等；右 255078、左 255102 个原三角面全部保留，零位重建误差 <1e-8 m。只生成独立派生 OBJ 并修改专用 XML 的引用，不改源 mesh，不把旧白色杆留在小腿上造成脱开假象。
- **验收**：15 项回归通过，包括左右 882 个全范围姿态闭合、J 差分/约束一致性、原本体初始姿态/惯量/Actor 观测不变、视觉面数守恒。单踝 PR/AB 各 18 s 通过。仍保留每新增辅助体 1 g 的数值惯量近似，视觉拆分不等于实测动力学质量拆分。
- **旧策略回放**：9 月 7 日 `model_29999`、旧 PD、固定 `[0.6,0,0]`，30 s 无摔倒/重置/数值异常，平均 body vx 0.699 m/s，电机峰值 34.237 Nm，最大闭合误差 0.152 mm，最大 J 条件数 3.152。最终 yaw 仍偏 83.10°；本轮不宣称解决走直问题。
- XML：`parallel_ankle/bsrl_closed_ankles.xml`；结果：`parallel_ankle/results/20260914_cad_zero_connected_30s`；全身录像：`videos/bsrl_model29999_cad_zero_connected_forward06_30s_20260914.mp4`；特写：`videos/bsrl_cad_zero_connected_ankle_closeup_30s_20260914.mp4`。原 sim2sim XML、训练 URDF、策略、T1/BSRL 运行中的训练均不变。

### 2026-09-14：PD 目标角按 URDF 硬限位裁剪（已撤回，无正式训练）

- **改动**：BSRL 的 12 维位置目标由 `[-100,100] rad` 宽松数值保护，改为从当前 `scene.robot.spawn.asset_path` 指向的 URDF 读取逐关节 `lower/upper`。沿用 `JointPositionAction`，在 `default_pos + action × scale` 后裁剪；旧配置保留为注释，不新增任务、执行器或 Actor 观测。
- **不变项**：raw action、scale `0.25`、PD（踝部 `100/1`）、电机参数及全部奖励保持不变。用户确认 pitch/roll 范围可达，本次直接使用 URDF 硬限位，没有额外缩小范围，也没有引入新的可达域惩罚。Isaac 训练端仍为原独立关节执行器，本次不等于接入闭链。
- **奖励澄清**：`dof_pos_limits=-0.5` 调用 Isaac Lab 的 `joint_pos_limits`，读取实际 `joint_pos` 与 `soft_joint_pos_limits`；软限位为 URDF 区间以中点为中心缩到 90%。它不读取 `joint_pos_target` 或原始策略动作，所以很远的目标不一定比略微超限的目标受到更大的限位惩罚。`action_rate_l2` 惩罚动作变化，不能替代目标角范围限制。
- **验证**：CPU 检查 12 个关节的 URDF 映射，执行实际 Isaac Lab `JointAction.process_actions` 内核，确认默认姿态、区间中点、上下界及极端越界目标均在 scale/offset 后正确裁剪，raw action 不变。实际奖励内核验证相同关节姿态、不同超限目标具有相同的限位惩罚；固定大动作的 action-rate 惩罚为零。配置语法及 `git diff --check` 通过，没有启动 Isaac 仿真或 smoke 训练。
- **T1 对照核查（只读）**：T1 AMP 原已通过 `T1_ACTION_CLIP` 将目标角裁到 URDF 两端各内缩 5% 行程的范围；`wbt_est` T1 mimic 的 `JointPositionActionCfg` 未显式设置目标角 clip，runner 的 raw-action clip 也为 None，依靠参考跟踪、动作平滑、实际角度软限位惩罚与物理约束。两者不能混为“都不裁剪”，本次不修改 T1。
- **实验隔离**：只影响后续创建的 BSRL 环境；没有启动或停止训练，没有修改已有 run 的 YAML/checkpoint，T1、MuJoCo 闭链回放及实机代码不变。历史策略回放仍应使用各自 run 的动作裁剪配置，不能用新裁剪范围代替旧实验条件。

### 2026-09-14：按用户要求取消 BSRL/T1 的 PD 目标角裁剪

- 核对本工程 G1 AMP 与 wbt_est T1 mimic 后，确认显式裁剪目标角不是训练必需条件；用户选择允许通过更大位置误差请求力矩。因此撤回上一个尚未启动训练的 BSRL URDF 目标裁剪，设为 `self.actions.joint_pos.clip=None`，连旧 `±100 rad` 数值范围也不再使用；两种旧配置均保留为注释。runner 显式记录 `clip_actions=None`，raw action 行为不变。
- BSRL 原来的 PD、机械角度限位、执行器力矩/速度/功率限幅和 `dof_pos_limits=-0.5` 全部保持；不启动新 BSRL 训练，已有 GPU 2 的 `pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192` 不停止、不热更新。
- 仅 T1 新开 `8192 env × 30000` 轮，取消目标裁剪并将其实际角度限位惩罚从 `-1.0` 调到 `-5.0`；GPU 3、tmux `t1_amp_noclip_jointlimit5_formal1`，细节写入 `debuglistT1.md`。不新增任务、不跑 smoke、不修改 wbt_est 或任何 URDF/XML。

### 2026-09-14：BSRL 实际角度限位惩罚提高到 -5.0（未启动新训练）

- **改动**：调整已有 `self.rewards.dof_pos_limits.weight`，从 `-0.5` 改为 `-5.0`，没有重复添加奖励项。旧权重保留注释；目标角/raw action 继续不裁剪，PD、电机参数及其余奖励不变。
- **计算**：对全部 12 个实际关节角求 `sum(max(lower_soft-q,0) + max(q-upper_soft,0))`，然后乘权重 `-5.0`。软区间由 URDF 硬区间以中点为中心缩到 90%，即两端各保留 5% 行程。奖励连续、分段线性，边界处斜率不连续；它不是硬限位，不裁剪目标，不直接施加约束力，也不会在越界瞬间固定扣 5 分。
- **量级例子**：hip roll 硬区间 `[-0.3,0.3]`、软区间 `[-0.27,0.27] rad`。单关节实际角 `0.28 rad` 时项值为 `-0.05`，`0.30 rad` 时为 `-0.15`（乘控制 dt 前）；当前 `dt=0.02 s`，对应每控制步 task reward 为 `-0.001/-0.003`，随后仍参与原 AMP task/style 混合。目标很大但实际角仍在软区间内时，这一项为零。
- **强度判断**：相同轨迹下惩罚扩大 10 倍，但仍是边界附近的软惩罚，不能仅凭权重 `-5.0` 判断过硬。多个关节同时贴限位会累加，参考动作若经常需要靠近机械限位，可能与风格奖励竞争；需要后续训练的实际限位项占比、关节轨迹和风格效果判断，不保证该权重一定最佳。
- **验证**：CPU 执行实际 Isaac Lab 奖励内核，验证区间内部、软限位边界及正负方向越界的 6 个例子；配置单一权重 `-5.0`、目标 clip=None、语法与 `git diff --check` 均通过。未进行 smoke 或新训练。
- **实验隔离**：本次只更新 BSRL 配置和本文，没有启动/停止训练。运行中的 BSRL 仍按其冻结 YAML 使用 `-0.5`；刚启动的 T1 no-clip 对照仍使用自身的 `-5.0`，不受本次改动影响。

### 2026-09-14：BSRL/T1 平滑奖励对齐 mimic（只改配置，暂不训练）

- **BSRL 新默认**：目标角和 raw action 均不裁剪，实际角度限位惩罚 `-5.0` 保持；一阶动作变化 `-0.005 → -0.1`，实际关节加速度平方 `-1.25e-7 → -1e-6`，新增二阶 `action_smoothness=-0.001`。旧参数保留注释，没有重复添加已有奖励，也没有新注册任务。
- **共用实现**：`amp/mdp/rewards.py:ActionSecondOrderL2` 沿用 wbt_est 的 `sum((a_t - 2*a_(t-1) + a_(t-2))²)`。使用原始策略动作，按每个环境分别保存/清理历史，与 mimic 相同采用 episode 开始的零填充历史；RewardManager 每控制步调用一次并统一乘权重和 dt，随后参与原 AMP task/style 混合，不重复乘 dt。
- **范围**：T1 同步改为一阶 `-0.1`、二阶 `-0.001`、关节加速度 `-1e-6`。BSRL 的 PD、URDF/XML、自碰撞、电机参数、观测/Estimator/判别器、力矩平方、足滑和其余奖励不变；G1 没有启用新奖励，wbt_est 本身未改。
- **验证**：`source/legged_lab/test/test_amp_action_smoothness.py` 的 12 项 CPU 回归通过，覆盖 12/25 维动作、8192 环境随机序列、线性/恒定大目标、部分及全量 reset、实例隔离、目标不裁剪、两机器人配置权重及 G1 隔离。已核对现有 RewardManager 的每步计算与 reset 接口；不启动 Kit、smoke 或新训练。
- **运行中的实验不变**：BSRL `2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192` 仍用目标保护 `±100 rad`、限位 `-0.5`、一阶 `-0.005`、加速度 `-1.25e-7`、无二阶；T1 no-clip 新 run 仍用启动时参数。此次没有启动/停止任何训练，待用户明确要求后再用新源码创建下一轮。

### 2026-09-14：Isaac 显式闭链 BSRL（8192 环境三次均未通过，默认已回到串联）

**最终状态：未稳定运行到100轮，三次正式启动均在首个采样窗口被物理检查器拦截，未产生可用训练checkpoint。旧BSRL及T1训练未停止。显式闭链代码/复制的CAD资源保留，但 `bsrl_amp_env_cfg.py:BSRL_EXPLICIT_CLOSED_ANKLES=False`，默认仍原串联资产；已获授权的mimic平滑奖励保持。以下按尝试顺序记录，不能将中途小规模通过读作正式训练通过。**

- **用户指定路线 A**：导入原 `export.urdf` 后由 `assets/bsrl_closed_ankle_spawn.py` 自动增加四个电机 hinge、四根 rod 的 spherical joint、四条 `excludeFromArticulation=True` 的末端球铰。树内共 `28` 个 DOF、`21` 个刚体；不是将 `Jᵀτ_motor` 施加回串联踝关节的等效模型，不维护另一套手工 USD。默认任务仍只有 `LeggedLab-Isaac-AMP-BSRL-v0`。
- **控制**：策略保持原 12 维关节目标；pitch/roll `100/1` PD（随机化 ×0.8–1.2）不做目标/关节空间力矩裁剪，以实际关节与电机状态的 CAD 约束 Jacobian 解 `Jᵀτ_motor=τ_joint`，仅向 motor hinge 施力。四个被动 pitch/roll 的直接驱动力矩为零。电机侧采用 `112 Nm`、`250 rpm`、`1250 W` 限幅，超过速度上限仍允许制动；原零位、URDF 机械限位不改。
- **CAD 与近似**：复用用户已确认的 `20260914_cad_zero_connected`，左右约 0.1 mm 的差异保留。14 份分拆 OBJ 和来源 manifest 已复制进 AMP，本仓库运行不依赖旁边的 `parallel_ankle`。旧 shank 的静态杆件视觉被分拆视觉替换，原本体碰撞和惯量保留；8 个辅助刚体各 `1 g`，总增加 `8 g`，辅助碰撞关闭，不新增电机 armature。它们是近理想连杆近似，**不是实测连杆惯量**。
- **复位/维度**：`ClosedAnkleArticulation` 在每次参考动作 reset 时同时装配 motor 和球铰位置/速度；rollout 中不手动投影/瞬移杆件。PhysX 球铰位置采用旋转向量，速度采用子刚体系角速度，不能把旋转向量差分直接当作关节速度。Actor 240D、动作 12D、Critic 与判别器仍用原观测合同；辅助 DOF 不加入关节加速度/限位奖励和观测。力矩平方成本统计实际被驱动的电机力矩（8 个原关节 +4 个踝电机）。
- **随机化**：原本体材料、质量/惯量、COM、12 关节默认角偏移、PD、armature、摩擦、0–4 物理步延迟和推扰继续保留；不独立随机化辅助杆件质量、球铰默认角或球铰增益。自碰撞仍开启，辅助杆件不引入新接触面。
- **求解器实测**：原 `8/4` 次 TGS 求解在大目标变化时出现约 `244 mm` 闭合误差；`64/8` +每迭代外力约 `13.6 mm`；`128/32` 约 `7.86 mm`；最终 `255/64` +`enable_external_forces_every_iteration=True` 在 1000 个 GPU 物理步、反复复位和满电机力矩测试下最大约 `2.72 mm`，未继续发散。物理 `dt=.005`、策略 `50 Hz`、PD 和电机限幅不变。此项只说明可进入并行训练检查，毫米级瞬态误差及性能仍需评估，不能等同 MuJoCo 精度。
- **代码验证**：新增几何测试 +已有平滑奖励测试共 `16 passed`；8192 组 URDF 限位内随机姿态可达，Jacobian 有限差分误差 `<2e-9`，虚功/电机限幅通过。单机非零姿态 GPU 装配位置残差约 `1e-7 m`，解析复位速度连接点残差约 `1e-7 m/s`。诊断脚本为 `scripts/rsl_rl/check_bsrl_closed_ankle.py`，不是另起 PPO smoke。
- **新训练**：tmux `bsrl_closedankle_explicit_formal1`，物理 GPU `4`，8192 env ×30000 轮、seed42，无 Estimator、Actor/Critic 不归一化、镜像损失0.1；run 后缀 `explicit_closedankle_cadzero_pd100kd1_mimicsmooth_noestimator_mirror01_formal1_8192`。本轮使用上一条已修改的 mimic 平滑奖励（一阶 -0.1、二阶 -0.001、加速度 -1e-6、限位 -5），因此与17:51旧训练不只是闭链单变量对照。为了留存早期样本，本轮启动覆盖保存间隔为100轮，源码默认仍1000轮。
- **首次并行失败（保留记录）**：`2026-09-14_22-58-07_explicit_closedankle_cadzero_pd100kd1_mimicsmooth_noestimator_mirror01_formal1_8192` 在首个96物理步中最大闭合误差约 `583 mm`，被检查器自动停止，**没有完成第0次PPO更新，更没有通过100轮**。随后16环境固定底座的 replicate physics 开/关对照相同，排除只复制了env0约束；完整机器人名义落地条件仍能复现约296mm瞬态及异常关节速度，非随机化或PPO损失导致。
- **求解顺序修正**：Isaac Sim 5.1 的 `solve_articulation_contact_last=True` 使外部约束/接触在内部关节限位之后求解。完整机器人16环境名义落地、随机动作的60控制步检查最大约 `3.76 mm`，不再复现爆开。继续保持 `dt=.005` 与原PD，只改数值求解顺序。
- **第二次正式启动**：tmux `bsrl_closedankle_explicit_formal2`、GPU4；run后缀 `explicit_closedankle_cadzero_contactlast_pd100kd1_mimicsmooth_noestimator_mirror01_formal2_8192`，启动输出 `logs/launches/bsrl_closedankle_explicit_formal2_20260914.log`，其余规模/参数同上。正在检查前100轮，尚不能宣称通过。
- **第二次同样未通过**：实际 run `2026-09-14_23-05-24_explicit_closedankle_cadzero_contactlast_pd100kd1_mimicsmooth_noestimator_mirror01_formal2_8192`，首个采样窗口最大闭合误差约572mm，自动停止、没有完成PPO更新。小规模名义通过不代表8192全随机化通过。
- **继续排查（不冒充正式训练成功）**：物理细化到1ms、仍保留200Hz PD（每5子步计算一次）与50Hz策略；命令延迟缓冲仍每5ms推进，保持0–20ms范围；电机力矩/功率/超速制动包络每物理子步检查。64环境DR检查曾出现19.6mm瞬态，被检查器拦截。发现新增电机的 `velocity_limit_sim=26.18` 引入了已验证MuJoCo模型没有的理想硬速度约束，会与闭环/触地约束争夺求解；现改为仅在电机驱动力包络使用额定速度、保留超速制动，solver仅保留1000rad/s数值上界，正在复核。没有降低112Nm/1250W额定包络，没有裁剪roll/pitch目标。
- **电机侧对齐修正的结果**：移除额外硬速度约束后，同一64环境DR的1ms/200Hz-PD检查起步误差由6.75mm降至0.076mm，完成60控制步、125次累计reset后最大约1.72mm，全部有限；5ms/200Hz-PD对照仍在48控制步内出现10.03mm瞬态。因此第三次正式启动选择物理1ms、PD200Hz、策略50Hz，其他参数同上。18项CPU测试通过，包含PD/延迟缓冲仅每5子步更新、子步继续执行电机功率/超速制动限制、部分reset不改变其他环境控制节律。
- **第三次正式启动也失败**：run `2026-09-14_23-14-00_explicit_closedankle_phys1ms_pd200_cadzero_noestimator_mirror01_formal3_8192`，原tmux `bsrl_closedankle_explicit_formal3`、GPU4；日志 `logs/launches/bsrl_closedankle_explicit_formal3_20260914.log`。每480物理步（24控制步/一轮PPO）审计一次。首窗口最大误差604.16mm、电机力矩112Nm、记录功率1250.000122W（float32舍入）、异常实际电机速度3608.21rad/s；被动pitch/roll直接力矩0，无NaN但物理明显不可信，自动中止，没有进入有效PPO更新。不能因总量有限就报告训练稳定。
- **已排除/尚不确定**：64环境同样DR/初始量级随机动作检查最大0.321mm；整体平移到世界X+100m的检查最大0.195mm；均完成30控制步。这两项不支持“世界原点太远”或“动作std本身”是充分根因。参考PKL全量关节角均在URDF范围，踝速度峰值walk约10.3rad/s、run约20.35rad/s，不是数千rad/s的非法标签。新增1g杆件与原聚合惯量的极端比例、少数接触/限位状态导致的GPU约束病态仍是怀疑对象，**尚未隔离为已确认根因**。不能未经许可任意加大杆件质量/惯量后声称与真实机器人对齐。
- **交接**：默认开关已置False，仅关闭这次尚未通过的源码实验，不创建替代新训练，不影响旧进程。可继续用 `scripts/rsl_rl/check_bsrl_closed_ankle.py --amp_env --num_envs 64 --steps 60 --headless --device cuda:0` 定位（默认物理1ms、PD200Hz），组件模式支持 `--powered`、`--no_replication`、`--world_offset`、`--action_std` 等明确对照。继续正式训练前应先找到大规模约束失稳原因/获得可靠辅助惯量；不是绕过检查器强行跑100轮。
- **最终回归与证据**：CPU测试共19项通过（含默认关闭未验证实验、原URDF仍12关节且零位不变），`git diff --check`通过。组件原始日志归档于 `reports/bsrl_closed_ankle_20260914/`，正式失败log保留在 `logs/launches/`，三个run的冻结YAML保留。确认没有这些新run的 `.pt`；交接时仅原有4个训练进程占用GPU0–3，GPU4/5诊断已退出。
- **验收规则**：每96个物理步记录 `[ClosedAnkleAudit]`，覆盖所有 decimation 子步与终止前状态，统计闭合误差、电机峰值、被动关节直接力矩、奇异/非有限数；超过1cm或出现数值异常时仅此新 run 抛错停止，不停止旧实验。异常时输出峰值环境及当时root/q/dq，不让reset掩盖物理失稳。
- **历史保护**：旧BSRL、两个wbt_est T1、T1 no-clip训练均保持；播放脚本根据冻结 run 的 actuator 列表恢复串联/闭链拓扑，避免旧BSRL checkpoint 因相同 URDF 路径被误放进新闭链。

### 2026-09-15：闭链失稳复现、PD时钟修复与PGS对照

- **从“8192才崩”缩小到单个确定现场**：seed42，env4054第6个1ms子步速度爆增、第7步闭合误差20.54mm；关TF32仍同样失败。关闭关节摩擦后仍15.50mm，取这个现场的状态/实际力矩/动力学参数单环境重放，可以逐步重现；移到原点也失败。抬离地面则早期异常消失，定位到接触/约束求解，而非批量复制、世界坐标偏移或PKL关节顺序整体错误。
- **明确代码bug已修**：闭链PD/延迟队列之前按`write_data_to_sim`调用计时，reset后的纯运动学`write+forward`也会消耗计数。现由`ClosedAnkleArticulation.begin_physics_step()`只在真实物理步前放行一次执行器计算；没有该回调的串联BSRL/T1路径不变。20项几何/控制/奖励CPU测试，以及真实Kit下4项AMP terminal/reset回归通过。历史“DelayBuffer不自动补零”只描述buffer本身，不能排除调用端先写了零目标的问题。
- **单改时钟不够**：TGS255/64仍第11步失败；减少速度迭代、取消直接关节硬速度夹紧也分别在8192环境中失败。辅助惯量×100只做了诊断重放，未写入训练模型。暂不把猜测的杆质量或额外armature当作修复。
- **PGS候选通过环境检查**：PGS128/16、1ms物理、200Hz PD、50Hz策略、全部DR/自碰撞，8192环境、动作std1、100控制步（2000物理子步）完成；最大闭合误差6.6439mm，motor≤112Nm，passive直接力矩0，未出现奇异/NaN。**不是100轮PPO**。退回5ms会在第一步产生44.10mm误差，因此保留1ms。
- **恢复原请求的正式尝试**：2026-09-15，tmux`bsrl_closedankle_pgs_formal1`，物理GPU4，8192×30000；每100轮保存；启动日志`logs/launches/bsrl_closedankle_pgs_formal1_20260915.log`。仅该进程设置`BSRL_EXPLICIT_CLOSED_ANKLES=1`，未新增任务、普通BSRL默认仍串联。100轮PPO结论待正式日志核验，不预先宣称通过。
- **硬件条件保持**：PD100/1踝、原URDF零位/限位、自碰撞、112Nm/250rpm/1250W电机侧包络、motor-only施力；保留原本体惯量和8个1g辅助体近似，不称为真实杆件质量已标定。旧BSRL PD训练自然结束；本次没有停止任何T1或旧训练。
- 原始诊断、失败现场和复现方法见`reports/bsrl_closed_ankle_20260915/README.md`；播放入口同时恢复run保存的solver_type与求解迭代数，避免用TGS播放新的PGS模型。
- **正式初始状态（14:43）**：run`2026-09-15_14-36-56_explicit_closedankle_pgs128v16_clockfix_pd200_noestimator_mirror01_formal1_8192`到iteration8，`model_0.pt`完整读取、98个浮点tensor有限；9个rollout物理审计最大7.7135mm、passive直接力矩0、motor≤112Nm、无NaN/奇异。冻结YAML确认Actor240/Critic350/disc156/action12。约36秒/轮，100轮需约1小时，30000轮按初始速度约12.5天；logger的ETA不显示天数，不能把显示的十几小时当作完整剩余时间。只读后台验收结果在`reports/bsrl_closed_ankle_20260915/formal1_acceptance100.json`，监控不停止训练；**正式100轮尚未宣称通过**。

### 2026-09-15：默认切换串联骨架＋并联踝等效传动（保留旧显式闭链实验）

- **原因**：显式闭链PGS128/16＋1ms在8192环境约36秒/轮，30000轮约12.5天。用户同意改用解析传动，要求检查兼容回退和针对机构误差的随机化。本次没有停止原显式闭链或任何T1训练，没有新注册任务。
- **新默认**：仍导入原`BSRL_urdf/urdf/export.urdf`，12关节/13刚体、自碰撞、URDF零位/机械限位/原本体惯量不变；恢复TGS8/4、dt=.005、decimation4（物理/PD200Hz、策略50Hz）。不生成电机/杆件/球铰，不运行闭合约束/杆端间隙审计。`BSRL_CFG`原始直接PD模板保留给旧模型，任务默认用`with_reduced_ankles(BSRL_CFG)`。
- **控制链**：12D策略目标→踝关节空间PD100/1（不裁目标或请求力矩）→`tau_m=J_nom^-T tau_q`→电机112Nm/26.18rad/s/1250W包络→`tau_q_applied=J_true^T tau_m_applied`→串联关节。名义`J_true=J_nom`；不饱和时还原原PD，饱和时是随姿态变化的双轴耦合力矩域，**不再把单个电机112Nm/250rpm直接当作roll/pitch各轴上限**。高速外力反驱仍允许，超过额定速度禁止同向加速力矩、允许制动；1250W为简化峰值包络，不是已知连续/热限模型。
- **避免双重限幅**：新踝执行器`effort_limit_sim=1e9`、`velocity_limit_sim=1000`仅数值保护；真实执行器限幅只在电机空间。保留原关节armature .01，不额外猜测转子惯量。hip yaw/roll/pitch150/2、knee200/2与其他电机参数不变。
- **应退出/应保留**：显式高迭代、1ms、额外拓扑、杆件复位只保留在显式开关路径；保留按实际物理步推进的执行器时钟修复（新模型也有延迟/可选力矩响应状态），保留12关节作用范围和历史播放恢复；用户已要求的软限位-5、动作变化-.1、二阶平滑-.001、实际加速度-1e-6均不回退。没有改全局IsaacLab、rsl_rl、wbt_est或T1配置。
- **播放兼容**：`bsrl_dynamics_restore.py`按run保存的`parallel_ankles`/`closed_ankles`标记选择新传动/显式闭链，否则恢复旧直接PD；同时恢复dt、decimation、solver/iterations、PD、延迟和新增电机参数。旧run绝不继承当前新增DR。`--nominal_eval`关闭传动DR以及原有DR。标量PD保存/恢复也支持，不只支持正则字典。
- **新增针对性DR（已实现，默认关闭）**：通过`BSRL_ANKLE_TRANSMISSION_DR=1`启用startup事件。以下是待硬件标定的保守假设范围，并非rosbag已测得的参数；左右脚/每个电机/各环境独立采样，参数在episode内固定，不逐帧乱跳。现有质量/摩擦/PD/armature/延迟随机化照旧；不加Actor特权观测。

| 新增误差 | 可选范围 | 实现与边界 |
|---|---|---|
| 传动模型比例误差 | 每电机0.97–1.03 | 控制器仍逆`J_nom`，仿真实际模型`m_true=diag(ratio)m_nom(q)`、`J_true=diag(ratio)J_nom`；实际速度和回传力矩均用`J_true`，保证虚功一致；不会前后用同一受扰动J而相互抵消 |
| 电机输出能力误差 | 每电机0.90–1.00 | 输出和力矩/功率上限按同一能力比例缩放，绝不超过原112Nm/1250W；不等同于PD增益随机化 |
| 电机力矩一阶响应 | 时间常数0–5ms | 精确离散一阶状态，真实物理步更新、局部reset清零；滤波后再次执行电机包络，避免旧力矩在超速后继续驱动；不同于原0–20ms目标通信延迟 |
| 编码器偏差、回差、弹性 | **未默认加入** | 需要测量范围/刚度/阻尼/滞回；原default-position偏置不等同真实编码器误差。以后若加编码器偏差，PD与Actor测量须一致，Critic保留真值；回差需要有记忆的模型，不可用白噪声替代 |

- **泛化边界**：等效传动表达姿态相关解算与电机限制，新增DR表达低阶模型/力矩响应不确定性，不能重现完整连杆惯性、杆弯曲、真实闭环结构模态或保证严重机械松动可由学习补偿。先保留名义传动实验，再独立打开DR做对照，避免一次混入两个主要变量。
- **新正式训练**：run`2026-09-15_15-14-55_reduced_parallelankle_nominal_pd200_noestimator_mirror01_formal1_8192`，tmux`bsrl_reducedankle_formal1`、物理GPU5、8192×30000、每100轮保存、不带Estimator、镜像损失.1、Actor/Critic归一化仍False。启动脚本`scripts/rsl_rl/launch_bsrl_reduced_ankle_formal1.sh`，日志`logs/launches/bsrl_reducedankle_formal1_20260915.log`；未跑PPO smoke。冻结YAML和主要来源代码/URDF在run的`params/`、`params/source_snapshot/`中。
- **首次启动修正**：IsaacLab初始化调用`update(0)`刷新缓存，不是物理推进；新增dt一致性审计已跳过该调用。首次在环境初始化阶段自动退出，没有PPO更新，失败日志保留`..._init_dt0.log`，重启正式流程。
- **验证**：29项CPU组件测试通过：全URDF范围、名义PD恒等/虚功、电机耦合限幅、DR不抵消、独立/非累积采样、局部reset、响应后重限幅、无效姿态报告、任意关节顺序/大目标不裁剪、初始化update(0)、旧闭链控制时钟和奖励回归。另外11项真实Kit导入/DelayBuffer/随机化事件/三种播放动力学恢复/AMP reset测试通过（其中4项为原有环境回归）。新工厂统一从`bsrl.py`继承PD/armature/摩擦，后续流水线修改只改一处；当前100/1数值未变，未因此重启训练。新增工厂从标量gain扩展为按原配置解析关节gain，播放兼容两种保存形式。
- **正式100轮已通过数值验收（15:20:44）**：当前107轮，108个rollout审计均无无效解算/NaN；`model_100.pt`完整读取、98个浮点tensor全部有限，所有已打印loss/reward/metric有限。电机≤112Nm、≤1250.000122W（浮点误差），踝关节力矩峰值185.54Nm，证明不是roll/pitch各自独立112Nm夹紧。最近20轮2.784s/轮，约比显式36s快13倍；30000轮按此速度约23.2小时，实际随episode变化。平均episode约1.42s，仍主要跌倒，**不代表已经学会走路**。早期探索存在最高243.90rad/s的电机外力反驱速度，超过额定26.18rad/s；执行器禁止超速同向驱动而允许制动，不把数值验收当作硬件速度/持续力矩合规结论。详见`reports/bsrl_reduced_ankle_20260915/README.md`。

### 2026-09-15：修复后补测显式闭链 TGS 8/4，仍未通过

- 用户要求补做之前缺失的低迭代对照。本次真实加载显式闭链（28DOF/21刚体），复位位置/速度一致性修复、物理步PD时钟修复、取消踝电机额定转速硬约束全部保留；运行配置/回调/PhysX失败现场均核验，不能沿用早期未修复时244mm的结果替代本次测试。
- 固定8192环境、seed42、踝PD100/1@200Hz、策略50Hz、动作std1、自碰撞、contact-last与逐迭代外力；TGS8/4分别测试1ms完整DR、5ms完整DR、1ms关闭DR/延迟。关闭DR组仍为随机参考帧复位与随机动作，不是默认站姿录像。

| 修复后TGS8/4 | 初始复位间隙 | 首物理步最大闭合误差 | 结果 |
|---|---:|---:|---|
| 1ms＋完整DR | 0.1302mm | 33.2632mm（env7998） | 失败 |
| 5ms＋完整DR | 0.1302mm | 543.2622mm（env4377） | 失败 |
| 1ms＋关闭DR/延迟 | 0.1189mm | 33.2517mm（env696） | 失败 |

- 三组均触发10mm首越界检查，保存物理现场并结束诊断，未完成首个控制步，更不是100轮PPO。电机额定转速并未重新硬夹紧，数值保护仍1000rad/s；1ms/5ms完整DR现场电机实际施力≤112Nm、passive直接力矩0、状态有限，不能据此掩盖闭合误差。
- 结论仅限当前模型/配置：修复后不能直接用这套TGS8/4，关闭DR也不能解决；不泛化为所有TGS/闭链都不可行。没有修改现有训练或默认等效传动配置，没有停止旧PGS或T1，没有增加任务。8项几何/闭链执行器回归通过。
- 配置、完整log、原始failure.pt和源码快照在`reports/bsrl_closed_ankle_tgs84_fixed_20260915/`。诊断脚本补齐冻结配置与初始连接点位置/速度误差打印。注意本机Kit异常清理仍可能返回shell code0，验收必须检查COMPLETE、异常日志和物理审计，不能只看退出码。

### 2026-09-15：停止旧PGS闭链，参考 wheeled-legged_RL 重新隔离测试

- **按用户要求停止** `bsrl_closedankle_pgs_formal1`，最后完成iteration131；`2026-09-15_14-36-56_explicit_closedankle_pgs128v16_clockfix_pd200_noestimator_mirror01_formal1_8192`中的model_0/model_100与所有日志保留。解析传动BSRL及全部T1训练未停止。没有新开PPO或新注册任务，没有改生产默认。
- **修正前一节结论边界**：旧TGS8/4首物理步33mm只是触发我们自设10mm门槛，不等于引擎当场发散。新诊断逐子步记录误差分布/持续时间，允许继续观察，reset截断与真实自行恢复分开统计；非有限、奇异、超过电机包络、巨大间隙或极端速度仍保存现场并停止诊断，不做运行时姿态投影。
- **固定真实条件**：8192环境、完整DR/参考帧复位、随机action std1、PD200Hz/策略50Hz、踝100/1、112Nm/1250W电机侧限幅、1g辅助体、原位置/速度reset与PD时钟修复；除明确单因素对照外保留自碰撞。物理测试不是已训练策略，跌倒会按原episode逻辑复位，不把“跑完”当成学会走路。

| 同seed42、2秒诊断 | 全程最大闭合误差 | 环境×物理步中>10mm比例 | 最长连续>10mm |
|---|---:|---:|---:|
| TGS8/4、1ms、逐迭代外力True | 42.594mm | 14.462% | 90ms |
| TGS12/6、1ms、逐迭代外力True | 20.715mm | 2.821% | 55ms |
| TGS12/6、1ms、仅关闭contact-last | 20.683mm | 2.812% | 69ms |
| TGS12/6、1ms、仅关闭逐迭代外力 | 2.725mm | 0 | 0 |
| TGS12/6、1ms、两个开关均False | 2.281mm | 0 | 0 |
| TGS8/4、1ms、仅关闭逐迭代外力 | 4.742mm | 0 | 0 |

- **关键隔离结果**：此前漏测`enable_external_forces_every_iteration=False`，固定True后主要加迭代的排查不充分。此选项在当前TGS闭链模型中显著影响闭合误差；不是断言该PhysX功能普遍错误。PGS本身忽略此选项。
- **没有照搬成功的条件**：5ms/TGS12/6关闭两个开关仍在10ms仿真时出现526mm间隙与约693rad/s主动关节速度；关闭自碰撞首两步无改善。照搬球铰Kd=.01/armature=.0001也未解决，且两者已读回确认实际生效，未写入生产默认。2.5ms/12/6能完成2s但峰值28.776mm、超10mm占比0.063%；提高到32/8的2.5ms/5ms组出现极端速度，不把更多迭代当作必然改善。
- **10秒延长复核**：8/4、1ms、逐迭代外力False、seed42完成10000物理步，最大4.742mm；12/6、1ms、两个开关均False、seed43完成10000物理步，最大9.626mm（6.409s时1个环境>5mm，该帧P99仅0.932mm）。均0次>10mm、无NaN/奇异/电机包络违规。seed不同，不能直接按这两个峰值给迭代配置排名；9.626mm接近原门槛，仍需后续观察。
- **速度检查**：恢复原生产审计、不装诊断recorder，8192环境×24控制步：旧PGS128/16为36.150s，新TGS12/6为7.833s，纯采样约快4.62倍，且均通过原10mm审计。不包含PPO网络推理/更新；不是正式训练速度承诺。长测主动关节/电机反驱峰值仍达211.73/358.86rad/s，不能把数值通过当作实机速度安全结论。
- **下一轮候选**：TGS12/6、dt=.001、PD200Hz、策略50Hz、`solve_articulation_contact_last=False`、`enable_external_forces_every_iteration=False`；自碰撞、PD、电机限幅、质量和原修复保持。仅记录候选，**当前未新开正式训练，也未替换解析传动默认**。
- **诊断工具**：新增持续误差统计及reset不冒充恢复测试，10项CPU回归和语法检查通过。并行相同seed导致Isaac默认临时USD目录碰撞，诊断入口改为独立`mkdtemp`目录后重跑；没有修改全局IsaacLab或生产导入器。完整16组有效物理对照、2组原审计计时、失败现场、源码快照和复现命令见`reports/bsrl_closed_ankle_repo_comparison_20260915/README.md`。

### 2026-09-15：正式启动 TGS12/6 显式闭链，闭合误差仅监控

- **用户明确授权**：TGS12/6、两个求解开关False、PhysX1ms、PD200Hz；保留已有修复和其他训练条件，不对闭合误差设硬停止阈值，注释解析传动随机化。
- **run**：`2026-09-15_16-30-14_explicit_closedankle_tgs12v6_phys1ms_pd200_gapmonitor_noestimator_mirror01_formal1_8192`；tmux `bsrl_closedankle_tgs126_formal1`，物理GPU4，8192env×30000轮，seed42，从头训练、每100轮保存。沿用唯一BSRL任务，无Estimator、Actor/Critic均不归一化、镜像损失0.1、AMP scale5/task-style混合与此前一致。不跑PPO smoke。
- **动力学**：`solver_type=1`、位置/速度迭代12/6、`enable_external_forces_every_iteration=False`、`solve_articulation_contact_last=False`、dt=.001、decimation20；全部直接驱动与踝PD/延迟每5个物理步更新，策略50Hz。保留复位位置与速度闭合、纯write/forward不推进PD/延迟时钟；电机侧112Nm/1250W/额定26.18rad/s超速禁同向驱动允许制动、1000rad/s数值保护。roll/pitch直接力矩0，自碰撞与原12关节/原本体范围的材料、质量/惯量、COM、PD、armature、摩擦、零位、推扰随机化不变。
- **闭合误差不再硬限制**：生产审计的`or gap > .01`留作注释，不因有限间隙触发停止、reset或额外奖励。逐物理步累积并每rollout打印`closure_max_mm`、`closure_alltime_mm`、`closure_mean_mm`、`closure_over10_percent`、`gap_monitor_only=1`；mean/占比基于每环境4条约束间隙最大值。非有限状态、Jacobian奇异、passive直接施力、电机力矩/功率违规保护继续生效。监控脚本识别新日志标志，同时保留历史run的10mm验收语义。
- **解析传动随机化已注释**：ratio/strength/response的EventTerm构建、环境变量启用表达式和相关导入留作注释，`events.ankle_transmission=None`；解算与历史播放恢复函数保留。没有把数值间隙当作可标定、独立同分布的真实机构误差，也没有运行时投影隐藏间隙。用户本轮接受有限残差作为鲁棒性实验的一部分，仍需看误差与动作/接触的相关性和策略是否利用误差。
- **实加载核对**：新旧env.yaml仅求解器/迭代/两个flag、ankle_transmission显式null和log_dir有变化；agent.yaml只有run_name变化。25项CPU回归通过（含500mm有限间隙仅记录不拒绝、NaN/奇异/力矩超限仍拒绝、dt0不推进审计、旧新monitor语义）；语法检查和diff检查通过。
- **启动观察**：已打印第14轮、15个rollout审计正常；model_0.pt完整读取、98个浮点tensor全部有限，约8.1s/轮，当前全程最大闭合误差3.147mm，超10mm比例0，已打印loss/reward/metric均有限，无NaN/奇异/包络违规；早期策略仍频繁跌倒，不能当成已学会走路。源码/URDF快照、启动脚本和记录见`reports/bsrl_closed_ankle_tgs126_formal1_20260915/`。现有解析传动BSRL以及全部T1任务未停止。

### 2026-09-15：昨日普通串联 PD 对照完成，按新 PD 做 MuJoCo 回放

- **已完成训练**：`2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192` 完成 30000 轮，最终 `model_29999.pt`。这是普通串联关节执行器训练，不是当天新增的解析传动或显式闭链训练。
- **导出与 PD**：严格加载该 checkpoint 的完整 ActorCritic 权重，复用 AMP ONNX exporter，生成该 run 的 `exported/policy.onnx`；输入240D、输出12D，无Estimator、Actor/Critic均不归一化。PD从该run冻结env.yaml读取：髋yaw/roll/pitch `150/2`、膝 `200/2`、踝pitch/roll `100/1`；没有覆盖旧run/ONNX或修改其他控制条件。
- **频率是有意保留的差别**：用户确认 Isaac PD200Hz、MuJoCo PD1000Hz；回放继续物理1ms/策略50Hz。沿用 `parallel_ankle/replay_policy.py` 的CAD闭链，并额外保留同PD串联对照；固定 `[0.6,0,0]`、默认静止姿态、无观测噪声/随机推力/随机延迟/自动reset。原XML、接触、辅助体、限幅逻辑及训练进程不改。
- **结果**：两者均完成30s（1500策略帧、30000物理步），无摔倒、NaN/Inf、MuJoCo warning。闭链/串联平均body vx分别 `0.686/0.684 m/s`，最终yaw分别 `107.26/107.30°`，因此仍明显走弯，不能把不摔倒等同于走直。闭链最大间隙 `0.141 mm`、踝电机力矩峰值 `28.731 Nm`，未触及电机包络。
- **文件**：`videos/bsrl_pd150_knee200_ankle100kd1_model29999_closed_sim2sim_forward06_30s_20260915.mp4`、`videos/bsrl_pd150_knee200_ankle100kd1_model29999_serial_sim2sim_forward06_30s_20260915.mp4`。原始轨迹/manifest位于兄弟目录 `parallel_ankle/results/20260915_pd150_knee200_ankle100kd1_model29999_{closed,serial}_forward06_30s/`；详情与复现命令见 `reports/bsrl_pd_serial_checkpoint_sim2sim_20260915/README.md`。

### 当前训练参数表

| 类别 | 当前值 |
|---|---|
| 环境与控制 | 默认原串联 `export.urdf`＋并联踝解析电机限幅，显式闭链开关False；hip roll `±0.3 rad`、knee `[-0.1,2.2] rad`；8192 env；TGS8/4、物理/PD200Hz（dt=.005）、策略50Hz（decimation4）；episode20s；12维位置动作、scale .25；延迟0–4个PD周期（0–20ms） |
| 显式闭链（GPU4新正式训练） | 新TGS12/6、物理1ms/PD200Hz/策略50Hz、逐迭代外力False/contact-last False，有限闭合误差只监控，不设停止阈值。roll/pitch被动、`J⁻ᵀ`解算仅驱动4个motor hinge，电机包络/本体惯量/新增1g辅助体不变；仍需显式开关启用。旧PGS128/16已停于131轮、model_100保留；解析传动GPU5继续对照 |
| 名义 PD（2026-09-14） | 左右 hip yaw/roll/pitch `150/2`；knee pitch `200/2`；ankle pitch/roll `100/1`；保留 Kp/Kd `×0.8–1.2` 随机化 |
| PD 目标角限幅（仅后续新环境） | `joint_pos.clip=None`、`clip_actions=None`；目标为 `default_pos + action × scale`，不按 URDF 或 `±100 rad` 裁剪；机械限位、执行器限制不变；旧裁剪代码保留注释 |
| 指令与课程 | 每 `10 s` 重采样，`2%` 站立，`heading_command=False`；第 `0/5000/12000` 轮切换速度范围：`vx ±0.6/0.9/1.2`、`vy ±0.4/0.6/0.8`、yaw `±0.25/0.375/0.5` |
| 参考动作 | `walk_and_run` 中的 PKL 按文件等权采样；AMP 使用连续 `4` 帧 |
| Rollout 与保存 | 每环境每轮 `24` 步；默认 `30000` 轮；每 `1000` 轮保存；日志根 `bsrl_amp` |
| PPO | epochs `5`、mini-batches `4`、LR `1e-4 adaptive`、clip `0.2`、gamma `0.99`、lambda `0.95`、desired KL `0.01`、max grad norm `1.0`、entropy `0.01` |
| 网络 | Actor/Critic `[512,256,128]` ELU；初始动作 std `1.0`；普通基准关闭 Actor/Critic 观测归一化；新训练使用 `--use_estimator` 时启用 `[256,128]` ELU Estimator，并分别归一化 Estimator 输入、增广 Actor 输入和 Critic 输入；历史 checkpoint 播放时恢复其冻结开关 |
| AMP | 判别器 `[1024,512]` ELU；LR `1e-4`；replay buffer `100`；gradient penalty `10`；trunk/linear weight decay `1e-4/1e-2`；LSGAN |
| 随机化 | 材料静/动摩擦 `0.5–1.5`、质量/惯量、Kp/Kd、armature 均 `×0.8–1.2`；零位 `±0.05 rad`；COM x `±0.025 m`、y/z `±0.05 m`；关节摩擦 `0–0.05`；每 `5 s` 推动 base XY 速度 `±0.2 m/s` |
| 可选传动DR（本轮关闭） | startup独立采样：实际传动比例0.97–1.03、电机输出能力0.90–1.00、力矩响应时间常数0–5ms；`BSRL_ANKLE_TRANSMISSION_DR=1`单独开启，不影响旧进程，不增加Actor观测 |
| Reset / 终止 | 从参考动作随机帧复位；20 秒超时、base 接触、base 高度 `<0.5 m` 或倾角 `>60°` 终止 |

## 当前问题与风险

- **已修正：Critic 缺少奖励所需状态。** 2026-09-01 起，base height、双足接触/计时/世界系 XY 速度、非足部接触计数和 base 接触标志只作为 Critic 特权输入；Actor 与 4D Estimator 合同不变。

- **已修正：GMR 参考动作风格污染**。IK 已加入默认姿态、hip yaw/roll 和 root-frame 足宽正则，全量数据的足宽及贴髋限位比例显著下降；本轮训练将验证这种数据修正能否转化为更紧凑的策略步态。
- **已修正：air-time 奖励曾偏向长单支撑**。`feet_air_time_positive_biped` 在单支撑期间按持续时间给分，它并不是步频奖励；现已从权重 `1.0`、阈值 `0.5 s` 改为宇树 AMP 的 `0.5`、`0.4 s`。
- **高优先级：训练复位与部署起步分布不一致**。训练全部从随机参考帧及其速度开始，播放/部署却从默认静止姿态开始；随机参考帧还与当次随机速度指令无条件配对。它不会破坏训练，但会造成开局弹跳、先调整平衡或岔腿的分布外行为。
- **待观察：横向指令仍可能放大支撑宽度**。最终范围已收窄到 `vy=±0.8 m/s`、`vx=±1.2 m/s`，但横向速度仍占前向上限的三分之二且均匀采样；本轮需要用足宽统计判断它是否仍会鼓励宽支撑和跨步。
- **已修正：base 复合质量随机化**。base `-1～+3 kg` 加法事件已关闭，当前只保留全身质量及惯量 `×0.8～1.2`。
- **中优先级：高方差和执行器饱和需要监控**。历史最终策略 std 约 `3.35`，乘 `0.25` scale 后单关节探索标准差约 `0.84 rad`，远大于 hip roll 等关节行程。2026-09-14 按用户要求，新环境不裁剪 PD 目标角；这是一项允许预载/更大力矩请求的控制选择，不等于取消机械限位。是否有问题应结合实际姿态、饱和比例和奖励判断，不能把目标超限本身当作 bug。
- **已修正：默认训练轮数与实验名不一致**。agent cfg 现默认 `max_iterations=30000`、`experiment_name=bsrl_amp`。
- **已排除：延迟首步零目标**。IsaacLab `DelayBuffer` 在 reset 后第一次 append 会用当前目标填满该环境的历史槽，不会先输出若干步全零 PD 目标。
- **已排除：当前速度坐标系、足部 link 与左右顺序错误**。这些链路目前一致；30 Hz 参考动作会按时间插值到 50 Hz 控制时刻，不是直接丢帧，虽然重新生成 50 Hz 数据仍有利于验证速度和步频细节。

## 下一轮建议：小幅度、高步频、便于部署

- **先评估本轮正则化数据训练**：GMR 数据修正、统计和人工播放确认已经完成。本轮应固定观察 root-frame 足宽、hip yaw/roll 贴限位比例、默认姿态起步和分档速度响应，再决定是否需要额外姿态奖励，避免在数据与奖励两侧同时继续加约束。
- **保留已经验证正确的核心**：新 URDF/USD、贴地默认姿态、base-frame 速度奖励、原生 ActorCritic、entropy `0.01`、不裁 raw action、自碰撞和跌倒终止均先保留。
- **实验 A，只恢复姿态约束**：将 `joint_pos_penalty` 从 `0` 恢复为 `-0.3`。这是历史上已经验证过的折中点：`-1.0` 幅度过小，`-0.3` 能稳定移动且明显比 `0` 更容易限制大幅岔腿。
- **实验 B，只收窄横向课程**：在 A 的基础上把三阶段 `vy` 改为 `±0.3/±0.4/±0.5 m/s`，先不改变 `vx`。如果部署主要前向行走，还应采用分层采样，让前向和低速档的样本多于后退/横移，而不是在大矩形范围内完全均匀采样。
- **实验 C，只减弱长单支撑偏好**：在 B 的基础上把 air-time 从权重 `1.0`、阈值 `0.5 s` 降到约 `0.5`、`0.3 s`。这只能移除低频大步的奖励偏置；若仍需精确控制 cadence，再增加经过测试的接触相位/步频项。
- **实验 D，再处理部署起步**：动作风格稳定后，单独测试 `10%～20%` 默认静止姿态复位，并确保其初始指令为站立或低速前向，再逐步放开。8 月 9 日的 `20%` 默认复位失败同时混入动作裁剪和软限位，不能据此否定默认姿态复位本身。
- **鲁棒性最后回加**：先用 delay `0` 和精简随机化确认能学到紧凑步态，再依次加入 `0～4` 延迟、质量/COM、PD 和材料随机化。直接用完整 DR 学风格，容易让策略优先选择宽支撑的保守解。
- **新增部署指标**：除速度误差外固定记录 root-frame 双脚横向间距的 mean/p90/max、hip yaw/roll 贴限位比例、每条腿步频、动作/力矩峰值以及默认静止起步成功率。这样可以在 TensorBoard 阶段发现“奖励提高但步态更难部署”。

## 2026-09-15：普通串联 BSRL 关闭 mirror 的单变量对照

- 用户要求串联、不加 mirror，再从头训练用于对比。严格基线为 `2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192`，不是解析传动或显式闭环并联版本。
- 唯一训练改动：`agent.algorithm.symmetry_cfg=None`；不使用 mirror loss、对称增广或仅记录用的 mirror 前向计算。其余冻结环境参数逐项恢复：12 个直接 PD 驱动串联关节、原 PD/电机限制/动作裁剪/DR、奖励、课程、动作数据及 PPO/AMP。
- 本轮没有加入初始航向坐标系速度奖励，没有改变指令采样比例；Actor 不增加 heading，仍为 240D；Critic 总输入 350D，判别器 156D；无 estimator/Actor-Critic normalization。
- 运行目录：`logs/rsl_rl/bsrl_amp/2026-09-15_19-55-26_serial_pd_hip150_knee200_ankle100_noestimator_nomirror_formal1_8192`。GPU 7，8192 env × 30000 轮，seed 42，从头训练，每 1000 轮保存；tmux `bsrl_serial_nomirror_formal1`，启动 PID 912657。
- 启动脚本 `scripts/rsl_rl/launch_bsrl_serial_no_mirror_formal1.sh`；日志 `logs/launches/bsrl_serial_nomirror_formal1_20260915.log`；唯一任务仍为 `LeggedLab-Isaac-AMP-BSRL-v0`。新增显式参数 `--bsrl_serial_no_mirror_baseline`，不改默认串联/并联分支，不影响已运行进程。
- 验收：新旧 `params/env.yaml` 除 log_dir 外完全一致；agent 除 run_name 与 symmetry_cfg 外完全一致。`model_0.pt` 首层 Actor `(512,240)` / Critic `(512,350)`，参数有限、无 estimator；已进入第 25 轮，PPO/AMP loss 有限，没有 mirror loss 日志。源 YAML SHA256 与审计在 run 的 `params/serial_no_mirror_audit.yaml`。
- T1 新的无 Actor heading 训练及其他所有原进程均保留；未修改 URDF/XML 或真机代码。先完成 GPU 7 上的 T1 历史策略录像，录像进程退出后才在该 GPU 启动本轮训练。

## 2026-09-16：接近完成策略的闭链 sim2sim

- 用户要求检查快训好的BSRL AMP。固定解析传动run `2026-09-15_15-14-55...` 的model26300、串联无mirror run `2026-09-15_19-55-26...` 的model24000，并重跑9/14 17:51普通串联mirror model29999作参照。显式闭链TGS训练当时约8343/30000，尚未接近完成，本次未评估。全部训练保持运行。
- 同一恢复CAD闭链XML，0.6m/s前进30s，物理/PD/策略1000/1000/50Hz，原名义PD150/2、200/2、100/1，无随机化/延迟/自动reset；Actor240D，无heading和estimator。独立CPU导出，不覆盖旧ONNX。
- 三组均完成1500策略帧/30000物理步，无摔倒/非有限值/MuJoCo warning。无mirror串联、解析传动mirror、旧串联mirror的末采样累计yaw分别为-6.02°、+87.05°、+107.26°；平均body vx为.667/.682/.686m/s；最大闭合间隙.145/.168/.141mm；踝电机未触发包络限幅。无mirror本次走得最直，最终横向偏移-.187m。
- 回放兼容解析传动冻结配置时，明确读取motor_effort/speed/power，不把1e9Nm/1000rad/s数值保护当真实电机上限；各run原目标裁剪条件保留。69项CPU回归通过、1项跳过；三份30s视频均完整解码，实际动作PyTorch/ONNX最大误差≤1.20e-6。
- 只代表当前checkpoint与该单一名义工况。无mirror/mirror本次轮数不同；解析传动还混有smoothness、关节限位惩罚等训练差异，不能据此证明mirror或机构是偏航原因。完整视频、轨迹、配置差异、模型来源及复现命令见 `reports/bsrl_nearfinal_closed_sim2sim_20260916/README.md`。

## 2026-09-16：启动解析并联传动的航向参考系速度奖励实验

- 用户要求用“并联解算、不闭链”新训；基线为9/15 15:14解析传动run，不是普通直接串联或TGS显式闭链。新run：`2026-09-16_10-51-46_reduced_parallelankle_headingframevel_criticpriv_mirror01_noactorheading_formal1_8192`；GPU3，8192×30000，seed42，从头训练，每100轮保存，tmux `bsrl_reduced_heading_formal1`，启动PID1480372。
- 严格恢复基线冻结环境/agent，仅加入权重1.0/std.4的航向参考系XY速度奖励及对应Critic观测。mirror=.1、Actor240D无航向/无estimator、原解析传动12关节骨架、PD/物理200Hz、策略50Hz、随机化/参考复位/奖励其余项/课程/PPO/AMP均保持。
- 与T1零转速门控项不同：原BSRL连续均匀采样转速，直接用.001门槛会使新增项极少覆盖运动指令。本轮事先说明后选择不改采样，用“复位yaw＋期望yaw-rate积分”的参考系覆盖全部指令；零转速时保持初始朝向，有转向指令正常旋转参考系，不跟随实际身体漂移，也不改变给Actor的速度命令。
- Critic新增参考朝向误差sin/cos2D和参考系真实XY速度2D，各5帧，总输入350→370D，Actor仍240D；新参考状态按物理步推进，奖励/观测重复访问不重复积分，复位后捕获真实yaw，重采样先结算旧命令的间隔。
- 65项相关回归通过；初始化前后配置审计通过；实际env.yaml仅新增上述项及元数据，agent仅run_name变化。12个关节、只有large_motors/small_motors/parallel_ankles；model_0 Actor(512,240)/Critic(512,370)且全部有限。观察至第30轮，镜像和新奖励均生效、无传动invalid/nonfinite；早期仍频繁跌倒，不代表已学会走直。
- 未停止其他训练、未改全局默认训练分支、未改Actor部署接口。来源、语义、限制和脚本见 `reports/bsrl_reduced_heading_formal1_20260916/README.md`，日志 `logs/launches/bsrl_reduced_heading_formal1_20260916.log`。

## 2026-09-16：ACCAD34 解析并联 mirror 双训练

- 用户要求模拟并联、无estimator，分别关闭/开启mirror从头训练两组。使用最新9/16 10:51解析并联航向参考系奖励run为冻结基线，保留航向奖励及critic观测，Actor240无heading、Critic370、disc156；12关节解析传动，不闭链。
- 无mirror：`2026-09-16_21-06-48_accad34_reduced_parallel_headingpriv_noestimator_nomirror_8192`，GPU1/PID1923791，tmux `bsrl_accad34_nomirror`。mirror0.1：`2026-09-16_21-07-06_accad34_reduced_parallel_headingpriv_noestimator_mirror01_8192`，GPU5/PID1924636，tmux `bsrl_accad34_mirror01`。均8192×30000、seed42、每100轮保存，不恢复旧checkpoint。
- 数据34段/10896帧/181.6s，完整ACCAD参考候选集，两组等权、clamp、不裁剪不平滑；训练格式wxyz/BSRL交错左右关节顺序，新grounding约定显式保留，旧训练默认和旧数据未改。8段角速度复查项、跑步接触几何风险保留并披露，不能将训练启动成功视为数据已清洗。
- 首次同秒初始化触发临时USD冲突与新增disc形状检查误判，均在PPO开始前退出；已修正并错开初始化重启。实际两份env只差log_dir，agent只差run_name/symmetry_cfg；分别观察到93/78轮，model0参数有限，无estimator，mirror仅对照组生效，传动invalid/nonfinite为0。57项测试通过，其他旧训练保持运行。
- 详情：`reports/bsrl_accad34_mirror_pair_20260916/README.md`；启动脚本 `scripts/rsl_rl/launch_bsrl_accad_mirror_pair.sh`；日志 `logs/launches/bsrl_accad34_{nomirror,mirror01}_20260916_retry1.log`。

## 跨实验结论

- **一次只改一个高影响变量**：延迟和高速指令曾同时加入，raw action 裁剪、目标限位、默认姿态复位也曾同时加入，导致无法区分贡献。后续应采用单变量消融。
- **总奖励和 episode 长度不能代表学会移动**：策略可能通过站立和存活提高总奖励。必须同时看 `Metrics/base_velocity/error_vel_xy`、速度奖励、终止分布和实际播放。
- **动作裁剪与熵奖励存在耦合**：只在环境入口裁动作，会让 PPO 无法感知裁剪外差异，而熵仍可通过增大 std 获益。当前最终可用版本选择恢复宇树方案：不裁 raw action、使用原生 ActorCritic；“有界 std + 动作裁剪”在现有组合下未训练成功。
- **区分 episode 终止与训练进程中断**：基座过低和姿态异常只会 reset 单个环境。8 月 5、6 日两次提前停止均由 PPO 数值发散导致，应检查 checkpoint、`Loss/value`、`Loss/learning_rate` 和 NaN/Inf，而不能只看 episode 长度。
- **低速响应与高速能力要分别评价**：`vx=1.0 m/s` 能启动不代表 `vx=0.6 m/s` 能慢走。固定速度分档播放比单一 mixed 视频更能发现启动死区。
- **速度奖励 std 会改变静止容忍度**：当 std 为 `0.5` 时，静止面对 `0.6 m/s` 指令仍可获得约 `0.71` 的加权线速度奖励，而 `1.0 m/s` 时仅约 `0.055`。低速严格跟踪需要调整 std 或低速采样分布。
- **慢走数据不会自动建立“指令到风格”的条件映射**：`walk1_subject1.pkl` 的慢走帧充足，但 AMP 判别器不观察速度指令或根线速度，walk/run 又无条件混合采样；actor 也没有直接根线速度观测。AMP 只能约束动作像数据集中的某种运动，低速指令与慢走风格的对应主要依赖任务奖励。
- **贴限位参考数据是控制风险项**：当前精确复核显示 hip roll 约有 `28%～40%` 帧贴 `±0.15 rad`，hip yaw 约有 `11%～18%` 帧贴近所选方向的 `0.4 rad` 限位。90% 软限位已确认不能覆盖这些状态；硬限位目标是否因缺少预载力矩而影响训练仍是待独立验证的假设。
