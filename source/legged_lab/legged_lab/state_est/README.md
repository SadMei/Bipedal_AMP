# BSRL / T1 estimator：对齐 wbtest run46/47（2026-09-18）

`train.py --use_estimator` 的新训练默认采用 `wbtest` profile。
未使用这个开关的 AMP、latent、正在运行的 clean-AMP 均不变；本次只完成实现和短测，
没有启动新的 estimator 正式训练。

## 对齐内容

- estimator 输入：AMP 自身观测的完整 **3 帧**，每个 term 内按旧→新排列，再拼接各 term。
- estimator：独立输入标准化，MLP **256 → 128 → 8**，ELU。
- 8 维顺序：基座高度、机体系 vx/vy/vz、左/右脚接触概率、左/右脚高度。
- Actor 直接取各 term **当前帧**，拼接 detached 的 8 维估计；不直接输入整个历史。
- Actor：独立标准化，**512 → 256 → 128 → 动作维度**，ELU。
- 接触输出训练时为 logits、送入 Actor 前 sigmoid；监督损失为六个连续维度的整体 mean MSE
  加两个接触维度的 mean BCE-with-logits，系数 1。不是三个分组 MSE 的相加。
- PPO/mirror 梯度不进入 estimator；estimator 由仿真当前帧真值监督。
  真值只在训练标签组，推理和 ONNX 输入不包含这些真值。
- Critic 独立标准化，保留 AMP 原有结构和观测；没有为本次更新扩展 Critic 特权量。

| AMP policy 布局 | estimator/ONNX 原始输入 | Actor 当前帧 | 拼接估计后的 Actor MLP 输入 |
| --- | ---: | ---: | ---: |
| BSRL 12 关节 | 144 | 48 | 56 |
| T1 25 关节、不带 heading_error | 261 | 87 | 95 |
| T1 25 关节、带 heading_error | 267 | 89 | 97 |

T1 普通任务模板仍带历史 estimator 基线的 `heading_error`；此更新**不改变该开关**。
传入基线关闭它时，保持关闭。正在训练的 T1 clean-AMP 仍不带 Actor/Critic heading，
且不使用此 estimator。没有增加 `wz=k×heading_error` 外部纠偏。

脚高度定义严格跟 wbtest 一致：**足部 link 原点 z − env origin z**，不是足底离地间隙。
接触标签采用传感器历史最大合力范数 >1 N，两个 selector 均固定为左→右：
BSRL `link_left/right_ankle_roll`，T1 `left/right_ankle_pitch_Link`。

## 保持 AMP 本身不变

保留 AMP 判别器和 PPOAMP，不使用 WBT 的动作跟踪命令、参考姿态奖励或 PPO 替代 AMP。
奖励、指令采样、参考数据、随机化、PD、关节模型、Critic 历史均不变。
mirror **开关和系数沿用调用者的配置**；仅使既有 mirror 映射同时支持 3 帧/5 帧。
因此这是 estimator 对齐，不是整个 WBT run46/47 的训练设置复刻（那两轮关闭 mirror）。

## 旧模型与使用入口

```bash
# 新 estimator 训练（实际运行前自行指定任务、环境数、轮数等）
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-T1-v0 --use_estimator ...
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-BSRL-v0 --use_estimator ...

# 旧 4 维基线仍可显式选择
python scripts/rsl_rl/train.py --use_estimator --estimator_profile legacy4d ...

# 回放从 checkpoint 旁的 params/agent.yaml、env.yaml 自动恢复版本、观测和网络
python scripts/rsl_rl/play.py --use_estimator --checkpoint /path/to/model.pt ...
```

`models.py:EstimatorActorCritic` 保留旧 4 维布局及旧权重迁移逻辑；
新类为 `wbtest.py:WbtEstimatorActorCritic`。不把旧 checkpoint 隐式转成新 8 维/3 帧网络。
`--estimator_profile` 默认 auto；有意指定与 checkpoint 不同的版本会明确报错。
没有 params 元数据时，auto 会报错，需要提供已知 profile 和相符的环境配置。
恢复的是网络观测契约；历史物理模型/奖励的精确复现仍应使用项目已有冻结配置/动力学恢复入口。

新 ONNX 仍为单输入、内部包含标准化和 estimator；sim2sim/实机必须按新 3 帧 term-major
布局组装输入，不能继续喂旧的 5 帧数组。ONNX 不负责跨调用缓存历史。

## 验证记录

- 专项 CPU 测试 22 项通过；加上 T1/BSRL latent、clean-AMP、冻结配置回归共 71 项通过。
- 验证 term-major 最新帧选择、左右脚/接触阈值/环境高度、真实 wbtest loss 及梯度数值一致性、
  监督/PPO 梯度隔离、特权量不泄漏、标准化独立、4 维加载、新旧 mirror 双重映射恒等。
- 使用 `play.py` 实际导出组合：Isaac JIT + 项目 ONNX exporter，T1/BSRL 数值一致。
- 两台机器人均完成 **128 环境 × 3 次 PPOAMP 更新**，保持普通基线 mirror 开启，
  运行时观测审计通过、checkpoint 参数有限。这只是链路验证，不代表步态训练效果。
- T1 短测：`logs/rsl_rl/men_t1_amp/2026-09-18_18-06-19_wbtest8_h3_smoke128_20260918`。
- BSRL 短测：`logs/rsl_rl/bsrl_amp/2026-09-18_18-06-20_wbtest8_h3_smoke128_20260918`。
- 日志：`logs/launchers/2026-09-18_{t1,bsrl}_wbtest_estimator_smoke.log`。
- 真实配置/权重加载检查通过：T1 9/7 的 `model_29999.pt`（旧 445→449 输入，Critic 595）、
  BSRL 9/1 的 `model_29999.pt`（旧 240→244 输入，Critic 350），以及上述两份新 `model_2.pt`。
  检查包括从任务模板恢复全部网络观测组/网络配置、严格加载、参数有限和推理有限；
  四份均通过，旧权重未被改写。

参考的是 `/root/mjw/wbt_est` 的本地最新文件，而非仅 Git HEAD（新改动尚未全部提交）：

| 参考文件（`source/whole_body_tracking/whole_body_tracking/state_est/`） | SHA256 |
| --- | --- |
| `t1_full_history.py` | `99aad7119b08d2f9d1c1f88f19b7da1857fcae6094441b7fb768e865a2c3c93d` |
| `reference_loss.py` | `af3070aca2afda4e8048236db20f948c184bd8f23a355cadefc362c48bcf72f6` |
| `models.py` | `ab32848000ad6a61a123af406aa311e21e93ca6691e1620e2b7ee7412a59a166` |
