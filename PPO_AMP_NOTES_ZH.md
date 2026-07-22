# PPO 与 AMP 复习笔记

这份笔记写给“已经学过，但一段时间没看细节”的自己。  
目标不是替代教科书，而是帮你在做项目、看日志、调参数时，快速想起：

- PPO 在优化什么
- AMP 比 PPO 多了什么
- 为什么 AMP 更容易只会站、不愿走
- 日志里那些量到底该怎么理解

本文尽量结合你当前这个项目：

- `/root/Desktop/Bipedal_AMP`
- `Isaac Lab + forked rsl_rl + AMP`

---

## 1. 先记住一句话

### PPO

> PPO 是一种“限制每次策略更新不要太猛”的策略梯度方法。

### AMP

> AMP = PPO 的任务学习 + 一个“动作像不像参考数据”的判别器奖励。

所以：

- PPO 只学“完成任务”
- AMP 同时学“完成任务”和“动作风格像参考数据”

---

## 2. PPO 到底在学什么

在强化学习里，我们想让策略 $\pi_\theta(a\mid s)$ 最大化长期回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \gamma^t r_t \right]
$$

其中：

- $s$：状态
- $a$：动作
- $r_t$：奖励
- $\gamma$：折扣因子

但是直接暴力更新策略容易不稳定，所以 PPO 的核心思想是：

> 允许策略变好，但不允许一次改太多。

---

## 3. PPO 的核心式子

PPO 最经典的是 clipped objective：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
 r_t(\theta)\hat A_t,\;
 \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t
\right)
\right]
$$

其中：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

$\hat A_t$ 是 advantage，表示：

> 这一步动作到底比“平均水平”好多少。

### 直观理解

- 如果新策略比旧策略只变一点点，那就正常更新
- 如果新策略偏得太狠，就把更新裁掉

这就是 PPO 的“保守更新”。

---

## 4. PPO 里 value function 是干什么的

PPO 一般是 actor-critic：

- actor 负责输出动作分布
- critic 负责估计状态价值 $V(s)$

优势函数常写成：

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

意思是：

- 如果这个动作比当前状态的平均预期更好，$A_t > 0$
- 如果更差，$A_t < 0$

实践里 PPO 常用 GAE：

$$
\hat A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 记忆版理解

- `critic` 不是为了控制机器人
- 它是为了帮助 actor 更稳定地知道“这一步到底值不值”

---

## 5. PPO 的总损失通常长什么样

一个常见的 PPO 总损失可以写成：

$$
L_{\text{PPO}} = L_{\text{policy}} + c_v L_{\text{value}} - c_e H(\pi)
$$

其中：

- `policy loss`：希望策略变好
- `value loss`：让 critic 估值更准
- `entropy`：鼓励探索，不要过早变得太确定

### 熵项的直觉

如果熵太低：

- 策略太早“自信”
- 容易陷入局部最优

如果熵太高：

- 策略一直乱试
- 学不稳

---

## 6. 你在训练日志里看到的 PPO 指标怎么理解

### `Mean reward`

- 越高通常越好
- 但必须结合 reward 构成看
- 单看一个总 reward 容易被误导

### `Mean episode length`

- 变长通常说明更不容易摔
- 但不代表任务就学好了
- 很可能只是“会站住了”

### `value loss`

- 反映 critic 学得好不好
- 不是越小越绝对好，但太大通常说明估值不稳

### `surrogate loss`

- PPO 的策略目标
- 绝对值小不一定坏
- 要结合 reward 趋势看

### `entropy loss` / `mean action noise std`

- 反映探索程度
- 太低：容易卡住
- 太高：动作可能太乱

---

## 7. PPO 为什么常常比 AMP 更容易收敛

因为 PPO 只需要解决：

> “怎么做能拿更高任务奖励？”

而 AMP 需要同时解决：

> “怎么做能拿更高任务奖励？”  
> “怎么做还得像参考动作？”

这让优化目标变得更复杂，也更容易互相冲突。

---

## 8. AMP 到底比 PPO 多了什么

AMP 的关键额外部分是：

1. 判别器 $D$
2. 参考动作数据 `demo`
3. style reward

### 核心思想

判别器学习区分：

- `demo` 里的参考动作片段
- 当前策略产生的动作片段

如果判别器觉得策略动作“像 demo”，就给策略更高的 style reward。

所以 AMP 不是直接监督学习，而是：

> 用一个对抗式判别器，把“像不像参考动作”转成奖励。

---

## 9. AMP 的直观结构

可以把 AMP 理解成：

$$
r_t^{\text{total}} = (1-\alpha) r_t^{\text{task}} + \alpha r_t^{\text{style}}
$$

这里：

- $r_t^{\text{task}}$：任务奖励，比如速度跟踪、别摔、动作平滑
- $r_t^{\text{style}}$：风格奖励，由判别器给
- $\alpha$：风格和任务的混合权重

在你这个项目里，对应的是 `task_style_lerp` 这类配置。

### 这就是为什么 AMP 常见现象是：

- 先学会站稳
- 动作“有点像”
- 但并不一定马上会走

因为 style 本身就可能奖励“看起来比较像站姿/基础节律”的解。

---

## 10. 判别器在 AMP 里学什么

判别器不是看单帧就够，而通常看一个短时间窗口：

$$
o_t^{\text{disc}} = [x_t, x_{t-1}, \dots, x_{t-k}]
$$

也就是说，判别器看的是“短时间动态”，不是只看一个姿势。

所以它能区分：

- 一个姿势像不像
- 以及动作变化节律像不像

在你当前项目里，`disc` 和 `disc_demo` 这两组观测就是干这个的。

---

## 11. AMP 判别器的损失在干什么

虽然不同实现细节略有区别，但核心目标都是：

- 让 `demo` 被判成“真”
- 让 policy 产生的 motion 被判成“假”

如果是更抽象地写，判别器在最大化：

$$
\log D(x_{\text{demo}}) + \log(1 - D(x_{\text{policy}}))
$$

实践里还常带：

- gradient penalty
- weight decay

就是为了让判别器更稳，不要太容易爆。

---

## 12. 为什么 AMP 更容易“只会站、不愿走”

这是做双足 AMP 时最常见的现象之一。

原因通常是这几个一起造成的：

1. 任务奖励和风格奖励冲突  
   例如：
   - 任务想让它往前走
   - style 数据却更像原地踏步/低速动作
2. 站立是低风险解
3. 终止条件太宽松
4. 参考数据质量有问题

如果 retarget 数据本身关节语义不对，style reward 反而会拖后腿。

---

## 13. 这就是为什么“会站但不走”不能简单理解为训练正常

如果一个 AMP 策略：

- `episode length` 很高
- `time_out` 很多
- 但 `track_lin_vel_xy_exp` 很低
- `error_vel_xy` 还很大

那更像是在学：

> “怎样安全地活到 episode 结束”

而不是：

> “怎样按照命令稳定行走”

这是你看日志时特别要警惕的局部最优。

---

## 14. 什么时候更像是“没训够”，什么时候更像是“卡住了”

### 更像没训够

- `track_lin_vel_xy_exp` 持续在涨
- `error_vel_xy` 在慢慢降
- `style` 也在涨
- `episode length` 增长同时任务指标也改善

### 更像卡住

- `episode length` 很高
- `time_out` 很高
- 但速度跟踪长期不上升
- 机器人主要就是稳站或小幅乱挪

---

## 15. AMP 为什么这么吃数据

因为 AMP 学的是一个更复杂的问题：

- 既要满足控制目标
- 又要满足动作先验
- 还要让判别器和策略互相博弈

所以它通常比纯 PPO 更吃：

- 总采样量
- motion data 质量
- 训练稳定性

这也是为什么你常看到：

- PPO `1500` 轮看起来已经差不多了
- AMP `1500` 轮还只是早期

---

## 16. 你当前项目里 PPO 和 AMP 的结构差异

在这个项目里，AMP 比 PPO 多出来的核心是：

- `disc` 观测
- `disc_demo` 观测
- `motion_data_manager`
- `animation_manager`
- `amp_discriminator`
- `PPOAMP` 算法

相关位置可以看：

- [amp_env_cfg.py](/root/Desktop/Bipedal_AMP/source/legged_lab/legged_lab/tasks/locomotion/amp/amp_env_cfg.py)
- [amp_cfg.py](/root/Desktop/Bipedal_AMP/source/legged_lab/legged_lab/rsl_rl/amp_cfg.py)
- [rl_cfg.py](/root/Desktop/Bipedal_AMP/source/legged_lab/legged_lab/rsl_rl/rl_cfg.py)

一句话：

- PPO 主要靠环境 reward
- AMP 多了一条 demo/style 学习链

---

## 17. 什么时候要怀疑 motion data

如果出现这些现象，优先怀疑 motion data：

- `style reward` 有值，但动作看起来很怪
- `joint_pos_penalty` 特别大
- 某些关节长期贴边
- 训练容易数值不稳定
- 机器人能站住，但步态很别扭

这通常意味着：

- retarget 坐标系没对好
- joint 正方向没对好
- 默认姿态和数据零位不一致

这类问题会让 AMP 比 PPO 更难训，因为 AMP 会额外被错误的 style 目标拉偏。

---

## 18. 标准差、探索和数值稳定

策略常用高斯分布采样动作：

$$
a_t \sim \mathcal{N}(\mu_\theta(s_t), \sigma_\theta(s_t))
$$

如果 `std` 参数化不好，就可能出问题。

### 直接学 `std`

风险：

- 数值可能被推到负数
- 不稳定

### 学 `log_std`

更常见更稳：

$$
\sigma = \exp(\log \sigma)
$$

这样天然保证：

$$
\sigma > 0
$$

所以你之前碰到的：

- `normal expects all elements of std >= 0.0`

本质上就是策略分布数值出问题了。  
这未必说明任务一定学坏了，但一定说明训练稳定性出问题了。

---

## 19. 训练日志里 AMP 相关项怎么读

### `Mean AMP total reward`

- AMP 风格侧总奖励
- 不是整个训练唯一目标

### `Mean AMP style reward`

- 越高通常表示判别器更认可当前动作像 demo
- 但要警惕“像得不一定对任务有帮助”

### `amp/disc_loss`

- 判别器分类相关损失
- 不是越小越绝对好，要看是否稳定

### `amp/disc_grad_penalty`

- 判别器梯度惩罚
- 太大通常说明判别器比较激进

### `amp/disc_demo_score`

- demo 被判真程度

### `amp/disc_score`

- policy motion 被判真的程度

---

## 20. 一份实用判断模板

看一段训练日志时，可以按这个顺序问自己：

1. 它更会活了吗？
   - 看 `episode length`
   - 看 `time_out`
2. 它更会做任务了吗？
   - 看 `track_lin_vel_xy_exp`
   - 看 `error_vel_xy`
3. 它更像参考动作了吗？
   - 看 `style`
   - 看 `amp style reward`
4. 它有没有学歪？
   - 只会站不走
   - 关节贴边
   - termination 很低但 tracking 也很差

---

## 21. 为什么 AMP 会比 PPO 更依赖 reward 设计边界

因为 AMP 已经有了一股“动作风格梯度”，所以任务 reward 不需要像纯 PPO 那么“什么都自己教”。

但如果 task reward 设计得太强、太硬，也会和 style 冲突。

所以 AMP 的 reward 调法和 PPO 不完全一样：

- PPO：更多靠 task reward 直接塑形
- AMP：task reward 更像“任务边界条件”，style 负责动作风格

---

## 22. 如果我是做双足 locomotion，该怎么理解 PPO 和 AMP 的关系

可以这样理解：

### PPO

像在教机器人：

> “你只要想办法走起来、别摔、按命令跑就行。”

### AMP

像在教机器人：

> “你不仅要走起来，还要走得像这批参考动作。”

所以 AMP 的上限更高，但前提是：

- 参考动作真的靠谱
- retarget 真的靠谱
- 训练稳定性也够

---

## 23. 一页纸记忆版

### PPO 关键词

- actor-critic
- advantage
- clipped objective
- entropy
- 保守更新

### AMP 关键词

- discriminator
- demo motion
- style reward
- task + style
- 比 PPO 更吃数据，也更依赖数据质量

### 看日志时最重要的三件事

- 是否更会活
- 是否更会完成任务
- 是否只是卡在保守局部最优

---

## 24. 和你当前项目最相关的提醒

结合你这段时间的实验，最该记住的是：

1. `会站` 不等于 `会走`
2. `style 有值` 不等于 `motion data 没问题`
3. `episode length 很长` 不等于已经收敛
4. `关节长期贴边` 往往说明 retarget 或 IK config 有问题
5. `AMP` 如果数据不对，会比 `PPO` 更容易被带歪

---

## 25. 最后一句提醒

当你以后再觉得“理论有点糊”时，先别急着看公式，先问自己这两个问题：

1. 这个方法到底在优化什么？
2. 它为什么会学成我现在看到的样子？

只要这两个问题能回答上来，很多调参和诊断都会顺很多。
