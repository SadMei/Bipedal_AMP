# BABEL + CMU 单文件 BSRL 预览

无需 AMASS 大包或 SMPL-X 模型。按 BABEL 的动作身份从 CMU 原站获取少量
ASF/AMC，用自己的 FK 适配到本地 GMR，输出四类运动学预览。

## 本次输出

- 源数据：`/root/mjw/GMR/motion_data/CMU_BABEL_source/`
- 最终选择：`manifest_selected.json`（初始 `manifest.json` 保留用于审计）
- 最终重定向：`/root/mjw/Bipedal_AMP/reports/bsrl_cmu_babel_20260916_preview/`
- 最终视频：`/root/mjw/GMR/videos/bsrl_cmu_babel_20260916_preview/`
- 按 `straight / turn / sideways / stand` 四目录保存，未生成总览视频。
- 其他同日前缀目录是中间诊断版本，不是最终预览。

| 输出 | BABEL split / ID | CMU | 原始时间段（秒） | 标注依据 |
|---|---|---|---|---|
| 直行 | train / 251 | 07_03 | [0.15, 3.30) | sequence `walk`，再以轨迹筛选直行 |
| 左转/转身 | train / 13202 | 46_01 | [0.82, 2.12) | frame `turn around`，实测正向 yaw |
| 右转 | train / 378 | 133_18 | [1.252, 2.315) | frame `turn right` |
| 右侧行 | train / 11492 | 113_18 | [0.50, 8.40) | frame `sidestep right` |
| 左侧行 | train / 11492 | 113_18 | [9.20, 17.10) | frame `sidestep left` |
| 站立 | val / 6750 | 77_02 | [0.10, 3.10) | frame `stand`；同时有 `look around` |

这里保留了 BABEL 的 train/val 身份，**没有把验证集自动混入训练**。
只把标注当作候选，不把 `walk` 自动等同于直行。初始 07_04 慢走片段的
骨盆 yaw 范围为 18.76°，超过本轮 15° 阈值，已排除。这并不证明它不能用于
其他走路任务。77_02 的后段身体晃动较大，改选前面的稳定站立区间。

## 复现

在 `Bipedal_AMP` 下执行。若已有同名不同输出，脚本拒绝覆盖；新试验应换输出目录。

```bash
/root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/prepare_cmu_babel.py \
  --babel /root/mjw/babel_v1-0_release.zip \
  --output /root/mjw/GMR/motion_data/CMU_BABEL_source \
  --manifest-name manifest_selected.json --previews

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/retarget_cmu_babel_bsrl.py \
  --source /root/mjw/GMR/motion_data/CMU_BABEL_source \
  --manifest-name manifest_selected.json \
  --output /root/mjw/Bipedal_AMP/reports/bsrl_cmu_babel_NEW

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/render_cmu_babel_bsrl.py \
  /root/mjw/Bipedal_AMP/reports/bsrl_cmu_babel_NEW \
  --video-dir /root/mjw/GMR/videos/bsrl_cmu_babel_NEW

/root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/audit_cmu_babel_bsrl.py \
  /root/mjw/Bipedal_AMP/reports/bsrl_cmu_babel_NEW \
  --video-dir /root/mjw/GMR/videos/bsrl_cmu_babel_NEW

/root/miniconda3/envs/isaaclab/bin/python -m pytest -q \
  source/legged_lab/test/test_cmu_babel_retarget.py
```

`MUJOCO_EGL_DEVICE_ID=5` 仅是本次空闲渲染 GPU，复现时应按机器资源选择。
IK 在 CPU 上运行；不启动/停止任何训练进程。

## 数据与时间轴

1. 仅使用用户提供的 BABEL 标注，不自动登录、不代替用户接受许可。
2. CMU 每个被试一份 ASF，每条动作一份 AMC；只下载显式选中的文件及页面。
   首轮所有下载加官方预览约 **5.79 MB**；最终选中数据加预览约 **5.28 MB**。
3. 单位遵循 CMU FAQ：`0.0254 / ASF.units.length` 米。坐标变换为
   X-forward/Y-left/Z-up，保持右手系。
4. 官方记录 07/46/113/133 为 120 Hz、77 为 60 Hz；从页面核对，不统一硬编码。
5. 原始动作时长与 BABEL 差异最大约 28 ms，没有拉伸、重定时或倒放。
   以 AMC 第 1 帧作为 t=0，按秒裁剪，再重采样到 60 Hz。
6. 保留源文件 SHA256、BABEL ID、标注原文、重叠标注、时间戳及官方描述。
   三条代表片段已与 BABEL 官方 30fps 预览做姿态相位目视抽查。
   **同名和时长一致不证明与 AMASS 拟合版逐帧完全一致**；正式训练前应继续核对。
7. 只移除每条片段的初始 XY 原点和初始 yaw。不会逐帧消除转向。

## 重定向与限制

- 沿用本地 BSRL 的 12 自由度、xyz 正向关节坐标模型，不改机器人资产。
- 按腿长缩放人体。为适应机器人脚宽，对过窄的左右踝目标做对称横向分离，
  最小目标踝间距为 0.22 m；实际位移幅度记录在指标中。
- 本轮只用于无腾空的走路/转身/侧步/站立。最低碰撞足底逐帧保持约 0.5 mm
  间隙；站立额外根据左右脚碰撞网格设置双脚支撑高度。
- 行走的支撑相估计仅作诊断，存在不少不确定帧，**不是测力板真值**。
  没有横向脚底锁定，可能仍有滑脚、踝关节限幅、步态变形；不能因为贴地就认为
  数据已经具有动力学可行性。
- 运动学限位、导出/渲染 FK、一脚最低间隙、脚间碰撞距离、视频完整解码均可检查。
  没有验证接触力、平衡、真机安全性或 AMP 训练效果。
- 视频 1440×720，0.5 倍速，三个视图为源骨架/机器人/足部；固定相机方位角，
  XY 轨迹小窗展示真实转向，没有掩盖路径变化。

## KIT 的单文件路线（尚未下载/适配）

已经按 BABEL `train/582` 对应到 KIT Subject 9 的具体动作页：

- 动作页：<https://motion-database.humanoids.kit.edu/details/motions/37/>
- `WalkingStraightForwards05.c3d`：<https://motion-database.humanoids.kit.edu/file_download/714/>
- `WalkingStraightForwards05.xml`（MMM）：<https://motion-database.humanoids.kit.edu/file_download/41039/>

这些入口虽然列为 PUBLIC，实际下载请求会跳转到 **KIT 自己的登录页面**。
需要用户登录 KIT 下载指定文件（不是 AMASS 十几 GB 包），之后再实现 MMM/C3D
骨架适配、检查参考 MMM 模型及时间轴。当前没有冒充已经完成 KIT 重定向，
也没有将登录页当作数据存盘。

## 许可与引用

- CMU 原始数据及使用说明：<http://mocap.cs.cmu.edu/faqs.php>
- BABEL 的非商业研究许可仍适用于其标注与相关材料：
  <https://babel.is.tue.mpg.de/license.html>。使用 CMU 原始数据不会消除 BABEL 许可限制。
- BABEL: Punnakkal et al., *BABEL: Bodies, Action and Behavior with English Labels*, CVPR 2021.
- CMU Graphics Lab Motion Capture Database: <http://mocap.cs.cmu.edu/>
