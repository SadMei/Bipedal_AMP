# ACCAD SMPL-X → BSRL：保留整体位移的运动学试转换

本工具读取用户提供的 `ACCAD.tar.bz2` 和 `models_smplx_v1_1.zip`，不下载数据，不修改训练配置，不自动安装 AMP 数据集。SMPL-X 模型及 ACCAD/AMASS 动作仍受各自原始许可约束。

## 本次结果

- 源数据：`/root/mjw/GMR/motion_data/ACCAD_smplx_source/manifest.json`
- 数据、逐段指标：`/root/mjw/Bipedal_AMP/reports/bsrl_accad_smplx_20260916_preview/`
- 独立视频：`/root/mjw/GMR/videos/bsrl_accad_smplx_20260916_preview/`
- 8 段、2215 帧、60 FPS，共 36.9167 秒原始时间；视频为 0.5 倍速，共 73.8333 秒。没有总览拼接视频。
- 1 段前行、左右转各 1 段、两名被试的左右侧行各 1 段、1 段站立。类别来自源文件名，只作为候选类别。

前行 `straight_female1_B3` 源根节点净移动 8.0932 米。按 BSRL/人体腿长比例 0.986 缩放后，机器人净移动 7.994 米，不是原地踏步。源轨迹相对首尾连线的最大横向偏离约 0.0596 米；骨盆周期摆动会触发严格的 yaw 范围筛选，但不能因此直接判断轨迹走弯。

左转 `turn_left_male2_B10` 中段确实左转，末尾又回转；整段净 yaw 变化不能代表中间的转弯角度。完整保留原片段，没有逐帧消除 yaw，也没有隐藏收尾动作。它和前行片段的类别筛选均标记为未通过，允许通过显式 `--include-unsegmented-preview` 生成预览；不表示通过了训练集筛选。

## 处理约定与限制

1. 使用用户提供的男女 SMPL-X 模型、每段自己的 `betas` 计算前 22 个身体关节。根位置是 `rest_pelvis + trans`。独立审计逐段抽取 5 帧与官方 `smplx` 模型核对，不以近似骨架替代。
2. 原始 120 FPS 精确隔帧抽取到 60 FPS，不拉伸时间、不循环拼接。只做一次初始 XY/yaw 对齐，并按人体/机器人腿长等比例缩放。整体水平位移和转弯均保留。
3. BSRL 使用现有 12 自由度模型、现有关节符号约定；踝目标对应 SMPL-X 的踝关节 7/8，不把脚趾关节 10/11 当踝。
4. 按机器人形态对目标踝宽作左右对称适配，最小间距 0.22 米；每段记录实际改变量。IK 有厘米级根位置跟踪误差，不能宣称输出与源平移逐帧完全相等。
5. 预览专用于非腾空的走/转/侧行/站立。每帧对根 Z 作统一校正，使最低脚部碰撞几何离地 0.5 毫米；不改变 XY。站立另加双足高度目标，单脚 IK 残差容限 1 毫米，共同 Z 投影后双足最大高度的理论界限为 `0.5 + 2×1 = 2.5` 毫米。本次站立实际最高约 2.21 毫米。
6. 这是明确的运动学高度修正，不是物理接触求解。支撑脚没有做水平锁定，有滑脚和脚掌摇摆；支撑判定仅依据几何高度，不是力传感器标签。不能用于跑步/腾空动作。
7. 不证明动力学平衡、电机角速度可行性或 AMP 可训练性。MuJoCo 某些 mesh 距离查询返回零，即使输出的两端点相距较远，因此不能把原始零值当成脚部相撞或无穿插的证据。

GMR PKL 中根四元数为 **xyzw**，另有 `dof_names`、关键点位置、源帧时间及 `kinematic_preview_only=True`。它们是预览数据，不是已经接入训练 loader 的正式数据集。视频中人体使用固定世界相机；机器人全身和脚部跟随根位置，并显示根 XY 轨迹和距起点的距离。

## 复现

解释器：`/root/miniconda3/envs/isaaclab/bin/python`。本次只补装 `smplx==0.1.28 --no-deps`，未升级 Torch/Isaac。脚本从 `Bipedal_AMP` 根目录运行。以下输出目录使用新名字；工具拒绝覆盖已有的不同结果。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/prepare_accad_bsrl.py --output /root/mjw/GMR/motion_data/ACCAD_smplx_source

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/retarget_accad_bsrl.py --source /root/mjw/GMR/motion_data/ACCAD_smplx_source --output /root/mjw/Bipedal_AMP/reports/bsrl_accad_smplx_new_preview --include-unsegmented-preview

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/render_cmu_babel_bsrl.py /root/mjw/Bipedal_AMP/reports/bsrl_accad_smplx_new_preview --video-dir /root/mjw/GMR/videos/bsrl_accad_smplx_new_preview --fixed-source-camera

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/audit_accad_bsrl.py /root/mjw/Bipedal_AMP/reports/bsrl_accad_smplx_new_preview --video-dir /root/mjw/GMR/videos/bsrl_accad_smplx_new_preview

/root/miniconda3/envs/isaaclab/bin/python -m pytest -q source/legged_lab/test/test_accad_smplx_retarget.py source/legged_lab/test/test_cmu_babel_retarget.py source/legged_lab/test/test_100style_retarget.py
```

审计包括原文件哈希、官方 SMPL-X 对照、源平移保留、精确抽帧时间、导出/viewer 正运动学一致性、关节限位、实际脚部高度以及全部视频逐帧解码。不会把类别纯度或动力学可行性计入这些通过项。

## 2026-09-16 完整 G1 参考清单扩充

新增独立目录，保留原 8 段预览不变：

- 源数据：`/root/mjw/GMR/motion_data/ACCAD_smplx_g1_full_source/`
- 转换结果：`/root/mjw/Bipedal_AMP/reports/bsrl_accad_g1full_20260916/`
- 视频：`/root/mjw/GMR/videos/bsrl_accad_g1full_20260916/`

读取本地 `g1_amp_env_cfg.py` 的 30 个动作名，不执行或改动训练配置。全部匹配 ACCAD；C5/C6 各有 Female1、Male2 同名来源，均保留而不假定原参考来源，因此得到 32 个参考候选，再加之前的直行 B3/站立 A1，共 34 段、10896 帧、181.6 秒。`dataset_catalog.json` 按 30 个参考名分组，明确标注歧义，不能把 32 个片段宣称为严格一对一的 30 条原参考复现。

扩充版包含跑步和走跑切换，**不适用上面第 5 条的逐帧强制贴地**。这些片段使用 `source_mesh_clearance_transfer_with_flight_v1`：官方 SMPL-X 足部网格最低点（踝+脚 skinning weight > 0.5）给出逐帧高度，缩放后第 5 百分位作为参考地面；最低脚目标间隙为 `0.5mm + max(0, source_min - floor - 5mm)`。与原网格几何一致，但没有测力标签，也没有支撑段地面漂移修正；部分片段的悬空比例偏高，必须复查，不能直接认定物理接触正常。

全片保留所有异常并记录复查标记：8 段最大关节速度超过人工复查阈值 20 rad/s；部分根位置相对缩放后源骨盆位移差超过 4cm。这里的差值包含不同根参考点和 IK 跟踪误差，不等于丢弃了 `trans`。数据审计将其记录为质量问题，并独立检查全部源平移、轨迹长度、导出 FK 一致性；不会宣称训练质量通过。

所有导出仍为 GMR 格式、xyzw 四元数，未调用现有 `bsrl_gmr_to_amp.py` 伪装成旧的强制贴地约定，也未更改训练入口以接受它们。正式接入 AMP 时需要处理新 grounding 约定、wxyz 四元数/关节顺序、片段纯度和接触质量。

复现时将下列 `RUN` 和 `VIDEOS` 换成新的具体输出路径：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/prepare_accad_bsrl.py --selection g1-full --output /root/mjw/GMR/motion_data/ACCAD_smplx_g1_full_source

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/retarget_accad_bsrl.py --source /root/mjw/GMR/motion_data/ACCAD_smplx_g1_full_source --output RUN --include-unsegmented-preview

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/render_cmu_babel_bsrl.py RUN --video-dir VIDEOS --fixed-source-camera

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/audit_accad_bsrl.py RUN --video-dir VIDEOS

/root/miniconda3/envs/isaaclab/bin/python scripts/tools/retarget/catalog_accad_bsrl.py RUN --video-dir VIDEOS
```
