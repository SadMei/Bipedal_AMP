# BSRL 并联踝工具

从工作区 `parallel_ankle/` 迁入的脚本、测试和 C++ 接口。CAD 零位几何、
控制与限幅算法、原始策略和仿真条件保持不变；这是独立 MuJoCo 诊断工具，
不修改 Isaac 训练或现有 sim2sim 默认入口。

## 误删后的模型恢复（2026-09-15）

脚本迁移后，原工作区 `parallel_ankle/` 数据目录已缺失。现已利用保留的生成逻辑
和训练工程内的 CAD 分件备份，把整机及左右单踝 XML 重建到工程内部的
`source/legged_lab/legged_lab/data/Robots/BSRL_closed_ankle/xml/`，配套网格按原字节恢复，
使用相对路径。详细来源及验证见该目录中的 `parallel_ankle_recovery.json`。

这是重建恢复，不是原 XML 的逐字节找回；旧轨迹、历史结果、归档示意 XML 和
原 C++/YAML 未恢复。策略回放的默认路径仍要求缺失的 `bsrl_ankle_core_flat`。
显式传入工程内 `BSRL_closed_ankle/xml/bsrl_closed_ankles.xml` 可在核对恢复清单后
使用 CAD 几何回放，不声称完成原 C++/YAML 核验；`--onnx` 可指定确切的 ONNX，
并核对其 checkpoint 校验和，避免误用另一个迭代的默认导出。
新入口 `restore_models.py` 仅显式从 CAD 重建模型，不代替硬件解算器或历史回放。

## 入口

- `real_ankle.py` / `core_bridge.cpp`：原始 C++ 接口、CAD 几何与闭链构造。
- `cad_assets.py`：提取 CAD 连接点、无损拆分视觉部件。
- `build_models.py`：生成左右单踝及整机闭链 MJCF。
- `restore_models.py`：从保留的 CAD 和网格备份重建可搬移的模型目录，拒绝覆盖已有文件。
- `parallel_ankle.py`：单踝 PR/AB 控制验证。
- `replay_policy.py`：ONNX 策略在 `serial`、`mapped`、`closed` 三种拓扑下对照回放。
- `export_checkpoint.py`：纯CPU严格导出240D、无estimator的BSRL checkpoint，核对轮数、SHA256及PyTorch/ONNX输出；拒绝覆盖已有导出目录。
- `render_replay.py`：从已记录状态生成整机或踝部特写视频，不重新跑策略。
  `--split-ankles --slowdown 2 --width 1920 --height 1080` 为左右分屏半速特写。
- `check_mounting.py`：独立检查 URDF/CAD 安装方向。
- `build_preview_urdf.py`：生成仅供查看的 fixed-visual URDF，不作为训练闭链模型。
- `parallel_ankle_approximate.py`：归档示意模型，不作为当前机器人几何依据。
- `test_closed_ankle.py`、`test_cad_visuals.py`、`test_preview_urdf.py`：原有 CPU 回归测试。
- `test_restore_models.py`：不依赖原 C++ 的恢复验证、约束雅可比及网格一致性测试。
- `_paths.py`：区分代码目录与外部数据目录，不依赖当前工作目录。

## 运行

使用 `/root/miniconda3/envs/isaaclab/bin/python`（或具备依赖的 Python）。
以下在 `Bipedal_AMP` 根目录执行，结果目录和视频名应使用新名称：

```bash
python scripts/tools/parallel_ankle/build_models.py \
  --output-dir /tmp/bsrl_parallel_ankle_models
python scripts/tools/parallel_ankle/parallel_ankle.py --side right --mode PR --no-view
python scripts/tools/parallel_ankle/parallel_ankle.py --side left --mode AB --no-view
python scripts/tools/parallel_ankle/check_mounting.py \
  --plot /tmp/bsrl_parallel_ankle_mounting.png
python scripts/tools/parallel_ankle/replay_policy.py --topology closed \
  --seconds 30 --command 0.6 0 0 --output /tmp/bsrl_parallel_ankle_replay
MUJOCO_GL=egl python scripts/tools/parallel_ankle/render_replay.py \
  /tmp/bsrl_parallel_ankle_replay videos/NEW_closed.mp4
python -m pytest -q scripts/tools/parallel_ankle
```

也支持包入口，例如 `python -m scripts.tools.parallel_ankle.replay_policy --help`。
脚本绝对路径可在任意目录执行；显式传入的相对参数仍相对当前工作目录。
回放默认仍为原 9 月 7 日 `model_29999` 基线，迁移没有更换 checkpoint。
仿真和 ONNX 推理使用 CPU，渲染另需可用的图形环境。

2026-09-16起，回放也支持冻结配置中的 `ReducedAnkleActuator`：按
`motor_effort/motor_speed/motor_power`读取实际电机包络，不使用关节层1e9/1000数值保护值。
动作目标裁剪从各run冻结配置恢复（旧串联±100rad，新解析传动None）；其他控制与模型不变。
此入口尚不支持显式闭链训练执行器配置，遇到未知类型会拒绝加载，不静默套用串联参数。
本次临近完成的两组与旧串联参照结果见
[闭链sim2sim报告](../../../reports/bsrl_nearfinal_closed_sim2sim_20260916/README.md)。

## 模型、资源与依赖

模型路径都在 `Bipedal_AMP` 仓库内：

- `source/legged_lab/legged_lab/data/Robots/BSRL_urdf/`：训练使用的串联 URDF、MJCF 与本地 mesh。
- `source/legged_lab/legged_lab/data/Robots/BSRL_closed_ankle/`：闭链 XML、恢复清单与本地 mesh。
- `scripts/tools/parallel_ankle/bsrl_observation.py`：仓库内 sim2sim 观测定义，不再导入 `mjw_simtool`。
- `Bipedal_AMP/logs/rsl_rl/bsrl_amp/`：原策略及训练参数。
- `bsrl_ankle_core_flat/`：原始 `AnkleKinematics.hpp`、`AnkleKinematics.cpp` 和 `bsrl_bridge_bringup.yaml`；可通过 `BSRL_ANKLE_CORE_DIR` 覆盖目录。

网格不由本仓库的 Git 跟踪。新环境只需将 `BSRL_urdf` 资产仓库中的
`urdf/`、`xml/`、`meshes/` 复制到上述机器人数据目录；训练核心路径保持不变。

Python 依赖为 `numpy`、`scipy`、`PyYAML`、`mujoco`、`trimesh`、`onnxruntime`、
`matplotlib`，录像另需 `imageio`、`imageio-ffmpeg`、`Pillow`；测试另需 `pytest`
和提供 `pinocchio` 模块的环境。编译原始接口需支持 C++17 的 `g++`。

迁移检查时（2026-09-15）工作区中缺少 `bsrl_ankle_core_flat`。建模、回放和
原有几何回归仍要求该原始依赖，以保留来源核验；缺失时会明确报错，不会替换
为别的解算器。命令行帮助、独立 CAD 读取、已录状态渲染及归档示意模型不依赖它。

归档示意入口仍读取 `parallel_ankle/parallel_ankle_approximate.xml`，该 XML 当前缺失；
其波形输出路径为 `parallel_ankle/results/approximate/`。旧目录中的历史说明也已缺失。

恢复验证（不要求原 C++ 解算器）：

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q scripts/tools/parallel_ankle/test_restore_models.py
```

迁移专项检查（仓库根目录执行，无需原始解算器）：

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q source/legged_lab/test/test_tools_relocation.py
```
