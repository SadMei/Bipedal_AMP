# BSRL rosbag 离线工具

从工作区 `bsrl_rosbags/analysis/` 迁入的脚本与 C++ 接口。只整理代码位置，
不搬运或修改原始 bag、配置快照、历史分析结果，也不改变分析算法。

## 入口

- `export_bags.py`：读取 ROS1 bag，导出 NPZ 和消息元数据；不需要安装 ROS。
- `analyze_bags.py` / `ankle_batch.cpp`：调用原始 C++ 并联踝解算器，检查映射、PD、滤波、通信并生成图表。
- `detail_checks.py`：读取已有导出及派生结果，复核固定目标振铃和力矩时间偏移。
- `_paths.py`：统一解析外部数据和原始解算器路径；不依赖当前工作目录。

## 运行

使用 `/root/miniconda3/envs/isaaclab/bin/python`（或具备依赖的 Python）。
以下命令在 `Bipedal_AMP` 根目录执行，`NEW_analysis` 应替换成新的结果目录，
避免覆盖历史结果：

```bash
python scripts/tools/bsrl_rosbags/export_bags.py \
  --root ../bsrl_rosbags --output ../bsrl_rosbags/analysis/NEW_analysis
OPENBLAS_NUM_THREADS=1 python scripts/tools/bsrl_rosbags/analyze_bags.py \
  --results ../bsrl_rosbags/analysis/NEW_analysis
OPENBLAS_NUM_THREADS=1 python scripts/tools/bsrl_rosbags/detail_checks.py \
  --results ../bsrl_rosbags/analysis/NEW_analysis
```

也支持包入口，例如 `python -m scripts.tools.bsrl_rosbags.export_bags --help`。
使用脚本绝对路径时可从任意目录运行；显式传入的相对参数仍相对当前工作目录。

## 数据与依赖

默认工作区为 `Bipedal_AMP` 的父目录：

- 原始数据：`<workspace>/bsrl_rosbags/<session>/control.bag` 及 `snapshots/`。
- 默认结果：`<workspace>/bsrl_rosbags/analysis/results_20260914/`（沿用历史默认值；运行时建议指定新目录）。
- 原始解算器：`<workspace>/bsrl_ankle_core_flat/`，需要 `AnkleKinematics.hpp`、`AnkleKinematics.cpp` 和 `bsrl_bridge_bringup.yaml`。
- Python 依赖：`rosbags`、`numpy`、`scipy`、`matplotlib`、`PyYAML`；映射分析另需支持 C++17 的 `g++`。

可用 `BSRL_ROSBAG_ROOT` 更改默认 bag 根目录（默认结果目录随之变化），
用 `BSRL_ANKLE_CORE_DIR` 或分析入口的 `--core` 指定原始解算器目录。
迁移检查时（2026-09-15）工作区中缺少 `bsrl_ankle_core_flat`，因此完整映射分析
需先恢复该依赖；工具会明确报错，不会自动替换解算器。
导出 bag、读取已有结果的细节检查不需要编译该解算器。

历史诊断报告保留在 [bsrl_rosbags/analysis/README.md](../../../../bsrl_rosbags/analysis/README.md)。

迁移专项检查（仓库根目录执行，无需原始解算器）：

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q source/legged_lab/test/test_tools_relocation.py
```
