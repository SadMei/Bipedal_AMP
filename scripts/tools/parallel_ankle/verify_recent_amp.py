"""Independently validate frozen AMP recordings and write their video index.

Rollout outcomes (including falls) are separate from export/video integrity.
Never modifies training, recorded arrays, existing videos, or previous reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
import torch

from .export_checkpoint import load_policy


TITLES = {
    'serial_mirror': '串联 + mirror',
    'serial_nomirror': '串联，无 mirror',
    'reduced_mirror': '解析并联 + mirror',
    'reduced_heading': '解析并联 + 初始航向速度奖励 + mirror',
    'explicit_closed': '显式闭链训练 + mirror',
    'accad_nomirror': 'ACCAD34 解析并联，无 mirror',
    'accad_mirror': 'ACCAD34 解析并联 + mirror',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_new(path, text):
    with path.open('x') as stream:
        stream.write(text)


def verify(record, report_dir, video_dir, selection, common_xml_sha):
    folder = report_dir / record['label']
    manifest = read_json(folder / 'replay/manifest.json')
    summary = read_json(folder / 'replay/summary.json')
    exported = read_json(folder / 'export/export_verification.json')
    assert manifest['checkpoint'] == record['checkpoint']
    assert manifest['checkpoint_sha256'] == record['checkpoint_sha256']
    assert exported['checkpoint_sha256'] == record['checkpoint_sha256']
    for config in ('env', 'agent'):
        assert sha(Path(record['run']) / f'params/{config}.yaml') == record[f'{config}_sha256']
        assert exported[f'{config}_sha256'] == record[f'{config}_sha256']
    assert manifest['recovered_model_sha256'] == common_xml_sha
    assert sha(manifest['recovered_model_xml']) == common_xml_sha
    assert manifest['topology'] == summary['topology'] == 'closed'
    assert manifest['passive_ankle_direct_actuators'] == 0
    assert manifest['automatic_resets'] == summary['automatic_resets'] == 0
    assert manifest['physics_dt'] == manifest['pd_dt'] == .001
    assert manifest['policy_dt'] == .02
    assert not summary['warnings'], summary['warnings']
    completed = summary['fall'] is None and summary['failure'] is None
    requested = selection['requested_seconds']
    assert 0 < summary['seconds'] <= requested + 1e-8
    if completed:
        np.testing.assert_allclose(summary['seconds'], requested, rtol=0, atol=1e-8)

    with np.load(folder / 'replay/policy.npz') as trajectory, np.load(folder / 'replay/physics.npz') as physics:
        for archive in (trajectory, physics):
            count = len(archive['time_s'])
            assert count > 0
            for name in archive.files:
                assert len(archive[name]) == count, name
                assert np.isfinite(archive[name]).all(), name
        n = len(trajectory['time_s'])
        steps = len(physics['time_s'])
        assert n == summary['policy_frames'] and steps == summary['physics_steps']
        assert trajectory['obs'].shape == (n, 240)
        assert trajectory['action'].shape == (n, 12)
        np.testing.assert_allclose(trajectory['time_s'], np.arange(n) * .02, rtol=0, atol=1e-8)
        np.testing.assert_allclose(physics['time_s'], np.arange(steps) * .001, rtol=0, atol=1e-8)
        assert abs(n * .02 - summary['seconds']) <= .02000001
        if completed:
            assert n == round(requested / .02) and steps == round(requested / .001)
        np.testing.assert_allclose(trajectory['command'], np.tile(selection['fixed_command'], (n, 1)))
        nominal = manifest['nominal']
        targets = np.asarray(nominal['default_pos']) + trajectory['action'] * nominal['action_scale']
        if nominal['target_lower'] is not None:
            targets = np.clip(targets, nominal['target_lower'], nominal['target_upper'])
        np.testing.assert_allclose(trajectory['target'], targets, atol=1e-6, rtol=0)
        torque, velocity = physics['motor_applied'], physics['motor_velocity']
        assert np.abs(torque).max() <= 112 + 1e-6
        assert np.abs(torque * velocity).max() <= 1250 + 1e-6
        assert not np.any((np.abs(velocity) >= 26.17993877991494) & (torque * velocity > 1e-6))
        assert np.max(physics['closure_gap']) <= .002
        # Replay stops BEFORE saving an over-limit closure sample. Therefore a
        # failed rollout must not be described as having valid closure throughout.
        closure_limit_stop = bool(summary['failure'] and 'Closure gap' in summary['failure'])
        policy, metadata = load_policy(Path(record['checkpoint']))
        assert metadata['checkpoint_sha256'] == record['checkpoint_sha256']
        onnx_path = folder / 'export/policy.onnx'
        assert sha(onnx_path) == manifest['policy_sha256'] == exported['onnx_sha256']
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
        indices = np.unique(np.linspace(0, n - 1, min(n, 100), dtype=int))
        replay_error = torch_error = 0.
        for index in indices:
            obs = trajectory['obs'][index:index + 1]
            actual = session.run(None, {session.get_inputs()[0].name: obs})[0][0]
            with torch.no_grad():
                expected = policy.actor(policy.actor_obs_normalizer(torch.from_numpy(obs))).numpy()[0]
            np.testing.assert_allclose(actual, trajectory['action'][index], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
            replay_error = max(replay_error, float(np.abs(actual - trajectory['action'][index]).max()))
            torch_error = max(torch_error, float(np.abs(actual - expected).max()))
        video = video_dir / (record['label'] + '.mp4')
        frames = 0
        with imageio.get_reader(video) as reader:
            info = reader.get_meta_data()
            for frame in reader:
                assert frame.shape == (720, 1440, 3), frame.shape
                frames += 1
        assert frames == n and info['fps'] == 50
        quat = trajectory['qpos'][:, 3:7]
        np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-6)
        w, x, y, z = quat.T
        yaw = np.rad2deg(np.unwrap(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))))
        yaw -= yaw[0]
        tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (x*x + y*y), -1, 1)))
        np.testing.assert_allclose(yaw[-1], summary['final_yaw_degrees'], atol=1e-8)
        return dict(
            integrity_verified=True, completed_requested_duration=completed,
            physical_outcome=summary['fall'] or summary['failure'] or 'completed',
            checkpoint=record['checkpoint'], iteration=record['iteration'],
            checkpoint_sha256=metadata['checkpoint_sha256'], onnx_sha256=sha(onnx_path),
            video=str(video), video_sha256=sha(video), decoded_frames=frames,
            fps=info['fps'], video_seconds=frames / info['fps'],
            actual_observation_samples=len(indices), replay_onnx_max_abs_error=replay_error,
            pytorch_onnx_max_abs_error=torch_error, actor_input_dim=240,
            final_yaw_deg=float(yaw[-1]), max_abs_yaw_deg=float(np.abs(yaw).max()),
            max_base_tilt_deg=float(tilt.max()),
            max_abs_lateral_displacement_m=float(np.abs(trajectory['qpos'][:, 1] - trajectory['qpos'][0, 1]).max()),
            max_motor_power_w=float(np.abs(torque * velocity).max()),
            max_motor_speed_rad_s=float(np.abs(velocity).max()),
            max_recorded_closure_gap_mm=float(np.max(physics['closure_gap']) * 1000),
            closure_limit_stop=closure_limit_stop, all_recorded_arrays_finite=True,
            motor_envelope_verified=True, summary=summary,
        )


def index_text(records, result, selection, report_dir):
    lines = [
        '# BSRL AMP 录制对比 — 2026-09-17', '',
        '本文件夹只收纳本次 7 组最近 BSRL 实验，不包含 T1 或更早的历史归档。原有视频未移动或覆盖。', '',
        '每个视频左侧是全身（视角方位固定、相机随机器人平移），右侧是足踝放大视图；画面显示模型轮数、累计航向偏差和位置。', '',
        '| 实验 / 视频 | checkpoint | 取样时是否最终轮 | 时长 | 仿真结果 | 最后航向偏差 |',
        '| --- | ---: | --- | ---: | --- | ---: |',
    ]
    for record in records:
        label = record['label']
        value = result[label]
        final = '是' if record['is_final_checkpoint'] else '否，训练中快照'
        title = f'[{TITLES[label]}]({label}.mp4)'
        if value.get('integrity_verified'):
            outcome = '完成，未跌倒' if value['completed_requested_duration'] else value['physical_outcome']
            lines.append(f"| {title} | {record['iteration']} | {final} | {value['video_seconds']:.2f}s | {outcome} | {value['final_yaw_deg']:+.2f}° |")
        else:
            lines.append(f"| {title} | {record['iteration']} | {final} | — | 校验失败，见报告 | — |")
    lines += [
        '', '## 统一评估条件', '',
        '- 同一 CAD 恢复的 MuJoCo 显式闭链并联踝模型，前进指令 `[0.6, 0, 0]`，目标 30 秒。训练模型类型仍在表内分别标明。',
        '- 从静态默认姿态开始；关闭噪声、随机化、延迟和推力，不自动 reset；跌倒或约束故障则原样记录并提前停止，不补帧凑时长。',
        '- 物理 / PD / 策略频率：1000 / 1000 / 50 Hz；这是沿用的 sim2sim 部署设置，PD 频率不同于这些训练的 200 Hz。',
        '- PD、默认关节位姿、动作缩放和裁剪取各 run 保存的配置；电机包络 112 Nm / 26.17994 rad/s / 1250 W；被动踝轴无直接驱动。',
        '- 均不使用 estimator，Actor 输入均为 240 维，不含累计 heading error；航向奖励版本和 ACCAD34 两组仅在 Critic 中有相应特权观测。',
        '- 1440 × 720、50 FPS、原速。离线渲染已保存的关节状态，不重新运行策略。',
        '', '## 如何理解结果', '',
        '这是各组最新保存 checkpoint 的单次确定性 sim2sim，不是多随机种子结论。未完成训练的轮数不同，且数据集、奖励和训练动力学有变化，因此不能把所有差异都归因于 mirror。', '',
        '闭链连接采用原 CAD 校准；辅助连杆质量/惯量是建模假设，未用缺失的硬件 C++ 解算器重新标定，不等同于真机验证。', '',
        f"模型选择时间：`{selection['selected_at']}`。每组的原训练目录、checkpoint / ONNX / 视频 SHA256、轨迹数组、导出日志和逐帧解码校验见 [{report_dir.name}]({report_dir})。", '',
        '所有报告与录制均独立保存，未修改训练或原来的导出文件。', '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    args = parser.parse_args()
    report_dir, video_dir = args.report_dir.resolve(), args.video_dir.resolve()
    for path in (report_dir / 'verification.json', report_dir / 'README.md', video_dir / 'README.md'):
        if path.exists():
            raise FileExistsError(path)
    torch.set_num_threads(1)
    selection = read_json(report_dir / 'selection.json')
    records = selection['records']
    baseline = read_json(report_dir / records[0]['label'] / 'replay/manifest.json')
    result = {}
    for record in records:
        try:
            value = verify(record, report_dir, video_dir, selection, baseline['recovered_model_sha256'])
        except Exception:
            value = dict(integrity_verified=False, error=traceback.format_exc())
        result[record['label']] = value
        print('[VERIFY]', record['label'], json.dumps({k: value.get(k) for k in (
            'integrity_verified', 'physical_outcome', 'decoded_frames', 'final_yaw_deg', 'error'
        )}, ensure_ascii=False), flush=True)
    write_new(report_dir / 'verification.json', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_new(video_dir / 'README.md', index_text(records, result, selection, report_dir))
    write_new(report_dir / 'README.md', '\n'.join([
        '# BSRL AMP 2026-09-17 录制审计', '',
        f'[视频与实验索引]({video_dir / "README.md"})', '',
        '- `selection.json`：启动时冻结的模型选择、配置与 checkpoint 哈希。',
        '- `simulation_status.json` / `render_status.json`：仿真和渲染状态。',
        '- `verification.json`：独立校验结果；区分录制完整性与物理运行结果。',
        '- 每组 `export/`：本次 ONNX 与严格完整模型加载 / 导出校验。',
        '- 每组 `replay/`：完整 manifest、原始 policy / physics 数组、模型 XML 和总结。',
        '- 每组 `preview.png`：同一视频约 5 秒位置的预览。',
        '- `source_snapshot/`：本次使用的脚本快照。', '',
        '校验包含：源文件哈希、实际观测上的 100 次 ONNX / PyTorch 对比、固定指令 / 动作目标、',
        '全数组有限值、已记录电机包络 / 闭链间隙、全部视频逐帧解码及帧数 / 尺寸 / FPS。', '',
    ]))
    if not all(value['integrity_verified'] for value in result.values()):
        raise SystemExit('Some recordings failed verification; inspect verification.json')


if __name__ == '__main__':
    main()
