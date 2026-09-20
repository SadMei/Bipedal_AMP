"""Package the full ACCAD conversion inventory without installing it into training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np

from accad_reference import reference_weights
from cmu_babel import save_json
from prepare_accad_bsrl import sha


def write_text_new(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--video-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve(); args.video_dir = args.video_dir.resolve()
    audit = json.loads((args.run/'audit.json').read_text())
    manifest = json.loads(Path(audit['source_manifest']).read_text())
    verification_path = args.run/'verification.json'
    verification = json.loads(verification_path.read_text())
    if not verification['success'] or not verification['videos_checked']:
        raise ValueError('Require complete data and video verification')
    reference = manifest['reference']
    if sha(reference['config_path']) != reference['config_sha256']:
        raise ValueError('Reference configuration has changed')
    weights = reference_weights(reference['config_path'])
    by_stem = {e['clip']['stem']: e for e in audit['clips']}
    checks = {e['stem']: e for e in verification['checks']}
    if set(checks) != set(by_stem) or len(by_stem) != len(manifest['clips']):
        raise ValueError('Missing or duplicate audited clips')
    groups = []
    for name in weights:
        clips = [e['clip'] for e in audit['clips'] if e['clip'].get('reference_name') == name]
        mapping = next(m for m in reference['mappings'] if m['reference_name'] == name)
        if {c['motion'] for c in clips} != set(mapping['source_candidates']):
            raise ValueError('Missing reference source candidate: '+name)
        groups.append(dict(reference_name=name, original_weight=weights[name],
                           source_identity_status=mapping['identity_status'],
                           converted_candidates=[dict(stem=c['stem'],source=c['motion'],
                               pkl=str(args.run/'gmr'/c['category']/(c['stem']+'.pkl'))) for c in clips]))
    counts = Counter(e['clip']['category'] for e in audit['clips'])
    duration = sum(e['duration_s'] for e in audit['clips'])
    joint_flags = [e['clip']['stem'] for e in audit['clips'] if e['joint_speed_max_rad_s'] > 20]
    review = dict(reference_name_count=len(groups), converted_source_count=len(by_stem),
        ambiguous_reference_names=reference['unresolved_subject_names'],
        reference_candidate_count=sum(e['clip']['in_local_g1_amp_name_list'] for e in audit['clips']),
        supplemental_count=sum(not e['clip']['in_local_g1_amp_name_list'] for e in audit['clips']),
        fps=60., frames=sum(e['frames'] for e in audit['clips']), duration_s=duration,
        category_counts=dict(counts), joint_speed_review_threshold_rad_s=20.,
        joint_speed_review_threshold_is_motor_limit=False, joint_speed_review_clips=joint_flags,
        all_clips_retained_no_silent_filter=True, training_configs_changed=False,
        amp_ready=False, verification=str(verification_path), reference_groups=groups)
    save_json(args.run/'dataset_catalog.json', review)
    headings = dict(straight='前行',turn='走路转向/掉头',sideways='侧行',backward='倒走',
                    stand='站立',run='跑步/跑步转向',run_transition='走跑/站跑切换')
    lines = ['# ACCAD → BSRL：完整 G1 参考清单扩充版', '',
        f'共 **{len(by_stem)} 段 / {duration:.1f} 秒 / 60 FPS**。覆盖本地开源 Legged Lab G1 配置的全部 **{len(groups)} 个参考动作名**。', '',
        '这是 ACCAD 源动作重新重定向到 BSRL，不是将 G1 的关节数据直接套到 BSRL，也不是已确认的宇树内部训练集。', '',
        '30 个名字对应 32 个源片段：C5（walk to run）和 C6（stand to run backwards）各有 Female1 / Male2 两个同名来源，无法仅凭配置判定原参考用了哪个，因此两者均保留。额外保留之前的直行 B3 和站立 A1，共 34 段。', '',
        '## 使用状态与质量提醒', '',
        '- 全部是 GMR 格式的完整运动学序列，根四元数 xyzw；尚未安装为正式 AMP 训练集，未更改训练配置。',
        '- 保留原始时间、整体 XY 位移和转向；只做初始坐标系对齐、腿长缩放、机器人目标踝宽适配。',
        '- 走路沿用前一版足底高度修正。跑步按源 SMPL-X 足部网格最低点的高度变化转移离地间隙，保留几何腾空，不强制每帧贴地。',
        '- 跑步地面基准取缩放后的最低足部网格高度第 5 百分位，并扣除 5mm 几何噪声余量；源脚部拟合有高度波动，因此该方法不是可靠的接触/腾空标签，可能造成偏高的悬空比例，训练前需要接触校正。',
        f'- {len(joint_flags)} 段出现 >20 rad/s 的关节速度，标记复查，未静默删除或平滑。20 rad/s 是本次人工复查阈值，不是电机上限。',
        '- 类别仅为源动作候选标签；完整序列包含起停和收尾回转。没有强制去转向、镜像、裁剪或循环拼接。',
        '- 通过项是源哈希、官方 SMPL-X 骨架对照、源平移、导出 FK、限位和视频完整性；不表示动力学平衡、无滑脚、训练质量已通过。', '',
        '## 分类', '', '| 类别 | 数量 | 原始时长 |', '|---|---:|---:|']
    for category,count in sorted(counts.items()):
        seconds=sum(e['duration_s'] for e in audit['clips'] if e['clip']['category']==category)
        lines.append(f'| {headings[category]} | {count} | {seconds:.2f} 秒 |')
    lines += ['', '## 独立视频和数据', '', '全部视频为 0.5 倍速，有固定世界视角、轨迹图和足部放大，无总览拼接。', '',
              '| 名称（保留被试目录） | 类别 | 秒 | 数据 | 视频 | 复查标记 |', '|---|---|---:|---|---|---|']
    detailed_quality = []
    for stem,e in by_stem.items():
        category=e['clip']['category']; pkl=args.run/'gmr'/category/(stem+'.pkl')
        video=args.video_dir/category/(stem+'.mp4')
        if sha(pkl)!=checks[stem]['pkl_sha256'] or sha(video)!=checks[stem]['video_sha256']:
            raise ValueError('Changed verified artifact')
        with pkl.open('rb') as stream:
            motion = pickle.load(stream)
        speed = np.abs(np.diff(motion['dof_pos'],axis=0)*motion['fps'])
        frame, joint = np.unravel_index(speed.argmax(),speed.shape)
        detailed_quality.append(dict(stem=stem,
            joint_speed_p95_p99_max_rad_s=np.quantile(speed,[.95,.99,1]).tolist(),
            peak_joint=motion['dof_names'][joint],
            peak_interval_source_s=motion['source_times'][frame:frame+2].tolist(),
            frame_intervals_over_20rad_s=int(np.sum(speed.max(axis=1)>20)),
            root_xy_displacement_error_max_m=e['root_xy_displacement_error_max_m'],
            source_grounding_details=e['grounding_details'],
            review_flags=checks[stem]['quality_review_flags']))
        flags=[]
        if stem in joint_flags: flags.append('关节速度')
        if e['clip'].get('reference_source_ambiguous'): flags.append('同名来源歧义')
        if e['geometry_pass'] is not True: flags.append('类别未通过/未评估')
        if e['root_xy_displacement_error_max_m']>.04: flags.append('根位置跟踪')
        lines.append(f'| {stem} | {headings[category]} | {e["duration_s"]:.3f} | [PKL]({pkl}) | [MP4]({video}) | {"、".join(flags) or "—"} |')
    lines += ['', '## 路径与复现', '', f'- 源清单：`{audit["source_manifest"]}`',
              f'- 转换数据：`{args.run}/gmr/`', f'- 精确对应与复查信息：`{args.run}/dataset_catalog.json`',
              f'- 逐段验证：`{verification_path}`',
              '- 脚本说明：`/root/mjw/Bipedal_AMP/scripts/tools/retarget/README_accad.md`',
              '- 原始 ACCAD/AMASS、SMPL-X、BABEL 的许可约束保留；本次没有联网下载或改动原始归档。', '']
    content='\n'.join(lines)
    save_json(args.run/'quality_review.json', dict(training_quality_passed=False,
        review_threshold_rad_s=20., threshold_is_hardware_limit=False, clips=detailed_quality))
    write_text_new(args.run/'README.md', content)
    write_text_new(args.video_dir/'README.md', content)
    print(f'Catalog complete: {len(groups)}/{len(weights)} reference names, {len(by_stem)} clips, {duration:.1f}s')


if __name__ == '__main__':
    main()
