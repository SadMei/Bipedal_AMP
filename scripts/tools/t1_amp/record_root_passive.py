"""Frozen user T1 ONNX: root-only passive dynamics ablation in MuJoCo.

Reuse the user's actual T1 runner and saved training contract. Do not export a new
policy or edit deployment/training defaults. Save MJB, not just MJCF, because
mj_saveLastXML does not preserve every in-memory runtime override.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

import record_comparison as reference

core = reference.core
ROOT = Path('/root/mjw/Bipedal_AMP')
SOURCES = {
    '0915': ROOT / 'reports/t1_amp_0914_0915_20260917/t1_0915_noheading',
    'latent14000': ROOT / 'reports/video_organization_20260920/retained_metadata/older_checkpoints_20260919/T1/latent_checkpoints_20260918_1025/t1_model14000',
}
PASSIVE_FIELDS = ('dof_damping', 'dof_frictionloss', 'dof_armature')


def configure(variant):
    folder = SOURCES[variant]
    prior = json.loads((folder / 'manifest.json').read_text())
    checkpoint = Path(prior['checkpoint'])
    env_path = checkpoint.parent / 'params/env.yaml'
    agent_path = checkpoint.parent / 'params/agent.yaml'
    env = reference.load_saved_config(env_path)
    agent = reference.load_saved_config(agent_path)
    assert reference.sha(checkpoint) == prior['checkpoint_sha256']
    actor = env['observations']['policy']
    assert reference.active(actor) == reference.BASE_TERMS
    assert actor['flatten_history_dim'] and actor['concatenate_terms']
    for term in reference.BASE_TERMS:
        item = actor[term]
        assert item['scale'] is None and item['clip'] is None and item['modifiers'] is None
        assert item['history_length'] == 0
    assert actor['joint_pos']['func'].endswith(':joint_pos')
    assert actor['joint_vel']['func'].endswith(':joint_vel')
    assert actor['root_local_rot_tan_norm']['func'].endswith(':root_local_rot_tan_norm')
    for term in ('joint_pos', 'joint_vel'):
        assert tuple(actor[term]['params']['asset_cfg']['joint_names']) == core.JOINT_NAMES
    assert agent['clip_actions'] is None
    if variant == '0915':
        widths, heading = reference.policy_contract(env, agent)
        assert not heading and widths['policy'] == 435
        assert reference.sha(folder / 'export/policy.onnx') == prior['onnx_sha256']
    else:
        contract = json.loads((folder / 'export/contract.json').read_text())
        assert contract['input_dim'] == 870 and contract['history_length'] == 10
        assert contract['checkpoint'] == str(checkpoint)
        assert agent['policy']['class_name'].endswith(':LatentActorCritic')
    nominal = reference.restore_nominal(env)
    core.OBS_HISTORY = int(actor['history_length'])
    core.OBS_DIM_NO_FEET = 87 * core.OBS_HISTORY
    session = reference.session(folder / 'export/policy.onnx')
    assert session.get_inputs()[0].shape == [1, core.OBS_DIM_NO_FEET]
    assert session.get_outputs()[0].shape == [1, 25]
    return session, nominal, prior, env_path, agent_path


def record(args):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    mujoco = core.mujoco
    engine, nominal, prior, env_path, agent_path = configure(args.variant)
    command = np.array([args.vx, 0, 0], dtype=np.float64)
    sim = core.BSRLMujoco(reference.XML, headless=True, command=command, delay_steps=0,
                         actor_include_key_body=False, actor_include_heading_error=False)
    m, d = sim.model, sim.data
    np.testing.assert_array_equal(sim.command, command)
    np.testing.assert_allclose(sim.effort, nominal['effort_limit_sim'])
    np.testing.assert_allclose(sim.velocity_limit, nominal['velocity_limit_sim'])
    jid = m.joint('floating_base_joint').id
    assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    start = int(m.jnt_dofadr[jid])
    root_ids = np.arange(start, start + 6)
    nonroot = np.setdiff1d(np.arange(m.nv), root_ids)
    before = {key: getattr(m, key).copy() for key in PASSIVE_FIELDS}
    assert all(np.array_equal(value[root_ids], np.zeros(6)) for value in before.values())
    if args.mode == 'vendor_root':
        for key, value in zip(PASSIVE_FIELDS, (1.0, .55, .1)):
            getattr(m, key)[root_ids] = value
    for key in PASSIVE_FIELDS:
        np.testing.assert_array_equal(getattr(m, key)[nonroot], before[key][nonroot])
    mujoco.mj_setConst(m, d)
    sim.reset()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Binary snapshot roundtrip preserves condim, actuator limit flags, passive
    # arrays, gain/bias parameters and all other effective runtime settings.
    mujoco.mj_saveModel(m, str(output / 'runtime_model.mjb'), None)
    roundtrip = mujoco.MjModel.from_binary_path(str(output / 'runtime_model.mjb'))
    for key in (*PASSIVE_FIELDS, 'geom_condim', 'geom_friction', 'actuator_gainprm',
                'actuator_biasprm', 'actuator_ctrllimited', 'actuator_forcerange'):
        np.testing.assert_array_equal(getattr(m, key), getattr(roundtrip, key))
    arrays = {key: getattr(m, key).copy() for key in (
        *PASSIVE_FIELDS, 'body_mass', 'body_inertia', 'body_ipos', 'geom_condim', 'geom_friction',
        'geom_solref', 'geom_solimp', 'actuator_gainprm', 'actuator_biasprm',
        'actuator_ctrllimited', 'actuator_forcelimited', 'actuator_forcerange')}
    np.savez_compressed(output / 'effective_model.npz', **arrays)
    paths = [Path(__file__), Path(reference.__file__), Path(reference.t1.__file__), Path(core.__file__),
             reference.XML, env_path, agent_path, Path(prior['checkpoint']), SOURCES[args.variant] / 'export/policy.onnx']
    hashes = {str(p): reference.sha(p) for p in paths}
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    for p in paths:
        if p.suffix in ('.py', '.yaml', '.xml'):
            shutil.copy2(p, snapshots / (p.parent.name + '_' + p.name))
    initial_qpos = d.qpos.copy()
    manifest = dict(variant=args.variant, mode=args.mode, checkpoint=prior['checkpoint'],
        policy=str(SOURCES[args.variant] / 'export/policy.onnx'), source_hashes=hashes,
        model_xml=str(reference.XML), runtime_model_mjb_sha256=reference.sha(output / 'runtime_model.mjb'),
        root_joint='floating_base_joint', root_dof_ids=root_ids.tolist(),
        root_passive_before={key: value[root_ids].tolist() for key, value in before.items()},
        root_passive_after={key: getattr(m, key)[root_ids].tolist() for key in PASSIVE_FIELDS},
        root_only_parameter_change_verified=True, command=command.tolist(), nominal=nominal,
        actor_input_dim=core.OBS_DIM_NO_FEET, history_frames=core.OBS_HISTORY,
        observation='term-major; gyro3, yaw-removed rotation6, command3, absolute q25, dq25, last_action25',
        actor_heading=False, yaw_compensation=False, automatic_resets=0,
        physics_dt=core.PHYSICS_DT, policy_dt=core.POLICY_DT,
        pd_implementation='user 1ms implicit position PD; each user policy keeps its own saved training gains',
        target_clipping=None, noise=False, domain_randomization=False, delay_steps=0,
        initial_qpos=initial_qpos.tolist(), requested_seconds=args.seconds, videos=[],
        fall_criteria='root height <0.65m or tilt >60deg; record but do not reset or terminate',
        caveats=['Root damping/friction/armature is a synthetic dynamics ablation, not a real-robot heading controller.',
                 'Latent model14000 was previously recorded in Isaac; this is a new MuJoCo baseline, not that Isaac trajectory.',
                 'One fixed-state rollout per setting; full three-parameter group changes together.'])
    history = {key: [] for key in ('time', 'obs', 'action', 'target', 'command')}
    physics = {key: [] for key in ('time', 'qpos', 'qvel', 'actuator_force')}
    frames = {key: [] for key in ('time', 'qpos', 'qvel', 'body_velocity', 'yaw_drift_deg', 'height', 'tilt_deg')}
    previous_yaw = total_yaw = 0.0
    fall = None
    steps = round(args.seconds / core.PHYSICS_DT)
    assert steps % core.DECIMATION == 0
    print('[CONTRACT]', args.variant, args.mode, manifest['actor_input_dim'], manifest['root_passive_after'], flush=True)
    try:
        for step in range(steps):
            # Match the previous user-policy MuJoCo recorder's state refresh.
            mujoco.mj_forward(m, d)
            if step % core.DECIMATION == 0:
                obs = sim.observation()
                action = engine.run(None, {engine.get_inputs()[0].name: obs})[0][0]
                assert np.isfinite(obs).all() and np.isfinite(action).all()
                sim.set_action(action)
                for key, value in zip(history, (d.time, obs[0].copy(), action.copy(), sim.target.copy(), command.copy())):
                    history[key].append(value)
            old_time = d.time
            sim.step()
            assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            assert math.isclose(d.time - old_time, core.PHYSICS_DT, abs_tol=1e-10)
            rotation = core._quat_to_mat_wxyz(d.qpos[3:7])
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            total_yaw += math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
            previous_yaw = yaw
            tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1, 1))))
            if fall is None and (d.qpos[2] < .65 or tilt > 60):
                fall = float(d.time)
            for key, value in zip(physics, (d.time, d.qpos.copy(), d.qvel.copy(), d.actuator_force.copy())):
                physics[key].append(value)
            if (step + 1) % core.DECIMATION == 0:
                values = (d.time, d.qpos.copy(), d.qvel.copy(), rotation.T @ d.qvel[:3],
                          math.degrees(total_yaw), float(d.qpos[2]), tilt)
                for key, value in zip(frames, values):
                    frames[key].append(value)
            if (step + 1) % 5000 == 0:
                print('[SIM]', args.variant, args.mode, round(d.time, 2), 'yaw', round(math.degrees(total_yaw), 2), 'fall', fall, flush=True)
        history, physics, frames = [{key: np.asarray(value) for key, value in group.items()}
                                   for group in (history, physics, frames)]
        for name, group in (('policy', history), ('physics', physics), ('frames', frames)):
            assert all(np.isfinite(value).all() for value in group.values())
            np.savez_compressed(output / f'{name}.npz', **group)
        steady = frames['time'] >= 5
        manifest.update(simulation_seconds=float(d.time), first_fall_time=fall,
            final_yaw_drift_deg=float(frames['yaw_drift_deg'][-1]),
            displacement_xy=(d.qpos[:2] - initial_qpos[:2]).tolist(),
            mean_body_vx=float(frames['body_velocity'][:, 0].mean()),
            mean_body_vx_after_5s=float(frames['body_velocity'][steady, 0].mean()) if steady.any() else None,
            fps=50, frames_per_view=len(frames['time']),
            warnings={mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(d.warning) if w.number})
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print('[RESULT]', args.variant, args.mode, manifest['mean_body_vx_after_5s'], manifest['final_yaw_drift_deg'], 'fall', fall, flush=True)
        if not args.no_video:
            render(m, d, frames, manifest, args, output, imageio, Image, ImageDraw, ImageFont)
        for path, expected in hashes.items():
            assert reference.sha(path) == expected, f'Source changed: {path}'
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print('[COMPLETE]', output, flush=True)
    finally:
        sim.close()


def render(m, d, frames, manifest, args, output, imageio, Image, ImageDraw, ImageFont):
    mujoco = core.mujoco
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1280)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 720)
    renderer = mujoco.Renderer(m, width=1280, height=720)
    camera = mujoco.MjvCamera()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    try:
        for view in ('whole', 'feet'):
            path = output / f't1_{args.variant}_{args.mode}_forward{round(args.vx*10):02d}_{view}.mp4'
            camera.azimuth = 130
            camera.elevation = -15 if view == 'whole' else -10
            camera.distance = 3.7 if view == 'whole' else 1.55
            with imageio.get_writer(str(path), fps=50, codec='libx264', quality=8,
                    macro_block_size=1, ffmpeg_log_level='error', output_params=['-threads', '2', '-movflags', '+faststart']) as writer:
                for i, t in enumerate(frames['time']):
                    d.qpos[:] = frames['qpos'][i]
                    d.qvel[:] = frames['qvel'][i]
                    d.time = t
                    mujoco.mj_forward(m, d)
                    camera.lookat[:] = [d.qpos[0], d.qpos[1], .85 if view == 'whole' else .25]
                    renderer.update_scene(d, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((8, 8, 1272, 124), fill=(18, 22, 28))
                    root_label = 'damping=1 frictionloss=.55 armature=.1' if args.mode == 'vendor_root' else 'all zero'
                    lines = [f'USER T1 {args.variant} ONNX | {args.mode} | {view} | OWN POLICY PD',
                             f'ROOT 6 DOFs: {root_label} | cmd=({args.vx:.2f}, 0, 0) | t={t:.2f}s',
                             f'NO heading obs / compensation / resets | vx={frames["body_velocity"][i,0]:+.3f} m/s',
                             f'yaw={frames["yaw_drift_deg"][i]:+.1f} deg | first fall={manifest["first_fall_time"] if manifest["first_fall_time"] is not None and manifest["first_fall_time"]<=t else None}']
                    for row, line in enumerate(lines):
                        draw.text((18, 12 + row*28), line, font=font, fill='white')
                    writer.append_data(np.asarray(frame))
                    if i in (0, 249, 749, len(frames['time']) - 1):
                        frame.save(output / f'preview_{view}_{i:04d}.png')
                    if (i + 1) % 500 == 0:
                        print('[VIDEO]', args.variant, args.mode, view, i + 1, flush=True)
            manifest['videos'].append(str(path))
    finally:
        renderer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(SOURCES), required=True)
    parser.add_argument('--mode', choices=('baseline', 'vendor_root'), required=True)
    parser.add_argument('--vx', type=float, default=.8)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.seconds) or args.seconds <= 0 or not np.isfinite(args.vx):
        parser.error('Require new output directory and finite command/positive duration')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
    reference.torch.set_num_threads(1)
    record(args)
