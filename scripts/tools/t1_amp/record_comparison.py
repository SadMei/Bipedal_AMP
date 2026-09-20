"""Record frozen estimator-free T1 AMP checkpoints without changing training.

Reuse the existing T1 AMP deployment, restore nominal values from saved YAML,
strictly export the actor, and render stored trajectories with provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[3]
SIMTOOL = REPO.parent / 'mjw_simtool'
sys.path.insert(0, str(REPO / 'external/rsl_rl'))
sys.path.insert(0, str(REPO / 'scripts/rsl_rl'))
sys.path.insert(0, str(SIMTOOL / 'scripts'))
from rsl_rl.modules import ActorCritic
from t1_heading_ablation import load_saved_config
import sim2simT1AMP.run_sim2sim_standalone as t1

core = t1.core
XML = SIMTOOL / 'scripts/sim2simT1/assets/men_t1_path_25dofs_collision_mc_amt.xml'
BASE_TERMS = ('base_ang_vel', 'root_local_rot_tan_norm', 'velocity_commands',
              'joint_pos', 'joint_vel', 'actions')
WIDTHS = dict(base_lin_vel=3, base_ang_vel=3, root_local_rot_tan_norm=6,
              velocity_commands=3, joint_pos=25, joint_vel=25, actions=25,
              key_body_pos_b=18, heading_error=2, base_height=1, foot_contact=2,
              foot_lin_vel_w_xy=4, undesired_contact_count=1, base_contact=1)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def active(group):
    return tuple(k for k, v in group.items() if isinstance(v, dict) and v.get('func'))


def policy_contract(env, agent):
    assert agent['policy']['class_name'] == 'ActorCritic'
    assert agent['obs_groups']['policy'] == ['policy']
    assert agent['obs_groups']['critic'] == ['critic', 'state_est_critic', 'reward_privileged']
    assert agent['clip_actions'] is None
    assert env['observations'].get('state_est_label') is None
    policy = env['observations']['policy']
    terms = active(policy)
    if terms not in (BASE_TERMS, BASE_TERMS + ('heading_error',)):
        raise ValueError(f'Unsupported actor terms: {terms}')
    with_heading = terms[-1] == 'heading_error'
    if with_heading:
        assert policy['heading_error']['func'].endswith(':integrated_heading_error')
    assert policy['joint_pos']['func'].endswith(':joint_pos')
    assert policy['joint_vel']['func'].endswith(':joint_vel')
    assert policy['root_local_rot_tan_norm']['func'].endswith(':root_local_rot_tan_norm')
    for name in terms:
        term = policy[name]
        assert term['scale'] is None and term['clip'] is None and term['modifiers'] is None
    for name in ('joint_pos', 'joint_vel'):
        assert tuple(policy[name]['params']['asset_cfg']['joint_names']) == core.JOINT_NAMES
    expected = {
        'policy': terms,
        'critic': ('base_lin_vel',) + BASE_TERMS + ('key_body_pos_b', 'heading_error'),
        'state_est_critic': ('base_height',),
        'reward_privileged': ('foot_contact', 'foot_lin_vel_w_xy', 'undesired_contact_count', 'base_contact'),
    }
    widths = {}
    for name, names in expected.items():
        group = env['observations'][name]
        assert active(group) == names
        assert group['history_length'] == 5 and group['flatten_history_dim']
        assert all(group[term]['history_length'] == 0 for term in names)
        widths[name] = sum(WIDTHS[term] for term in names) * 5
    assert widths['policy'] == (445 if with_heading else 435)
    assert sum(widths[name] for name in agent['obs_groups']['critic']) == 595
    return widths, with_heading


def load_policy(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    env = load_saved_config(checkpoint.parent / 'params/env.yaml')
    agent = load_saved_config(checkpoint.parent / 'params/agent.yaml')
    widths, heading = policy_contract(env, agent)
    before = sha(checkpoint)
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert sha(checkpoint) == before
    assert checkpoint.stem == f"model_{int(loaded['iter'])}"
    assert all(torch.isfinite(value).all() for value in loaded['model_state_dict'].values())
    kwargs = dict(agent['policy'])
    kwargs.pop('class_name')
    obs = TensorDict({key: torch.zeros(1, width) for key, width in widths.items()}, batch_size=[1])
    policy = ActorCritic(obs, agent['obs_groups'], 25, **kwargs).cpu()
    policy.load_state_dict(loaded['model_state_dict'], strict=True)
    policy.eval()
    return policy, env, agent, dict(
        checkpoint=str(checkpoint), checkpoint_sha256=before, iteration=int(loaded['iter']),
        env_sha256=sha(checkpoint.parent / 'params/env.yaml'),
        agent_sha256=sha(checkpoint.parent / 'params/agent.yaml'),
        actor_heading=heading, actor_input_dim=widths['policy'], critic_input_dim=595,
        history_frames=5, estimator=False, strict_full_state_load=True,
    )


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def export_policy(policy, metadata, output):
    spec = importlib.util.spec_from_file_location('t1_amp_exporter', REPO / 'source/legged_lab/legged_lab/rsl_rl/exporter.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.export_policy_as_onnx(policy, str(output), normalizer=policy.actor_obs_normalizer)
    path = output / 'policy.onnx'
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    engine = session(path)
    assert engine.get_inputs()[0].shape == [1, metadata['actor_input_dim']]
    assert engine.get_outputs()[0].shape == [1, 25]
    error = 0.
    for obs in np.random.default_rng(42).normal(size=(64, 1, metadata['actor_input_dim'])).astype(np.float32):
        actual = engine.run(None, {engine.get_inputs()[0].name: obs})[0]
        with torch.no_grad():
            expected = policy.actor(policy.actor_obs_normalizer(torch.from_numpy(obs))).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
        error = max(error, float(np.abs(actual - expected).max()))
    save_json(output / 'export_verification.json', dict(**metadata, onnx_sha256=sha(path),
        synthetic_samples=64, max_abs_error=error))
    return engine


def resolve(value, name):
    if not isinstance(value, dict):
        return float(value)
    matches = [number for pattern, number in value.items() if re.fullmatch(pattern, name)]
    assert len(matches) == 1, (name, value)
    return float(matches[0])


def restore_nominal(env):
    robot = env['scene']['robot']
    action = env['actions']['joint_pos']
    assert tuple(action['joint_names']) == core.JOINT_NAMES
    assert action['use_default_offset'] and action['preserve_order']
    assert action['clip'] is None, 'These two no-clip runs must not get legacy target clipping'
    assert float(env['sim']['dt']) * env['decimation'] == .02
    fields = ('stiffness', 'damping', 'armature', 'effort_limit_sim', 'velocity_limit_sim')
    nominal = {field: [] for field in fields}
    for name in core.JOINT_NAMES:
        matches = [value for value in robot['actuators'].values()
                   if any(re.fullmatch(pattern, name) for pattern in value['joint_names_expr'])]
        assert len(matches) == 1
        actuator = matches[0]
        assert actuator['class_type'].endswith(':DelayedImplicitActuator')
        for field in fields:
            nominal[field].append(resolve(actuator[field], name))
    nominal['default_pos'] = [resolve(robot['init_state']['joint_pos'], name) for name in core.JOINT_NAMES]
    nominal['action_scale'] = [resolve(action['scale'], name) for name in core.JOINT_NAMES]
    for constant, field in (('KP', 'stiffness'), ('KD', 'damping'), ('ARMATURE', 'armature'),
                            ('VELOCITY_LIMIT', 'velocity_limit_sim'), ('DEFAULT_JOINT_POS', 'default_pos'),
                            ('ACTION_SCALE', 'action_scale')):
        np.testing.assert_allclose(getattr(core, constant), nominal[field], atol=1e-8)
        setattr(core, constant, np.asarray(nominal[field]))
    core.START_HEIGHT = float(robot['init_state']['pos'][2])
    core.TARGET_LOWER = core.TARGET_UPPER = None
    return nominal


def rollout(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    policy, env, agent, metadata = load_policy(args.checkpoint)
    nominal = restore_nominal(env)
    args.output.mkdir(parents=True)
    engine = export_policy(policy, metadata, args.output / 'export')
    sim = core.BSRLMujoco(XML, headless=True, command=args.command, delay_steps=0,
                         actor_include_key_body=False, actor_include_heading_error=metadata['actor_heading'])
    np.testing.assert_allclose(sim.effort, nominal['effort_limit_sim'])
    np.testing.assert_allclose(sim.velocity_limit, nominal['velocity_limit_sim'])
    np.testing.assert_allclose(sim.command, args.command)
    assert sim.observation().shape == (1, metadata['actor_input_dim'])
    sim.reset()
    mujoco = core.mujoco
    mujoco.mj_saveLastXML(str(args.output / 'model.xml'), sim.model)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for source in (Path(__file__), Path(t1.__file__), Path(core.__file__)):
        shutil.copy2(source, snapshot / (source.parent.name + '_' + source.name))
    manifest = dict(**metadata, label=args.label, model_xml=str(XML), model_xml_sha256=sha(XML),
        runtime_model_sha256=sha(args.output / 'model.xml'), onnx_sha256=sha(args.output / 'export/policy.onnx'),
        training_urdf=env['scene']['robot']['spawn']['asset_path'],
        training_urdf_sha256=sha(env['scene']['robot']['spawn']['asset_path']),
        command=list(args.command), requested_seconds=args.seconds, nominal=nominal,
        physics_dt=core.PHYSICS_DT, policy_dt=core.POLICY_DT, training_physics_dt=env['sim']['dt'],
        target_clipping=None, automatic_resets=0, yaw_command_compensation=False,
        evaluation='MuJoCo T1 AMP deployment, ordinary feet, implicit position PD; nominal static reset; no DR/noise/delay/push',
        caveats=['MuJoCo uses 1ms physics vs 5ms Isaac training; not an exact training-dynamics replay.',
                 'Retains existing deployment joint-velocity clamps after integration.',
                 '9/14 uses ground-truth simulated heading in its trained actor input; 9/15 does not.'])
    save_json(args.output / 'manifest.json', manifest)
    rows = {key: [] for key in ('time_s', 'qpos', 'qvel', 'obs', 'action', 'target', 'command', 'base_velocity_b')}
    physics = {key: [] for key in ('time_s', 'torque', 'joint_velocity', 'qpos', 'qvel')}
    stop = None
    completed_steps = 0
    try:
        for step in range(round(args.seconds / core.PHYSICS_DT)):
            mujoco.mj_forward(sim.model, sim.data)
            if step % core.DECIMATION == 0:
                obs = sim.observation()
                action = engine.run(None, {engine.get_inputs()[0].name: obs})[0][0]
                assert np.isfinite(obs).all() and np.isfinite(action).all()
                sim.set_action(action)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY,
                                        sim.model.body(core.ROOT_BODY_NAME).id, velocity, 1)
                values = (sim.data.time, sim.data.qpos.copy(), sim.data.qvel.copy(), obs[0].copy(),
                          action.copy(), sim.target.copy(), sim.command.copy(), velocity)
                for key, value in zip(rows, values):
                    rows[key].append(value)
            values = (sim.data.time, sim.data.actuator_force[sim._act_ids].copy(),
                      sim.data.qvel[sim._qvel_adr].copy(), sim.data.qpos.copy(), sim.data.qvel.copy())
            for key, value in zip(physics, values):
                physics[key].append(value)
            sim.step()
            completed_steps += 1
            if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                stop = 'nonfinite'; break
            rotation = core._quat_to_mat_wxyz(sim.data.qpos[3:7])
            if sim.data.qpos[2] < .5 or rotation[2, 2] < .5:
                stop = 'base_height' if sim.data.qpos[2] < .5 else 'bad_orientation'
                break
            if (step + 1) % 5000 == 0:
                print('[SIM]', args.label, round(sim.data.time, 2), flush=True)
    except Exception as error:
        stop = f'{type(error).__name__}: {error}'
    finally:
        sim.close()
    rows = {key: np.asarray(value) for key, value in rows.items()}
    physics = {key: np.asarray(value) for key, value in physics.items()}
    np.savez_compressed(args.output / 'policy.npz', **rows)
    np.savez_compressed(args.output / 'physics.npz', **physics)
    w, x, y, z = rows['qpos'][:, 3:7].T
    yaw = np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))
    yaw -= yaw[0]
    summary = dict(seconds=float(sim.data.time), physics_steps=completed_steps, policy_frames=len(yaw),
        stop=stop, final_yaw_deg=float(yaw[-1]), max_abs_yaw_deg=float(np.abs(yaw).max()),
        final_base_pos=sim.data.qpos[:3].tolist(), mean_body_vx=float(rows['base_velocity_b'][:, 3].mean()),
        max_abs_lateral_m=float(np.abs(rows['qpos'][:, 1]-rows['qpos'][0, 1]).max()),
        warnings={mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(sim.data.warning) if w.number})
    save_json(args.output / 'summary.json', summary)
    print('[SUMMARY]', json.dumps(summary), flush=True)
    assert sha(args.checkpoint) == metadata['checkpoint_sha256']


def render(args):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    mujoco = core.mujoco
    if args.video.exists():
        raise FileExistsError(args.video)
    manifest = json.loads((args.record / 'manifest.json').read_text())
    summary = json.loads((args.record / 'summary.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(args.record / 'model.xml'))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = 720; model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, width=720, height=720)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    font = ImageFont.truetype('DejaVuSans.ttf', 17)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    try:
        with np.load(args.record / 'policy.npz') as rows, imageio.get_writer(
                str(args.video), fps=50, codec='libx264', quality=8, macro_block_size=1,
                ffmpeg_params=['-threads', '2']) as writer:
            w, x, y, z = rows['qpos'][:, 3:7].T
            yaw = np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))
            yaw -= yaw[0]
            for i, time in enumerate(rows['time_s']):
                data.qpos[:] = rows['qpos'][i]; data.qvel[:] = rows['qvel'][i]
                data.time = float(time); mujoco.mj_forward(model, data)
                camera.lookat[:] = data.body(core.ROOT_BODY_NAME).xpos + [0, 0, -.16]
                camera.distance, camera.azimuth, camera.elevation = 3.3, 125, -14
                renderer.update_scene(data, camera=camera, scene_option=option)
                body = renderer.render().copy()
                left = data.body('left_ankle_pitch_Link').xpos
                right = data.body('right_ankle_pitch_Link').xpos
                camera.lookat[:] = (left+right)/2 + [0, 0, .20]
                camera.distance = max(1.3, 1.7*(np.linalg.norm(left-right)+.35))
                camera.azimuth, camera.elevation = yaw[i] + 125, -15
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(np.concatenate([body, renderer.render().copy()], axis=1))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1440, 62), fill=(15, 15, 15))
                heading = 'ON (trained input)' if manifest['actor_heading'] else 'OFF'
                draw.text((12, 8), f"T1 {manifest['label']} | model_{manifest['iteration']} | Actor heading {heading} | {manifest['actor_input_dim']}D", fill='white', font=font)
                draw.text((12, 35), f"MuJoCo sim2sim | cmd={manifest['command']} | t={time:.2f}s | body vx={rows['base_velocity_b'][i, 3]:.2f} m/s | no reset / no yaw-command compensation", fill='white', font=font)
                draw.line((720, 62, 720, 720), fill='white', width=2)
                draw.rectangle((0, 678, 1440, 720), fill=(15, 15, 15))
                draw.text((12, 689), f"Body view | yaw {yaw[i]:+.1f} deg | XY {data.qpos[0]:+.2f}/{data.qpos[1]:+.2f} m", fill='white', font=font)
                draw.text((734, 689), 'Feet close-up | actual recorded states | 1x', fill='white', font=font)
                if summary['stop'] and i >= len(yaw)-25:
                    draw.rectangle((10, 72, 1430, 110), fill=(105, 25, 20))
                    draw.text((20, 82), f"Stopped at {summary['seconds']:.3f}s: {summary['stop']}", fill='white', font=font)
                if i == min(250, len(yaw)-1):
                    frame.save(args.record / 'preview.png')
                writer.append_data(np.asarray(frame))
                if (i+1) % 250 == 0:
                    print('[RENDER]', args.video.name, i+1, '/', len(yaw), flush=True)
    finally:
        renderer.close()


def verify(args):
    import imageio.v2 as imageio
    manifest = json.loads((args.record / 'manifest.json').read_text())
    summary = json.loads((args.record / 'summary.json').read_text())
    policy, env, agent, metadata = load_policy(Path(manifest['checkpoint']))
    assert metadata['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert sha(args.record / 'export/policy.onnx') == manifest['onnx_sha256']
    assert sha(args.record / 'model.xml') == manifest['runtime_model_sha256']
    engine = session(args.record / 'export/policy.onnx')
    with np.load(args.record / 'policy.npz') as rows, np.load(args.record / 'physics.npz') as physics:
        for archive in (rows, physics):
            for key in archive.files:
                assert np.isfinite(archive[key]).all(), key
                assert len(archive[key]) == len(archive['time_s']), key
        n = len(rows['time_s'])
        assert rows['obs'].shape == (n, manifest['actor_input_dim'])
        np.testing.assert_allclose(rows['command'], np.tile(manifest['command'], (n, 1)))
        np.testing.assert_allclose(rows['time_s'], np.arange(n)*.02, rtol=0, atol=1e-8)
        np.testing.assert_allclose(rows['target'], np.asarray(manifest['nominal']['default_pos']) +
                                   rows['action']*manifest['nominal']['action_scale'], atol=1e-7, rtol=0)
        assert (np.abs(physics['torque']) <= np.asarray(manifest['nominal']['effort_limit_sim']) + 1e-6).all()
        assert (np.abs(physics['joint_velocity']) <= np.asarray(manifest['nominal']['velocity_limit_sim']) + 1e-6).all()
        if not summary['stop']:
            assert n == 1500 and summary['physics_steps'] == 30000
        indices = np.unique(np.linspace(0, n-1, min(100, n), dtype=int))
        max_error = 0.
        for i in indices:
            obs = rows['obs'][i:i+1]
            actual = engine.run(None, {engine.get_inputs()[0].name: obs})[0][0]
            with torch.no_grad():
                expected = policy.actor(policy.actor_obs_normalizer(torch.from_numpy(obs))).numpy()[0]
            np.testing.assert_allclose(actual, rows['action'][i], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
            max_error = max(max_error, float(np.abs(actual-expected).max()))
        frames = 0
        with imageio.get_reader(args.video) as reader:
            info = reader.get_meta_data()
            for frame in reader:
                assert frame.shape == (720, 1440, 3)
                frames += 1
        assert frames == n and info['fps'] == 50
    result = dict(integrity_verified=True, decoded_frames=frames, video_seconds=frames/50,
        actual_observation_samples=len(indices), pytorch_onnx_max_abs_error=max_error,
        video_sha256=sha(args.video), summary=summary)
    save_json(args.record / 'verification.json', result)
    print('[VERIFIED]', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sim = sub.add_parser('rollout')
    sim.add_argument('--checkpoint', type=Path, required=True)
    sim.add_argument('--output', type=Path, required=True)
    sim.add_argument('--label', required=True)
    sim.add_argument('--command', type=float, nargs=3, default=[.6, 0, 0])
    sim.add_argument('--seconds', type=float, default=30)
    for mode in ('render', 'verify'):
        child = sub.add_parser(mode)
        child.add_argument('--record', type=Path, required=True)
        child.add_argument('--video', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dict(rollout=rollout, render=render, verify=verify)[args.mode](args)


if __name__ == '__main__':
    main()
