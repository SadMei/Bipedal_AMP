"""CPU-only, strict export of the verified 240D estimator-free BSRL policy.

Uses the repository ActorCritic and AMP exporter without starting Isaac. Output
must be a new directory; neither training runs nor previous exports are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

if __package__:
    from ._paths import REPO_ROOT
else:
    from _paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "external/rsl_rl"))
sys.path.insert(0, str(REPO_ROOT / "scripts/rsl_rl"))
from rsl_rl.modules import ActorCritic
from t1_heading_ablation import load_saved_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def critic_widths(env):
    """The two verified critic contracts; actor/deployment input stays 240D."""
    critic = env['observations']['critic']
    active = tuple(k for k,v in critic.items() if isinstance(v,dict) and v.get('func'))
    base = ('base_lin_vel','base_ang_vel','root_local_rot_tan_norm','velocity_commands',
            'joint_pos','joint_vel','actions','key_body_pos_b')
    heading = ('heading_frame_error','heading_frame_lin_vel_xy')
    if critic['history_length'] != 5 or active not in (base,base+heading):
        raise ValueError(f'Unsupported frozen critic observation contract: {active}')
    if active == base+heading:
        for name in heading:
            if critic[name]['func'] != 'legged_lab.tasks.locomotion.amp.mdp.heading_frame:'+name:
                raise ValueError('Unsupported privileged heading function')
    widths = {'policy':240,'critic':305 if active==base+heading else 285,
              'state_est_critic':5,'reward_privileged':60}
    return widths


def load_policy(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    run = checkpoint.parent
    agent = load_saved_config(run / "params/agent.yaml")
    env = load_saved_config(run / "params/env.yaml")
    kwargs = dict(agent["policy"])
    if kwargs.pop("class_name") != "ActorCritic" or kwargs.get("state_dependent_std"):
        raise ValueError("Only ordinary estimator-free ActorCritic supported")
    if agent["obs_groups"]["policy"] != ["policy"] or agent["obs_groups"]["critic"] != [
        "critic", "state_est_critic", "reward_privileged"
    ]:
        raise ValueError("Unsupported observation groups")
    active = tuple(k for k, v in env["observations"]["policy"].items()
                   if isinstance(v, dict) and v.get("func"))
    if active != ("base_ang_vel", "root_local_rot_tan_norm", "velocity_commands",
                  "joint_pos", "joint_vel", "actions"):
        raise ValueError(f"Unsupported actor layout: {active}")
    if env["observations"].get("state_est_label") is not None:
        raise ValueError("Estimator input is not supported")
    if env["observations"]["policy"]["history_length"] != 5:
        raise ValueError("Expected five-frame actor history")
    before = sha(checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if before != sha(checkpoint):
        raise RuntimeError("Checkpoint changed during loading")
    state = loaded["model_state_dict"]
    widths = critic_widths(env)
    total_critic = sum(widths[name] for name in agent['obs_groups']['critic'])
    if tuple(state["actor.0.weight"].shape) != (512, 240) or tuple(state["critic.0.weight"].shape) != (512, total_critic):
        raise ValueError("Unexpected Actor/Critic dimensions")
    if not all(torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError("Non-finite model parameters")
    if checkpoint.stem != f"model_{int(loaded['iter'])}":
        raise ValueError("Checkpoint filename and internal iteration differ")
    # Group widths from the frozen 240D BSRL observation contract. Strict
    # complete-state loading also checks the combined 350/370D critic, although
    # deployment only uses the actor.
    obs = TensorDict({k: torch.zeros(1,n) for k,n in widths.items()},batch_size=[1])
    policy = ActorCritic(obs, agent["obs_groups"], 12, **kwargs).cpu()
    policy.load_state_dict(state, strict=True)
    policy.eval()
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=before,
                    iteration=str(loaded["iter"]), run=run.name,
                    env_sha256=sha(run / "params/env.yaml"),
                    agent_sha256=sha(run / "params/agent.yaml"),
                    critic_total_dim=str(total_critic),
                    policy_input="240D term-major 5-step history; no actor heading; no estimator")
    return policy, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    policy, metadata = load_policy(args.checkpoint)
    path = REPO_ROOT / "source/legged_lab/legged_lab/rsl_rl/exporter.py"
    spec = importlib.util.spec_from_file_location("bsrl_cpu_amp_exporter", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    args.output.mkdir(parents=True)
    exporter.export_policy_as_onnx(policy, str(args.output), normalizer=policy.actor_obs_normalizer)
    output = args.output / "policy.onnx"
    model = onnx.load(output)
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 240]
    assert session.get_outputs()[0].shape == [1, 12]
    max_error = 0.
    for sample in np.random.default_rng(42).normal(size=(64, 1, 240)).astype(np.float32):
        actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
        with torch.no_grad():
            expected = policy.actor(policy.actor_obs_normalizer(torch.from_numpy(sample))).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    report = dict(**metadata, onnx_sha256=sha(output), synthetic_samples=64,
                  max_abs_error=max_error, strict_full_state_load=True)
    (args.output / "export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
