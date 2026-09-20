"""BSRL latent contract, reset-safe PPO/AMP, export and T1 compatibility tests."""

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import test_t1_latent as shared

ROOT = shared.ROOT
models, algorithm = shared.models, shared.algorithm
NEXT, VALID = shared.NEXT, shared.VALID
BASELINE = ROOT / "logs/rsl_rl/bsrl_amp/2026-09-16_21-06-48_accad34_reduced_parallel_headingpriv_noestimator_nomirror_8192"


def observations(n=8):
    return TensorDict({"policy": torch.randn(n, 480), "critic": torch.randn(n, 370),
                       "disc": torch.randn(n, 4, 39), "disc_demo": torch.randn(n, 4, 39)}, [n])


def policy(obs=None):
    return models.BSRLLatentActorCritic(observations() if obs is None else obs, shared.GROUPS, 12,
        encoder_dims=(32, 16), decoder_dims=(16,), actor_hidden_dims=(32, 16),
        critic_hidden_dims=(16,), noise_std_type="scalar")


def test_bsrl_history_and_recent_frames():
    frames = torch.arange(480).reshape(1, 10, 48).float()
    raw = torch.cat([part.flatten(1) for part in frames.split(models.BSRL_TERM_DIMS, -1)], -1)
    p = policy()
    torch.testing.assert_close(p.actor.layout(raw), frames)
    torch.testing.assert_close(p.actor.layout.latest(raw), frames[:, -1])
    captured = []
    hook = p.actor.policy.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    p.actor.eval()(raw)
    torch.testing.assert_close(captured[0][:, 20:], p.actor.normalizer(frames)[:, -2:].flatten(1))
    assert captured[0].shape == (1, 116)
    hook.remove()


def test_prediction_target_gradient_and_no_privileged_leak():
    obs = observations()
    p = policy(obs)
    action = p.act_inference(obs)
    obs["critic"][:] = float("nan")
    obs[NEXT] = torch.randn(8, 48)
    obs[VALID] = torch.ones(8, 1)
    torch.testing.assert_close(p.act_inference(obs), action)
    assert p.decoder[-1].out_features == 33
    torch.testing.assert_close(p.prediction_indices, torch.tensor(list(range(9)) + list(range(12, 36))))
    obs[VALID][0] = 0
    obs[NEXT][0] = float("nan")
    loss, metrics = p.compute_auxiliary_loss(obs)
    assert torch.isfinite(loss)
    assert metrics["latent/valid_fraction"] == 0.875
    loss.backward()
    for layer in (p.actor.encoder[0], p.actor.latent_mu, p.latent_logvar, p.decoder[0]):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    obs[VALID][:] = 0
    assert p.compute_auxiliary_loss(obs)[0] == 0


@pytest.mark.parametrize("width", [240, 500, 870])
def test_bad_bsrl_width_rejected(width):
    obs = observations()
    obs["policy"] = torch.randn(8, width)
    with pytest.raises(ValueError, match="48-feature"):
        policy(obs)


def test_bsrl_real_ppoamp_update():
    from rsl_rl.storage import RolloutStorage, CircularBuffer
    obs = observations()
    p = policy(obs)
    amp = {"loss_type": "LSGAN", "disc_obs_dim": 39, "disc_obs_steps": 4,
           "disc_trunk_weight_decay": 1e-4, "disc_linear_weight_decay": 1e-2,
           "disc_learning_rate": 1e-4, "step_dt": 0.02, "grad_penalty_scale": 10,
           "amp_discriminator": {"hidden_dims": [16, 8], "task_style_lerp": 0.4}}
    alg = algorithm.LatentPPOAMP(p, RolloutStorage("rl", 8, 4, obs, [12]),
        CircularBuffer(4, 8, "cpu"), CircularBuffer(4, 8, "cpu"), amp_cfg=amp,
        num_learning_epochs=2, num_mini_batches=4)
    before = p.actor.encoder[0].weight.detach().clone()
    count = p.actor.normalizer.count.item()
    with torch.inference_mode():
        for step in range(4):
            alg.act(obs)
            next_obs = observations()
            dones = torch.tensor([True, False, False, False, False, False, False, False])
            alg.process_env_step(next_obs, torch.rand(8), dones, {"time_outs": dones})
            stored = alg.storage.observations[step]
            torch.testing.assert_close(stored["policy"], obs["policy"])
            torch.testing.assert_close(stored[NEXT][1:], p.actor.layout.latest(next_obs["policy"])[1:])
            assert not stored[VALID][0] and torch.all(stored[NEXT][0] == 0)
            assert NEXT not in obs and NEXT not in next_obs
            obs = next_obs
        alg.compute_returns(obs)
    assert p.actor.normalizer.count == count
    metrics = alg.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["latent/valid_fraction"] == 0.875
    assert "symmetry" not in metrics and "state_est" not in metrics
    assert p.actor.normalizer.count > count
    assert not torch.equal(before, p.actor.encoder[0].weight)


def test_bsrl_export_has_no_decoder_or_random_latent(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    p = policy().eval()
    raw = torch.randn(1, 480)
    traced = torch.jit.script(p.actor)
    torch.testing.assert_close(traced(raw), p.actor(raw))
    path = tmp_path / "policy.onnx"
    torch.onnx.export(p.actor, raw, str(path), opset_version=17, dynamo=False,
                      input_names=["obs"], output_names=["actions"])
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert not any("Random" in node.op_type for node in graph.graph.node)
    assert not any("decoder" in v.name or "logvar" in v.name for v in graph.graph.initializer)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {"obs": raw.numpy()})[0],
                               p.actor(raw).detach().numpy(), rtol=1e-4, atol=1e-5)


def saved_baseline():
    from t1_heading_ablation import load_saved_config
    if not (BASELINE / "params/env.yaml").is_file():
        pytest.skip("Local frozen BSRL baseline unavailable")
    return load_saved_config(BASELINE / "params/env.yaml"), load_saved_config(BASELINE / "params/agent.yaml")


def test_baseline_and_only_documented_config_differences(monkeypatch):
    config = __import__(shared.PKG + ".bsrl_config", fromlist=["*"])
    monkeypatch.setitem(sys.modules, "legged_lab.latent.bsrl_config", config)
    from bsrl_latent_ablation import validate_baseline, expected_configuration
    from t1_heading_ablation import configuration_differences
    env, agent = saved_baseline()
    validate_baseline(env, agent)
    e, a = expected_configuration(env, agent, run_name="latent", num_envs=8192, max_iterations=25000, device="cuda:0")
    for key in ("commands", "rewards", "scene", "sim", "motion_data", "actions", "events", "curriculum"):
        assert configuration_differences(e[key], env[key]) == []
    assert e["observations"]["critic"] == env["observations"]["critic"]
    assert a["algorithm"]["symmetry_cfg"] is None
    assert a["algorithm"]["amp_cfg"] == agent["algorithm"]["amp_cfg"]
    assert a["algorithm"]["num_mini_batches"] == 4
    assert a["policy"]["actor_hidden_dims"] == [2048, 1024, 512, 128]
    assert env["observations"]["policy"]["history_length"] == 5
    e["log_dir"] = env["log_dir"]
    assert {d["path"] for d in configuration_differences(env, e)} == {"observations.policy.history_length"}


@pytest.mark.parametrize("corruption", ["mirror", "closed_loop", "heading", "labels", "dataset"])
def test_wrong_baselines_rejected(corruption):
    from bsrl_latent_ablation import validate_baseline
    e, a = saved_baseline()
    if corruption == "mirror":
        a["algorithm"]["symmetry_cfg"] = {"use_mirror_loss": True}
    elif corruption == "closed_loop":
        e["scene"]["robot"]["class_type"] = "ClosedAnkleArticulation"
    elif corruption == "heading":
        e["observations"]["policy"]["heading_error"] = {"func": "integrated_heading_error"}
    elif corruption == "labels":
        e["observations"]["state_est_label"] = {"label": True}
    else:
        e["motion_data"]["motion_dataset"]["motion_data_weights"] = {"clip": 1.0}
    with pytest.raises(ValueError):
        validate_baseline(e, a)


def test_existing_t1_checkpoint_and_outputs_are_exactly_compatible():
    run = ROOT / "logs/rsl_rl/men_t1_amp/2026-09-17_18-03-20_latent20_h10_recent2_predict59_kl001_nomirror_noactorheading_8192"
    old_source = run / "latent_source/source/legged_lab/legged_lab/latent/models.py"
    if not old_source.is_file():
        pytest.skip("Frozen T1 source unavailable")
    spec = importlib.util.spec_from_file_location("t1_latent_frozen", old_source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    obs = shared.observations()
    from t1_heading_ablation import load_saved_config
    options = load_saved_config(run / "params/agent.yaml")["policy"]
    options.pop("class_name")
    original = old.LatentActorCritic(obs, shared.GROUPS, 25, **options)
    original.load_state_dict(torch.load(run / "model_0.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    current = models.LatentActorCritic(obs, shared.GROUPS, 25, **options)
    current.load_state_dict(original.state_dict())
    torch.testing.assert_close(current.act_inference(obs), original.act_inference(obs), rtol=0, atol=0)
    target = shared.labels(obs)
    torch.manual_seed(52)
    old_loss = original.compute_auxiliary_loss(target)[0]
    torch.manual_seed(52)
    new_loss = current.compute_auxiliary_loss(target)[0]
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
