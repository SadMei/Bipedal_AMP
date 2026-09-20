"""CPU tests for the opt-in latent branch, without importing Isaac's UI."""

import copy
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "external/rsl_rl"))
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
# A unique package name keeps these CPU tests from polluting legged_lab imports.
PKG = "t1_latent_test_package"
package = types.ModuleType(PKG)
package.__path__ = [str(ROOT / "source/legged_lab/legged_lab/latent")]
sys.modules[PKG] = package
models = __import__(PKG + ".models", fromlist=["*"])
algorithm = __import__(PKG + ".algorithm", fromlist=["*"])
LatentActorCritic = models.LatentActorCritic
NEXT, VALID = models.NEXT_FRAME_KEY, models.VALID_KEY
torch.set_num_threads(1)


def observations(n=8):
    return TensorDict({"policy": torch.randn(n, 870), "critic": torch.randn(n, 595),
                       "disc": torch.randn(n, 4, 7), "disc_demo": torch.randn(n, 4, 7)}, [n])


GROUPS = {"policy": ["policy"], "critic": ["critic"], "discriminator": ["disc"],
          "discriminator_demonstration": ["disc_demo"]}


def policy(obs=None, **kwargs):
    return LatentActorCritic(observations() if obs is None else obs, GROUPS, 25,
                            encoder_dims=(32, 16), decoder_dims=(16,),
                            actor_hidden_dims=(32, 16), critic_hidden_dims=(16,),
                            noise_std_type="log", **kwargs)


def labels(obs, valid=True):
    result = obs.clone()
    result[NEXT] = torch.randn(obs.batch_size[0], 87)
    result[VALID] = torch.full((obs.batch_size[0], 1), float(valid))
    return result


def test_term_major_history_chronology_and_latest():
    frames = torch.arange(10 * 87).reshape(1, 10, 87).float()
    term_major = torch.cat([part.flatten(1) for part in frames.split(models.T1_TERM_DIMS, -1)], -1)
    layout = models.HistoryLayout()
    assert torch.equal(layout(term_major), frames)
    assert torch.equal(layout.latest(term_major), frames[:, -1])


def test_actual_isaac_history_reset_does_not_leak_previous_episode():
    buffer_path = Path("/root/IsaacLab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py")
    if not buffer_path.is_file():
        pytest.skip("IsaacLab source unavailable")
    spec = importlib.util.spec_from_file_location("latent_test_isaac_buffer", buffer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = module.CircularBuffer(10, 2, "cpu")
    for value in range(12):
        buffer.append(torch.full((2, 87), float(value)))
    torch.testing.assert_close(buffer.buffer[1, :, 0], torch.arange(2, 12).float())
    buffer.reset([0])
    buffer.append(torch.stack((torch.full((87,), 100.0), torch.full((87,), 12.0))))
    assert torch.all(buffer.buffer[0] == 100)
    torch.testing.assert_close(buffer.buffer[1, :, 0], torch.arange(3, 13).float())


def test_recent_two_frames_are_chronological():
    actor = policy().actor
    obs = observations()["policy"]
    captured = []
    hook = actor.policy.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    actor(obs)
    frames, _, mu = actor.encode(obs)
    torch.testing.assert_close(captured[0][:, :20], mu)
    torch.testing.assert_close(captured[0][:, 20:], frames[:, -2:].flatten(1))
    hook.remove()


def test_actor_deterministic_and_labels_do_not_leak():
    p = policy()
    obs = labels(observations())
    expected = p.act_inference(obs)
    torch.testing.assert_close(p.act_inference(obs), expected, rtol=0, atol=0)
    p.act(obs)
    torch.testing.assert_close(p.action_mean, expected)
    obs["critic"][:] = 1e5
    obs[NEXT][:] = float("nan")
    obs[VALID][:] = 0
    torch.testing.assert_close(p.act_inference(obs), expected)
    p.eval()
    torch.testing.assert_close(p.act_inference(obs), expected)


def test_auxiliary_trains_encoder_logvar_and_decoder():
    p = policy()
    loss, metrics = p.compute_auxiliary_loss(labels(observations()))
    loss.backward()
    for layer in (p.actor.encoder[0], p.actor.latent_mu, p.latent_logvar, p.decoder[0]):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert p.actor.policy[0].weight.grad is None
    assert metrics["latent/kl"] >= 0


def test_ppo_actor_gradient_reaches_encoder():
    p = policy()
    p.act_inference(observations()).square().mean().backward()
    assert p.actor.encoder[0].weight.grad.abs().sum() > 0


def test_activation_recomputation_preserves_outputs_and_gradients():
    actor = policy().actor
    reference = copy.deepcopy(actor)
    raw = torch.randn(8, 870)
    checkpointed = actor(raw)
    direct = reference._forward_impl(raw)
    torch.testing.assert_close(checkpointed, direct, rtol=0, atol=0)
    checkpointed.square().mean().backward()
    direct.square().mean().backward()
    for p, q in zip(actor.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_reset_mask_excludes_invalid_nan_targets():
    p = policy()
    obs = labels(observations())
    obs[VALID][:4] = 0
    obs[NEXT][:4] = float("nan")
    loss, metrics = p.compute_auxiliary_loss(obs)
    assert torch.isfinite(loss)
    assert metrics["latent/valid_fraction"] == 0.5
    obs[VALID][:] = 0
    loss, _ = p.compute_auxiliary_loss(obs)
    assert loss == 0
    loss.backward()


def test_kl_and_mse_reductions():
    p = policy()
    with torch.no_grad():
        for param in p.parameters():
            param.zero_()
    obs = labels(observations())
    obs[NEXT] = p.actor.normalizer._mean.expand(8, -1).clone()
    loss, metrics = p.compute_auxiliary_loss(obs)
    assert loss == 0 and metrics["latent/kl"] == 0
    with torch.no_grad():
        p.actor.latent_mu.bias.fill_(1)
    loss, metrics = p.compute_auxiliary_loss(obs)
    assert metrics["latent/kl"] == 10  # sum over 20 latent dimensions
    torch.testing.assert_close(loss, torch.tensor(0.1))


def test_normalizer_fixed_until_commit_and_checkpoint_roundtrip(tmp_path):
    p = policy()
    mean = p.actor.normalizer.mean.clone()
    obs = observations()
    p.update_normalization(obs)
    assert torch.equal(mean, p.actor.normalizer.mean)
    p.commit_normalization(obs["policy"] + 10)
    assert not torch.equal(mean, p.actor.normalizer.mean)
    torch.save(p.state_dict(), tmp_path / "model.pt")
    other = policy()
    other.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    torch.testing.assert_close(other.act_inference(obs), p.act_inference(obs), rtol=0, atol=0)


def test_onnx_and_torchscript_export_parity(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    p = policy().eval()
    raw = torch.randn(1, 870)
    torch.jit.script(copy.deepcopy(p.actor)).save(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(raw), p.actor(raw))
    torch.onnx.export(p.actor, raw, str(tmp_path / "policy.onnx"), opset_version=17,
                      input_names=["obs"], output_names=["actions"], dynamo=False)
    graph = onnx.load(tmp_path / "policy.onnx")
    assert not any("Random" in node.op_type for node in graph.graph.node)
    assert not any("decoder" in item.name or "logvar" in item.name for item in graph.graph.initializer)
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {"obs": raw.numpy()})[0],
                               p.actor(raw).detach().numpy(), rtol=1e-4, atol=1e-5)


def make_algorithm(obs):
    from rsl_rl.storage import RolloutStorage, CircularBuffer
    p = policy(obs)
    storage = RolloutStorage("rl", 8, 4, obs, [25])
    amp = {"loss_type": "LSGAN", "disc_obs_dim": 7, "disc_obs_steps": 4,
           "disc_trunk_weight_decay": 1e-4, "disc_linear_weight_decay": 1e-2,
           "disc_learning_rate": 2e-6, "step_dt": 0.02, "grad_penalty_scale": 10,
           "amp_discriminator": {"hidden_dims": [16, 8], "task_style_lerp": 0.4}}
    return algorithm.LatentPPOAMP(p, storage, CircularBuffer(4, 8, "cpu"),
                                 CircularBuffer(4, 8, "cpu"), amp_cfg=amp,
                                 num_learning_epochs=2, num_mini_batches=2)


def test_actual_ppoamp_rollout_update_alignment_and_normalization():
    obs = observations()
    alg = make_algorithm(obs)
    mean = alg.policy.actor.normalizer.mean.clone()
    encoder_before = alg.policy.actor.encoder[0].weight.detach().clone()
    with torch.inference_mode():
        for step in range(4):
            alg.act(obs)
            next_obs = observations()
            dones = torch.tensor([True, False, False, False, False, False, False, False])
            alg.process_env_step(next_obs, torch.rand(8), dones, {"time_outs": dones})
            stored = alg.storage.observations[step]
            torch.testing.assert_close(stored["policy"], obs["policy"])
            torch.testing.assert_close(stored[NEXT][1:], alg.policy.actor.layout.latest(next_obs["policy"])[1:])
            assert stored[NEXT][0].abs().sum() == 0 and stored[VALID][0] == 0
            assert NEXT not in obs and NEXT not in next_obs
            obs = next_obs
        alg.compute_returns(obs)
    assert torch.equal(mean, alg.policy.actor.normalizer.mean)
    metrics = alg.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["latent/valid_fraction"] == 0.875
    assert "symmetry" not in metrics and "state_est" not in metrics
    assert not torch.equal(mean, alg.policy.actor.normalizer.mean)
    assert not torch.equal(encoder_before, alg.policy.actor.encoder[0].weight)
    assert alg.storage.step == 0


def test_privileged_group_rejected():
    obs = observations()
    obs["policy"] = torch.randn(8, 890)
    with pytest.raises(ValueError, match="87-feature"):
        policy(obs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_constructor_accepts_gpu_observations():
    obs = observations().to("cuda:0")
    p = policy(obs)
    assert p.act_inference(obs).device == obs.device


def test_frozen_baseline_preserved_except_allowlist(monkeypatch):
    config = __import__(PKG + ".config", fromlist=["*"])
    monkeypatch.setitem(sys.modules, "legged_lab.latent.config", config)
    from t1_heading_ablation import load_saved_config, configuration_differences
    from t1_latent_ablation import expected_configuration

    baseline = ROOT / "logs/rsl_rl/men_t1_amp/2026-09-15_19-40-00_resetmatch_noactorheading_noestimator_mirror01_noclip_jointlimit5_formal1_8192/params"
    if not baseline.is_dir():
        pytest.skip("Local baseline unavailable")
    env, agent = load_saved_config(baseline / "env.yaml"), load_saved_config(baseline / "agent.yaml")
    actual_env, actual_agent = expected_configuration(env, agent, run_name="latent_test", num_envs=8192,
                                                     max_iterations=25000, device="cuda:0")
    for key in ("rewards", "events", "commands", "curriculum", "actions", "scene", "sim"):
        assert configuration_differences(env[key], actual_env[key]) == []
    assert actual_env["observations"]["critic"] == env["observations"]["critic"]
    assert actual_env["observations"]["policy"]["heading_error"] is None
    assert actual_env["observations"]["policy"]["history_length"] == 10
    assert actual_agent["algorithm"]["symmetry_cfg"] is None
    assert actual_agent["algorithm"]["amp_cfg"] == agent["algorithm"]["amp_cfg"]
    assert env["observations"]["policy"]["history_length"] == 5
    assert agent["algorithm"]["symmetry_cfg"]["mirror_loss_coeff"] == 0.1
