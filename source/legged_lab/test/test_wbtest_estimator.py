"""CPU contract tests for both AMP robots and the legacy estimator path."""

import ast
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
PKG = "wbtest_estimator_test_package"
package = types.ModuleType(PKG)
package.__path__ = [str(ROOT / "source/legged_lab/legged_lab/state_est")]
sys.modules[PKG] = package
models = __import__(PKG + ".wbtest", fromlist=["*"])
legacy = __import__(PKG + ".models", fromlist=["*"])
torch.set_num_threads(1)
GROUPS = {"policy": ["policy"], "critic": ["critic"], "discriminator": ["disc"],
          "discriminator_demonstration": ["disc_demo"]}


def dims(joints=25, heading=False):
    return [3, 6, 3, joints, joints, joints] + ([2] if heading else [])


def observations(joints=25, heading=False, label_dim=8, history=3):
    target = torch.randn(8, label_dim)
    if label_dim == 8:
        target[:, 4:6] = torch.randint(0, 2, (8, 2)).float()
    return TensorDict({"policy": torch.randn(8, sum(dims(joints, heading)) * history),
                       "critic": torch.randn(8, 119), "state_est_label": target,
                       "disc": torch.randn(8, 4, 7), "disc_demo": torch.randn(8, 4, 7)}, [8])


def policy(joints=25, heading=False, **kwargs):
    obs = observations(joints, heading)
    return models.WbtEstimatorActorCritic(
        obs, GROUPS, joints, estimator_term_dims=dims(joints, heading),
        actor_hidden_dims=[32, 16], critic_hidden_dims=[16], estimator_hidden_dims=[32, 16], **kwargs)


@pytest.mark.parametrize("joints,heading", [(12, False), (25, False), (25, True)])
def test_term_major_current_frame_and_width(joints, heading):
    widths = dims(joints, heading)
    frames = torch.arange(3 * sum(widths)).reshape(1, 3, sum(widths)).float()
    raw = torch.cat([term.flatten(1) for term in frames.split(widths, -1)], -1)
    actor = policy(joints, heading).actor
    augmented = actor.augmented_observation(raw)
    torch.testing.assert_close(augmented[:, :-8], frames[:, -1])
    assert augmented.shape[-1] == sum(widths) + 8
    assert actor.estimator[0].in_features == sum(widths) * 3
    assert torch.all((augmented[:, -4:-2] >= 0) & (augmented[:, -4:-2] <= 1))


@pytest.mark.parametrize("joints", [12, 25])
def test_history_only_reaches_actor_via_estimator_and_ppo_is_detached(joints):
    p = policy(joints)
    obs = observations(joints)
    raw = obs["policy"].detach().requires_grad_(True)
    p.actor(raw).square().mean().backward()
    assert all(param.grad is None for param in p.actor.estimator.parameters())
    current_indices = set(models.latest_actor_indices(dims(joints)))
    old_indices = [i for i in range(raw.shape[-1]) if i not in current_indices]
    assert torch.all(raw.grad[:, old_indices] == 0)
    assert raw.grad[:, list(current_indices)].abs().sum() > 0
    p.zero_grad(set_to_none=True)
    loss, metrics = p.compute_estimator_loss(obs)
    loss.backward()
    assert p.actor.estimator[0].weight.grad.abs().sum() > 0
    assert p.actor.mlp[0].weight.grad is None
    assert len(metrics) == 6


def test_privileged_values_cannot_leak_to_actor():
    p, obs = policy(), observations()
    expected = p.act_inference(obs)
    obs["state_est_label"][:] = float("nan")
    obs["critic"][:] = float("nan")
    torch.testing.assert_close(p.act_inference(obs), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="never privileged"):
        models.WbtEstimatorActorCritic(obs, {**GROUPS, "policy": ["policy", "critic"]}, 25,
                                       estimator_term_dims=dims())


def test_exact_wbt_run46_loss_and_gradient_parity():
    path = ROOT.parent / "wbt_est/source/whole_body_tracking/whole_body_tracking/state_est/reference_loss.py"
    if not path.is_file():
        pytest.skip("Local wbtest reference unavailable")
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "continuous_mse_contact_bce")
    namespace = {"torch": torch, "F": torch.nn.functional, "ESTIMATOR_OUTPUT_DIM": 8,
                 "BASE_HEIGHT_SLICE": slice(0, 1), "BASE_LIN_VEL_SLICE": slice(1, 4),
                 "FOOT_CONTACT_SLICE": slice(4, 6), "FOOT_HEIGHT_SLICE": slice(6, 8)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    raw = torch.randn(8, 8, requires_grad=True)
    target = observations()["state_est_label"]
    actual, metrics = models.continuous_mse_contact_bce(raw, target)
    expected, reference_metrics = namespace["continuous_mse_contact_bce"](raw, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in metrics:
        torch.testing.assert_close(metrics[key], reference_metrics[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, raw)[0], torch.autograd.grad(expected, raw)[0])
    # Large contact logits use numerically stable BCE-with-logits.
    raw = torch.full((8, 8), 1000., requires_grad=True)
    loss, _ = models.continuous_mse_contact_bce(raw, target)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, raw)[0]).all()


@pytest.mark.parametrize("joints", [12, 25])
def test_independent_normalizers_and_checkpoint_roundtrip(joints):
    p, obs = policy(joints), observations(joints)
    p.update_normalization(obs)
    assert p.actor.estimator_input_normalizer.mean.shape[-1] == 3 * sum(dims(joints))
    assert p.actor.obs_normalizer.mean.shape[-1] == sum(dims(joints)) + 8
    assert p.critic_obs_normalizer.mean.shape[-1] == 119
    other = policy(joints)
    assert other.load_state_dict(p.state_dict()) is True
    torch.testing.assert_close(other.act_inference(obs), p.act_inference(obs), rtol=0, atol=0)


@pytest.mark.parametrize("joints", [12, 25])
def test_project_play_onnx_and_isaac_jit_export(joints, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path = Path("/root/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py")
    spec = importlib.util.spec_from_file_location("wbtest_native_exporter", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    onnx_spec = importlib.util.spec_from_file_location(
        "wbtest_project_exporter", ROOT / "source/legged_lab/legged_lab/rsl_rl/exporter.py")
    onnx_exporter = importlib.util.module_from_spec(onnx_spec)
    onnx_spec.loader.exec_module(onnx_exporter)
    p, obs = policy(joints).eval(), observations(joints)
    p.update_normalization(obs)
    raw = obs["policy"][:1]
    exporter.export_policy_as_jit(p, p.actor_obs_normalizer, str(tmp_path))
    onnx_exporter.export_policy_as_onnx(p, str(tmp_path), p.actor_obs_normalizer)
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(raw), p.actor(raw))
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {session.get_inputs()[0].name: raw.numpy()})[0],
                               p.actor(raw).detach().numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("joints", [12, 25])
def test_legacy_four_state_contract_unchanged(joints):
    obs = observations(joints, label_dim=4, history=5)
    options = dict(actor_hidden_dims=[16], critic_hidden_dims=[16], estimator_hidden_dims=[16])
    old = legacy.EstimatorActorCritic(obs, GROUPS, joints, **options)
    restored = legacy.EstimatorActorCritic(obs, GROUPS, joints, **options)
    assert restored.load_state_dict(old.state_dict()) is True
    torch.testing.assert_close(restored.act_inference(obs), old.act_inference(obs), rtol=0, atol=0)
    assert old.actor.mlp[0].in_features == sum(dims(joints)) * 5 + 4
    assert old.compute_estimator_loss(obs)[0].isfinite()
    with pytest.raises(RuntimeError):
        policy(joints).load_state_dict(old.state_dict())


@pytest.mark.parametrize("robot,widths", [("bsrl", [144, 240]), ("t1", [261, 267, 435, 445])])
def test_mirror_three_and_five_frame_involution(robot, widths):
    path = ROOT / f"source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/symmetry/{robot}.py"
    spec = importlib.util.spec_from_file_location(f"wbtest_symmetry_{robot}", path)
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    for width in widths:
        raw = torch.randn(8, width)
        torch.testing.assert_close(mirror._mirror_policy_observation(mirror._mirror_policy_observation(raw)),
                                   raw, rtol=0, atol=0)


@pytest.mark.parametrize("joints", [12, 25])
def test_actual_ppoamp_update_learns_estimator(joints):
    from rsl_rl.algorithms.ppo_amp import PPOAMP
    from rsl_rl.storage import RolloutStorage, CircularBuffer
    obs, p = observations(joints), policy(joints)
    storage = RolloutStorage("rl", 8, 4, obs, [joints])
    amp = {"loss_type": "LSGAN", "disc_obs_dim": 7, "disc_obs_steps": 4,
           "disc_trunk_weight_decay": 1e-4, "disc_linear_weight_decay": 1e-2,
           "disc_learning_rate": 2e-6, "step_dt": 0.02, "grad_penalty_scale": 10,
           "amp_discriminator": {"hidden_dims": [16, 8], "task_style_lerp": 0.4}}
    alg = PPOAMP(p, storage, CircularBuffer(4, 8, "cpu"), CircularBuffer(4, 8, "cpu"), amp_cfg=amp,
                 state_est_loss_required=True, num_learning_epochs=2, num_mini_batches=2)
    before = p.actor.estimator[0].weight.detach().clone()
    with torch.inference_mode():
        for _ in range(4):
            alg.act(obs)
            obs = observations(joints)
            alg.process_env_step(obs, torch.rand(8), torch.zeros(8, dtype=torch.bool), {})
        alg.compute_returns(obs)
    metrics = alg.update()
    assert all(np.isfinite(value) for value in metrics.values())
    assert "state_est/contact_bce" in metrics
    assert not torch.equal(before, p.actor.estimator[0].weight)


def test_checkpoint_profile_detection_and_mismatch():
    from estimator_checkpoint import checkpoint_profile, PROFILE_CLASSES
    for cls, profile in PROFILE_CLASSES.items():
        assert checkpoint_profile({"policy": {"class_name": cls}}) == profile
        with pytest.raises(ValueError, match="no implicit"):
            checkpoint_profile({"policy": {"class_name": cls}}, "wbtest" if profile == "legacy4d" else "legacy4d")
    with pytest.raises(ValueError, match="Not a supported"):
        checkpoint_profile({"policy": {"class_name": "ActorCritic"}})


@pytest.fixture
def isaac_free_config(monkeypatch):
    """Isolate small simulator-label/config functions from Kit's import side effects."""
    class Cfg:
        def __init__(self, **kwargs):
            for cls in reversed(type(self).__mro__):
                for name, value in vars(cls).items():
                    if not name.startswith("_") and not callable(value):
                        setattr(self, name, copy.deepcopy(value))
            self.__dict__.update(kwargs)

    for name, attrs in {
        "isaaclab.assets": {"Articulation": object},
        "isaaclab.managers": {"SceneEntityCfg": Cfg},
        "isaaclab.sensors": {"ContactSensor": object},
        "isaaclab.utils": {"configclass": lambda cls: cls},
        "isaaclab_rl.rsl_rl": {"RslRlPpoActorCriticCfg": Cfg},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    # Separate package means monkeypatch cleanup cannot leave a stale fake
    # config module in the production namespace or another test's package.
    name = "wbtest_config_test_package"
    pkg = types.ModuleType(name)
    pkg.__path__ = package.__path__
    monkeypatch.setitem(sys.modules, name, pkg)
    for suffix in ("observations", "config"):
        monkeypatch.delitem(sys.modules, name + "." + suffix, raising=False)
    config = __import__(name + ".config", fromlist=["*"])
    labels = __import__(name + ".observations", fromlist=["*"])
    return Cfg, config, labels


def test_eight_labels_left_right_contact_history_and_origin(isaac_free_config):
    Cfg, _, labels = isaac_free_config
    forces = torch.zeros(2, 3, 4, 3)
    forces[0, 0, 2, 2] = 1.1  # left foot, past contact, not latest
    forces[1, 2, 0, 2] = 1.0  # threshold is strictly greater than 1
    data = Cfg(root_pos_w=torch.tensor([[0., 0., 4.], [0., 0., 6.]]),
               root_lin_vel_b=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
               body_pos_w=torch.tensor([[[0., 0., 3.2], [0., 0., 0.], [0., 0., 3.1], [0., 0., 0.]],
                                        [[0., 0., 5.2], [0., 0., 0.], [0., 0., 5.1], [0., 0., 0.]]]))
    class Scene(dict):
        pass
    scene = Scene(robot=Cfg(data=data))
    scene.env_origins = torch.tensor([[0., 0., 3.], [0., 0., 5.]])
    scene.sensors = {"contact_forces": Cfg(data=Cfg(net_forces_w_history=forces))}
    env = Cfg(scene=scene)
    asset = Cfg(name="robot", body_ids=[2, 0])
    sensor = Cfg(name="contact_forces", body_ids=[2, 0])
    expected = torch.tensor([[1., 1., 2., 3., 1., 0., .1, .2],
                             [1., 4., 5., 6., 0., 0., .1, .2]])
    torch.testing.assert_close(labels.wbtest_estimator_label(env, asset, sensor), expected)
    torch.testing.assert_close(labels.state_estimator_label(env, asset), expected[:, :4])


@pytest.mark.parametrize("joints,heading", [(12, False), (25, False), (25, True)])
def test_config_changes_only_estimator_contract(isaac_free_config, joints, heading):
    Cfg, config, _ = isaac_free_config
    def term(**kwargs):
        return Cfg(func=lambda *args: None, **kwargs)
    policy_cfg = Cfg(base_ang_vel=term(), root_local_rot_tan_norm=term(), velocity_commands=term(),
                     joint_pos=term(params={"asset_cfg": Cfg(joint_names=list(range(joints)))}),
                     joint_vel=term(), actions=term(), history_length=5)
    if heading:
        policy_cfg.heading_error = term()
    selector = Cfg(body_names=["left_foot", "right_foot"], preserve_order=True)
    class ObsCfg(Cfg):
        StateEstLabelCfg = lambda self=None: Cfg(label=term(), enable_corruption=False)
        StateEstCriticCfg = lambda self=None: Cfg()
    obs_cfg = ObsCfg(policy=policy_cfg, state_est_label=None, state_est_critic=Cfg(base_height=term()),
                     critic=Cfg(history_length=5), reward_privileged=Cfg(
                         foot_contact=term(params={"sensor_cfg": selector}),
                         foot_lin_vel_w_xy=term(params={"asset_cfg": selector})))
    env = Cfg(observations=obs_cfg, rewards=object(), events=object(), scene=object(), commands=object())
    original_other = {name: getattr(env, name) for name in ("rewards", "events", "scene", "commands")}
    original_critic = obs_cfg.critic
    agent = Cfg(obs_groups=copy.deepcopy(GROUPS), algorithm=Cfg(symmetry_cfg={"use_mirror_loss": True}),
                policy=Cfg(init_noise_std=1., noise_std_type="log", state_dependent_std=False,
                           actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu"))
    symmetry = agent.algorithm.symmetry_cfg
    config.enable_state_estimator(env, agent)
    assert agent.policy.estimator_term_dims == dims(joints, heading)
    assert agent.policy.estimator_output_dim == 8 and policy_cfg.history_length == 3
    assert agent.policy.actor_hidden_dims == [512, 256, 128]
    assert obs_cfg.state_est_label.label.func.__name__ == "wbtest_estimator_label"
    assert obs_cfg.state_est_label.history_length == 0
    assert agent.algorithm.symmetry_cfg is symmetry
    assert obs_cfg.critic is original_critic and obs_cfg.critic.history_length == 5
    for name, value in original_other.items():
        assert getattr(env, name) is value
    assert (getattr(policy_cfg, "heading_error", None) is not None) == heading
