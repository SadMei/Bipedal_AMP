"""Exact policy selection and explicit recovered-model replay checks."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

if __package__:
    from ._paths import REPO_ROOT, WORKSPACE
    from .real_ankle import Geometry
    from .render_replay import ankle_camera, video_label
    from .replay_policy import Replay, action_target, load_nominal, policy_history_length, policy_provenance, recovered_model_xml, sha
else:
    from _paths import REPO_ROOT, WORKSPACE
    from real_ankle import Geometry
    from render_replay import ankle_camera, video_label
    from replay_policy import Replay, action_target, load_nominal, policy_history_length, policy_provenance, recovered_model_xml, sha


def test_policy_history_contract_supports_legacy_and_latent():
    assert policy_history_length([1, 240], [1, 12]) == 5
    assert policy_history_length([1, 480], [1, 12]) == 10
    with pytest.raises(ValueError, match="input"):
        policy_history_length([1, 360], [1, 12])
    with pytest.raises(ValueError, match="output"):
        policy_history_length([1, 480], [1, 25])


def test_onnx_selects_its_actual_checkpoint(tmp_path):
    checkpoint = tmp_path / "model_25000.pt"
    checkpoint.write_bytes(b"checkpoint25000")
    wrong = tmp_path / "model_29999.pt"
    wrong.write_bytes(b"checkpoint29999")
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"policy")
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), run=tmp_path.name)
    result = policy_provenance(tmp_path, onnx, metadata)
    assert result["checkpoint_label"] == "model_25000"
    assert result["checkpoint_sha256"] == sha(checkpoint)
    with pytest.raises(ValueError, match="checksum differ"):
        policy_provenance(tmp_path, onnx, metadata, wrong)
    with pytest.raises(ValueError, match="nominal training run differ"):
        policy_provenance(tmp_path, onnx, {**metadata, "run": "another_training_run"})


def test_changed_recovered_model_is_rejected(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>")
    (tmp_path / "parallel_ankle_recovery.json").write_text(json.dumps(dict(
        geometry_revision=Geometry.revision, xml_sha256={xml.name: "wrong-checksum"})))
    with pytest.raises(ValueError, match="XML checksum differs"):
        recovered_model_xml(xml, .001, .002)


@pytest.fixture(scope="module")
def replay():
    run = REPO_ROOT / "logs/rsl_rl/bsrl_amp/2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192"
    xml = WORKSPACE / "BSRL_urdf/xml/bsrl_closed_ankles.xml"
    if not xml.is_file() or not (run / "params/env.yaml").is_file():
        pytest.skip("Recovered model and frozen September 14 training configuration required")
    return Replay(run, "closed", None, [.6, 0, 0], model_xml=xml)


def test_recovered_replay_preserves_control_and_passive_ankles(replay):
    assert replay.model.nu == 12 and replay.model.neq == 4
    assert len(replay.direct_ids) == 8
    assert replay.initial_gaps.max() < 1e-12
    assert replay.model.opt.timestep == .001
    assert replay.nominal["policy_dt"] == .02
    assert replay.observation().shape == (1, 240)
    for indices in replay.ankle_indices.values():
        np.testing.assert_array_equal(replay.nominal["stiffness"][indices], [100., 100.])
        np.testing.assert_array_equal(replay.nominal["damping"][indices], [1., 1.])
    replay.target[replay.ankle_indices["right"]] += [.1, .05]
    replay.apply()
    mujoco.mj_forward(replay.model, replay.data)
    for indices in replay.ankle_indices.values():
        np.testing.assert_array_equal(replay.data.qfrc_actuator[replay._qvel_adr[indices]], 0.)
    replay.reset()


@pytest.mark.parametrize("side", ["left", "right"])
def test_closeup_tracks_each_ankle(replay, side):
    camera = mujoco.MjvCamera()
    ankle_camera(camera, replay.data, side)
    np.testing.assert_allclose(camera.lookat,
                               replay.data.body(f"link_{side}_ankle_roll").xpos + [0, 0, .105])
    assert camera.distance < .7


def test_video_label_does_not_hardcode_model29999():
    assert video_label(dict(checkpoint_label="model_25000")) == "model_25000"
    assert video_label(dict(checkpoint="/run/model_123.pt")) == "model_123"
    assert video_label({}) == "checkpoint unspecified"


def test_reduced_run_uses_physical_motor_envelope_not_numerical_guards():
    run = REPO_ROOT / "logs/rsl_rl/bsrl_amp/2026-09-15_15-14-55_reduced_parallelankle_nominal_pd200_noestimator_mirror01_formal1_8192"
    if not (run / "params/env.yaml").is_file():
        pytest.skip("Frozen reduced-ankle training configuration required")
    nominal, _ = load_nominal(run)
    np.testing.assert_array_equal(nominal["effort_limit"][8:], [112.] * 4)
    np.testing.assert_allclose(nominal["velocity_limit"][8:], [26.17993877991494] * 4)
    np.testing.assert_array_equal(nominal["peak_power"][8:], [1250.] * 4)
    np.testing.assert_array_equal(nominal["stiffness"][8:], [100.] * 4)
    assert nominal["target_lower"] is nominal["target_upper"] is None
    raw = np.full(12, 1000.)
    np.testing.assert_array_equal(action_target(nominal, raw), nominal["default_pos"] + raw * .25)


def test_legacy_action_clipping_is_preserved(replay):
    np.testing.assert_array_equal(action_target(replay.nominal, np.full(12, 1000.)), [100.] * 12)
    np.testing.assert_array_equal(action_target(replay.nominal, np.full(12, -1000.)), [-100.] * 12)
