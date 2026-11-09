"""LAFAN/ACCAD style-data ablations on the completed R2 robustness baseline."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

from frozen_run_config import restore_run_config
from frozen_run_config import configuration_differences, load_saved_config, resolved_scene_config
from bsrl_scale01_ablation import validate_baseline


VARIANTS = ("lafan_only_style60", "lafan80_accad20_style50", "accad_forward_style60")
BASELINE_R2_RUN = (
    "logs/rsl_rl/bsrl_amp/"
    "2026-09-28_15-29-53_accad34_scale01_deploypd_robust_r2_ankle_transmission_"
    "40000it_8192env_20260928_1531"
)
LAFAN_SOURCE = "source/legged_lab/legged_lab/data/MotionData/bsrl_12dof/amp/walk_and_run"
ACCAD_SOURCE = "source/legged_lab/legged_lab/data/MotionData/bsrl/amp/accad_g1full_34_20260916"

LAFAN_MOTIONS = ("run1_subject2", "walk1_subject1")
ACCAD_FORWARD_MOTIONS = (
    "Male2Running_c3d__C1_-_stand_to_run_stageii",
    "Male2Running_c3d__C3_-_run_stageii",
    "Male2Running_c3d__C4_-_run_to_walk_a_stageii",
    "Female1Running_c3d__C5_-_walk_to_run_stageii",
    "Male2Running_c3d__C5_-_walk_to_run_stageii",
    "straight_female1_B3",
    "stand_male2_A1",
)
TASK_STYLE_LERP = {
    "lafan_only_style60": 0.4,
    "lafan80_accad20_style50": 0.5,
    "accad_forward_style60": 0.4,
}


def motion_weights(variant: str) -> dict[str, float]:
    if variant == "lafan_only_style60":
        return {name: 0.5 for name in LAFAN_MOTIONS}
    if variant == "lafan80_accad20_style50":
        weights = {name: 0.4 for name in LAFAN_MOTIONS}
        weights.update({name: 0.2 / len(ACCAD_FORWARD_MOTIONS) for name in ACCAD_FORWARD_MOTIONS})
        return weights
    if variant == "accad_forward_style60":
        return {name: 1.0 / len(ACCAD_FORWARD_MOTIONS) for name in ACCAD_FORWARD_MOTIONS}
    raise ValueError(f"Unknown variant: {variant}")


def source_files(root: Path, variant: str) -> dict[str, Path]:
    root = root.resolve()
    files = {}
    if variant != "accad_forward_style60":
        files.update({name: root / LAFAN_SOURCE / f"{name}.pkl" for name in LAFAN_MOTIONS})
    if variant != "lafan_only_style60":
        files.update({name: root / ACCAD_SOURCE / f"{name}.pkl" for name in ACCAD_FORWARD_MOTIONS})
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing style data: {missing}")
    return files


def prepare_dataset(root: Path, variant: str) -> tuple[Path, dict]:
    """Create an immutable symlink view without copying or altering source motions."""
    files = source_files(root, variant)
    dataset = root.resolve() / "logs/datasets/bsrl_r2_lafan_ablation_v1" / variant
    dataset.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{name}.pkl" for name in files}
    unexpected = [path.name for path in dataset.glob("*.pkl") if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(f"Unexpected files in frozen dataset view {dataset}: {unexpected}")
    manifest_files = []
    for name, source in files.items():
        source = source.resolve(strict=True)
        link = dataset / f"{name}.pkl"
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve(strict=True) != source:
                raise RuntimeError(f"Dataset view collision: {link}")
        else:
            link.symlink_to(source)
        manifest_files.append({
            "stem": name,
            "source": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "weight": motion_weights(variant)[name],
        })
    manifest = {
        "variant": variant,
        "dataset_dir": str(dataset),
        "sampling_mass": {
            "lafan": sum(row["weight"] for row in manifest_files if row["stem"] in LAFAN_MOTIONS),
            "accad_forward_stand": sum(
                row["weight"] for row in manifest_files if row["stem"] in ACCAD_FORWARD_MOTIONS
            ),
        },
        "files": manifest_files,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = dataset / "manifest.json"
    if manifest_path.exists() and manifest_path.read_text() != payload:
        raise RuntimeError(f"Frozen dataset manifest changed: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(payload)
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return dataset, manifest


def validate_r2_baseline(env: dict, agent: dict) -> None:
    validate_baseline(env, agent)
    if agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"] != 0.6:
        raise ValueError("Expected completed R2 with task_style_lerp=0.6")
    transmission = env["events"].get("ankle_transmission")
    expected_ranges = {
        "ratio_range": (0.97, 1.03),
        "strength_range": (0.9, 1.0),
        "response_time_range": (0.0, 0.005),
    }
    if not transmission or transmission["mode"] != "startup" or transmission["params"] != expected_ranges:
        raise ValueError("Expected the R2 analytical ankle transmission randomization")
    reset = env["events"]["reset_from_ref"]
    if (not reset["func"].endswith(":reset_from_ref_with_default_mixture")
            or reset["params"].get("default_start_probability") != 0.2):
        raise ValueError("Expected the R2 default/reference reset mixture")
    if env["observations"]["policy"]["base_ang_vel"].get("modifiers") is not None:
        raise ValueError("R2 baseline must not include the R3 observation perturbations")


def expected_configuration(saved_env, saved_agent, *, variant, dataset_dir, run_name,
                           num_envs, max_iterations, device):
    validate_r2_baseline(saved_env, saved_agent)
    if variant not in VARIANTS or not 1 <= max_iterations <= 40000 or num_envs < 7:
        raise ValueError("Invalid variant, environment count, or iteration count")
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    dataset_cfg = env["motion_data"]["motion_dataset"]
    dataset_cfg["motion_data_dir"] = str(Path(dataset_dir).resolve())
    dataset_cfg["motion_data_weights"] = motion_weights(variant)
    # This mirrors the historical LAFAN AMP setup: the style prior is sampled
    # independently while the unchanged R2 task rewards cover all commands.
    animation = env["animation"]["animation"]
    animation["command_name"] = None
    animation["command_motion_patterns"] = None
    animation["command_component_scales"] = (1.0, 1.0, 1.0)
    animation["match_command_speed"] = False
    agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"] = TASK_STYLE_LERP[variant]
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent.update(run_name=run_name, max_iterations=max_iterations, device=device, resume=False)
    return resolved_scene_config(env), agent


def apply_ablation(cfg, agent, baseline, variant, *, dataset_dir, dataset_manifest,
                   run_name, num_envs, max_iterations, device):
    from isaaclab.managers import EventTermCfg
    from legged_lab.assets.bsrl_reduced_ankle import randomize_ankle_transmission

    baseline = Path(baseline).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    expected_env, expected_agent = expected_configuration(
        saved_env, saved_agent, variant=variant, dataset_dir=dataset_dir,
        run_name=run_name, num_envs=num_envs, max_iterations=max_iterations, device=device,
    )
    # The current source config intentionally defaults this optional R2 event
    # to ``None``.  Give the strict YAML restorer a typed manager-term
    # container; otherwise it can only reconstruct the saved mapping as a
    # plain dict, which IsaacLab's EventManager correctly rejects.
    transmission = saved_env["events"]["ankle_transmission"]
    cfg.events.ankle_transmission = EventTermCfg(
        func=randomize_ankle_transmission,
        mode=transmission["mode"],
        params=copy.deepcopy(transmission["params"]),
    )
    audit = restore_run_config(cfg, agent, baseline)
    cfg.motion_data.motion_dataset.motion_data_dir = str(Path(dataset_dir).resolve())
    cfg.motion_data.motion_dataset.motion_data_weights = motion_weights(variant)
    cfg.animation.animation.command_name = None
    cfg.animation.animation.command_motion_patterns = None
    cfg.animation.animation.command_component_scales = (1.0, 1.0, 1.0)
    cfg.animation.animation.match_command_speed = False
    agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp = TASK_STYLE_LERP[variant]
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.log_dir = None
    agent.run_name = run_name
    agent.max_iterations = max_iterations
    agent.device = device
    agent.resume = False
    differences = configuration_differences(expected_env, resolved_scene_config(cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent.to_dict(), "agent")
    if differences:
        raise RuntimeError(f"Incomplete LAFAN ablation restoration: {differences}")
    audit.update(
        variant=variant, fresh_training=True, resume=False, r2_plant_and_robustness_preserved=True,
        max_iterations=max_iterations, num_envs=num_envs,
        task_style_lerp=TASK_STYLE_LERP[variant], style_reward_fraction=1.0 - TASK_STYLE_LERP[variant],
        reference_sampling="dataset-wide_unconditioned_like_historical_lafan",
        motion_weights=motion_weights(variant), dataset_manifest=dataset_manifest,
        requested_configuration_differences=(
            configuration_differences(saved_env, expected_env, "env")
            + configuration_differences(saved_agent, expected_agent, "agent")
        ),
        unrequested_configuration_differences=differences,
    )
    return audit


def runtime_audit(env, agent, variant):
    dims = env.observation_manager.group_obs_dim
    critic_width = sum(dims[name][0] for name in agent.obs_groups["critic"])
    if tuple(dims["policy"]) != (480,) or critic_width != 370:
        raise RuntimeError(f"Unexpected R2 observation contract: {dims}, critic={critic_width}")
    if agent.resume or agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp != TASK_STYLE_LERP[variant]:
        raise RuntimeError("Training is not fresh or task/style blend is incorrect")
    if "ankle_transmission" not in env.event_manager.active_terms.get("startup", []):
        raise RuntimeError("R2 ankle transmission randomization is not active")
    animation = env.animation_manager.get_term("animation")
    if animation._command_category_motion_ids is not None:
        raise RuntimeError("Style reference must be dataset-wide and command-independent")
    motion = env.motion_data_manager.get_term("motion_dataset")
    expected_names = list(motion_weights(variant))
    if motion.motion_names != expected_names:
        raise RuntimeError(f"Wrong runtime style dataset: {motion.motion_names}")
    expected_weights = [motion_weights(variant)[name] for name in expected_names]
    actual_weights = motion.motion_weights.cpu().tolist()
    total = sum(expected_weights)
    expected_weights = [value / total for value in expected_weights]
    if any(abs(left - right) > 1e-6 for left, right in zip(actual_weights, expected_weights)):
        raise RuntimeError(f"Wrong normalized motion weights: {actual_weights}")
    return {
        "variant": variant, "policy_width": 480, "critic_width": critic_width,
        "fresh_training": True, "task_style_lerp": TASK_STYLE_LERP[variant],
        "style_reward_fraction": 1.0 - TASK_STYLE_LERP[variant],
        "ankle_transmission": True, "reference_command_conditioned": False,
        "motion_names": motion.motion_names, "normalized_motion_weights": actual_weights,
        "motion_total_duration_s": motion.get_total_duration(),
    }


def snapshot_sources(log_dir):
    root = Path(__file__).resolve().parents[2]
    paths = [
        "scripts/rsl_rl/bsrl_r2_lafan_ablation.py",
        "scripts/rsl_rl/train_bsrl_r2_lafan_ablation.py",
        "scripts/rsl_rl/launch_bsrl_r2_lafan_ablation.py",
        "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/events.py",
        "source/legged_lab/legged_lab/assets/bsrl_reduced_ankle.py",
    ]
    hashes = {}
    for relative in paths:
        source = root / relative
        target = Path(log_dir) / "ablation_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
