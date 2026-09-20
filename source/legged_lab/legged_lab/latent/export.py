"""Export a latent checkpoint without starting Isaac or constructing the Critic.

Run this file directly with the isaaclab Python environment (see README).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "external/rsl_rl"))
    from models import LatentActor, T1_TERM_DIMS, BSRL_TERM_DIMS
else:
    from .models import LatentActor, T1_TERM_DIMS, BSRL_TERM_DIMS


def export_checkpoint(checkpoint: Path, output_dir: Path):
    import onnx
    import onnxruntime as ort

    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / "scripts/rsl_rl"))
    from t1_heading_ablation import load_saved_config

    checkpoint = checkpoint.resolve(strict=True)
    config = load_saved_config(checkpoint.parent / "params/agent.yaml")["policy"]
    contracts = {"legged_lab.latent.models:LatentActorCritic": (25, T1_TERM_DIMS),
                 "legged_lab.latent.models:BSRLLatentActorCritic": (12, BSRL_TERM_DIMS)}
    if config["class_name"] not in contracts:
        raise ValueError("Not a supported T1/BSRL latent checkpoint")
    num_actions, term_dims = contracts[config["class_name"]]
    for name in ("policy.onnx", "policy.pt", "contract.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    actor = LatentActor(num_actions, config["history_length"], config["recent_frames"], config["latent_dim"],
                        config["encoder_dims"], config["actor_hidden_dims"], term_dims=term_dims).eval()
    actor.load_state_dict({key[len("actor."):]: value for key, value in state.items() if key.startswith("actor.")})
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Nonfinite actor checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = torch.randn(1, actor.input_dim)
    torch.jit.script(actor).save(str(output_dir / "policy.pt"))
    torch.onnx.export(actor, raw, str(output_dir / "policy.onnx"), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    graph = onnx.load(output_dir / "policy.onnx")
    onnx.checker.check_model(graph)
    if any("Random" in node.op_type for node in graph.graph.node):
        raise RuntimeError("Random latent operation entered deployment graph")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    onnx.helper.set_model_props(graph, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "run": checkpoint.parent.name,
        "history_length": str(config["history_length"]),
        "frame_dim": str(sum(term_dims)),
    })
    onnx.save(graph, output_dir / "policy.onnx")
    session = ort.InferenceSession(str(output_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    max_error = 0.0
    for _ in range(5):
        raw = torch.randn(1, actor.input_dim)
        expected = actor(raw).detach().numpy()
        actual = session.run(None, {"obs": raw.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        max_error = max(max_error, float(np.abs(actual - expected).max()))
    contract = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
                "input_dim": actor.input_dim, "output_dim": num_actions,
                "history_length": config["history_length"], "frame_dim": sum(term_dims),
                "layout": "term-major, chronological oldest-first within each term",
                "terms": {"base_ang_vel": 3, "root_local_rot_tan_norm": 6, "velocity_commands": 3,
                          "joint_pos": num_actions, "joint_vel": num_actions, "actions": num_actions},
                "normalization": "embedded and frozen", "latent_inference": "deterministic mean",
                "onnx_max_absolute_error": max_error,
                "actor_parameters": sum(p.numel() for p in actor.parameters())}
    (output_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output_dir or args.checkpoint.parent / "exported_latent" / args.checkpoint.stem)
