"""Standalone BSRL WM shape and serial-MJX one-step smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from legged_lab.world_model import (
    FrameHistory,
    MJXNominalPhysics,
    bsrl_serial_mjx_cfg,
    bsrl_world_model_state_spec,
    default_standing_frame,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default_xml = (
        root
        / "source/legged_lab/legged_lab/data/Robots/BSRL_urdf/xml/bsrl_scene.xml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    xml = args.xml.resolve(strict=True)
    spec = bsrl_world_model_state_spec()
    nominal = MJXNominalPhysics(spec, bsrl_serial_mjx_cfg(xml))

    class ZeroReference:
        def get(self, index):
            return SimpleNamespace(
                joint_pos=torch.zeros(spec.num_joints),
                joint_vel=torch.zeros(spec.num_joints),
            )

    frame, _ = default_standing_frame(
        nominal, ZeroReference(), spec, device=args.device
    )
    history = FrameHistory.create(spec, frame).tensor
    with torch.no_grad():
        next_frame = nominal(history, torch.zeros(1, spec.action_dim, device=args.device))
    if next_frame.shape != frame.shape or not torch.isfinite(next_frame).all():
        raise RuntimeError("BSRL serial MJX returned an invalid next frame.")
    print(
        json.dumps(
            {
                "scope": "serial-interface-test_not_r2_parallel_ankle",
                "xml": str(xml),
                "nq": nominal._cpu_model.nq,
                "nv": nominal._cpu_model.nv,
                "nu": nominal._cpu_model.nu,
                "frame_dim": spec.frame_dim,
                "model_input_dim": 47,
                "generalized_force_dim": spec.generalized_force_dim,
                "finite": True,
                "root_height_before": float(frame[0, spec.root_height_slice]),
                "root_height_after": float(next_frame[0, spec.root_height_slice]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
