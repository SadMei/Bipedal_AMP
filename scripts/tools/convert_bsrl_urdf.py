"""Regenerate the BSRL USD asset from the canonical export.urdf file."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--urdf",
    default="source/legged_lab/legged_lab/data/Robots/BSRL/export.urdf",
    help="Canonical BSRL URDF path.",
)
parser.add_argument(
    "--output_dir",
    default="source/legged_lab/legged_lab/data/Robots/BSRL/export",
    help="Directory for the generated USD asset.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


cfg = UrdfConverterCfg(
    asset_path=os.path.abspath(args.urdf),
    usd_dir=os.path.abspath(args.output_dir),
    usd_file_name="export.usd",
    force_usd_conversion=True,
    make_instanceable=True,
    fix_base=False,
    merge_fixed_joints=False,
    joint_drive=None,
    collision_from_visuals=False,
    self_collision=False,
    replace_cylinders_with_capsules=False,
)
converter = UrdfConverter(cfg)
print(f"Generated BSRL USD: {converter.usd_path}")
simulation_app.close()
