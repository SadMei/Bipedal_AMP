"""Source paths are independent of external models and historical results."""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WORKSPACE = REPO_ROOT.parent
ROBOT_DATA_ROOT = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots"
BSRL_ROOT = ROBOT_DATA_ROOT / "BSRL_urdf"
CLOSED_ANKLE_ROOT = ROBOT_DATA_ROOT / "BSRL_closed_ankle"
DATA_DIR = CLOSED_ANKLE_ROOT
CORE = Path(os.environ.get("BSRL_ANKLE_CORE_DIR", WORKSPACE / "bsrl_ankle_core_flat")).expanduser().resolve()
CALIBRATION = CORE / "bsrl_bridge_bringup.yaml"
URDF = BSRL_ROOT / "urdf" / "export.urdf"
SERIAL_XML = BSRL_ROOT / "xml" / "bsrl_scene.xml"
