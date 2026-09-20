"""Source paths are independent of external models and historical results."""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WORKSPACE = REPO_ROOT.parent
DATA_DIR = WORKSPACE / "parallel_ankle"
CORE = Path(os.environ.get("BSRL_ANKLE_CORE_DIR", WORKSPACE / "bsrl_ankle_core_flat")).expanduser().resolve()
CALIBRATION = CORE / "bsrl_bridge_bringup.yaml"
URDF = WORKSPACE / "BSRL_urdf" / "urdf" / "export.urdf"
