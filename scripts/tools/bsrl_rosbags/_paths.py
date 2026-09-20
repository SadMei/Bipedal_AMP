"""Keep raw bags and historical results outside the relocated source folder."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
BAG_ROOT = Path(os.environ.get("BSRL_ROSBAG_ROOT", WORKSPACE_ROOT / "bsrl_rosbags")).expanduser().resolve()
RESULTS_ROOT = BAG_ROOT / "analysis" / "results_20260914"
CORE_ROOT = Path(os.environ.get("BSRL_ANKLE_CORE_DIR", WORKSPACE_ROOT / "bsrl_ankle_core_flat")).expanduser().resolve()
