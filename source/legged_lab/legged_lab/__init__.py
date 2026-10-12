"""Legged Lab package.

Set ``LEGGED_LAB_SKIP_TASK_REGISTRATION=1`` for standalone replay/world-model
tools that do not launch Isaac Sim. Normal Isaac entry points retain automatic
task and UI registration.
"""

import os

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.getenv("LEGGED_LAB_SKIP_TASK_REGISTRATION") != "1":
    # Register Gym environments and UI extensions only after an Isaac app is
    # available. Standalone WM tools deliberately avoid pxr/Kit imports.
    from .tasks import *
    from .ui_extension_example import *
