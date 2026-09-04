from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class AnimationTermCfg:
    """Configuration for an animation."""

    motion_data_term: str = MISSING
    """The motion data term to use for this animation term."""

    motion_data_components: list[str] = MISSING
    """The components of the motion data to use for this animation term."""

    num_steps_to_use: int = 1
    """Number of steps of motion data to extract from the motion data term.
        If positive, extracts current and future steps.
        If negative, extracts current and past steps.
        1 and -1 both extract only the current step.
        0 is invalid.
    """

    random_initialize: bool = False
    """Whether to randomly initialize the starting point in the motion data term."""

    random_fetch: bool = False
    """Whether to randomly fetch the motion data at each step."""

    command_name: str | None = None
    """Velocity command used to select a matching motion category, if enabled."""

    command_motion_patterns: dict[str, tuple[str, ...]] | None = None
    """Glob patterns for forward/backward/lateral/turning motion categories."""

    command_component_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Expected maximum absolute x, y, and yaw commands used to compare dominant components."""

    match_command_speed: bool = False
    """Whether to choose the closest-speed clip inside the selected command category."""

    enable_visualization: bool = True
    """Whether to enable visualization for this animation term."""

    vis_root_offset: list[float] = (0.0, 0.0, 0.0)
    """Root position offset for visualization (x, y, z)."""
