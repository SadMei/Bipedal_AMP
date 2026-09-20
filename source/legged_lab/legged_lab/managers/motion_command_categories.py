"""Tensor-only command categorization, shared by sampler and launch audits."""
import torch


def command_category_ids(command, component_scales, *, standing=False):
    if len(component_scales) != 3 or min(component_scales) <= 0:
        raise ValueError("Three positive command component scales are required")
    sx, sy, sw = component_scales
    scores = torch.stack((command[:,0].clamp_min(0)/sx, (-command[:,0]).clamp_min(0)/sx,
                          command[:,1].clamp_min(0)/sy, (-command[:,1]).clamp_min(0)/sy,
                          command[:,2].clamp_min(0)/sw, (-command[:,2]).clamp_min(0)/sw), dim=1)
    speeds, ids = scores.max(dim=1)
    if standing:
        ids = torch.where(command[:, :3].abs().amax(dim=1) <= 1e-3, 6, ids)
    return ids, speeds
