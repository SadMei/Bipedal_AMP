# BSRL state estimator

The optional estimator consumes three frames of the 48-D BSRL policy state
(144 values) and predicts eight simulator-only quantities: base height, body
linear velocity, ordered left/right foot contact, and ordered left/right foot
height. The Actor receives only the current 48-D frame plus the eight predicted
values; privileged labels never enter deployment inference.

Enable it with:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-BSRL-v0 --use_estimator ...
```

The `legacy4d` profile remains available for compatible BSRL checkpoints.
