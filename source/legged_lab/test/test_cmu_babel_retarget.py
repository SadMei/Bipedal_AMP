"""Synthetic numerical tests; no licensed dataset or body model required."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/tools/retarget'))
from cmu_babel import (AXES, align_initial_heading, category_passes, cmu_identity,
                       forward_kinematics, parse_amc, parse_asf, sample_interval, trajectory_metrics)
from audit_cmu_babel_bsrl import check_source_fk
from retarget_cmu_babel_bsrl import enforce_ankle_width, source_support

ASF = '''
:units
mass 1
length 0.45
angle deg
:root
order TX TY TZ RX RY RZ
axis XYZ
position 0 0 0
orientation 0 0 0
:bonedata
begin
id 1
name bone
direction 0 -1 0
length 10
axis 0 0 90 XYZ
dof rx
limits (-180 180)
end
:hierarchy
begin
root bone
end
'''
AMC = ':FULLY-SPECIFIED\n:DEGREES\n1\nroot 1 2 3 0 0 0\nbone 0\n2\nroot 1 2 3 0 0 0\nbone 90\n'


class TestCMUBabel(unittest.TestCase):
    def test_units_and_axes(self):
        s = parse_asf(ASF)
        np.testing.assert_allclose(AXES@AXES.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(AXES), 1.)
        p, _ = forward_kinematics(s, parse_amc(AMC, s))
        np.testing.assert_allclose(p[0, 0], np.array([3, 1, 2])*.0254/.45)
        np.testing.assert_allclose(p[0, 1]-p[0, 0], [0, 0, -10*.0254/.45])

    def test_axis_conjugation_and_lengths(self):
        s = parse_asf(ASF)
        p, r = forward_kinematics(s, parse_amc(AMC, s))
        # C=Rz(90) changes rotation about X into rotation about Y, leaving
        # the bone along -Y unchanged. Missing C^-1 would break this result.
        np.testing.assert_allclose(p[0], p[1], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(p[:, 1]-p[:, 0], axis=1), 10*.0254/.45)
        np.testing.assert_allclose(r @ np.swapaxes(r, -1, -2), np.broadcast_to(np.eye(3), r.shape), atol=1e-12)

    def test_root_rotates_hierarchy(self):
        s = parse_asf(ASF)
        motion = parse_amc(AMC.replace('1 2 3 0 0 0', '1 2 3 0 0 90'), s)
        p, _ = forward_kinematics(s, motion)
        np.testing.assert_allclose(p[0, 1]-p[0, 0], [0, 10*.0254/.45, 0], atol=1e-12)

    def test_fk_audit_accepts_rounded_source_directions(self):
        s = parse_asf(ASF.replace('direction 0 -1 0', 'direction 0.34202 -0.939693 0'))
        p, r = forward_kinematics(s, parse_amc(AMC, s))
        metrics = check_source_fk(s, p, r)
        self.assertLess(metrics['max_bone_length_error_m'], 1e-12)
        self.assertGreater(metrics['max_nominal_length_rounding_error_m'], 1e-9)

    def test_fk_audit_rejects_changed_bone_length(self):
        s = parse_asf(ASF)
        p, r = forward_kinematics(s, parse_amc(AMC, s))
        p[1, 1, 2] += .001
        with self.assertRaises(ValueError):
            check_source_fk(s, p, r)

    def test_fk_audit_rejects_nonunit_source_directions(self):
        s = parse_asf(ASF.replace('direction 0 -1 0', 'direction 0 -2 0'))
        p, r = forward_kinematics(s, parse_amc(AMC, s))
        with self.assertRaises(ValueError):
            check_source_fk(s, p, r)

    def test_reject_missing_channels(self):
        with self.assertRaises(ValueError):
            parse_amc(AMC.replace('bone 90', ''), parse_asf(ASF))

    def test_reject_noncontiguous_frames(self):
        with self.assertRaises(ValueError):
            parse_amc(AMC.replace('\n2\n', '\n3\n'), parse_asf(ASF))

    def test_reject_dialect(self):
        with self.assertRaises(ValueError):
            parse_asf(ASF.replace('angle deg', 'angle rad'))

    def test_path_identity_and_traversal(self):
        self.assertEqual(cmu_identity('CMU/CMU/7/07_03_poses.npz'), ('07', '07_03'))
        for value in ['../07/07_03_poses.npz', 'CMU/CMU/08/07_03_poses.npz', '/tmp/07_03.npz']:
            with self.assertRaises(ValueError):
                cmu_identity(value)

    def test_resampling_does_not_stretch(self):
        times = np.arange(121)/120
        p = np.zeros((121, 1, 3)); p[:, 0, 0] = times
        r = Rotation.from_euler('z', times).as_matrix()[:, None]
        points, rotations, t = sample_interval(p, r, 120, .125, .875, 60)
        np.testing.assert_allclose(np.diff(t), 1/60)
        np.testing.assert_allclose(points[:, 0, 0], t)
        np.testing.assert_allclose(Rotation.from_matrix(rotations[:, 0]).as_rotvec()[:, 2], t)
        self.assertLess(t[-1], .875)

    def test_constant_alignment_keeps_turn(self):
        p = np.zeros((3, 1, 3)); p[:, 0] = [[3, 4, 1], [3, 5, 1], [4, 5, 1]]
        r = Rotation.from_euler('z', [90, 120, 150], degrees=True).as_matrix()[:, None]
        points, rotations, yaw = align_initial_heading(p, r)
        np.testing.assert_allclose(points[0, 0], [0, 0, 1])
        self.assertAlmostEqual(yaw, np.pi/2)
        np.testing.assert_allclose(Rotation.from_matrix(rotations[:, 0]).as_euler('xyz', degrees=True)[:, 2], [0, 30, 60], atol=1e-12)

    def test_category_gates(self):
        t = np.arange(121)/60
        r = np.broadcast_to(np.eye(3), (121, 3, 3))
        forward = np.column_stack([t, np.zeros_like(t), np.ones_like(t)])
        lateral = forward[:, [1, 0, 2]]
        self.assertTrue(category_passes('straight', trajectory_metrics(forward, r, 60)))
        self.assertFalse(category_passes('straight', trajectory_metrics(lateral, r, 60)))
        self.assertTrue(category_passes('sideways', trajectory_metrics(lateral, r, 60)))
        self.assertTrue(category_passes('stand', trajectory_metrics(forward*0, r, 60)))

    def test_ankle_width_is_symmetric_in_body_frame(self):
        rotation = Rotation.from_euler('z', [90], degrees=True).as_matrix()
        root = np.array([[5., 3., 1.]])
        left, right, correction = enforce_ankle_width(root.copy(), root.copy(), root, rotation)
        np.testing.assert_allclose(.5*(left+right), root)
        np.testing.assert_allclose(left-right, [[-.22, 0., 0.]], atol=1e-12)
        np.testing.assert_allclose(correction, [.11])

    def test_ankle_width_does_not_change_wide_stance(self):
        root = np.zeros((1, 3)); rot = np.eye(3)[None]
        left, right = np.array([[0, .2, .1]]), np.array([[.1, -.2, .2]])
        l, r, correction = enforce_ankle_width(left, right, root, rot)
        np.testing.assert_allclose(l, left); np.testing.assert_allclose(r, right)
        np.testing.assert_allclose(correction, 0.)

    def test_stand_support_is_bilateral(self):
        support, fallback = source_support(np.zeros((10, 4, 3)), [], 60, 'stand')
        self.assertTrue(support.all())
        self.assertEqual(fallback, 0)

    def test_support_detects_high_swing_foot(self):
        points = np.zeros((120, 4, 3)); points[30:90, :2, 2] = .2
        support, fallback = source_support(points, ['lfoot', 'ltoes', 'rfoot', 'rtoes'], 60, 'straight')
        self.assertFalse(support[60, 0]); self.assertTrue(support[60, 1])


if __name__ == '__main__':
    unittest.main()
