"""Numerical ACCAD adapter tests; no licensed data or GPU required."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/tools/retarget'))
from accad_smplx import (SMPL_TO_ROBOT_BASIS, limb_reference_frame, motion_fk,
                        sample_indices, skeleton_fk, flight_clearance_target)
from accad_reference import resolve_reference, category_for_name


class TestAccadSmplx(unittest.TestCase):
    def test_semantic_basis(self):
        basis = SMPL_TO_ROBOT_BASIS
        np.testing.assert_allclose(basis.T @ basis, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(basis), 1.)
        np.testing.assert_equal(basis[:, 0], [0, 0, 1])  # SMPL forward
        np.testing.assert_equal(basis[:, 2], [0, 1, 0])  # SMPL up

    def test_exact_decimation(self):
        idx = sample_indices(917, 120, 60)
        np.testing.assert_array_equal(idx, np.arange(0, 917, 2))
        np.testing.assert_allclose(np.diff(idx/120), 1/60)
        self.assertEqual(idx[-1], 916)

    def test_invalid_rates(self):
        for count, source, output in [(1,120,60), (5,120,0), (5,0,60),
                                      (5,120,50), (5,60,120), (5,np.nan,60),
                                      (5,120,np.inf)]:
            with self.subTest(values=(count,source,output)), self.assertRaises(ValueError):
                sample_indices(count,source,output)

    def test_root_rest_offset_is_not_rotated(self):
        rest = np.array([[.1, .2, .3], [.1, -.8, .3]])
        pose = np.zeros((2, 2, 3)); pose[1, 0, 2] = np.pi/2
        trans = np.array([[1., 2., 3.], [2., 4., 3.]])
        p, r = skeleton_fk(rest, [-1, 0], pose, trans)
        np.testing.assert_allclose(p[:, 0], trans+rest[0])
        np.testing.assert_allclose(p[1, 1]-p[1, 0], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(p[:, 1]-p[:, 0], axis=1), 1.)
        np.testing.assert_allclose(r[1, 1], Rotation.from_euler('z', 90, degrees=True).as_matrix())

    def test_parent_first_required(self):
        with self.assertRaises(ValueError):
            skeleton_fk(np.zeros((2, 3)), [-1, 1], np.zeros((2, 2, 3)), np.zeros((2, 3)))

    def test_global_translation_changes_every_joint_equally(self):
        rest = np.array([[0., .1, 0.], [.1, -.2, 0.], [.1, -.6, 0.]])
        rng = np.random.default_rng(1)
        poses = rng.normal(size=(10, 3, 3))
        trans = rng.normal(size=(10, 3))
        p0, r0 = skeleton_fk(rest, [-1, 0, 1], poses, np.zeros_like(trans))
        p1, r1 = skeleton_fk(rest, [-1, 0, 1], poses, trans)
        np.testing.assert_allclose(p1-p0, np.broadcast_to(trans[:, None], p0.shape), atol=1e-12)
        np.testing.assert_allclose(r1, r0)

    def test_motion_fk_retains_translation_after_decimation(self):
        rest = np.zeros((22, 3)); rest[:, 1] = -np.arange(22)*.01
        parents = np.arange(22)-1
        trans = np.arange(15).reshape(5, 3)/100
        motion = dict(trans=trans, fps=120, betas=np.zeros(16),
                      root_orient=np.zeros((5, 3)), pose_body=np.zeros((5, 63)))
        with patch('accad_smplx.shaped_skeleton', return_value=(rest, parents)):
            p, _, _, _, idx = motion_fk(motion, 'unused')
        np.testing.assert_array_equal(idx, [0, 2, 4])
        np.testing.assert_allclose(p[:, 0]-p[0, 0], trans[idx]-trans[0])

    def test_limb_basis(self):
        down = np.array([.1, -.9, .02])
        frame = limb_reference_frame(down)
        np.testing.assert_allclose(frame.T@frame, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(frame[:, 2], -down/np.linalg.norm(down))
        self.assertAlmostEqual(np.linalg.det(frame), 1.)
        with self.assertRaises(ValueError):
            limb_reference_frame([0., 0., -1.])

    def test_running_clearance_keeps_flight(self):
        heights = np.full((100,2), .03)
        heights[40:60] += .12
        target, audit = flight_clearance_target(heights, .8)
        np.testing.assert_allclose(target[:40], .0005)
        np.testing.assert_allclose(target[40:60], .0005+.12*.8-.005)
        self.assertAlmostEqual(audit['geometric_air_fraction_over_1cm'], .2)
        # A constant capture floor offset must not change the transferred rise.
        target2, _ = flight_clearance_target(heights+1.3, .8)
        np.testing.assert_allclose(target2, target)

    def test_flight_uses_lower_foot(self):
        heights = np.column_stack([np.full(100,.3), np.zeros(100)])
        target, _ = flight_clearance_target(heights, 1.)
        np.testing.assert_allclose(target, .0005)

    def test_reference_duplicate_subjects_are_not_overwritten(self):
        name = 'C5_-_walk_to_run_stageii'
        candidates = [f'ACCAD/{subject}/{name}.npz' for subject in ('Male2Running_c3d','Female1Running_c3d')]
        clips, mapping = resolve_reference({name: 1.}, candidates)
        self.assertEqual(len(clips), 2)
        self.assertEqual(len({c['stem'] for c in clips}), 2)
        self.assertTrue(all(c['reference_source_ambiguous'] for c in clips))
        self.assertEqual(mapping[0]['identity_status'], 'ambiguous_subject_not_resolved')

    def test_reference_missing_fails(self):
        with self.assertRaises(ValueError):
            resolve_reference({'C3_-_run_stageii':1.}, [])

    def test_reference_extras_deduplicated_by_full_path(self):
        name = 'B22_-__side_step_left_stageii'
        relative = 'Male2Walking_c3d/'+name+'.npz'
        clips, _ = resolve_reference({name:1.}, ['ACCAD/'+relative],
            [('old_stem','sideways',relative,'side step',True)])
        self.assertEqual(len(clips),1)

    def test_reference_categories(self):
        self.assertEqual(category_for_name('C3_-_run_stageii'),'run')
        self.assertEqual(category_for_name('C5_-_walk_to_run_stageii'),'run_transition')
        self.assertEqual(category_for_name('Walk_B4_-_Stand_to_Walk_Back_stageii'),'backward')
        self.assertEqual(category_for_name('B22_-__side_step_left_stageii'),'sideways')
        self.assertEqual(category_for_name('B15_-__Walk_turn_around_stageii'),'turn')


if __name__ == '__main__':
    unittest.main()
