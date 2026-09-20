"""Pure numerical tests for the standalone 100STYLE preview adapter."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/tools/retarget/retarget_100style_bsrl.py"
SPEC = importlib.util.spec_from_file_location("retarget_100style_bsrl", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Test100StyleBasis(unittest.TestCase):
    def test_right_handed_orthonormal(self):
        matrix = MODULE.AXIS_CHANGE
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(matrix), 1.)

    def test_axes_and_centimeters(self):
        q = np.tile([1., 0., 0., 0.], (1, 3, 1))
        _, pos = MODULE.convert_global_frames(q, 100*np.eye(3)[None])
        np.testing.assert_allclose(pos[0], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def test_identity_stays_identity(self):
        q = np.tile([1., 0., 0., 0.], (2, 3, 1))
        output, _ = MODULE.convert_global_frames(q, np.zeros((2, 3, 3)))
        np.testing.assert_allclose(output, q)

    def test_source_yaw_becomes_robot_yaw(self):
        q = Rotation.from_euler("y", 30, degrees=True).as_quat(scalar_first=True)[None, None]
        converted, _ = MODULE.convert_global_frames(q, np.zeros((1, 1, 3)))
        expected = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(converted[0, 0], scalar_first=True).as_matrix(), expected, atol=1e-12)

    def test_rotation_and_position_agree(self):
        rotation = Rotation.from_euler("YXZ", [23, -17, 42], degrees=True)
        point = np.array([12., 35., -27.])
        q = rotation.as_quat(scalar_first=True)[None, None]
        converted, pos = MODULE.convert_global_frames(q, point[None, None])
        lhs = Rotation.from_quat(converted[0, 0], scalar_first=True).apply(pos[0, 0])
        rhs = MODULE.AXIS_CHANGE @ rotation.apply(point) * .01
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_distances_preserved_with_scale(self):
        random = np.random.default_rng(42)
        points = random.normal(size=(5, 7, 3))*100
        q = Rotation.random(35, random_state=random).as_quat(scalar_first=True).reshape(5, 7, 4)
        converted, output = MODULE.convert_global_frames(q, points)
        np.testing.assert_allclose(np.linalg.norm(output, axis=-1), .01*np.linalg.norm(points, axis=-1))
        np.testing.assert_allclose(np.linalg.norm(converted, axis=-1), 1.)


if __name__ == "__main__":
    unittest.main()
