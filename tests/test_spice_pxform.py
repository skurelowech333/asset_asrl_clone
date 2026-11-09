"""
Unit tests for the native SPICE frame transformation VectorFunctions.

Covers ext.SpicePxform (ET -> flattened 3x3 rotation), ext.SpicePxformApply
(ET, r -> R(t) r) and their inverse() counterparts.

The tests need spiceypy and a working ASSET build that includes the SPICE
extension. If the SPICE nodes are not available the tests are skipped.

A minimal test PCK for IAU_MOON is written to a temporary directory, so the
tests do not depend on external NAIF kernels. Set SPICE_KERNELS to a
space-separated list of kernel paths to also run against real kernels.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    import spiceypy as spice
except ImportError:  # pragma: no cover
    spice = None

try:
    import asset_asrl.Extensions as ext
except ImportError:  # pragma: no cover
    ext = None

TEST_PCK = """KPL/PCK

\\begindata
BODY301_POLE_RA  = ( 269.9949   0.0031   0.0 )
BODY301_POLE_DEC = (  66.5392   0.0130   0.0 )
BODY301_PM       = (  38.3213  13.17635815  -1.4e-12 )
\\begintext
"""

FROM_FRAME = "J2000"
TO_FRAME = "IAU_MOON"


def _skip_reason():
    if spice is None:
        return "spiceypy is not installed"
    if ext is None or not hasattr(ext, "SpicePxform"):
        return "ASSET build does not include SpicePxform"
    return None


SKIP_REASON = _skip_reason()


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TestSpicePxform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="asset_spice_")
        pck = os.path.join(cls._tmpdir, "moon_test.tpc")
        with open(pck, "w") as f:
            f.write(TEST_PCK)

        kernels = [pck] + os.environ.get("SPICE_KERNELS", "").split()
        ext.SpiceLoadKernels(kernels)

        cls.rot = ext.SpicePxform(FROM_FRAME, TO_FRAME)
        cls.apply = ext.SpicePxformApply(FROM_FRAME, TO_FRAME)
        cls.ets = [0.0, 1.0e8, -3.2e7, 5.0e8]
        cls.r = np.array([1.2e6, -0.7e6, 0.5e6])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _ref_rotation(self, et):
        return np.array(spice.pxform(FROM_FRAME, TO_FRAME, et))

    # Rotation node

    def test_rotation_matches_spiceypy(self):
        for et in self.ets:
            with self.subTest(et=et):
                R = np.array(self.rot.compute(np.array([et]))).reshape(3, 3)
                np.testing.assert_allclose(R, self._ref_rotation(et),
                                           rtol=0, atol=1e-13)

    def test_rotation_is_orthogonal(self):
        for et in self.ets:
            with self.subTest(et=et):
                R = np.array(self.rot.compute(np.array([et]))).reshape(3, 3)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_rotation_output_is_flattened_row_major(self):
        et = 1.0e8
        out = np.array(self.rot.compute(np.array([et])))
        self.assertEqual(out.shape, (9,))
        np.testing.assert_allclose(out.reshape(3, 3), self._ref_rotation(et),
                                   atol=1e-13)

    def test_rotation_jacobian_matches_finite_difference(self):
        h = 1.0
        for et in self.ets:
            with self.subTest(et=et):
                J = np.array(self.rot.jacobian(np.array([et])))
                J = J.reshape(9)
                fd = (np.array(self.rot.compute(np.array([et + h])))
                      - np.array(self.rot.compute(np.array([et - h])))) / (2 * h)
                scale = max(np.max(np.abs(fd)), 1e-30)
                self.assertLess(np.max(np.abs(J - fd)) / scale, 1e-6)

    # Apply node

    def test_apply_matches_rotation_times_vector(self):
        for et in self.ets:
            with self.subTest(et=et):
                x = np.concatenate([[et], self.r])
                r_pa = np.array(self.apply.compute(x))
                np.testing.assert_allclose(r_pa, self._ref_rotation(et) @ self.r,
                                           rtol=0, atol=1e-6)

    def test_apply_preserves_norm(self):
        x = np.concatenate([[1.0e8], self.r])
        r_pa = np.array(self.apply.compute(x))
        self.assertAlmostEqual(np.linalg.norm(r_pa),
                               np.linalg.norm(self.r), delta=1e-6)

    def test_apply_is_linear_in_vector(self):
        et = 2.0e8
        a = np.array([1.0e6, 2.0e6, -3.0e6])
        b = np.array([-4.0e5, 5.0e5, 6.0e5])
        fa = np.array(self.apply.compute(np.concatenate([[et], a])))
        fb = np.array(self.apply.compute(np.concatenate([[et], b])))
        fab = np.array(self.apply.compute(np.concatenate([[et], a + 2 * b])))
        np.testing.assert_allclose(fab, fa + 2 * fb, rtol=0, atol=1e-6)

    def test_apply_jacobian_shape_and_fd(self):
        h = 1.0
        for et in self.ets:
            with self.subTest(et=et):
                x = np.concatenate([[et], self.r])
                J = np.array(self.apply.jacobian(x))
                self.assertEqual(J.shape, (3, 4))

                J_fd = np.zeros((3, 4))
                for k in range(4):
                    dx = np.zeros(4)
                    dx[k] = h
                    J_fd[:, k] = (np.array(self.apply.compute(x + dx))
                                  - np.array(self.apply.compute(x - dx))) / (2 * h)
                scale = max(np.max(np.abs(J_fd)), 1e-30)
                self.assertLess(np.max(np.abs(J - J_fd)) / scale, 1e-6)

    def test_apply_jacobian_position_block_is_rotation(self):
        et = 1.0e8
        J = np.array(self.apply.jacobian(np.concatenate([[et], self.r])))
        np.testing.assert_allclose(J[:, 1:], self._ref_rotation(et), atol=1e-13)

    # Inverse

    def test_inverse_round_trip(self):
        et = 4.0e7
        inv = self.apply.inverse()
        r_pa = np.array(self.apply.compute(np.concatenate([[et], self.r])))
        r_back = np.array(inv.compute(np.concatenate([[et], r_pa])))
        np.testing.assert_allclose(r_back, self.r, rtol=0, atol=1e-6)

    def test_inverse_equals_transpose(self):
        et = 1.5e8
        inv = self.apply.inverse()
        out = np.array(inv.compute(np.concatenate([[et], self.r])))
        np.testing.assert_allclose(out, self._ref_rotation(et).T @ self.r,
                                   rtol=0, atol=1e-6)

    # Error handling

    def test_unknown_frame_raises(self):
        with self.assertRaises(Exception):
            bad = ext.SpicePxform(FROM_FRAME, "NOT_A_FRAME_XYZ")
            bad.compute(np.array([0.0]))


MU_MOON = 4.9028e12  # m^3/s^2


TEST_EARTH_PCK = """KPL/PCK

\\begindata
BODY399_POLE_RA  = ( 0.0   0.0   0.0 )
BODY399_POLE_DEC = ( 90.0  0.0   0.0 )
BODY399_PM       = ( 190.147  360.9856235  0.0 )
\\begintext
"""

EARTH_TO_FRAME = "IAU_EARTH"
MU_EARTH = 398600.4418
R_EARTH = 6378.1363
EGM96_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "native", "data", "egm96.nor")


@unittest.skipIf(SKIP_REASON is not None or not os.path.exists(EGM96_PATH),
                 SKIP_REASON or "EGM96 coefficients not found")
class TestGraphComposedPines(unittest.TestCase):
    """Pines acceleration composed at graph level with native SPICE frame nodes.

    a_J2000 = R^T(et) * PinesGravity(R(et) * r), built from separate nodes.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="asset_spice_earth_")
        pck = os.path.join(cls._tmpdir, "earth_test.tpc")
        with open(pck, "w") as f:
            f.write(TEST_EARTH_PCK)
        ext.SpiceLoadKernels([pck])
        cls.pines = ext.PinesGravity(os.path.abspath(EGM96_PATH), 8,
                                     MU_EARTH, R_EARTH, "fully_normalized")
        cls.to_body = ext.SpicePxformApply(FROM_FRAME, EARTH_TO_FRAME)
        cls.to_inertial = cls.to_body.inverse()
        cls.r = np.array([6.9e3, -4.1e3, 2.3e3])  # km

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _ref_accel(self, et, r):
        R = np.array(spice.pxform(FROM_FRAME, EARTH_TO_FRAME, et))
        a_body = np.array(self.pines.compute(R @ r))
        return R.T @ a_body

    def test_composed_matches_spiceypy_reference(self):
        for et in [0.0, 1.0e6, -2.5e5]:
            with self.subTest(et=et):
                r_body = self.to_body.compute(np.concatenate([[et], self.r]))
                a_body = self.pines.compute(np.array(r_body))
                a = np.array(self.to_inertial.compute(
                    np.concatenate([[et], a_body])))
                np.testing.assert_allclose(a, self._ref_accel(et, self.r),
                                           rtol=1e-12, atol=1e-18)

    def test_pines_is_plain_vectorfunction(self):
        x = np.array([6.9e3, -4.1e3, 2.3e3])
        J = np.array(self.pines.jacobian(x))
        self.assertEqual(J.shape, (3, 3))
        J_fd = np.zeros((3, 3))
        for k in range(3):
            dx = np.zeros(3)
            dx[k] = 1.0e-3
            J_fd[:, k] = (np.array(self.pines.compute(x + dx))
                          - np.array(self.pines.compute(x - dx))) / 2e-3
        np.testing.assert_allclose(J, J_fd, rtol=1e-5,
                                   atol=1e-9 * np.max(np.abs(J_fd)))


class TestSpiceForceFrame(unittest.TestCase):
    """Generic wrapper: any 3->3 force kernel in a SPICE body frame."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="asset_spice_force_")
        pck = os.path.join(cls._tmpdir, "earth_test.tpc")
        with open(pck, "w") as f:
            f.write(TEST_EARTH_PCK)
        ext.SpiceLoadKernels([pck])
        cls.pines = ext.PinesGravity(os.path.abspath(EGM96_PATH), 8,
                                     MU_EARTH, R_EARTH, "fully_normalized")
        cls.r = np.array([6.9e3, -4.1e3, 2.3e3])
        cls.wrap = staticmethod(lambda k, a, b: ext.SpiceForceFrame(k, a, b))
        cls.to_body = ext.SpicePxformApply(FROM_FRAME, EARTH_TO_FRAME)
        cls.to_inertial = cls.to_body.inverse()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _check(self, kernel, ref):
        f = self.wrap(kernel, FROM_FRAME, EARTH_TO_FRAME)
        for et in [0.0, 1.0e6, -2.5e5]:
            with self.subTest(et=et):
                x = np.concatenate([[et], self.r])
                a = np.array(f.compute(x))
                np.testing.assert_allclose(a, ref(et, self.r), rtol=1e-12, atol=1e-18)
        J = np.array(f.jacobian(np.concatenate([[1.0e6], self.r])))
        J_fd = np.zeros((3, 4))
        x0 = np.concatenate([[1.0e6], self.r])
        for k in range(4):
            dx = np.zeros(4)
            dx[k] = 1.0e-3
            J_fd[:, k] = (np.array(f.compute(x0 + dx))
                          - np.array(f.compute(x0 - dx))) / 2e-3
        np.testing.assert_allclose(J, J_fd, rtol=1e-5,
                                   atol=1e-9 * np.max(np.abs(J_fd)))

    def test_pines_kernel(self):
        def ref(et, r):
            R = np.array(spice.pxform(FROM_FRAME, EARTH_TO_FRAME, et))
            return R.T @ np.array(self.pines.compute(R @ r))
        self._check(self.pines, ref)

    def test_fused_node_matches_composed(self):
        fused = self.wrap(self.pines, FROM_FRAME, EARTH_TO_FRAME)
        for et in [0.0, 1.0e6, -2.5e5]:
            x = np.concatenate([[et], self.r])
            r_body = np.array(self.to_body.compute(x))
            a_body = np.array(self.pines.compute(r_body))
            composed = np.array(self.to_inertial.compute(
                np.concatenate([[et], a_body])))
            with self.subTest(et=et):
                np.testing.assert_allclose(np.array(fused.compute(x)), composed,
                                           rtol=1e-12, atol=1e-18)


if __name__ == "__main__":
    unittest.main()
