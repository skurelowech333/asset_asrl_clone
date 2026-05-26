import unittest
import numpy as np

import asset_asrl as ast
from asset_asrl.Forces.Gravity import GravityVF as GravityVF_CY

vf = ast.VectorFunctions


""" 
Colors and logging functions
"""
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def log(msg, color=C.BLUE):
    print(color + msg + C.RESET)


def ok(msg):
    print(C.GREEN + msg + C.RESET)


def warn(msg):
    print(C.YELLOW + msg + C.RESET)


def fail(msg):
    print(C.RED + msg + C.RESET)

""" 
Constants
"""
MU_EARTH = 398600.435507
R_EARTH = 6378.137


""" 
Kernel wrapper
"""
def make_accel_from_kernel(kernel):

    ode_vf = kernel.ode()

    def accel(x):
        x = np.asarray(x)

        if hasattr(ode_vf, "compute"):
            return np.asarray(ode_vf.compute(x))

        if callable(ode_vf):
            try:
                return np.asarray(ode_vf(x))
            except TypeError:
                return np.asarray(ode_vf())

        raise RuntimeError("Cannot evaluate VF")

    return accel


""" 
Tests
"""
class TestGravityKernelPhysics(unittest.TestCase):

    def setUp(self):
        log("\n================ SETUP ================", C.HEADER)
        self.kernel = GravityVF_CY(MU_EARTH)
        self.accel = make_accel_from_kernel(self.kernel)
        ok("Kernel ready")

    def test_inverse_square_law(self):
        log("\n[TEST] inverse square law", C.BOLD)
        
        r1 = np.array([7000.0, 0, 0, 0, 0, 0])
        r2 = np.array([14000.0, 0, 0, 0, 0, 0])

        a1 = np.linalg.norm(self.accel(r1)[3:])
        a2 = np.linalg.norm(self.accel(r2)[3:])

        ratio = a1 / a2

        log(f"ratio = {ratio:.6f}", C.BLUE)
        self.assertAlmostEqual(ratio, 4.0, places=6)
        ok("inverse square passed")

    def test_radial_alignment(self):
        log("\n[TEST] radial alignment", C.BOLD)

        r = np.array([8000, 3000, -1000, 0, 0, 0])

        a = self.accel(r)[3:]
        cross = np.cross(r[:3], a)

        err = np.linalg.norm(cross)

        log(f"cross error = {err:.3e}", C.BLUE)
        self.assertLess(err, 1e-6)
        ok("radial alignment passed")

    def test_surface_gravity(self):
        log("\n[TEST] surface gravity", C.BOLD)

        r = np.array([R_EARTH, 0, 0, 0, 0, 0])

        a = self.accel(r)[3:]
        amag = np.linalg.norm(a)

        expected = MU_EARTH / (R_EARTH ** 2)

        log(f"model    = {amag:.6e}", C.BLUE)
        log(f"expected = {expected:.6e}", C.BLUE)
        self.assertAlmostEqual(amag, expected, places=6)
        ok("surface gravity passed")


""" 
Unit test summary prints
"""
class SummaryRunner(unittest.TextTestRunner):

    def run(self, test):

        result = super().run(test)

        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total - failures - errors

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        print(f"Total tests : {total}")
        print(f"Passed      : {passed}")
        print(f"Failures    : {failures}")
        print(f"Errors      : {errors}")

        if failures == 0 and errors == 0:
            print(C.GREEN + "\nALL TESTS PASSED" + C.RESET)
        else:
            print(C.RED + "\nSOME TESTS FAILED" + C.RESET)

        print("=" * 60)

        return result

if __name__ == "__main__":
    log("STARTING GRAVITY KERNEL TESTS", C.HEADER)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestGravityKernelPhysics
    )

    SummaryRunner(verbosity=2).run(suite)