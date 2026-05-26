import unittest
import numpy as np

import asset_asrl as ast
from asset_asrl.Forces.Gravity import GravityVF as GravityVF_CY
from common import C, log, ok, warn, fail
from common import MU_EARTH, R_EARTH
from common import SummaryRunner

vf = ast.VectorFunctions

"""
=========================================================
TWO-BODY IDEAL EARTH GRAVITY KERNEL TEST SUITE
=========================================================

PURPOSE
---------------------------------------------------------
This file validates the correctness of a Cython gravity
kernel implementing a classical two-body Newtonian Earth
model.

The goal is to ensure that the kernel matches known
analytical orbital mechanics results under idealized
conditions.

=========================================================
PHYSICAL MODEL
=========================================================

This test suite assumes the following ideal model:

    a(r) = -μ * r / |r|^3

Where:
- μ  = Earth gravitational parameter
- r  = position vector in inertial space

This is the canonical inverse-square gravitational field.

This represents a pure central-force Kepler system.

=========================================================
THEORETICAL BACKGROUND
=========================================================

This model is derived from:

- Newton’s Law of Universal Gravitation
- Classical two-body orbital mechanics
- Kepler’s laws of planetary motion

Key identities used in validation:

1. Circular orbit velocity:
   v = sqrt(μ / r)

2. Escape velocity:
   v_escape = sqrt(2μ / r)

3. Orbital period:
   T = 2π * sqrt(r^3 / μ)

4. Inverse-square scaling:
   a is proportional to  1 / r²

=========================================================
REFERENCE SOURCES
=========================================================
    ---------------------------------------------------------
    1. Earth gravitational parameter (μ)
    ---------------------------------------------------------
    NASA / JPL Solar System Dynamics Group
    IAU 2015 Resolution (IAU WG Report)
    
    https://ssd.jpl.nasa.gov/astro_par.html
    
    Value:
    μ = 398600.435507 km^3/s^2
    
    ---------------------------------------------------------
    2. Earth reference radius (WGS84)
    ---------------------------------------------------------
    IERS Conventions (2010)
    NGA Earth Model Documentation
    
    https://earth-info.nga.mil/
    
    Value:
    Rₑ = 6378.137 km (equatorial radius)
    
    ---------------------------------------------------------
    3. Standard gravity (g_0)
    ---------------------------------------------------------
    ISO 80000-3 (Quantities and Units)
    CIPM definitions
    
    Value:
    g₀ = 9.80665 m/s² (exact)
    
    ---------------------------------------------------------
    4. Orbital mechanics theory
    ---------------------------------------------------------
    - Bate, Mueller & White (1971)
      "Fundamentals of Astrodynamics"
    
    - Vallado (2013)
      "Fundamentals of Astrodynamics and Applications"

"""

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
Tests for Gravity Kernel
"""
class TestGravityKernelPhysics(unittest.TestCase):

    def setUp(self):
        log("\n================ SETUP ================", C.HEADER)
        self.kernel = GravityVF_CY(MU_EARTH)
        self.accel = make_accel_from_kernel(self.kernel)
        ok("Kernel ready")

    def test_inverse_square_law(self):
        """ 
        Tests inverse-square scaling of gravitational field.

        Verified using:
            a(r1)/a(r2) = (r2/r1)^2

        Based on:
        - Newton’s law of gravitation
        - Bate, Mueller & White (1971)
        """
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

if __name__ == "__main__":
    log("STARTING GRAVITY KERNEL TESTS", C.HEADER)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestGravityKernelPhysics
    )

    SummaryRunner(verbosity=2).run(suite, 'GRAVITY')