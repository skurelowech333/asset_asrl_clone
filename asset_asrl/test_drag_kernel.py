# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:49:07 2026

@author: Sarah
"""

import unittest
import numpy as np

import asset_asrl as ast
from DragKernel import DragKernel
from common import C, log, ok, warn, fail
from common import R_EARTH
from common import SummaryRunner


vf = ast.VectorFunctions

Re = R_EARTH * 1000

Cd = 2.2
A_over_m = 0.01
rho0 = 1.225


"""
=========================================================
DRAG KERNEL PHYSICS TEST SUITE
=========================================================

PURPOSE
---------------------------------------------------------
This test suite validates the correctness of a Cython
atmospheric drag kernel implemented in ASSET-ASRL.

The goal is to ensure that the numerical kernel matches
known aerodynamic and orbital decay physics under a
simplified exponential atmosphere model.

=========================================================
PHYSICAL MODEL
=========================================================

Drag acceleration model implemented:

    a_drag = -1/2 * Cd * (A/m) * ρ(h) * |v| * v

Where:
- Cd      = drag coefficient
- A/m     = area-to-mass ratio
- ρ(h)    = atmospheric density model
- v       = velocity vector relative to inertial frame

Atmospheric model used:
---------------------------------------------------------
    ρ(h) = ρ0 * exp(-(h / 8500))

This corresponds to a simplified single-scale-height
exponential atmosphere.

=========================================================
REFERENCE SOURCES
=========================================================
    ---------------------------------------------------------
    1. Atmospheric Drag Model
    ---------------------------------------------------------
    Vallado, D. A.
    "Fundamentals of Astrodynamics and Applications"
    4th Edition, Microcosm Press
    
    - Standard exponential atmosphere approximation
    - Used widely in LEO orbit propagation
    
    ---------------------------------------------------------
    2. Drag Force Equation
    ---------------------------------------------------------
    NASA Orbital Mechanics Handbook
    
    https://www.nasa.gov
    
    Canonical form:
        F_d = 1/2 ρ v^2 Cd A
    
    Converted to acceleration:
        a_d = F_d / m
    
    ---------------------------------------------------------
    3. Standard atmosphere approximation
    ---------------------------------------------------------
    U.S. Standard Atmosphere 1976
    
    https://ntrs.nasa.gov/citations/19770009539
    
    Scale height near LEO:
        H ≈ 7–9 km effective (simplified here as 8500 m)
"""

class TestDragKernelPhysics(unittest.TestCase):

    def setUp(self):
        log("\n================ DRAG SETUP ================", C.HEADER)

        self.kernel = DragKernel(
            Cd,
            A_over_m,
            rho0,
            Re
        )

        self.drag_vf = self.kernel.vf(None, None)

        ok("Drag kernel initialized")

    def accel(self, r, v):
        X = np.hstack([r, v])
        ode = self.kernel.vf(None, None)

        if hasattr(ode, "compute"):
            return np.asarray(ode.compute(X))

        if callable(ode):
            try:
                return np.asarray(ode(X))
            except TypeError:
                return np.asarray(ode())

        raise RuntimeError("Cannot evaluate drag VF")

    def test_drag_opposes_velocity(self):
        """
        Physical law:
        Drag force always opposes velocity vector.

        Proof condition:
            v · a_drag < 0
        """

        log("\n[TEST] drag direction", C.BOLD)

        r = np.array([Re + 200e3, 0, 0])
        v = np.array([7500.0, 1000.0, 0.0])

        a = self.accel(r, v)

        dot = np.dot(a, v)

        log(f"dot(v, a_drag) = {dot:.6e}", C.BLUE)

        self.assertLess(dot, 0.0)

        ok("drag opposes velocity")

    def test_density_scaling(self):
        """
        Atmospheric model property:

        ρ(h) decreases exponentially with altitude.

        Therefore:
            |a_drag(low orbit)| > |a_drag(high orbit)|
        """

        log("\n[TEST] density scaling", C.BOLD)

        r_low = np.array([Re + 200e3, 0, 0])
        r_high = np.array([Re + 800e3, 0, 0])

        v = np.array([7500.0, 0, 0])

        a_low = np.linalg.norm(self.accel(r_low, v))
        a_high = np.linalg.norm(self.accel(r_high, v))

        log(f"low altitude drag  = {a_low:.6e}", C.BLUE)
        log(f"high altitude drag = {a_high:.6e}", C.BLUE)

        self.assertGreater(a_low, a_high)

        ok("density scaling correct")

    def test_zero_velocity(self):
        """
        Physical law:
        If velocity is zero, drag force is zero.

        a_drag ∝ |v| v
        """

        log("\n[TEST] zero velocity", C.BOLD)

        r = np.array([Re + 400e3, 0, 0])
        v = np.array([0.0, 0.0, 0.0])

        a = self.accel(r, v)

        mag = np.linalg.norm(a)

        log(f"drag magnitude = {mag:.6e}", C.BLUE)

        self.assertAlmostEqual(mag, 0.0, places=12)

        ok("zero velocity behavior correct")

    def test_velocity_scaling(self):
        """
        Physical law:

        Drag magnitude scales quadratically with velocity:

            |a| is proportional to |v|^2

        because:
            a is proportional to |v| v
        """
        
        log("\n[TEST] velocity scaling", C.BOLD)

        r = np.array([Re + 300e3, 0, 0])

        v1 = np.array([500.0, 0, 0])
        v2 = np.array([1000.0, 0, 0])

        a1 = np.linalg.norm(self.accel(r, v1))
        a2 = np.linalg.norm(self.accel(r, v2))

        ratio = a2 / a1

        log(f"drag ratio (v2/v1) = {ratio:.6f}", C.BLUE)
        self.assertAlmostEqual(ratio, 4.0, places=1)
        ok("velocity quadratic scaling correct")

if __name__ == "__main__":

    log("STARTING DRAG KERNEL TESTS", C.HEADER)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestDragKernelPhysics
    )

    SummaryRunner(verbosity=2).run(suite, 'DRAG')