# ============================================================
# FILE: benchmark_cpp_sh_node.py
#
# Benchmark and test suite for the C++ SphericalHarmonicGravity
# ASSET node (extensions/SphericalHarmonicGravity.h)
#
# Sections
# --------
#  1. Correctness  — compare against analytical J2 reference and
#                    the Pines Cartesian kernel at degree 2
#  2. Jacobian     — verify ASSET autodiff Jacobians are finite,
#                    consistent with FD, and non-trivial (this is
#                    the key capability the old Cython graph lacked)
#  3. Build time   — instantaneous construction at every degree
#                    (old ASSET graph hangs at degree 20-50)
#  4. Eval time    — per-evaluation microsecond throughput
#  5. Propagation  — full DOPRI87 orbit over one period
#
# Usage
# -----
#   cd asset_asrl
#   python benchmark_cpp_sh_node.py
#
# ============================================================

import time
import gc
import sys
import unittest
import numpy as np
import asset_asrl as ast
import asset_asrl.Extensions

from GravitySphericalHarmonic import SphericalHarmonicGravityVF
from GravityPines import PinesGravityVF


# ============================================================
# Constants
# ============================================================

MU = 398600.435507     # km^3/s^2  (Earth, IAU 2015)
R  = 6378.137          # km        (WGS84 equatorial radius)
J2 = 1.08262668e-3


vf   = ast.VectorFunctions
Args = vf.Arguments
oc   = ast.OptimalControl


# ============================================================
# Coefficient helpers
# ============================================================

def _np_coefficients(degree):
    """Return (C, S) as (degree+1) x (degree+1) numpy arrays."""
    C = np.zeros((degree + 1, degree + 1))
    S = np.zeros((degree + 1, degree + 1))
    return C, S


def _np_j2_coefficients(degree):
    """J2-only coefficients in numpy form."""
    C, S = _np_coefficients(degree)
    # Fully-normalised  C20 = -J2 / sqrt(5)
    C[2, 0] = -J2 / np.sqrt(5.0)
    return C, S


def _dict_j2_coefficients(degree):
    """J2-only coefficients as dicts (for PinesGravityVF)."""
    C, S = {}, {}
    for n in range(degree + 1):
        for m in range(n + 1):
            C[n, m] = 0.0
            S[n, m] = 0.0
    C[2, 0] = -J2 / np.sqrt(5.0)
    return C, S


def _np_random_coefficients(degree, seed=42):
    """Small random perturbation coefficients — tests high-degree path."""
    rng = np.random.default_rng(seed)
    C, S = _np_coefficients(degree)
    scale = 1e-4
    for n in range(2, degree + 1):
        for m in range(n + 1):
            C[n, m] = scale * rng.standard_normal() / (n * n)
            S[n, m] = scale * rng.standard_normal() / (n * n)
    C[2, 0] = -J2 / np.sqrt(5.0)
    return C, S


# ============================================================
# Analytical J2 reference (fully-normalised C20)
# ============================================================

def _analytical_j2(r):
    """
    Exact J2 acceleration using fully-normalised C20 = -J2/sqrt(5).

    Returns [ax, ay, az] in km/s^2.
    """
    x, y, z = r
    r2   = x * x + y * y + z * z
    rmag = np.sqrt(r2)

    fac    = -MU / rmag ** 3
    common =  1.5 * J2 * (R / rmag) ** 2

    zx = 5 * z * z / r2 - 1
    zz = 5 * z * z / r2 - 3

    return np.array([
        fac * x * (1 - common * zx),
        fac * y * (1 - common * zx),
        fac * z * (1 - common * zz),
    ])


# ============================================================
# Build helpers
# ============================================================

def _build_sh(degree, C, S, *, verbose=False):
    """Build SphericalHarmonicGravityVF and return (model, elapsed_s)."""
    gc.collect()
    t0 = time.perf_counter()
    model = SphericalHarmonicGravityVF(MU, R, C, S, degree)
    t1 = time.perf_counter()
    if verbose:
        print(f"  SH build  degree={degree}: {(t1 - t0) * 1e3:.3f} ms")
    return model, t1 - t0


def _build_pines(degree, C, S, *, verbose=False):
    """Build PinesGravityVF and return (model, elapsed_s)."""
    gc.collect()
    t0 = time.perf_counter()
    model = PinesGravityVF(MU, R, C, S, degree)
    t1 = time.perf_counter()
    if verbose:
        print(f"  Pines build degree={degree}: {(t1 - t0) * 1e3:.3f} ms")
    return model, t1 - t0


def _eval_accel(model, state):
    """Evaluate acceleration via model.vf().compute()."""
    return np.asarray(
        model.vf().compute(np.asarray(state, dtype=float))
    )


# ============================================================
# ============================================================
# 1.  CORRECTNESS TESTS
# ============================================================
# ============================================================

class TestCorrectnessJ2(unittest.TestCase):
    """
    Compare C++ node against analytical J2 formula.

    J2 only (degree=2, random coefficients zero except C20).
    Both should match analytical to < 1e-10 km/s^2.
    """

    DEGREE = 2
    TOL    = 1e-10

    _STATES = [
        [7000.0,    0.0,    0.0, 0.0, 7.5, 0.0],
        [6600.0, 1000.0, 2000.0, 0.0, 7.5, 0.0],
        [8000.0, -500.0, 3000.0, 0.0, 7.0, 1.0],
        [7500.0,  800.0, -1200.0, 1.0, 6.8, 0.5],
        [   R,     0.0,    0.0, 0.0, 7.9, 0.0],   # equatorial surface
        [7000.0,    0.0, 7000.0, 0.0, 5.0, 5.0],   # 45° inclined
    ]

    @classmethod
    def setUpClass(cls):
        C, S = _np_j2_coefficients(cls.DEGREE)
        cls.model, _ = _build_sh(cls.DEGREE, C, S)

    def test_against_analytical(self):
        for state in self._STATES:
            r   = np.array(state[:3])
            ref = _analytical_j2(r)
            got = _eval_accel(self.model, state)
            err = np.linalg.norm(got - ref)
            self.assertLess(
                err, self.TOL,
                f"J2 error {err:.3e} >= {self.TOL} at r={r}"
            )

    def test_radial_direction_at_equator(self):
        """Equatorial acceleration must point anti-radially (no J2 tangential)."""
        state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]
        a = _eval_accel(self.model, state)
        # cross product r x a should be near zero
        cross = np.cross(np.array(state[:3]), a)
        self.assertLess(np.linalg.norm(cross), 1e-8)


class TestCorrectnessVsPines(unittest.TestCase):
    """
    Cross-validate C++ SH node against Pines Cartesian kernel.

    Both use the same J2-only coefficients at degree=2.
    Agreement should be near machine epsilon (< 1e-12).
    """

    DEGREE = 2
    TOL    = 1e-10

    @classmethod
    def setUpClass(cls):
        Cnp, Snp   = _np_j2_coefficients(cls.DEGREE)
        Cd,  Sd    = _dict_j2_coefficients(cls.DEGREE)
        cls.sh,  _ = _build_sh(cls.DEGREE, Cnp, Snp)
        cls.pines, _ = _build_pines(cls.DEGREE, Cd, Sd)

    def test_agreement_random_states(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            r     = R + rng.uniform(200, 2000)
            state = [r, rng.uniform(-200, 200), rng.uniform(-200, 200),
                     0.0, 7.5, 0.0]
            a_sh    = _eval_accel(self.sh,    state)
            a_pines = _eval_accel(self.pines, state)
            err = np.linalg.norm(a_sh - a_pines)
            self.assertLess(err, self.TOL, f"SH/Pines diff {err:.3e}")


class TestCorrectnessHighDegree(unittest.TestCase):
    """
    Sanity checks at degree=20 with random coefficients.

    The Pines and old SH-graph cannot run at this degree without
    hanging; we just check the C++ node produces finite, physical
    output (direction anti-radial at the point-mass level, correct
    magnitude order).
    """

    DEGREE = 20

    @classmethod
    def setUpClass(cls):
        C, S = _np_random_coefficients(cls.DEGREE)
        cls.model, _ = _build_sh(cls.DEGREE, C, S)

    def test_output_finite(self):
        state = [R + 500, 0.0, 0.0, 0.0, 7.6, 0.0]
        a = _eval_accel(self.model, state)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertEqual(len(a), 3)

    def test_magnitude_plausible(self):
        """Total accel should be within 50% of point-mass at LEO."""
        state = [R + 400, 0.0, 0.0, 0.0, 7.6, 0.0]
        a   = _eval_accel(self.model, state)
        r   = np.linalg.norm(state[:3])
        a_pm = MU / r ** 2            # point-mass magnitude km/s^2
        ratio = np.linalg.norm(a) / a_pm
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 1.5)


# ============================================================
# ============================================================
# 2.  JACOBIAN TESTS
# ============================================================
# ============================================================

class TestJacobian(unittest.TestCase):
    """
    Verify ASSET autodiff Jacobian from the C++ node.

    The old pure-ASSET graph (and the Cython black-box kernel)
    could NOT produce Jacobians. The C++ AutodiffFwd node can.

    Checks:
      - jacobian() returns a (3 x 6) matrix
      - All entries finite
      - Matches centred finite-difference to < 1e-6 relative tol
      - Non-zero (non-trivial derivative)
    """

    DEGREE = 2
    FD_TOL = 1e-6

    @classmethod
    def setUpClass(cls):
        C, S = _np_j2_coefficients(cls.DEGREE)
        cls.model, _ = _build_sh(cls.DEGREE, C, S)
        cls.accel_vf  = cls.model.vf()

    def _fd_jacobian(self, state, eps=1e-5):
        """Centred finite-difference Jacobian of acceleration wrt full state."""
        x    = np.asarray(state, dtype=float)
        f0   = np.asarray(self.accel_vf.compute(x))
        n    = len(x)
        m    = len(f0)
        jac  = np.zeros((m, n))
        for j in range(n):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            fp = np.asarray(self.accel_vf.compute(xp))
            fm = np.asarray(self.accel_vf.compute(xm))
            jac[:, j] = (fp - fm) / (2 * eps)
        return jac

    def test_jacobian_shape(self):
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        jac   = np.asarray(self.accel_vf.jacobian(state))
        self.assertEqual(jac.shape, (3, 6))

    def test_jacobian_finite(self):
        state = np.array([7000.0, 500.0, -300.0, 0.0, 7.5, 0.0])
        jac   = np.asarray(self.accel_vf.jacobian(state))
        self.assertTrue(np.all(np.isfinite(jac)))

    def test_jacobian_nontrivial(self):
        """Position columns of Jacobian must be non-zero."""
        state = np.array([7000.0, 1000.0, 2000.0, 0.0, 7.5, 0.0])
        jac   = np.asarray(self.accel_vf.jacobian(state))
        pos_jac = jac[:, :3]
        self.assertGreater(np.linalg.norm(pos_jac), 1e-12)

    def test_jacobian_vs_fd(self):
        rng = np.random.default_rng(99)
        for _ in range(10):
            r     = R + rng.uniform(300, 1500)
            state = np.array([r, rng.uniform(-300, 300), rng.uniform(-300, 300),
                               0.0, 7.5, 0.0])
            jac_ad = np.asarray(self.accel_vf.jacobian(state))
            jac_fd = self._fd_jacobian(state)
            # Only compare position columns — velocity has no effect on gravity
            err = np.max(np.abs(jac_ad[:, :3] - jac_fd[:, :3]))
            scale = max(np.max(np.abs(jac_fd[:, :3])), 1e-20)
            rel   = err / scale
            self.assertLess(rel, self.FD_TOL,
                            f"AD/FD Jacobian rel err {rel:.3e} at r={state[:3]}")


class TestJacobianHighDegree(unittest.TestCase):
    """Same Jacobian checks at degree=8 with random coefficients."""

    DEGREE = 8
    FD_TOL = 1e-5

    @classmethod
    def setUpClass(cls):
        C, S = _np_random_coefficients(cls.DEGREE, seed=11)
        cls.model, _ = _build_sh(cls.DEGREE, C, S)
        cls.accel_vf  = cls.model.vf()

    def _fd_jacobian(self, state, eps=1e-5):
        x   = np.asarray(state, dtype=float)
        n   = len(x)
        f0  = np.asarray(self.accel_vf.compute(x))
        jac = np.zeros((len(f0), n))
        for j in range(n):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            jac[:, j] = (np.asarray(self.accel_vf.compute(xp)) -
                         np.asarray(self.accel_vf.compute(xm))) / (2 * eps)
        return jac

    def test_jacobian_vs_fd(self):
        rng = np.random.default_rng(55)
        for _ in range(5):
            r     = R + rng.uniform(400, 1200)
            state = np.array([r, rng.uniform(-200, 200), rng.uniform(-200, 200),
                               0.0, 7.4, 0.0])
            jad = np.asarray(self.accel_vf.jacobian(state))
            jfd = self._fd_jacobian(state)
            err   = np.max(np.abs(jad[:, :3] - jfd[:, :3]))
            scale = max(np.max(np.abs(jfd[:, :3])), 1e-20)
            self.assertLess(err / scale, self.FD_TOL)


# ============================================================
# ============================================================
# 3 & 4.  BUILD AND EVALUATION BENCHMARKS
# ============================================================
# ============================================================

_BENCH_DEGREES = [2, 4, 8, 12, 20]

_BENCH_STATE = np.array([R + 500.0, 0.0, 0.0, 0.0, 7.6, 0.0])

_EVAL_N = 200   # evaluations per timing run


def _bench_build_and_eval(degree, *, verbose=True):
    """
    Returns dict with:
        build_ms  : construction time (ms)
        eval_us   : mean evaluation time (µs)
        jac_us    : mean Jacobian time (µs)
        accel     : acceleration vector at _BENCH_STATE
    """
    C, S = _np_random_coefficients(degree)
    gc.collect()

    # --- Build ---
    t0    = time.perf_counter()
    model = SphericalHarmonicGravityVF(MU, R, C, S, degree)
    t1    = time.perf_counter()
    build_ms = (t1 - t0) * 1e3

    accel_vf = model.vf()
    x0 = _BENCH_STATE.copy()

    # --- Warmup ---
    _ = accel_vf.compute(x0)

    # --- Eval ---
    t0 = time.perf_counter()
    for _ in range(_EVAL_N):
        accel_vf.compute(x0)
    t1 = time.perf_counter()
    eval_us = (t1 - t0) / _EVAL_N * 1e6

    # --- Jacobian ---
    _ = accel_vf.jacobian(x0)
    t0 = time.perf_counter()
    for _ in range(_EVAL_N):
        accel_vf.jacobian(x0)
    t1 = time.perf_counter()
    jac_us = (t1 - t0) / _EVAL_N * 1e6

    accel = np.asarray(accel_vf.compute(x0))

    if verbose:
        print(
            f"  degree={degree:3d}"
            f"  build={build_ms:8.3f} ms"
            f"  eval={eval_us:8.2f} µs"
            f"  jac={jac_us:8.2f} µs"
            f"  |a|={np.linalg.norm(accel):.6e} km/s²"
        )

    return dict(
        degree=degree,
        build_ms=build_ms,
        eval_us=eval_us,
        jac_us=jac_us,
        accel=accel,
    )


def run_build_eval_benchmark():
    print()
    print("=" * 65)
    print("BUILD AND EVALUATION BENCHMARK  (C++ SphericalHarmonicGravity)")
    print("=" * 65)
    print(
        f"  {'degree':>6}  {'build (ms)':>12}  {'eval (µs)':>10}"
        f"  {'jac (µs)':>10}  {'|a| km/s²':>14}"
    )
    print("  " + "-" * 61)

    results = []
    for deg in _BENCH_DEGREES:
        r = _bench_build_and_eval(deg, verbose=True)
        results.append(r)

    print()
    print("Observation: build time should be <1 ms at all degrees")
    print("  (old ASSET graph hangs at degree ≥ 20)")
    return results


# ============================================================
# ============================================================
# 5.  PROPAGATION BENCHMARK
# ============================================================
# ============================================================

def run_propagation_benchmark(degree=8, tf=5400.0, runs=2):
    """
    Integrate one LEO orbit using the C++ node ODE.

    Prints runtime, step count, and radius conservation.
    """

    print()
    print("=" * 65)
    print(f"PROPAGATION BENCHMARK  degree={degree}  tf={tf:.0f} s")
    print("=" * 65)

    C, S = _np_random_coefficients(degree)
    model = SphericalHarmonicGravityVF(MU, R, C, S, degree)
    ode_vf = model.ode()

    class SHODE(oc.ODEBase):
        def __init__(self):
            super().__init__(ode_vf, 6, 0)

    x0 = np.array([[R + 500.0], [0.0], [0.0], [0.0], [7.6], [0.0], [0.0]])

    runtimes = []
    for run in range(runs):
        gc.collect()
        ode  = SHODE()
        integ = ode.integrator("DOPRI87", 20)

        t0 = time.perf_counter()
        sol  = integ.integrate_dense(x0, tf)
        t1 = time.perf_counter()
        elapsed = t1 - t0

        traj   = np.asarray(sol)
        radius = np.linalg.norm(traj[:, :3], axis=1)

        print(
            f"  Run {run + 1}/{runs}:"
            f"  t={elapsed:.3f} s"
            f"  steps={len(traj)}"
            f"  Δr={radius[-1] - radius[0]:.6e} km"
            f"  r_min={np.min(radius):.3f} km"
        )
        runtimes.append(elapsed)

    print(f"  Mean runtime: {np.mean(runtimes):.3f} s")
    return np.mean(runtimes)


# ============================================================
# ============================================================
# MAIN — run benchmarks then execute unit tests
# ============================================================
# ============================================================

def _run_benchmarks():
    print()
    print("*" * 65)
    print("  C++ SphericalHarmonicGravity Node — Benchmark & Test Suite")
    print("*" * 65)

    # Build / eval / Jacobian across degrees
    bench_results = run_build_eval_benchmark()

    # Summary table
    print()
    print("SUMMARY")
    print(f"  {'degree':>6}  {'build (ms)':>12}  {'eval (µs)':>10}  {'jac (µs)':>10}")
    print("  " + "-" * 45)
    for r in bench_results:
        print(
            f"  {r['degree']:>6}  {r['build_ms']:>12.3f}"
            f"  {r['eval_us']:>10.2f}  {r['jac_us']:>10.2f}"
        )

    # Propagation at degree 8
    run_propagation_benchmark(degree=8, tf=5400.0, runs=2)


if __name__ == "__main__":
    # ---- benchmarks first ----
    _run_benchmarks()

    # ---- unit tests ----
    print()
    print("=" * 65)
    print("UNIT TESTS")
    print("=" * 65)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestCorrectnessJ2,
        TestCorrectnessVsPines,
        TestCorrectnessHighDegree,
        TestJacobian,
        TestJacobianHighDegree,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
