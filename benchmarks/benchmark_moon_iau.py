"""
Moon body-fixed (IAU_MOON) benchmark: ASSET vs Orekit, in two parts.

Part 1, direct coordinate transforms:
    ASSET SpicePxformApply (J2000 -> IAU_MOON, SPICE pck00010.tpc) vs Orekit
    EME2000 -> MoonIAU transformVector (IAU 2009 model). Reports per-call time
    and agreement, with the residual frame offset fitted as a small rotation.

Part 2, propagation with transforms in the dynamics loop:
    A spacecraft orbits the Moon with point-mass gravity evaluated in IAU_MOON:
        r_pa = R(t) r_in,   a_pa = -mu r_pa / |r_pa|^3,   a_in = R(t)^T a_pa
    ASSET: SpiceForceFrame(PinesGravity degree 0, J2000, IAU_MOON) inside an
    ODEBase, integrated with DOPRI54. Orekit: NumericalPropagator with
    DormandPrince54 and a Python ForceModel doing the same transforms.

Both parts use the same frames and tolerances. Known frame and model
differences are listed in OREKIT_MOON_AGREEMENT.md.

Requires OREKIT_DATA (default /root/orekit-data) and SPICE_PCK (default
/root/spice_kernels/pck00010.tpc). Run with the ASSET dev environment on
PYTHONPATH.

Spyder: press F5. Edit the constants below, or pass CLI args:
    python benchmark_moon_iau.py [n_samples] [repeats] [durations_hours_csv]
"""

import os
import sys
import time

import numpy as np
import orekit_jpype

orekit_jpype.initVM()

from jpype import JArray, JDouble, JImplements, JOverride  # noqa: E402
from java.io import File  # noqa: E402
from java.util import ArrayList  # noqa: E402
from java.util.stream import Stream  # noqa: E402
from org.hipparchus.geometry.euclidean.threed import Vector3D  # noqa: E402
from org.hipparchus.ode.nonstiff import DormandPrince54Integrator  # noqa: E402
from org.orekit.bodies import CelestialBodyFactory  # noqa: E402
from org.orekit.data import DataContext, DirectoryCrawler  # noqa: E402
from org.orekit.forces import ForceModel  # noqa: E402,F401
from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import CartesianOrbit, OrbitType  # noqa: E402
from org.orekit.propagation import SpacecraftState  # noqa: E402
from org.orekit.propagation.numerical import NumericalPropagator  # noqa: E402
from org.orekit.time import AbsoluteDate  # noqa: E402
from org.orekit.utils import PVCoordinates  # noqa: E402

import asset_asrl as ast  # noqa: E402
import asset_asrl.Extensions as ext  # noqa: E402

vf = ast.VectorFunctions
oc = ast.OptimalControl

# Defaults; CLI args override them.
N_SAMPLES = 20000          # transform calls for part 1
REPEATS = 3                # best-of timing repeats
DURATIONS_H = (10.0, 100.0, 1000.0)  # part 2; repeats only for durations <= 100 h

MOON_RADIUS_KM = 1738.0
EGM96_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "native", "data", "egm96.nor")


def time_best(fn, repeats):
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


# ----------------------------------------------------------------------------
# Part 1: direct coordinate transforms
# ----------------------------------------------------------------------------

def transform_benchmark(n, repeats):
    rng = np.random.default_rng(0)
    ets = rng.uniform(0.0, 3.0e8, size=n)
    rs = rng.uniform(-2.0e6, 2.0e6, size=(n, 3))
    xs = np.column_stack([ets, rs])

    asset_vf = ext.SpicePxformApply("J2000", "IAU_MOON")
    eme = FramesFactory.getEME2000()
    moon = CelestialBodyFactory.getMoon().getBodyOrientedFrame()
    epoch = AbsoluteDate.J2000_EPOCH

    # Agreement on a subset (Orekit calls are slow relative to ASSET).
    m = min(n, 200)
    asset_out = np.array([np.array(asset_vf.compute(x)) for x in xs[:m]])
    orekit_out = np.empty((m, 3))
    for i in range(m):
        tr = eme.getTransformTo(moon, epoch.shiftedBy(float(ets[i])))
        v = tr.transformVector(Vector3D(*map(float, rs[i])))
        orekit_out[i] = [v.getX(), v.getY(), v.getZ()]

    diff = asset_out - orekit_out
    # Fit a rotation vector w with diff_i ~= w x x_i (small-angle frame error).
    A = np.zeros((3 * m, 3))
    for i, x in enumerate(asset_out):
        A[3 * i:3 * i + 3] = -np.array([[0, -x[2], x[1]],
                                        [x[2], 0, -x[0]],
                                        [-x[1], x[0], 0]])
    w, *_ = np.linalg.lstsq(A, diff.reshape(-1), rcond=None)
    fit = (A @ w).reshape(m, 3)
    resid = np.linalg.norm(diff - fit, axis=1)
    norm = np.linalg.norm(diff, axis=1)
    w_mas = np.linalg.norm(w) * 206264806.0

    print("Part 1: coordinate transforms, J2000 -> IAU_MOON")
    print(f"  agreement over {m} samples (ASSET vs Orekit MoonIAU):")
    print(f"    |diff| max / median     : {norm.max():.4f} / {np.median(norm):.4f} m")
    print(f"    fitted frame rotation   : {w_mas:.2f} mas, "
          f"axis {np.round(w / np.linalg.norm(w), 4)}")
    print(f"    residual after rotation : max {resid.max():.2e} m, "
          f"median {np.median(resid):.2e} m")

    def asset_run():
        for x in xs:
            asset_vf.compute(x)

    def orekit_run():
        for et, r in zip(ets, rs):
            tr = eme.getTransformTo(moon, epoch.shiftedBy(float(et)))
            tr.transformVector(Vector3D(float(r[0]), float(r[1]), float(r[2])))

    t_asset, _ = time_best(asset_run, repeats)
    t_ore, _ = time_best(orekit_run, repeats)
    print(f"  timing, {n} calls, best of {repeats}:")
    print(f"    ASSET SpicePxformApply.compute : {t_asset / n * 1e6:8.2f} us/call")
    print(f"    Orekit transformVector         : {t_ore / n * 1e6:8.2f} us/call")
    print(f"    speedup (Orekit / ASSET)       : {t_ore / t_asset:.1f}x")


# ----------------------------------------------------------------------------
# Part 2: propagation with transforms in the loop
# ----------------------------------------------------------------------------

class MoonPointMassBodyFrame(oc.ODEBase):
    """State [x, y, z, vx, vy, vz] in J2000 (km, km/s); TVar is ET (s)."""

    def __init__(self, mu):
        XtU = oc.ODEArguments(6, 0)
        x, y, z, vx, vy, vz = XtU.XVec().tolist()
        t = XtU.TVar()
        pines0 = ext.PinesGravity(os.path.abspath(EGM96_PATH), 0, mu,
                                  MOON_RADIUS_KM, "fully_normalized")
        grav = ext.SpiceForceFrame(pines0, "J2000", "IAU_MOON")
        a_in = grav([t, x, y, z])
        super().__init__(vf.stack([vx, vy, vz, a_in[0], a_in[1], a_in[2]]), 6, 0)


@JImplements("org.orekit.forces.ForceModel")
class MoonPointMassBodyFrameForce:
    """Point-mass Moon gravity evaluated in MoonIAU, acceleration rotated back."""

    def __init__(self, mu_m, inertial, body):
        self.mu = mu_m
        self.inertial = inertial
        self.body = body

    @JOverride
    def acceleration(self, state, params):
        if not isinstance(state, SpacecraftState):
            raise NotImplementedError("field propagation not used here")
        to_body = self.inertial.getTransformTo(self.body, state.getDate())
        r_body = to_body.transformPosition(state.getPosition())
        a_body = r_body.scalarMultiply(-self.mu / r_body.getNormSq() ** 1.5)
        return to_body.getInverse().transformVector(a_body)

    @JOverride
    def dependsOnPositionOnly(self):
        return True

    @JOverride
    def getParametersDrivers(self):
        return ArrayList()

    @JOverride
    def init(self, *args):
        return None

    @JOverride
    def addContribution(self, state, eqns):
        eqns.addNonKeplerianAcceleration(self.acceleration(state, None))

    @JOverride
    def getMassDerivative(self, *args):
        return 0.0

    @JOverride
    def dependsOnAttitudeRate(self):
        return False

    @JOverride
    def getEventDetectors(self):
        return Stream.empty()


def run_duration(duration_hours, repeats, ctx):
    """Propagate one duration on both sides; return timing and final-state agreement."""
    gm_m, mu_km, inertial, body, r0_km, v0_km, t0 = ctx
    tf = t0 + duration_hours * 3600.0

    # ASSET
    ode = MoonPointMassBodyFrame(mu_km)
    integ = ode.integrator("DOPRI54", 1.0)
    integ.setAbsTol(1.0e-9)  # km; matches Orekit 1e-6 m
    integ.setRelTol(1.0e-12)
    integ.setStepSizes(1.0, 1.0e-3, 600.0)  # matches Orekit min/max step
    X0 = [*r0_km, *v0_km, t0]
    asset_t, asset_final = time_best(lambda: integ.integrate(X0, tf), repeats)
    asset_r = np.array(asset_final[0:3]) * 1000.0  # m
    asset_v = np.array(asset_final[3:6]) * 1000.0  # m/s

    # Orekit. The orbit mu is set to a negligible value so the body-frame
    # force is the only gravity term (setIgnoreCentralAttraction freezes
    # position in Orekit 13.1).
    date0 = AbsoluteDate.J2000_EPOCH.shiftedBy(t0)
    date1 = AbsoluteDate.J2000_EPOCH.shiftedBy(tf)
    pv0 = PVCoordinates(Vector3D(*(r0_km * 1000.0)), Vector3D(*(v0_km * 1000.0)))
    orbit0 = CartesianOrbit(pv0, inertial, date0, 1.0e-3)
    abs_tol = JArray(JDouble)([1.0e-6, 1.0e-6, 1.0e-6,
                               1.0e-9, 1.0e-9, 1.0e-9, 1.0e-6])
    rel_tol = JArray(JDouble)([1.0e-12] * 7)
    force = MoonPointMassBodyFrameForce(gm_m, inertial, body)

    def run_orekit():
        integrator = DormandPrince54Integrator(1.0e-3, 600.0, abs_tol, rel_tol)
        prop = NumericalPropagator(integrator)
        prop.setOrbitType(OrbitType.CARTESIAN)
        prop.addForceModel(force)
        prop.setInitialState(SpacecraftState(orbit0))
        return prop.propagate(date1)

    orekit_t, orekit_state = time_best(run_orekit, repeats)
    pv1 = orekit_state.getPVCoordinates(inertial)
    p, v = pv1.getPosition(), pv1.getVelocity()
    orekit_r = np.array([p.getX(), p.getY(), p.getZ()])
    orekit_v = np.array([v.getX(), v.getY(), v.getZ()])

    return {"duration_h": duration_hours, "asset_s": asset_t, "orekit_s": orekit_t,
            "dr_m": np.linalg.norm(asset_r - orekit_r),
            "dv_mps": np.linalg.norm(asset_v - orekit_v)}


def propagation_benchmark(durations_hours, repeats):
    moon = CelestialBodyFactory.getMoon()
    gm_m = float(moon.getGM())  # m^3/s^2, shared by both sides
    mu_km = gm_m / 1.0e9        # km^3/s^2
    inertial = moon.getInertiallyOrientedFrame()
    body = moon.getBodyOrientedFrame()

    r0_km = np.array([1837.4, 0.0, 0.0])
    v_circ = np.sqrt(mu_km / np.linalg.norm(r0_km))
    v0_km = 1.02 * v_circ * np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    t0 = 3.0e8  # ET seconds (TDB)
    ctx = (gm_m, mu_km, inertial, body, r0_km, v0_km, t0)

    rows = []
    for dur in durations_hours:
        reps = repeats if dur <= 100.0 else 1
        print(f"  running {dur:.1f} h ({reps} repeat(s))...", flush=True)
        rows.append(run_duration(dur, reps, ctx))

    print(f"\nPart 2: propagation, DOPRI54 both sides, mu(Moon) = {gm_m:.6e} m^3/s^2")
    print(f"  {'Duration':>10} {'ASSET ms':>12} {'Orekit ms':>12} "
          f"{'Orekit/ASSET':>13} {'|dr| m':>12} {'|dv| m/s':>12}")
    for r in rows:
        print(f"  {r['duration_h']:>8.1f} h {r['asset_s'] * 1e3:>12.2f} "
              f"{r['orekit_s'] * 1e3:>12.2f} {r['orekit_s'] / r['asset_s']:>12.2f}x "
              f"{r['dr_m']:>12.4e} {r['dv_mps']:>12.4e}")
    return rows


def main(n=N_SAMPLES, repeats=REPEATS, durations_hours=DURATIONS_H):
    data_dir = os.environ.get("OREKIT_DATA", "/root/orekit-data")
    pck = os.environ.get("SPICE_PCK", "/root/spice_kernels/pck00010.tpc")
    DataContext.getDefault().getDataProvidersManager().addProvider(
        DirectoryCrawler(File(data_dir)))
    ext.SpiceLoadKernels([pck])

    transform_benchmark(n, repeats)
    print()
    propagation_benchmark(durations_hours, repeats)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPEATS
    durs = (tuple(float(x) for x in sys.argv[3].split(","))
            if len(sys.argv) > 3 else DURATIONS_H)
    main(n, reps, durs)
