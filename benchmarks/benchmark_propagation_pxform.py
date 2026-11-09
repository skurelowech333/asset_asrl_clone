"""
Propagation benchmark with coordinate transforms inside the dynamics loop.

Both propagators integrate the same problem: a spacecraft orbiting the Moon
with point-mass lunar gravity. Gravity is evaluated in the Moon body-fixed
frame, and the acceleration is rotated back to the inertial frame on every
evaluation. This is the same pattern as the GRAIL case:

    r_pa = R(t) r_in
    a_pa = -mu r_pa / |r_pa|^3
    a_in = R(t)^T a_pa

ASSET: fused SpiceForceFrame wrapping PinesGravity at degree 0 (point-mass lunar gravity, one SPICE sxform call per RHS evaluation), composed into an ODEBase
graph and integrated with DOPRI54. Frames: J2000 -> IAU_MOON (SPICE, NAIF
pck00010.tpc).

Orekit: NumericalPropagator (DormandPrince54) with a Python ForceModel doing
the same transforms. Frames: Moon-centered inertially oriented frame ->
MoonIAU (Orekit IAU 2009 model).

Reports timing for both sides and an agreement check on the final state.
Known frame and model differences are listed in OREKIT_MOON_AGREEMENT.md.

Requires OREKIT_DATA (default /root/orekit-data) and SPICE_PCK (default
/root/spice_kernels/pck00010.tpc). Run with the ASSET dev environment on
PYTHONPATH.

Usage:
    python benchmark_propagation_pxform.py [durations_hours_csv] [repeats]

    e.g. python benchmark_propagation_pxform.py 10,100,1000 3
    Repeats are used for durations up to 100 h; longer runs use one repeat.
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
from org.orekit.forces import ForceModel  # noqa: E402
from org.orekit.orbits import CartesianOrbit, OrbitType  # noqa: E402
from org.orekit.propagation import SpacecraftState  # noqa: E402
from org.orekit.propagation.numerical import NumericalPropagator  # noqa: E402
from org.orekit.time import AbsoluteDate  # noqa: E402
from org.orekit.utils import PVCoordinates  # noqa: E402

import asset_asrl as ast  # noqa: E402
import asset_asrl.Extensions as ext  # noqa: E402

vf = ast.VectorFunctions
oc = ast.OptimalControl

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


class MoonPointMassBodyFrame(oc.ODEBase):
    """
    State: [x, y, z, vx, vy, vz] in J2000 (km, km/s), time ET (TVar) in s.
    """

    def __init__(self, mu):
        XtU = oc.ODEArguments(6, 0)
        x, y, z, vx, vy, vz = XtU.XVec().tolist()
        t = XtU.TVar()

        pines0 = ext.PinesGravity(os.path.abspath(EGM96_PATH), 0, mu, MOON_RADIUS_KM,
                                  "fully_normalized")
        grav = ext.SpiceForceFrame(pines0, "J2000", "IAU_MOON")
        a_in = grav([t, x, y, z])
        ode = vf.stack([vx, vy, vz,
                        a_in[0], a_in[1], a_in[2]])
        super().__init__(ode, 6, 0)


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
        date = state.getDate()
        to_body = self.inertial.getTransformTo(self.body, date)
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
    """Propagate one duration on both sides and return timing and agreement."""
    gm_m, mu_km, inertial, body, r0_km, v0_km, t0, ode_mu = ctx
    tf = t0 + duration_hours * 3600.0

    # ----- ASSET ---------------------------------------------------------
    ode = MoonPointMassBodyFrame(ode_mu)
    integ = ode.integrator("DOPRI54", 1.0)
    integ.setAbsTol(1.0e-9)  # km; matches Orekit 1e-6 m
    integ.setRelTol(1.0e-12)  # matches Orekit rel_tol
    integ.setStepSizes(1.0, 1.0e-3, 600.0)  # matches Orekit min/max step
    X0 = [*r0_km, *v0_km, t0]

    def run_asset():
        return integ.integrate(X0, tf)  # final state only, like Orekit propagate()

    asset_t, asset_final = time_best(run_asset, repeats)
    asset_r = np.array(asset_final[0:3]) * 1000.0      # m
    asset_v = np.array(asset_final[3:6]) * 1000.0      # m/s

    # ----- Orekit --------------------------------------------------------
    date0 = AbsoluteDate.J2000_EPOCH.shiftedBy(t0)
    date1 = AbsoluteDate.J2000_EPOCH.shiftedBy(tf)
    pv0 = PVCoordinates(Vector3D(*(r0_km * 1000.0)),
                        Vector3D(*(v0_km * 1000.0)))
    # Orekit adds Newtonian attraction using the orbit's mu. Setting it to a
    # negligible value (1e-3 m^3/s^2 vs 4.9e12) makes the body-frame point-mass
    # force the only gravity term on the Orekit side, matching ASSET.
    # (setIgnoreCentralAttraction(True) freezes position in Orekit 13.1.)
    orbit0 = CartesianOrbit(pv0, inertial, date0, 1.0e-3)

    # Orekit state is [pos (m), vel (m/s), mass (kg)].
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
    orekit_r = np.array([pv1.getPosition().getX(), pv1.getPosition().getY(),
                         pv1.getPosition().getZ()])
    orekit_v = np.array([pv1.getVelocity().getX(), pv1.getVelocity().getY(),
                         pv1.getVelocity().getZ()])

    dr = np.linalg.norm(asset_r - orekit_r)
    dv = np.linalg.norm(asset_v - orekit_v)
    return {"duration_h": duration_hours, "asset_s": asset_t,
            "orekit_s": orekit_t, "dr_m": dr, "dv_mps": dv,
            "asset_r": asset_r, "orekit_r": orekit_r}


def main(durations_hours=(10.0, 100.0, 1000.0), repeats=3):
    data_dir = os.environ.get("OREKIT_DATA", "/root/orekit-data")
    pck = os.environ.get("SPICE_PCK", "/root/spice_kernels/pck00010.tpc")
    DataContext.getDefault().getDataProvidersManager().addProvider(
        DirectoryCrawler(File(data_dir)))
    ext.SpiceLoadKernels([pck])

    moon = CelestialBodyFactory.getMoon()
    gm_m = float(moon.getGM())          # m^3/s^2, shared by both sides
    mu_km = gm_m / 1.0e9                # km^3/s^2

    inertial = moon.getInertiallyOrientedFrame()
    body = moon.getBodyOrientedFrame()

    # Initial state, identical numbers on both sides (km, km/s).
    r0_km = np.array([1837.4, 0.0, 0.0])
    v_circ = np.sqrt(mu_km / np.linalg.norm(r0_km))
    v0_km = 1.02 * v_circ * np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    t0 = 3.0e8                          # ET seconds (TDB)
    ctx = (gm_m, mu_km, inertial, body, r0_km, v0_km, t0, mu_km)

    rows = []
    for dur in durations_hours:
        # Long runs are slow enough that repeating them is not worth the time.
        reps = repeats if dur <= 100.0 else 1
        print(f"Running {dur:.1f} h ({reps} repeat(s))...", flush=True)
        rows.append(run_duration(dur, reps, ctx))

    print(f"\nMoon point-mass propagation, DOPRI54 on both sides, mu(Moon) = {gm_m:.6e} m^3/s^2")
    print(f"{'Duration':>10} {'ASSET ms':>12} {'Orekit ms':>12} {'Orekit/ASSET':>13} "
          f"{'|dr| m':>12} {'|dv| m/s':>12}")
    for r in rows:
        print(f"{r['duration_h']:>8.1f} h {r['asset_s'] * 1e3:>12.2f} "
              f"{r['orekit_s'] * 1e3:>12.2f} {r['orekit_s'] / r['asset_s']:>12.2f}x "
              f"{r['dr_m']:>12.4e} {r['dv_mps']:>12.4e}")
    return rows


if __name__ == "__main__":
    durs = (tuple(float(x) for x in sys.argv[1].split(","))
            if len(sys.argv) > 1 else (10.0, 100.0, 1000.0))
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(durs, rep)
