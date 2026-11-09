"""
Pines gravity in a SPICE body-fixed frame, as ASSET graph nodes.

Shows two ways to put a force kernel in a SPICE body frame:

  * composed_accel(pines, from, to): graph composition of native
    SpicePxformApply nodes around PinesGravity.
  * SpiceForceFrame(pines, from, to): one fused node with the same result,
    about 3x faster per call in benchmark_transform_isolation.py.

It checks both against spiceypy (a_J2000 = R^T PinesGravity(R r)), then
propagates a satellite with the fused node using ODEBase and DOPRI54.

Usage:
    python SpicePinesFrameExample.py kernel1 [kernel2 ...]

Kernels must give the SPICE frame IAU_EARTH (for example an Earth binary PCK
and a leapseconds kernel). The EGM96 coefficients are read from
native/data/egm96.nor.
"""

import os
import sys

import numpy as np
import spiceypy as spice

import asset_asrl as ast
import asset_asrl.Extensions as ext
from asset_asrl.Astro import SpiceForceFrame

vf = ast.VectorFunctions
oc = ast.OptimalControl

FROM_FRAME = "J2000"
BODY_FRAME = "IAU_EARTH"
MU_EARTH = 398600.4418        # km^3/s^2
R_EARTH = 6378.1363           # km
DEGREE = 8
EGM96_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "native", "data", "egm96.nor")
# SPICE kernels for the Earth body-fixed frame (IAU_EARTH). Edit this list when
# running from Spyder, or set SPICE_KERNELS to a space-separated list.
KERNEL_FILES = os.environ.get("SPICE_KERNELS", "").split()


def composed_accel(pines, from_frame, to_frame):
    """Same J2000 acceleration built by graph composition, not one fused node."""
    args = vf.Arguments(4)
    xs = [args[i] for i in range(4)]
    to_body = ext.SpicePxformApply(from_frame, to_frame)
    a_body = pines(to_body(xs))
    to_inertial = to_body.inverse()
    return to_inertial([xs[0], a_body[0], a_body[1], a_body[2]])


class EarthPinesBodyFrame(oc.ODEBase):
    """State [x, y, z, vx, vy, vz] (km, km/s) in J2000; TVar is ET (s)."""

    def __init__(self, pines):
        XtU = oc.ODEArguments(6, 0)
        x, y, z, vx, vy, vz = XtU.XVec().tolist()
        t = XtU.TVar()
        accel = SpiceForceFrame(pines, FROM_FRAME, BODY_FRAME)
        a = accel([t, x, y, z])
        super().__init__(vf.stack([vx, vy, vz, a[0], a[1], a[2]]), 6, 0)


def main(kernel_files):
    ext.SpiceLoadKernels(list(kernel_files))

    pines = ext.PinesGravity(os.path.abspath(EGM96_PATH), DEGREE,
                             MU_EARTH, R_EARTH, "fully_normalized")

    # Two equivalent ways to build the same J2000 acceleration.
    composed = composed_accel(pines, FROM_FRAME, BODY_FRAME)
    fused = SpiceForceFrame(pines, FROM_FRAME, BODY_FRAME)

    et = 1.0e6
    r = np.array([6.9e3, -4.1e3, 2.3e3])  # km, J2000

    # 1. Both match the spiceypy reference.
    R = np.array(spice.pxform(FROM_FRAME, BODY_FRAME, et))
    a_ref = R.T @ np.array(pines.compute(R @ r))
    x = np.concatenate([[et], r])
    a_composed = np.array(composed.compute(x))
    a_fused = np.array(fused.compute(x))
    assert np.allclose(a_composed, a_ref, rtol=1e-12, atol=1e-18), "composed mismatch"
    assert np.allclose(a_fused, a_ref, rtol=1e-12, atol=1e-18), "fused mismatch"

    # 2. The fused Jacobian matches the composed one (both include d/dET).
    J_fused = np.array(fused.jacobian(x))
    J_composed = np.array(composed.jacobian(x))
    assert np.allclose(J_fused, J_composed, rtol=1e-8, atol=1e-18), "Jacobian mismatch"

    print("composed_accel and SpiceForceFrame match spiceypy reference")
    print(f"  a_J2000 (km/s^2)        = {a_fused}")
    print(f"  |a_fused - a_composed|  = {np.linalg.norm(a_fused - a_composed):.3e}")

    # 3. Propagate with the fused node.
    ode = EarthPinesBodyFrame(pines)
    integ = ode.integrator("DOPRI54", 1.0)
    integ.setAbsTol(1.0e-9)
    integ.setRelTol(1.0e-12)
    integ.setStepSizes(1.0, 1.0e-3, 600.0)

    v_circ = np.sqrt(MU_EARTH / np.linalg.norm(r))
    v0 = 1.02 * v_circ * np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    X0 = [*r, *v0, et]
    final = integ.integrate(X0, et + 3.0 * 3600.0)
    print("Propagated 3 h with DOPRI54 + SpiceForceFrame:")
    print(f"  final r (km) = {np.array(final[0:3])}")
    print(f"  final v (km/s) = {np.array(final[3:6])}")


if __name__ == "__main__":
    kernels = sys.argv[1:] or KERNEL_FILES
    if not kernels:
        print(__doc__)
        print("No SPICE kernels given. Pass them on the command line, set SPICE_KERNELS, "
              "or edit KERNEL_FILES at the top of this file.")
    else:
        main(kernels)
