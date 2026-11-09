#!/usr/bin/env python3
"""Isolate SPICE frame-transform cost from gravity cost (Earth, Pines degree 8).

Times, per call, via VectorFunction.compute:
  1. SpicePxformApply alone (ET, r -> body frame)
  2. PinesGravity alone (body-frame position -> acceleration)
  3. Composed graph: to_inertial(pines(to_body(et, r)))
  4. Fused node: SpiceForceFrame(pines, from, to)(et, r)

Run from the repo root with the staged asset module on PYTHONPATH.
"""
import os
import tempfile
import time

import numpy as np
import asset_asrl as ast
import asset.Extensions as ext

from asset_asrl.Astro import SpiceForceFrame

vf = ast.VectorFunctions

FROM_FRAME = "J2000"
EARTH_TO_FRAME = "IAU_EARTH"
MU_EARTH = 398600.4418
R_EARTH = 6378.1363
DEGREE = 8
N_CALLS = 20000
REPEATS = 5
EGM96 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native", "data", "egm96.nor")
TEST_EARTH_PCK = """KPL/PCK

\\begindata
BODY399_POLE_RA  = ( 0.0   0.0   0.0 )
BODY399_POLE_DEC = ( 90.0  0.0   0.0 )
BODY399_PM       = ( 190.147  360.9856235  0.0 )
\\begintext
"""


def composed_accel(pines, from_frame, to_frame):
    """Same J2000 acceleration built by graph composition, not one fused node."""
    args = vf.Arguments(4)
    xs = [args[i] for i in range(4)]
    to_body = ext.SpicePxformApply(from_frame, to_frame)
    a_body = pines(to_body(xs))
    to_inertial = to_body.inverse()
    return to_inertial([xs[0], a_body[0], a_body[1], a_body[2]])


def time_calls(fn, args):
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(N_CALLS):
            fn(args)
        samples.append((time.perf_counter() - t0) / N_CALLS)
    return float(np.median(samples))


def main():
    tmp = tempfile.mkdtemp(prefix="asset_iso_")
    pck = os.path.join(tmp, "earth.tpc")
    with open(pck, "w") as f:
        f.write(TEST_EARTH_PCK)
    ext.SpiceLoadKernels([pck])

    to_body = ext.SpicePxformApply(FROM_FRAME, EARTH_TO_FRAME)
    to_inertial = to_body.inverse()
    pines = ext.PinesGravity(os.path.abspath(EGM96), DEGREE, MU_EARTH, R_EARTH, "fully_normalized")
    composed = composed_accel(pines, FROM_FRAME, EARTH_TO_FRAME)
    fused = SpiceForceFrame(pines, FROM_FRAME, EARTH_TO_FRAME)

    r = np.array([6.9e3, -4.1e3, 2.3e3])
    et = 1.0e6
    x_et = np.concatenate([[et], r])
    r_body = np.array(to_body.compute(x_et))

    rows = [
        ("SpicePxformApply alone (to body)", time_calls(to_body.compute, x_et)),
        ("SpicePxformApply alone (to inertial)", time_calls(to_inertial.compute, x_et)),
        ("PinesGravity alone (body frame)", time_calls(pines.compute, r_body)),
        ("Composed: to_inertial(pines(to_body))", time_calls(composed.compute, x_et)),
        ("Fused: SpiceForceFrame", time_calls(fused.compute, x_et)),
    ]
    print(f"Pines degree {DEGREE}, {N_CALLS} calls x {REPEATS} repeats, median per call")
    for name, t in rows:
        print(f"  {name:42s} {t * 1e6:9.2f} us")
    print(f"  Transform overhead in composed graph: ~{(rows[3][1] - rows[2][1]) * 1e6:.2f} us "
          f"(composed minus bare Pines)")


if __name__ == "__main__":
    main()
