"""
ASSET native SpicePxformApply vs Orekit, for the Moon body-fixed frame.

Frame pair: J2000 -> IAU_MOON (SPICE, from NAIF pck00010.tpc) vs
EME2000 -> MoonIAU (Orekit, IAU 2009 model). Both are the IAU Moon body-fixed
frame. Timing and agreement are reported together.

Agreement is expected to be a constant offset of about 0.25 m on a 1e6-3e6 m
vector (~17 mas, the J2000/ICRF frame-bias size). Known discrepancies are
documented in the report printed by this script.

Requires OREKIT_DATA (default /root/orekit-data) and the NAIF kernel at
SPICE_PCK (default /root/spice_kernels/pck00010.tpc). Run with the ASSET dev
environment on PYTHONPATH.

Usage:
    python benchmark_orekit_pxform.py [n_samples] [repeats]
"""

import os
import sys
import time

import numpy as np
import orekit_jpype

orekit_jpype.initVM()

from java.io import File  # noqa: E402
from org.hipparchus.geometry.euclidean.threed import Vector3D  # noqa: E402
from org.orekit.bodies import CelestialBodyFactory  # noqa: E402
from org.orekit.data import DataContext, DirectoryCrawler  # noqa: E402
from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.time import AbsoluteDate  # noqa: E402

import asset_asrl.Extensions as ext  # noqa: E402


def time_best(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(n=20000, repeats=3):
    data_dir = os.environ.get("OREKIT_DATA", "/root/orekit-data")
    pck = os.environ.get("SPICE_PCK", "/root/spice_kernels/pck00010.tpc")
    DataContext.getDefault().getDataProvidersManager().addProvider(
        DirectoryCrawler(File(data_dir)))
    ext.SpiceLoadKernels([pck])

    rng = np.random.default_rng(0)
    ets = rng.uniform(0.0, 3.0e8, size=n)
    rs = rng.uniform(-2.0e6, 2.0e6, size=(n, 3))
    xs = np.column_stack([ets, rs])

    asset_vf = ext.SpicePxformApply("J2000", "IAU_MOON")
    eme = FramesFactory.getEME2000()
    moon = CelestialBodyFactory.getMoon().getBodyOrientedFrame()
    epoch = AbsoluteDate.J2000_EPOCH

    m = min(n, 200)
    asset_out = np.array([np.array(asset_vf.compute(x)) for x in xs[:m]])
    orekit_out = np.empty((m, 3))
    for i in range(m):
        tr = eme.getTransformTo(moon, epoch.shiftedBy(float(ets[i])))
        v = tr.transformVector(Vector3D(*map(float, rs[i])))
        orekit_out[i] = [v.getX(), v.getY(), v.getZ()]

    diff = asset_out - orekit_out
    # Fit a rotation vector w with diff_i ~= w x x_i (small-angle frame error).
    xs_out = asset_out
    A = np.zeros((3 * m, 3))
    for i, x in enumerate(xs_out):
        A[3 * i:3 * i + 3] = -np.array([[0, -x[2], x[1]],
                                        [x[2], 0, -x[0]],
                                        [-x[1], x[0], 0]])
    w, *_ = np.linalg.lstsq(A, diff.reshape(-1), rcond=None)
    fit = (A @ w).reshape(m, 3)
    resid = np.linalg.norm(diff - fit, axis=1)
    norm = np.linalg.norm(diff, axis=1)
    w_mas = np.linalg.norm(w) * 206264806.0
    print(f"agreement over {m} samples, ASSET IAU_MOON vs Orekit MoonIAU:")
    print(f"  |diff| max / median       : {norm.max():.4f} / {np.median(norm):.4f} m")
    print(f"  fitted frame rotation     : {w_mas:.2f} mas, axis {np.round(w / np.linalg.norm(w), 4)}")
    print(f"  residual after rotation   : max {resid.max():.2e} m, median {np.median(resid):.2e} m")
    print(f"  ET span                   : {ets[:m].min():.3e} .. {ets[:m].max():.3e} s")
    print("  discrepancy notes: see benchmarks/OREKIT_MOON_AGREEMENT.md")

    def asset_run():
        for x in xs:
            asset_vf.compute(x)

    def orekit_run():
        for et, r in zip(ets, rs):
            tr = eme.getTransformTo(moon, epoch.shiftedBy(float(et)))
            tr.transformVector(Vector3D(float(r[0]), float(r[1]), float(r[2])))

    t_asset = time_best(asset_run, repeats)
    t_ore = time_best(orekit_run, repeats)

    print(f"samples={n}, best of {repeats}")
    print(f"  ASSET SpicePxformApply.compute: {t_asset:8.4f} s "
          f"({t_asset / n * 1e6:8.2f} us/call)")
    print(f"  Orekit transformVector        : {t_ore:8.4f} s "
          f"({t_ore / n * 1e6:8.2f} us/call)")
    print(f"  speedup (Orekit / ASSET): {t_ore / t_asset:.1f}x")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(n, reps)
