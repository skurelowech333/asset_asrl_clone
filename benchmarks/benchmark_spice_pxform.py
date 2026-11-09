"""
Speed benchmark: native ASSET SPICE pxform vs a Python callback.

Compares evaluating r_PA = R(et) r for many (et, r) samples using:
  * ASSET native:   ext.SpicePxformApply.compute / .jacobian (C++ node)
  * Python callback: spiceypy.pxform(...) @ r on every call

Usage:
    python benchmark_spice_pxform.py [n_samples] [repeats]
Set SPICE_KERNELS to a space-separated list of kernels to use real frames;
otherwise a temporary test PCK for IAU_MOON is generated.
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np
import spiceypy as spice

import asset_asrl.Extensions as ext

TEST_PCK = """KPL/PCK

\\begindata
BODY301_POLE_RA  = ( 269.9949   0.0031   0.0 )
BODY301_POLE_DEC = (  66.5392   0.0130   0.0 )
BODY301_PM       = (  38.3213  13.17635815  -1.4e-12 )
\\begintext
"""

FROM_FRAME, TO_FRAME = "J2000", "IAU_MOON"


def load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="asset_bench_")
    pck = os.path.join(tmpdir, "moon_test.tpc")
    with open(pck, "w") as f:
        f.write(TEST_PCK)
    kernels = [pck] + os.environ.get("SPICE_KERNELS", "").split()
    ext.SpiceLoadKernels(kernels)
    return tmpdir


def make_samples(n, rng):
    ets = rng.uniform(0.0, 5.0e8, size=n)
    rs = rng.uniform(-2.0e6, 2.0e6, size=(n, 3))
    return ets, rs


def time_best(fn, repeats):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(n=20000, repeats=3):
    tmpdir = load_kernels()
    try:
        rng = np.random.default_rng(0)
        ets, rs = make_samples(n, rng)
        xs = np.column_stack([ets, rs])

        apply_vf = ext.SpicePxformApply(FROM_FRAME, TO_FRAME)

        def asset_apply():
            for x in xs:
                apply_vf.compute(x)

        def asset_jacobian():
            for x in xs:
                apply_vf.jacobian(x)

        def python_callback():
            for et, r in zip(ets, rs):
                R = np.array(spice.pxform(FROM_FRAME, TO_FRAME, et))
                R @ r

        # Correctness spot check so the timing compares equal work.
        ref = np.array([np.array(spice.pxform(FROM_FRAME, TO_FRAME, e)) @ r
                        for e, r in zip(ets[:50], rs[:50])])
        got = np.array([np.array(apply_vf.compute(x)) for x in xs[:50]])
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)

        t_asset = time_best(asset_apply, repeats)
        t_jac = time_best(asset_jacobian, repeats)
        t_py = time_best(python_callback, repeats)

        print(f"samples={n}, best of {repeats}")
        print(f"  ASSET SpicePxformApply.compute : {t_asset:8.4f} s "
              f"({t_asset / n * 1e6:7.2f} us/call)")
        print(f"  ASSET SpicePxformApply.jacobian: {t_jac:8.4f} s "
              f"({t_jac / n * 1e6:7.2f} us/call)")
        print(f"  Python spiceypy callback       : {t_py:8.4f} s "
              f"({t_py / n * 1e6:7.2f} us/call)")
        print(f"  speedup (callback / ASSET compute): {t_py / t_asset:.1f}x")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(n, reps)
