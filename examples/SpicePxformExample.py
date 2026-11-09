"""
Native SPICE frame transformations in ASSET.

Shows how to use ext.SpicePxform (ET -> flattened 3x3 rotation) and
ext.SpicePxformApply (ET, r -> R(t) r) and verifies the results:

  * rotation matches spiceypy.pxform
  * apply matches spiceypy.pxform @ r
  * analytic Jacobian matches finite differences in ET and r
  * inverse() recovers the original vector

Usage:
    python SpicePxformExample.py kernel1 [kernel2 ...]

Kernels must include a frame definition for the target frame. For the
GRAIL case use moon_pa_de440_200625.bpc and moon_080317.tf.
"""

import sys

import numpy as np
import spiceypy as spice

import asset_asrl.Extensions as ext


def main(kernel_files):
    # Load kernels once into the global SPICE pool.
    ext.SpiceLoadKernels(list(kernel_files))

    from_frame, to_frame = "J2000", "IAU_MOON"

    # Native VectorFunctions.
    rot = ext.SpicePxform(from_frame, to_frame)
    apply_vf = ext.SpicePxformApply(from_frame, to_frame)

    et = 1.0e8
    r = np.array([1.2e6, -0.7e6, 0.5e6])

    # 1. Rotation matches spiceypy.
    R_ref = np.array(spice.pxform(from_frame, to_frame, et))
    R_asset = np.array(rot.compute(np.array([et]))).reshape(3, 3)
    assert np.allclose(R_asset, R_ref, rtol=0, atol=1e-13), "rotation mismatch"

    # 2. Apply matches R @ r.
    x = np.concatenate([[et], r])
    r_pa = np.array(apply_vf.compute(x))
    assert np.allclose(r_pa, R_ref @ r, rtol=0, atol=1e-6), "apply mismatch"

    # 3. Jacobian [dR/dt * r | R] matches finite differences.
    J = np.array(apply_vf.jacobian(x))
    h = 1.0  # seconds for ET, meters for position
    J_fd = np.zeros((3, 4))
    for k in range(4):
        dx = np.zeros(4)
        dx[k] = h
        J_fd[:, k] = (np.array(apply_vf.compute(x + dx))
                      - np.array(apply_vf.compute(x - dx))) / (2 * h)
    scale = np.max(np.abs(J_fd))
    assert np.max(np.abs(J - J_fd)) < 1e-6 * scale, "Jacobian mismatch"

    # 4. Inverse recovers the original vector.
    inv = apply_vf.inverse()
    r_back = np.array(inv.compute(np.concatenate([[et], r_pa])))
    assert np.allclose(r_back, r, rtol=0, atol=1e-6), "inverse mismatch"

    print("SpicePxform / SpicePxformApply verification passed")
    print(f"  R @ r          = {r_pa}")
    print(f"  recovered r    = {r_back}")
    print(f"  Jacobian max rel. error vs FD = "
          f"{np.max(np.abs(J - J_fd)) / scale:.2e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
