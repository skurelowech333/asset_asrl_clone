# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Classical spherical harmonic gravity — C++ ASSET node
#
# The heavy computation is done by the C++ SphericalHarmonicGravity
# VectorFunction node registered in asset_asrl.Extensions.
# Using AutodiffFwd, ASSET can propagate dual numbers through the
# C++ kernel to obtain analytic Jacobians, enabling:
#
#   ✅ ASSET automatic differentiation
#   ✅ ASSET Jacobians / sparsity propagation
#   ✅ ASSET optimal control derivatives
#   ✅ ASSET ODE integration
#
# The pure-ASSET graph loop (degree iterations creating ASSET
# nodes) has been removed; at degree 20-50 it caused hangs.
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import numpy as np
import asset_asrl as ast
import asset_asrl.Extensions  # initialises ast.Extensions from asset.Extensions C++ submodule

cimport cython


vf = ast.VectorFunctions
Args = vf.Arguments


# ============================================================
# Early check: verify the C++ extension is available.
# If the asset module has not been recompiled with the
# SphericalHarmonicGravity registration (extensions/
# ASSET_Extensions.cpp) this raises a clear error instead
# of a confusing AttributeError deep in _build_acceleration.
# ============================================================

if not hasattr(ast.Extensions, 'SphericalHarmonicGravity'):
    raise RuntimeError(
        "SphericalHarmonicGravity C++ node is not registered in asset.Extensions.\n"
        "The 'asset' C++ module must be rebuilt after adding the extension in\n"
        "extensions/ASSET_Extensions.cpp and extensions/SphericalHarmonicGravity.h.\n"
        "Run cmake --build <build_dir> and reinstall, then retry."
    )



# ============================================================
# CLASSICAL SPHERICAL HARMONIC (C++ node backend)
# ============================================================


cdef class SphericalHarmonicGravityVF:


    cdef:

        double mu
        double radius

        int degree

        object Cnm
        object Snm

        object _acc_vf
        object _ode_vf



    def __cinit__(
        self,
        double mu,
        double radius,
        object Cnm,
        object Snm,
        int degree
    ):

        self.mu = mu
        self.radius = radius

        self.Cnm = Cnm
        self.Snm = Snm

        self.degree = degree


        self._acc_vf = (
            self._build_acceleration()
        )

        self._ode_vf = (
            self._build_ode()
        )



    # ========================================================
    # Acceleration
    #
    # Converts Cnm / Snm (2-D array-like indexed as [n, m])
    # to flat vectors and calls the C++ ASSET node.
    # ========================================================


    cdef object _build_acceleration(self):

        cdef int n
        cdef int m
        cdef int N1 = self.degree + 1


        C_flat = np.zeros(N1 * N1, dtype=np.float64)
        S_flat = np.zeros(N1 * N1, dtype=np.float64)


        for n in range(N1):

            for m in range(n + 1):

                C_flat[n * N1 + m] = float(self.Cnm[n, m])
                S_flat[n * N1 + m] = float(self.Snm[n, m])


        return ast.Extensions.SphericalHarmonicGravity(
            self.mu,
            self.radius,
            C_flat.tolist(),
            S_flat.tolist(),
            self.degree
        )



    # ========================================================
    # ODE  [x, y, z, vx, vy, vz] -> [vx, vy, vz, ax, ay, az]
    # ========================================================


    cdef object _build_ode(self):


        X = Args(6)

        v = X.tail3()

        a = self._acc_vf


        return vf.stack(
            [
                v[0],
                v[1],
                v[2],

                a[0],
                a[1],
                a[2]
            ]
        )



    # ========================================================
    # API
    # ========================================================


    cpdef object vf(self):

        return self._acc_vf


    cpdef object build_accel(self):

        return self._acc_vf


    cpdef object ode(self):

        return self._ode_vf


    def __call__(self):

        return self._ode_vf



    def __repr__(self):

        return (
            f"SphericalHarmonicGravityVF("
            f"degree={self.degree})"
        )
