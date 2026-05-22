# -*- coding: utf-8 -*-
"""
GravityVF Enhanced Model (Cython-safe + ASSET-compatible)
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast
cimport cython

vf = ast.VectorFunctions
Args = vf.Arguments


# =========================================================
# GRAVITY VF CLASS
# =========================================================
cdef class GravityVF:

    # -----------------------------------------------------
    # C-level scalars
    # -----------------------------------------------------
    cdef double mu
    cdef double Re
    cdef double J2
    cdef double Cd
    cdef double A_over_m
    cdef double rho0

    # -----------------------------------------------------
    # Python object VF graphs (must be cdef object)
    # -----------------------------------------------------
    cdef object _acc_vf
    cdef object _ode_vf


    # =====================================================
    # CONSTRUCTOR
    # =====================================================
    def __cinit__(self,
                  double mu,
                  double Re=6378137.0,
                  double J2=1.08263e-3,
                  double Cd=2.2,
                  double A_over_m=0.01,
                  double rho0=1.225):

        self.mu = mu
        self.Re = Re
        self.J2 = J2
        self.Cd = Cd
        self.A_over_m = A_over_m
        self.rho0 = rho0

        self._acc_vf = self._build_acceleration()
        self._ode_vf = self._build_ode()


    # =====================================================
    # DENSITY MODEL (SAFE PYTHON / VF MIX)
    # =====================================================
    def _density(self, r):

        h = r.norm() - self.Re
        return self.rho0 * vf.exp(-h / 8500.0)


    # =====================================================
    # ACCELERATION MODEL
    # =====================================================
    def _build_acceleration(self):

        X = Args(6)
        r = X.head3()
        v = X.tail3()

        rnorm = r.norm()
        z = r[2]

        # -------------------------------------------------
        # CENTRAL GRAVITY
        # -------------------------------------------------
        central = (-self.mu) * r.normalized_power3()

        # -------------------------------------------------
        # J2 PERTURBATION
        # -------------------------------------------------
        r2 = rnorm * rnorm
        z2 = z * z

        factor = (
            1.5
            * self.J2
            * self.mu
            * (self.Re * self.Re)
            / (r2 * rnorm * rnorm * rnorm)
        )

        j2 = vf.stack(
            factor * r[0] * (5 * z2 / r2 - 1),
            factor * r[1] * (5 * z2 / r2 - 1),
            factor * r[2] * (5 * z2 / r2 - 3)
        )

        # -------------------------------------------------
        # ATMOSPHERIC DRAG
        # -------------------------------------------------
        vnorm = v.norm()
        rho = self._density(r)

        drag = -0.5 * self.Cd * self.A_over_m * rho * vnorm * v

        # -------------------------------------------------
        # TOTAL ACCELERATION
        # -------------------------------------------------
        return central + j2 + drag


    # =====================================================
    # FULL ODE
    # =====================================================
    def _build_ode(self):

        X = Args(6)
        v = X.tail3()

        return vf.stack(
            v,
            self._acc_vf
        )


    # =====================================================
    # INTERFACE
    # =====================================================
    cpdef object vf(self):
        return self._acc_vf

    cpdef object ode(self):
        return self._ode_vf

    def __call__(self):
        return self._ode_vf


    # =====================================================
    # DEBUG
    # =====================================================
    def __repr__(self):
        return (
            f"GravityVF("
            f"mu={self.mu}, "
            f"J2={self.J2}, "
            f"Cd={self.Cd})"
        )