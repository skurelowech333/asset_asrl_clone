# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:30:19 2026

@author: Sarah
"""

# =========================================================
# FILE: GravityVF.pyx
# CLEAN MODULAR Cython VF KERNEL
# =========================================================
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
# SYMBOLIC BUILDER (PURE VF CONSTRUCTION)
# =========================================================
cdef object build_gravity_vf(double mu,
                             double Re,
                             double J2,
                             double Cd,
                             double A_over_m,
                             double rho0):

    cdef object X, r, v
    cdef object rnorm, z, r2, z2
    cdef object central, j2, drag, acc

    X = Args(6)
    r = X.head3()
    v = X.tail3()

    # -----------------------------------------------------
    # CENTRAL GRAVITY
    # -----------------------------------------------------
    central = (-mu) * r.normalized_power3()

    # -----------------------------------------------------
    # J2 PERTURBATION
    # -----------------------------------------------------
    rnorm = r.norm()
    z = r[2]

    r2 = rnorm * rnorm
    z2 = z * z

    cdef double Re2 = Re * Re

    factor = (
        1.5 * J2 * mu * Re2
        / (r2 * rnorm * rnorm * rnorm)
    )

    j2 = vf.stack(
        factor * r[0] * (5 * z2 / r2 - 1),
        factor * r[1] * (5 * z2 / r2 - 1),
        factor * r[2] * (5 * z2 / r2 - 3)
    )

    # -----------------------------------------------------
    # DRAG MODEL
    # -----------------------------------------------------
    vnorm = v.norm()

    h = rnorm - Re
    rho = rho0 * vf.exp(-h / 8500.0)

    drag = -0.5 * Cd * A_over_m * rho * vnorm * v

    # -----------------------------------------------------
    # TOTAL ACCELERATION
    # -----------------------------------------------------
    acc = central + j2 + drag

    return acc


# =========================================================
# FULL ODE BUILDER
# =========================================================
cdef object build_ode_vf(object acc_vf):

    cdef object X, v, ode

    X = Args(6)
    v = X.tail3()

    ode = vf.stack(
        v,
        acc_vf
    )

    return ode


# =========================================================
# Cython WRAPPER CLASS (THIN + FAST)
# =========================================================
cdef class GravityVF:

    cdef double mu
    cdef double Re
    cdef double J2
    cdef double Cd
    cdef double A_over_m
    cdef double rho0

    cdef object _acc_vf
    cdef object _ode_vf


    # -----------------------------------------------------
    # CONSTRUCTOR
    # -----------------------------------------------------
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

        # build once
        self._acc_vf = build_gravity_vf(
            mu, Re, J2, Cd, A_over_m, rho0
        )

        self._ode_vf = build_ode_vf(self._acc_vf)


    # -----------------------------------------------------
    # INTERFACE
    # -----------------------------------------------------
    cpdef object vf(self):
        return self._acc_vf

    cpdef object ode(self):
        return self._ode_vf

    def __call__(self):
        return self._ode_vf


    def __repr__(self):
        return (
            f"GravityVF("
            f"mu={self.mu}, "
            f"J2={self.J2}, "
            f"Cd={self.Cd})"
        )