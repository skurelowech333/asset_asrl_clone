# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:58:32 2026

@author: Sarah
"""

# =========================================================
# FILE: NBodyGravity.pyx
# FULL ODE FACTORY VERSION (N-BODY VF BENCHMARK)
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


cdef class NBodyGravityVF:

    cdef:
        double mu
        object bodies

        object _acc_vf
        object _ode_vf

    # =====================================================
    # CONSTRUCTOR
    # =====================================================
    def __cinit__(self, double mu, object bodies):

        self.mu = mu
        self.bodies = bodies

        self._acc_vf = self._build_acceleration()
        self._ode_vf = self._build_ode()


    # =====================================================
    # BUILD ACCELERATION
    # =====================================================
    cdef object _build_acceleration(self):

        cdef object X
        cdef object r

        cdef object m
        cdef object r_i

        cdef object dr
        cdef object term
        cdef object acc

        cdef bint first = True

        # -------------------------------------------------
        # SYMBOLIC INPUT
        # -------------------------------------------------
        X = Args(6)
        r = X.head3()

        # -------------------------------------------------
        # BUILD N-BODY SUM
        # -------------------------------------------------
        for m, r_i in self.bodies:

            # relative vector
            dr = r - r_i

            # gravity term
            term = (-self.mu * m) * dr.normalized_power3()

            # initialize accumulator with first term
            if first:
                acc = term
                first = False
            else:
                acc = acc + term

        return acc


    # =====================================================
    # BUILD FULL ODE
    # =====================================================
    cdef object _build_ode(self):

        cdef object X
        cdef object v
        cdef object ode

        X = Args(6)

        v = X.tail3()

        ode = vf.stack(
            v,
            self._acc_vf
        )

        return ode


    # =====================================================
    # RETURN ACCELERATION VF
    # =====================================================
    cpdef object vf(self):

        return self._acc_vf


    # =====================================================
    # RETURN FULL ODE VF
    # =====================================================
    cpdef object ode(self):

        return self._ode_vf


    # =====================================================
    # CALLABLE INTERFACE
    # =====================================================
    def __call__(self):

        return self._ode_vf


    # =====================================================
    # DEBUG STRING
    # =====================================================
    def __repr__(self):

        return f"NBodyGravityVF(mu={self.mu}, bodies={len(self.bodies)})"