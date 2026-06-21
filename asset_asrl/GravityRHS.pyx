# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:30:21 2026

@author: Sarah
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cdef class GravityRHS:

    cdef double mu

    def __init__(self, double mu):
        self.mu = mu

    # -----------------------------------------------------
    # CORE NUMERICAL RHS (FAST)
    # -----------------------------------------------------
    cpdef void eval(self,
                    double x, double y, double z,
                    double vx, double vy, double vz,
                    double[:] out):

        cdef double r2 = x*x + y*y + z*z + 1e-12
        cdef double inv_r = 1.0 / (r2 ** 0.5)
        cdef double inv_r3 = inv_r * inv_r * inv_r

        # acceleration
        out[0] = vx
        out[1] = vy
        out[2] = vz

        out[3] = -self.mu * x * inv_r3
        out[4] = -self.mu * y * inv_r3
        out[5] = -self.mu * z * inv_r3