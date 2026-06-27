# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:50:42 2026

@author: Sarah
"""

# =========================================================
# FILE: pines_kernel.pyx
# =========================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np
cimport numpy as np


cpdef accel_j2(
        double x,
        double y,
        double z,
        double mu,
        double Re,
        double J2):


    cdef double r2
    cdef double r
    cdef double r3
    cdef double r5

    cdef double fac
    cdef double zz


    r2 = x*x+y*y+z*z
    r = r2**0.5

    r3 = r2*r
    r5 = r3*r2


    zz = z*z/r2


    fac = (
        1.5 *
        J2 *
        mu *
        Re*Re /
        r5
    )


    cdef double ax
    cdef double ay
    cdef double az


    ax = (
        -mu*x/r3 *
        (1 - fac*(5*zz-1))
    )

    ay = (
        -mu*y/r3 *
        (1 - fac*(5*zz-1))
    )

    az = (
        -mu*z/r3 *
        (1 - fac*(5*zz-3))
    )


    return np.array(
        [ax,ay,az],
        dtype=float
    )