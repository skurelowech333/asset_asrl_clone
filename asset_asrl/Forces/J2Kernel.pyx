# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:41:18 2026

@author: Sarah
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


cdef class J2Kernel:

    cdef double mu, Re, J2
    cdef object _vf

    def __cinit__(self, double mu, double Re, double J2):
        self.mu = mu
        self.Re = Re
        self.J2 = J2
        self._vf = self._build()

    cdef object _build(self):

        X = Args(6)
        r = X.head3()

        rnorm = r.norm()
        z = r[2]

        r2 = rnorm * rnorm
        z2 = z * z

        factor = (
            1.5 * self.J2 * self.mu * (self.Re * self.Re)
            / (r2 * rnorm * rnorm * rnorm)
        )

        return vf.stack(
            factor * r[0] * (5*z2/r2 - 1),
            factor * r[1] * (5*z2/r2 - 1),
            factor * r[2] * (5*z2/r2 - 3),
        )

    cpdef object vf(self):
        return self._vf

    def __repr__(self):
        return "J2Kernel"