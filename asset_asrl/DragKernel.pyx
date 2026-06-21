# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:41:34 2026

@author: Sarah
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# =========================================================
# DragKernel.pyx
# =========================================================

import asset_asrl as ast
cimport cython

vf = ast.VectorFunctions
Args = vf.Arguments


cdef class DragKernel:

    cdef double Cd
    cdef double A_over_m
    cdef double rho0
    cdef double Re

    cdef object _vf

    def __cinit__(self,
                  double Cd,
                  double A_over_m,
                  double rho0,
                  double Re):

        self.Cd = Cd
        self.A_over_m = A_over_m
        self.rho0 = rho0
        self.Re = Re

        self._vf = self._build()

    cdef object _build(self):

        X = Args(6)
        r = X.head3()
        v = X.tail3()

        h = r.norm() - self.Re
        rho = self.rho0 * vf.exp(-h / 8500.0)

        vnorm = v.norm()

        return -0.5 * self.Cd * self.A_over_m * rho * vnorm * v

    cpdef object vf(self, object r, object v):
        return self._vf

    def __call__(self):
        return self._vf