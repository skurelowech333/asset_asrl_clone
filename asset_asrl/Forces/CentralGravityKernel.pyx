# =========================================================
# CentralGravityKernel.pyx
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


cdef class CentralGravityKernel:

    cdef double mu
    cdef object _vf

    def __cinit__(self, double mu):
        self.mu = mu
        self._vf = self._build()

    cdef object _build(self):

        X = Args(6)
        r = X.head3()

        return (-self.mu) * r.normalized_power3()

    cpdef object vf(self, object r, object v):
        return self._vf

    cpdef object expr(self):
        return self._vf

    def __call__(self):
        return self._vf