# =========================================================
# GravitySystem.pyx
# =========================================================

import asset_asrl as ast
cimport cython

vf = ast.VectorFunctions
Args = vf.Arguments


cdef class GravitySystem:

    cdef object kernels
    cdef object _ode

    def __cinit__(self, object kernels):

        self.kernels = kernels
        self._ode = self._build_ode()

    # =====================================================
    # BUILD FUSED ACCELERATION
    # =====================================================
    cdef object _build_acc(self):

    X = Args(6)
    r = X.head3()
    v = X.tail3()

    cdef object acc = None
    cdef object term

    for k in self.kernels:

        # -------------------------------------------------
        # CASE 1: k.vf(r, v) style kernel
        # -------------------------------------------------
        if hasattr(k.vf, "__call__"):
            try:
                term = k.vf(r, v)
            except TypeError:
                # -------------------------------------------------
                # CASE 2: k.vf() style kernel (closure/factory)
                # -------------------------------------------------
                term = k.vf()

        else:
            raise TypeError("Kernel must implement vf or be callable")

        if acc is None:
            acc = term
        else:
            acc = acc + term

    return acc

    # =====================================================
    # FULL ODE
    # =====================================================
    cdef object _build_ode(self):

        X = Args(6)
        v = X.tail3()

        acc = self._build_acc()

        return vf.stack(v, acc)

    cpdef object ode(self):
        return self._ode

    def __call__(self):
        return self._ode
    
    def build_acc(self):
        return self._build_acc()