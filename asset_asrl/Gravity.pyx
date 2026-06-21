# =========================================================
# FILE: Gravity.pyx
# FULL ODE FACTORY VERSION (CLEAN + TESTABLE)
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
# GRAVITY VECTOR FUNCTION WRAPPER
# =========================================================
cdef class GravityVF:

    cdef:
        double mu
        object _acc_vf
        object _ode_vf

    # =====================================================
    # CONSTRUCTOR
    # =====================================================
    def __cinit__(self, double mu):

        self.mu = mu

        self._acc_vf = self._build_acceleration()
        self._ode_vf = self._build_ode()


    # =====================================================
    # BUILD ACCELERATION VF (PRIMARY PUBLIC CONTRACT)
    # =====================================================
    cpdef object build_accel(self):
        return self._acc_vf


    # =====================================================
    # BUILD ACCELERATION VF (internal)
    # =====================================================
    cdef object _build_acceleration(self):

        cdef object X = Args(6)
        cdef object r = X.head3()

        # central gravity
        return (-self.mu) * r.normalized_power3()


    # =====================================================
    # BUILD FULL ODE VF
    # =====================================================
    cdef object _build_ode(self):

        cdef object X = Args(6)
        cdef object v = X.tail3()

        return vf.stack(v, self._acc_vf)


    # =====================================================
    # RETURN ACCELERATION VF (legacy alias)
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
    # DEBUG
    # =====================================================
    def __repr__(self):
        return f"GravityVF(mu={self.mu})"