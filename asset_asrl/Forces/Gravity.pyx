# =========================================================
# FILE: Gravity.pyx
# FULL ODE FACTORY VERSION
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

        # ---------------------------------------------
        # BUILD ONCE
        # ---------------------------------------------
        self._acc_vf = self._build_acceleration()

        self._ode_vf = self._build_ode()


    # =====================================================
    # BUILD ACCELERATION VF
    # returns:
    # [ax, ay, az]
    # =====================================================
    cdef object _build_acceleration(self):

        cdef object X
        cdef object r
        cdef object acc

        # ---------------------------------------------
        # SYMBOLIC INPUT
        # ---------------------------------------------
        X = Args(6)

        r = X.head3()

        # ---------------------------------------------
        # CENTRAL GRAVITY
        # ---------------------------------------------
        acc = (-self.mu) * r.normalized_power3()

        return acc


    # =====================================================
    # BUILD FULL ODE VF
    # returns:
    # [vx, vy, vz, ax, ay, az]
    # =====================================================
    cdef object _build_ode(self):

        cdef object X
        cdef object v
        cdef object ode

        # ---------------------------------------------
        # SYMBOLIC INPUT
        # ---------------------------------------------
        X = Args(6)

        # velocity state
        v = X.tail3()

        # ---------------------------------------------
        # FULL ODE
        # ---------------------------------------------
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

        return f"GravityVF(mu={self.mu})"