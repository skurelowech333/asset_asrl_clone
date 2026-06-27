# =========================================================
# FILE: GravityPines.pyx
# Pines-style gravity kernel
# ASSET VectorFunction Factory
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



cdef class PinesGravityVF:


    cdef:
        double mu
        object _acc_vf
        object _ode_vf



    def __cinit__(self, double mu):

        self.mu = mu

        self._acc_vf = (
            self._build_acceleration()
        )

        self._ode_vf = (
            self._build_ode()
        )



    # =====================================================
    # acceleration
    # =====================================================

    cdef object _build_acceleration(self):

        cdef object X = Args(6)

        cdef object r = X.head3()


        # Pines compatible central recurrence form
        #
        # replaces:
        # sqrt(x*x+y*y+z*z)
        #
        # with ASSET normalized_power3


        return (
            -self.mu
            *
            r.normalized_power3()
        )



    # =====================================================
    # full ODE
    # =====================================================

    cdef object _build_ode(self):

        cdef object X = Args(6)

        cdef object v = X.tail3()


        a = self._acc_vf


        return vf.stack(
            [
                v[0],
                v[1],
                v[2],

                a[0],
                a[1],
                a[2]
            ]
        )



    # =====================================================
    # public contract
    # =====================================================

    cpdef object build_accel(self):

        return self._acc_vf



    cpdef object vf(self):

        return self._acc_vf



    cpdef object ode(self):

        return self._ode_vf



    def __call__(self):

        return self._ode_vf



    def __repr__(self):

        return (
            f"PinesGravityVF(mu={self.mu})"
        )