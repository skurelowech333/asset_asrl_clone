# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Regular spherical harmonic gravity kernel
# ASSET VectorFunction factory version
#
# ============================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast

cimport cython


vf = ast.VectorFunctions
Args = vf.Arguments



# ============================================================
# SPHERICAL HARMONIC GRAVITY VF
# ============================================================

cdef class SphericalHarmonicGravityVF:


    cdef:

        double mu
        double radius
        int degree

        object Cnm
        object Snm

        object _acc_vf
        object _ode_vf



    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __cinit__(
        self,
        double mu,
        double radius,
        object Cnm,
        object Snm,
        int degree
    ):

        self.mu = mu
        self.radius = radius
        self.degree = degree

        self.Cnm = Cnm
        self.Snm = Snm


        self._acc_vf = (
            self._build_acceleration()
        )

        self._ode_vf = (
            self._build_ode()
        )



    # =====================================================
    # ACCELERATION BUILDER
    # =====================================================

    # ============================================================
    # ACCELERATION BUILDER
    # ============================================================
    
    cdef object _build_acceleration(self):
    
        X = Args(6)
    
        r = X.head3()
    
    
        # ---------------------------------
        # central gravity
        #
        # a = -mu*r/r^3
        #
        # ---------------------------------
    
        mu = X.Constant(self.mu)
    
        acc = (
            -mu
            *
            r.normalized_power3()
        )
    
    
        # ASSET scalar zero
        correction = X.Constant(0.0)
    
    
        for n in range(self.degree + 1):
    
            for m in range(n + 1):
    
                # placeholder harmonic coefficient
                #
                # make it symbolic-safe
                #
                correction = correction + (
                    X.Constant(float(self.Cnm[n,m]))
                    *
                    X.Constant(0.0)
                )
    
    
        # convert scalar correction to vector
        correction_vec = vf.stack(
            [
                correction,
                correction,
                correction
            ]
        )
    
    
        acc = acc + correction_vec
    
    
        return acc
    

    # =====================================================
    # ODE
    # =====================================================

    cdef object _build_ode(self):


        X = Args(6)

        v = X.tail3()


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
    # PUBLIC API
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
            f"SphericalHarmonicGravityVF("
            f"degree={self.degree})"
        )