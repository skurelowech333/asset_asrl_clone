# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Classical spherical harmonic gravity kernel
#
# Direct trig evaluation
# Baseline comparison against Pines recurrence
#
# ============================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast

cimport cython


from pines_math import (
    SphericalToCartesian
)

from Legendre import AssociatedLegendre



vf = ast.VectorFunctions
Args = vf.Arguments



# ============================================================
# CLASS
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



    # ========================================================
    # Constructor
    # ========================================================

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


        self._acc_vf = self._build_acceleration()

        self._ode_vf = self._build_ode()



    # ========================================================
    # Acceleration
    # ========================================================

    cdef object _build_acceleration(self):


        X = Args(6)


        rvec = X.head3()


        x = rvec[0]
        y = rvec[1]
        z = rvec[2]


        rxy = (
            x*x+y*y
        ).sqrt()


        r = (
            x*x+y*y+z*z
        ).sqrt()



        #
        # spherical coordinates
        #

        sinphi = z/r

        cosphi = rxy/r


        sinlam = y/rxy

        coslam = x/rxy



        rho = (
            X.Constant(self.radius)/r
        )



        #
        # longitude angle
        #

        lam = y.atan2(x)



        #
        # Legendre
        #

        P = AssociatedLegendre(
            sinphi,
            self.degree
        )



        zero = x*0.0

        one = zero + 1.0


        radial_sum = one

        latitude_sum = zero

        longitude_sum = zero



        cdef int n
        cdef int m


        cdef object rho_n
        cdef object trig
        cdef object cp
        cdef object sp
        cdef object dP



        #
        # Direct spherical harmonic summation
        #

        for n in range(2,self.degree+1):


            rho_n = rho**n


            for m in range(n+1):


                #
                # DIRECT TRIG
                #
                # This is where Pines gains speed.
                #

                cp = (
                    (m*lam).cos()
                )

                sp = (
                    (m*lam).sin()
                )


                trig = (

                    X.Constant(
                        float(self.Cnm[n,m])
                    )
                    *
                    cp

                    +

                    X.Constant(
                        float(self.Snm[n,m])
                    )
                    *
                    sp
                )



                #
                # derivative of P wrt latitude
                #

                if m == n:

                    dP = (

                        -n
                        *
                        sinphi
                        *
                        P[n,m]

                        /

                        cosphi

                    )

                else:

                    dP = (

                        (
                            (n+m)
                            *
                            P[n-1,m]
                        )

                        -

                        (
                            n
                            *
                            sinphi
                            *
                            P[n,m]
                        )

                    ) / cosphi



                radial_sum += (

                    (n+1)
                    *
                    rho_n
                    *
                    P[n,m]
                    *
                    trig

                )



                latitude_sum += (

                    rho_n
                    *
                    dP
                    *
                    trig

                )



                longitude_sum += (

                    rho_n
                    *
                    P[n,m]
                    *
                    m

                    *

                    (

                        X.Constant(
                            float(self.Snm[n,m])
                        )
                        *
                        cp

                        -

                        X.Constant(
                            float(self.Cnm[n,m])
                        )
                        *
                        sp

                    )

                )



        mu_r2 = (

            X.Constant(self.mu)

            /

            (r*r)

        )



        ar = (

            -mu_r2
            *
            radial_sum

        )


        aph = (

            mu_r2
            *
            latitude_sum

        )


        alam = (

            mu_r2
            *
            longitude_sum

            /

            cosphi

        )



        ax,ay,az = SphericalToCartesian(

            ar,
            aph,
            alam,

            sinphi,
            cosphi,

            sinlam,
            coslam

        )



        return vf.stack(
            [
                ax,
                ay,
                az
            ]
        )



    # ========================================================
    # ODE
    # ========================================================

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



    # ========================================================
    # API
    # ========================================================

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