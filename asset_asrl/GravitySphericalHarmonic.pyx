# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Classical spherical harmonic gravity kernel
#
# Reference implementation
#
# Compared against:
#     GravityPines.pyx
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
    HarmonicTrig,
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

        self.Cnm = Cnm
        self.Snm = Snm

        self.degree = degree


        self._acc_vf = (
            self._build_acceleration()
        )

        self._ode_vf = (
            self._build_ode()
        )



    # ========================================================
    # Acceleration
    # ========================================================

    cdef object _build_acceleration(self):


        X = Args(6)


        rvec = X.head3()


        x = rvec[0]
        y = rvec[1]
        z = rvec[2]



        #
        # radius
        #

        rxy = (
            x*x +
            y*y
        ).sqrt()


        r = (
            x*x +
            y*y +
            z*z
        ).sqrt()



        #
        # spherical coordinates
        #

        sinphi = z/r


        cosphi = rxy/r


        coslam = x/rxy

        sinlam = y/rxy



        rho = (
            X.Constant(self.radius)
            /
            r
        )



        #
        # Legendre functions
        #

        P = AssociatedLegendre(
            sinphi,
            self.degree
        )



        #
        # trig is NOT cached here
        #
        # This is the reference kernel
        #

        zero = x*0.0

        one = zero + 1.0


        radial_sum = one

        latitude_sum = zero

        longitude_sum = zero



        cdef int n
        cdef int m


        cdef object rho_n
        cdef object Ctrig
        cdef object Strig

        cdef object trig
        cdef object dP



        # ----------------------------------------------------
        # harmonic summation
        # ----------------------------------------------------

        for n in range(2,self.degree+1):


            rho_n = rho**n


            #
            # intentionally rebuild trig
            #

            Ctrig,Strig = HarmonicTrig(
                coslam,
                sinlam,
                n
            )



            for m in range(n+1):


                trig = (

                    X.Constant(
                        float(self.Cnm[n,m])
                    )

                    *

                    Ctrig[m]


                    +

                    X.Constant(
                        float(self.Snm[n,m])
                    )

                    *

                    Strig[m]

                )



                #
                # Legendre derivative
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



                #
                # radial derivative
                #

                radial_sum += (

                    (n+1)

                    *

                    rho_n

                    *

                    P[n,m]

                    *

                    trig

                )



                #
                # latitude derivative
                #

                latitude_sum += (

                    rho_n

                    *

                    dP

                    *

                    trig

                )



                #
                # longitude derivative
                #

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

                        Ctrig[m]


                        -

                        X.Constant(
                            float(self.Cnm[n,m])
                        )

                        *

                        Strig[m]

                    )

                )



        #
        # acceleration
        #

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
    # Public API
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