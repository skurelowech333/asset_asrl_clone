# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Classical spherical harmonic gravity kernel
#
# Fair comparison with Pines:
#   - Same Legendre
#   - Same coefficients
#   - Same normalization
#   - Same trig recurrence
#   - Same spherical conversion
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast

cimport cython


from Legendre import AssociatedLegendreGradient
from pines_math import (
    HarmonicTrig,
    SphericalToCartesian
)



vf = ast.VectorFunctions
Args = vf.Arguments



# ============================================================
# CLASSICAL SPHERICAL HARMONIC
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


        cdef int n
        cdef int m
        cdef int idx


        X = Args(6)


        rvec = X.head3()


        x = rvec[0]
        y = rvec[1]
        z = rvec[2]



        #
        # Geometry
        #

        r2 = (
            x*x
            +
            y*y
            +
            z*z
        )


        r = r2.sqrt()


        inv_r = 1.0/r



        xy = (
            x*x
            +
            y*y
        ).sqrt()



        inv_xy = 1.0/xy



        sinphi = z*inv_r

        cosphi = xy*inv_r


        coslam = x*inv_xy

        sinlam = y*inv_xy



        rho = (
            self.radius
            *
            inv_r
        )



        #
        # Same trig recurrence as Pines
        #

        Ctrig,Strig = HarmonicTrig(
            coslam,
            sinlam,
            self.degree
        )



        #
        # Same Legendre
        #

        P,dP = AssociatedLegendreGradient(
            sinphi,
            self.degree
        )



        #
        # rho powers
        #

        rho_power = [
            None
        ]*(self.degree+1)


        one = X.Constant(1.0)


        rho_power[0] = one


        for n in range(1,self.degree+1):

            rho_power[n] = (
                rho_power[n-1]
                *
                rho
            )



        #
        # harmonic sums
        #

        radial_sum = one

        latitude_sum = X.Constant(0.0)

        longitude_sum = X.Constant(0.0)



        N1 = self.degree+1



        #
        # classical harmonic loop
        #

        for n in range(2,self.degree+1):


            for m in range(n+1):


                idx = (
                    n*N1
                    +
                    m
                )


                Pnm = P[idx]

                dPnm = dP[idx]



                C = X.Constant(
                    float(
                        self.Cnm[n,m]
                    )
                )


                S = X.Constant(
                    float(
                        self.Snm[n,m]
                    )
                )



                trig = (

                    C*Ctrig[m]

                    +

                    S*Strig[m]

                )



                dtrig = (

                    m*

                    (
                        S*Ctrig[m]
                        -
                        C*Strig[m]
                    )

                )



                term = (

                    rho_power[n]
                    *
                    Pnm
                    *
                    trig

                )



                radial_sum += (

                    (n+1)
                    *
                    term

                )



                latitude_sum += (

                    rho_power[n]
                    *
                    dPnm
                    *
                    trig

                )



                longitude_sum += (

                    rho_power[n]
                    *
                    Pnm
                    *
                    dtrig

                )



        #
        # Convert acceleration
        #

        mu_r2 = (

            X.Constant(self.mu)
            /
            r2

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


    cpdef object vf(self):

        return self._acc_vf


    cpdef object build_accel(self):

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