# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Classical spherical harmonic gravity kernel
#
# Reference implementation for comparison with Pines
#
# ============================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast

cimport cython


from Legendre import AssociatedLegendre

from pines_math import SphericalToCartesian


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
    # Build acceleration
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



        zero = x*0.0

        one = zero + 1.0



        radial_sum = one

        latitude_sum = zero

        longitude_sum = zero



        cdef int n
        cdef int m
        cdef int k



        cdef object rho_n

        cdef object C_m
        cdef object S_m

        cdef object C_prev
        cdef object S_prev

        cdef object trig

        cdef object dP



        #
        # Classical implementation
        #
        # Harmonics are rebuilt for every n,m
        #

        for n in range(2,self.degree+1):


            rho_n = rho**n



            for m in range(0,n+1):


                #
                # ------------------------------------
                # Recompute cos(m lambda), sin(m lambda)
                #
                # No cached recurrence
                #
                # ------------------------------------
                #

                C_m = one

                S_m = zero


                if m > 0:


                    C_m = coslam

                    S_m = sinlam


                    for k in range(2,m+1):


                        C_prev = C_m

                        S_prev = S_m


                        C_m = (

                            coslam*C_prev

                            -

                            sinlam*S_prev

                        )


                        S_m = (

                            sinlam*C_prev

                            +

                            coslam*S_prev

                        )



                trig = (

                    X.Constant(
                        float(self.Cnm[n,m])
                    )

                    *

                    C_m


                    +

                    X.Constant(
                        float(self.Snm[n,m])
                    )

                    *

                    S_m

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

                        C_m


                        -

                        X.Constant(
                            float(self.Cnm[n,m])
                        )

                        *

                        S_m

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