# ============================================================
# FILE: GravityPines.pyx
#
# Pines Cartesian J2-only implementation
#
# Purpose:
#   Validate Cartesian Pines formulation
#
# No spherical coordinates
# No Legendre functions
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import asset_asrl as ast


vf = ast.VectorFunctions
Args = vf.Arguments



cdef class PinesGravityVF:


    cdef:

        double mu
        double radius
        double C20

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


        #
        # J2 coefficient
        #

        self.C20 = float(
            Cnm[2,0]
        )


        self._acc_vf = (
            self._build_acceleration()
        )


        self._ode_vf = (
            self._build_ode()
        )



    cdef object _build_acceleration(self):


        X = Args(6)


        x = X[0]
        y = X[1]
        z = X[2]



        r2 = (

            x*x
            +
            y*y
            +
            z*z

        )


        r = r2.sqrt()



        #
        # J2 polynomial
        #

        V20 = (

            0.5
            *
            (
                3*z*z
                -
                r2
            )

        )



        #
        # derivative of V20
        #

        dVdx = -x

        dVdy = -y

        dVdz = 2*z



        #
        # r^-5
        #

        invr5 = (

            1.0
            /
            (r2*r2*r)

        )



        #
        # derivative of V20/r^5
        #

        common = (

            self.mu
            *
            self.C20
            *
            self.radius*self.radius

        )



        ax = (

            -common
            *
            (
                dVdx*invr5
                -
                5*x*V20/(r2*r2*r2*r)
            )

        )


        ay = (

            -common
            *
            (
                dVdy*invr5
                -
                5*y*V20/(r2*r2*r2*r)
            )

        )


        az = (

            -common
            *
            (
                dVdz*invr5
                -
                5*z*V20/(r2*r2*r2*r)
            )

        )



        #
        # add central gravity
        #

        central = (

            -self.mu
            /
            (r2*r)

        )



        ax += central*x

        ay += central*y

        az += central*z



        return vf.stack(
            [
                ax,
                ay,
                az
            ]
        )



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



    cpdef object vf(self):

        return self._acc_vf



    cpdef object ode(self):

        return self._ode_vf



    cpdef object build_accel(self):

        return self._acc_vf



    def __call__(self):

        return self._ode_vf