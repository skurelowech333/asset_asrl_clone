# ============================================================
# FILE: GravityPines.pyx
#
# Pines (1973) Cartesian gravity kernel
#
# Cartesian formulation:
#   U = mu/r * sum(Cnm Vnm + Snm Wnm)
#
# Acceleration:
#   a = -grad(U)
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast


from PinesVW import BuildVW
from PinesDerivatives import BuildDerivatives



vf = ast.VectorFunctions
Args = vf.Arguments



cdef class PinesGravityVF:

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
        int degree):

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



    #
    # Sparse coefficient lookup
    #

    cdef double get_C(
        self,
        int n,
        int m
    ):

        if (n,m) in self.Cnm:

            return float(
                self.Cnm[n,m]
            )

        return 0.0



    cdef double get_S(
        self,
        int n,
        int m
    ):

        if (n,m) in self.Snm:

            return float(
                self.Snm[n,m]
            )

        return 0.0



    cdef object _build_acceleration(self):


        cdef int n
        cdef int m


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
        # Central gravity
        #

        mu_r3 = (

            -self.mu
            /
            (r2*r)

        )


        ax = mu_r3*x
        ay = mu_r3*y
        az = mu_r3*z



        #
        # Pines Cartesian basis
        #

        V,W = BuildVW(
            x,
            y,
            z,
            self.radius,
            self.degree
        )



        #
        # Cartesian gradients
        #

        Vx,Vy,Vz,Wx,Wy,Wz = BuildDerivatives(
            V,
            W,
            x,
            y,
            z,
            self.radius,
            self.degree
        )



        #
        # Harmonic acceleration
        #

        scale = (

            self.mu
            /
            self.radius

        )



        for n in range(
            2,
            self.degree+1
        ):


            for m in range(
                0,
                n+1
            ):


                C = X.Constant(
                    self.get_C(n,m)
                )


                S = X.Constant(
                    self.get_S(n,m)
                )


                ax -= scale*(

                    C*Vx[n,m]
                    +
                    S*Wx[n,m]

                )


                ay -= scale*(

                    C*Vy[n,m]
                    +
                    S*Wy[n,m]

                )


                az -= scale*(

                    C*Vz[n,m]
                    +
                    S*Wz[n,m]

                )



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



    cpdef object build_accel(self):

        return self._acc_vf



    cpdef object ode(self):

        return self._ode_vf



    def __call__(self):

        return self._ode_vf



    def __repr__(self):

        return (
            f"PinesGravityVF("
            f"degree={self.degree})"
        )