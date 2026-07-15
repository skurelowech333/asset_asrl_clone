# ============================================================
# FILE: GravityPines.pyx
#
# Pines (1973) Cartesian gravity formulation
#
# Uses:
#       Vnm / Wnm Cartesian harmonic recursion
#
# No:
#       latitude
#       longitude
#       Legendre functions
#       spherical derivatives
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast

cimport cython


from PinesVW import BuildVW


vf = ast.VectorFunctions
Args = vf.Arguments



# ============================================================
# PINES GRAVITY CLASS
# ============================================================

cdef class PinesGravityVF:


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


        self._acc_vf = self._build_acceleration()

        self._ode_vf = self._build_ode()



    # ========================================================
    # Build Cartesian Pines acceleration
    # ========================================================

    cdef object _build_acceleration(self):


        cdef int n
        cdef int m



        X = Args(6)


        x = X[0]
        y = X[1]
        z = X[2]



        #
        # Pines Cartesian recursion
        #

        (
            V,
            W,
            Vx,
            Vy,
            Vz,
            Wx,
            Wy,
            Wz
        ) = BuildVW(
            x,
            y,
            z,
            self.radius,
            self.degree
        )



        #
        # Initialize acceleration
        #

        ax = X.Constant(0.0)
        ay = X.Constant(0.0)
        az = X.Constant(0.0)



        #
        # Harmonic summation
        #

        for n in range(
            2,
            self.degree + 1
        ):


            for m in range(
                0,
                n + 1
            ):


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



                #
                # Cartesian derivatives
                #

                ax += (

                    C*Vx[n,m]
                    +
                    S*Wx[n,m]

                )


                ay += (

                    C*Vy[n,m]
                    +
                    S*Wy[n,m]

                )


                az += (

                    C*Vz[n,m]
                    +
                    S*Wz[n,m]

                )



        #
        # Gravity scale
        #

        scale = X.Constant(
            self.mu
        )



        return vf.stack(
            [
                scale*ax,
                scale*ay,
                scale*az
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
            f"PinesGravityVF("
            f"degree={self.degree}, "
            f"mu={self.mu})"
        )