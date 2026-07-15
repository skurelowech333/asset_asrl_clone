# ============================================================
# FILE: GravityPines.pyx
#
# Pines-style gravity kernel
#
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

# These modules live in the same package.  The former absolute import used a
# different (and differently cased) module name, so it failed as soon as the
# extension was imported from ``asset_asrl``.
from pines_math import (
    HarmonicTrig,
    SphericalToCartesian
)
from Legendre import AssociatedLegendre


vf = ast.VectorFunctions
Args = vf.Arguments



# ============================================================
# PINES GRAVITY VF
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


        self._acc_vf = (
            self._build_acceleration()
        )


        self._ode_vf = (
            self._build_ode()
        )



    # ========================================================
    # Build acceleration
    # ========================================================

    cdef object _build_acceleration(self):


        X = Args(6)


        rvec = X.head3()


        x = rvec[0]
        y = rvec[1]
        z = rvec[2]



        #
        # Radius
        #

        r = (
            x*x
            +
            y*y
            +
            z*z
        ).sqrt()



        #
        # spherical coordinates
        #

        sinphi = z/r


        cosphi = (

            (
                x*x
                +
                y*y
            ).sqrt()

            /

            r

        )


        rho = (

            X.Constant(self.radius)

            /

            r

        )



        #
        # longitude
        #

        xy = (

            x*x
            +
            y*y

        ).sqrt()



        coslam = x/xy

        sinlam = y/xy



        #
        # recurrence
        #

        Ctrig, Strig = HarmonicTrig(
            coslam,
            sinlam,
            self.degree
        )



        P = AssociatedLegendre(
            sinphi,
            self.degree
        )



        #
        # Pines harmonic summation and its spherical gradient.
        #
        # U = mu/r * [1 + sum(rho**n * P[n,m] *
        #                    (C[n,m] cos(m*lambda) + S[n,m] sin(m*lambda)))]
        #
        # The associated Legendre recurrence used by this package is the
        # unnormalised, no-Condon-Shortley convention.  Consequently, Cnm and
        # Snm must use that same convention (for example C[2,0] = -J2).
        #

        zero = X.Constant(0.0)
        one = X.Constant(1.0)
        radial_sum = one
        latitude_sum = zero
        longitude_sum = zero

        cdef int n
        cdef int m
        cdef object rho_n
        cdef object trig
        cdef object dP_dphi

        for n in range(2, self.degree + 1):

            rho_n = rho ** n

            for m in range(n + 1):

                trig = (
                    X.Constant(float(self.Cnm[n, m])) * Ctrig[m]
                    +
                    X.Constant(float(self.Snm[n, m])) * Strig[m]
                )

                # dP[n,m](sin(phi))/dphi.  P[n-1,m] is zero when m == n.
                if m == n:
                    dP_dphi = -n * sinphi * P[n, m] / cosphi
                else:
                    dP_dphi = (
                        (n + m) * P[n - 1, m]
                        - n * sinphi * P[n, m]
                    ) / cosphi

                radial_sum = radial_sum + (n + 1) * rho_n * P[n, m] * trig
                latitude_sum = latitude_sum + rho_n * dP_dphi * trig
                longitude_sum = longitude_sum + (
                    rho_n * P[n, m] * m * (
                        X.Constant(float(self.Snm[n, m])) * Ctrig[m]
                        - X.Constant(float(self.Cnm[n, m])) * Strig[m]
                    )
                )

        mu_over_r2 = X.Constant(self.mu) / (r * r)

        ar = -mu_over_r2 * radial_sum
        aph = mu_over_r2 * latitude_sum
        alam = mu_over_r2 * longitude_sum / cosphi

        ax, ay, az = SphericalToCartesian(
            ar, aph, alam, sinphi, cosphi, sinlam, coslam
        )

        return vf.stack([ax, ay, az])



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
            f"PinesGravityVF("
            f"degree={self.degree}, "
            f"mu={self.mu})"
        )
