# ============================================================
# FILE: PinesMath.pyx
#
# Pines-style gravity math utilities
#
# Demo implementation:
#   - trigonometric recurrence
#   - Legendre recursion
#   - harmonic summation
#   - spherical -> Cartesian conversion
#
# ASSET VectorFunction compatible
#
# ============================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


cimport cython


# ============================================================
# Trigonometric recurrence
# ============================================================

cpdef tuple HarmonicTrig(
    object coslam,
    object sinlam,
    int N
):

    cdef int m

    C = [None]*(N+1)
    S = [None]*(N+1)


    # ScalarFunction does not expose ``Constant``.  Construct symbolic
    # constants from the supplied expression instead, preserving its input
    # dimension and automatic-differentiation context.
    zero = coslam * 0.0
    one = zero + 1.0


    C[0] = one
    S[0] = zero


    if N == 0:
        return C,S


    C[1] = coslam
    S[1] = sinlam


    for m in range(2,N+1):

        C[m] = (
            coslam*C[m-1]
            -
            sinlam*S[m-1]
        )

        S[m] = (
            sinlam*C[m-1]
            +
            coslam*S[m-1]
        )


    return C,S



# ============================================================
# Associated Legendre recursion
#
# Demo normalized-like recurrence
# ============================================================

cpdef dict AssociatedLegendre(
    object sinphi,
    int N
):

    cdef int n
    cdef int m


    P = {}


    one = sinphi * 0.0 + 1.0


    cosphi = (
        one
        -
        sinphi*sinphi
    ).sqrt()



    P[0,0] = one



    if N == 0:
        return P



    #
    # diagonal
    #

    for n in range(1,N+1):

        P[n,n] = (

            (2*n-1)

            *

            cosphi

            *

            P[n-1,n-1]

        )



    #
    # first off diagonal
    #

    for n in range(1,N+1):

        P[n,n-1] = (

            (2*n-1)

            *

            sinphi

            *

            P[n-1,n-1]

        )



    #
    # remaining
    #

    for n in range(2,N+1):

        for m in range(0,n-1):

            P[n,m] = (

                (
                    (2*n-1)
                    *
                    sinphi
                    *
                    P[n-1,m]
                )

                -

                (
                    (n+m-1)
                    *
                    P[n-2,m]
                )

            )/(n-m)



    return P



# ============================================================
# Harmonic summation
# ============================================================

cpdef object HarmonicPotential(
    dict P,
    list Ctrig,
    list Strig,
    object Cnm,
    object Snm,
    int degree,
    object rho
):

    cdef int n
    cdef int m


    result = P[0,0].Constant(0.0)



    for n in range(2,degree+1):

        scale = rho**n


        for m in range(0,n+1):

            coeff = (

                Cnm[n,m]*Ctrig[m]

                +

                Snm[n,m]*Strig[m]

            )


            result += (

                scale

                *

                coeff

                *

                P[n,m]

            )


    return result



# ============================================================
# Spherical acceleration conversion
# ============================================================

cpdef tuple SphericalToCartesian(
    object ar,
    object aph,
    object alam,
    object sinphi,
    object cosphi,
    object sinlam,
    object coslam
):


    ax = (

        cosphi*coslam*ar

        -

        sinphi*coslam*aph

        -

        sinlam*alam

    )


    ay = (

        cosphi*sinlam*ar

        -

        sinphi*sinlam*aph

        +

        coslam*alam

    )


    az = (

        sinphi*ar

        +

        cosphi*aph

    )


    return ax,ay,az
