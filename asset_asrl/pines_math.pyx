# ============================================================
# FILE: PinesMath.pyx
#
# Pines gravity math utilities
#
# ASSET VectorFunction compatible
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


cimport cython



# ============================================================
# Harmonic trigonometric recurrence
#
# Computes:
#
# cos(m lambda)
# sin(m lambda)
#
# ============================================================


cpdef tuple HarmonicTrig(
    object coslam,
    object sinlam,
    int N
):

    cdef int m


    C = [None]*(N+1)
    S = [None]*(N+1)



    #
    # symbolic constants
    #

    zero = coslam*0.0
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
# Spherical acceleration conversion
#
# Inputs:
#
# ar      radial acceleration
# aph     latitude acceleration
# alam    longitude acceleration
#
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