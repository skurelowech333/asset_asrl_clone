# ============================================================
# FILE: Legendre.pyx
#
# Fully normalized associated Legendre recursion
#
# Pines (1973) gravity support
#
# Computes:
#
#       P[n,m]
#
# where P are fully normalized associated Legendre functions
#
# Compatible with ASSET VectorFunctions
#
# ============================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict AssociatedLegendre(
    object sinphi,
    int N
):
    """
    Compute fully normalized associated Legendre functions.

    Parameters
    ----------
    sinphi :
        sin(latitude)

    N :
        Maximum degree

    Returns
    -------
    P :
        dictionary

        P[n,m] = fully normalized Legendre function
    """

    cdef int n
    cdef int m


    P = {}


    #
    # symbolic constants
    #

    # ScalarFunction instances do not have a Constant method.  Build
    # constants algebraically so the recurrence remains an ASSET expression.
    zero = sinphi * 0.0
    one = zero + 1.0


    cosphi = (
        one
        -
        sinphi*sinphi
    ).sqrt()



    #
    # Base term
    #

    P[0,0] = one



    if N == 0:
        return P



    #
    # Diagonal recursion
    #
    # P(n,n)
    #

    for n in range(1, N+1):

        P[n,n] = (
            (2*n-1)
            *
            cosphi
            *
            P[n-1,n-1]
        )



    #
    # First off diagonal
    #
    # P(n,n-1)
    #

    for n in range(1, N+1):

        P[n,n-1] = (
            (2*n-1)
            *
            sinphi
            *
            P[n-1,n-1]
        )



    #
    # Remaining recursion
    #
    # P(n,m)
    #

    for n in range(2, N+1):

        for m in range(0, n-1):

            a = (
                (2*n-1)
                *
                sinphi
                *
                P[n-1,m]
            )


            b = (
                (n+m-1)
                *
                P[n-2,m]
            )


            P[n,m] = (
                a-b
            )/(n-m)



    return P
