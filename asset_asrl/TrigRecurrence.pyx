# ============================================================
# FILE: TrigRecurrence.pyx
#
# Pines (1973) Trigonometric Recurrence
#
# Computes
#
#     C[m] = cos(mλ)
#     S[m] = sin(mλ)
#
# using
#
#     C[m] = cosλ*C[m-1] - sinλ*S[m-1]
#     S[m] = sinλ*C[m-1] + cosλ*S[m-1]
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
cpdef tuple HarmonicTrig(
    object coslam,
    object sinlam,
    int N
):
    """
    Build the trigonometric recurrence used by the
    Pines spherical harmonic gravity algorithm.

    Parameters
    ----------
    coslam
        cos(lambda)

    sinlam
        sin(lambda)

    N
        Maximum harmonic order.

    Returns
    -------
    C
        C[m] = cos(m*lambda)

    S
        S[m] = sin(m*lambda)
    """

    cdef int m

    if N < 0:
        raise ValueError("N must be non-negative.")

    cdef list C = [None] * (N + 1)
    cdef list S = [None] * (N + 1)

    one = coslam.Constant(1.0)
    zero = coslam.Constant(0.0)

    #
    # m = 0
    #

    C[0] = one
    S[0] = zero

    if N == 0:
        return C, S

    #
    # m = 1
    #

    C[1] = coslam
    S[1] = sinlam

    #
    # Higher-order recurrence
    #

    for m in range(2, N + 1):

        C[m] = (
            coslam * C[m - 1]
            -
            sinlam * S[m - 1]
        )

        S[m] = (
            sinlam * C[m - 1]
            +
            coslam * S[m - 1]
        )

    return C, S