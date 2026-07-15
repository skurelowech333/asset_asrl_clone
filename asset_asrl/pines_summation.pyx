# ============================================================
# FILE: PinesSummation.pyx
#
# Pines (1973) Gravity Harmonic Summation
#
# Computes spherical harmonic gravity terms from:
#
#       P[n,m]
#       C[n,m]
#       S[n,m]
#       cos(m lambda)
#       sin(m lambda)
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
cpdef tuple PinesSum(
    object P,
    object Ctrig,
    object Strig,
    object Cnm,
    object Snm,
    int degree
):
    """
    Compute Pines spherical harmonic sums.

    Parameters
    ----------
    P :
        Normalized Legendre functions

    Ctrig :
        cos(m lambda)

    Strig :
        sin(m lambda)

    Cnm :
        cosine gravity coefficients

    Snm :
        sine gravity coefficients

    degree :
        maximum degree


    Returns
    -------
    V :
        potential summation

    dVdr :
        radial derivative

    dVdphi :
        latitude derivative

    dVdlambda :
        longitude derivative
    """


    cdef int n
    cdef int m



    #
    # symbolic zero
    #

    zero = P[0,0] * 0.0


    V = zero
    dVdr = zero
    dVdphi = zero
    dVdlambda = zero



    #
    # Harmonic summation
    #

    for n in range(2, degree+1):

        for m in range(0, n+1):


            coeff = (
                Cnm[n,m] * Ctrig[m]
                +
                Snm[n,m] * Strig[m]
            )



            #
            # potential term
            #

            term = (
                coeff
                *
                P[n,m]
            )


            V = (
                V
                +
                term
            )



            #
            # longitude derivative
            #
            # d/dlambda
            #

            dterm_lambda = (

                m
                *
                (
                    Snm[n,m] * Ctrig[m]
                    -
                    Cnm[n,m] * Strig[m]
                )
                *
                P[n,m]

            )


            dVdlambda = (
                dVdlambda
                +
                dterm_lambda
            )



    return (
        V,
        dVdr,
        dVdphi,
        dVdlambda
    )
