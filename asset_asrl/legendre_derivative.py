# ============================================================
# FILE: LegendreDerivative.pyx
#
# Pines (1973) Gravity Support
#
# Computes latitude derivatives:
#
#       dP[n,m]/dphi
#
# for fully normalized associated Legendre functions.
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
cpdef dict LegendreDerivative(
    object P,
    object sinphi,
    object cosphi,
    int N
):
    """
    Compute derivatives of normalized associated
    Legendre functions with respect to latitude.

    Parameters
    ----------
    P :
        Legendre functions

    sinphi :
        sin(latitude)

    cosphi :
        cos(latitude)

    N :
        maximum degree


    Returns
    -------
    dP :

        dP[n,m] =
            d(P[n,m])/dphi
    """


    cdef int n
    cdef int m


    dP = {}



    #
    # symbolic zero
    #

    zero = sinphi.Constant(0.0)



    #
    # Base derivative
    #

    dP[0,0] = zero



    #
    # Recurrence
    #
    # dP(n,m)
    #
    # = n*tan(phi)*P(n,m)
    #
    #   -
    #   ((n+m)/(cos(phi))) P(n-1,m)
    #
    #

    for n in range(1,N+1):

        for m in range(0,n+1):


            if n-1 >= m:

                dP[n,m] = (

                    n
                    *
                    sinphi
                    /
                    cosphi
                    *
                    P[n,m]

                    -

                    (
                        n+m
                    )
                    /
                    cosphi
                    *
                    P[n-1,m]

                )


            else:

                #
                # diagonal terms
                #

                dP[n,m] = (

                    n
                    *
                    sinphi
                    /
                    cosphi
                    *
                    P[n,m]

                )



    return dP