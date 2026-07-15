# ============================================================
# FILE: Legendre.pyx
#
# Shared Legendre recursion for:
#   Classical SH
#   Pines
#
# Flat storage:
#
# idx = n*(N+1)+m
#
# ============================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


cimport cython



cpdef tuple AssociatedLegendreGradient(
    object sinphi,
    int N
):

    cdef int n
    cdef int m
    cdef int idx


    size = (N+1)*(N+1)


    P = [None]*size
    dP = [None]*size


    zero = sinphi*0.0
    one = zero + 1.0


    cosphi = (
        one -
        sinphi*sinphi
    ).sqrt()



    #
    # P00
    #

    P[0] = one

    dP[0] = zero



    #
    # diagonal
    #

    for n in range(1,N+1):

        idx = n*(N+1)+n

        P[idx] = (
            (2*n-1)
            *
            cosphi
            *
            P[(n-1)*(N+1)+(n-1)]
        )

        dP[idx] = (
            -n*sinphi*P[idx]/cosphi
        )



    #
    # first off diagonal
    #

    for n in range(1,N+1):

        idx = n*(N+1)+(n-1)

        P[idx] = (
            (2*n-1)
            *
            sinphi
            *
            P[(n-1)*(N+1)+(n-1)]
        )


        dP[idx] = (
            (
                n*sinphi*P[idx]
                -
                (n+n-1)*P[(n-1)*(N+1)+(n-1)]
            )
            /
            cosphi
        )



    #
    # remaining terms
    #

    for n in range(2,N+1):

        for m in range(0,n-1):

            idx = n*(N+1)+m


            P[idx] = (

                (
                    (2*n-1)
                    *
                    sinphi
                    *
                    P[(n-1)*(N+1)+m]
                )

                -

                (
                    (n+m-1)
                    *
                    P[(n-2)*(N+1)+m]
                )

            )/(n-m)


            dP[idx] = (

                (
                    n*sinphi*P[idx]
                    -
                    (n+m)
                    *
                    P[(n-1)*(N+1)+m]
                )

                /

                cosphi

            )


    return P,dP