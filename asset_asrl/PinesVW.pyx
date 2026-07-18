# ============================================================
# FILE: PinesVW.pyx
#
# Pines (1973) Cartesian harmonic recursion
#
# Generates:
#
#       V(n,m)
#       W(n,m)
#
# Normalized Cartesian coordinates:
#
#       X=x/R
#       Y=y/R
#       Z=z/R
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


cimport cython



cpdef tuple BuildVW(
    object x,
    object y,
    object z,
    double radius,
    int degree):

    cdef int n
    cdef int m

    V = {}
    W = {}

    zero = x*0.0
    one = zero + 1.0

    #
    # normalized coordinates
    #

    X = x/radius
    Y = y/radius
    Z = z/radius



    #
    # n=0
    #

    V[0,0] = one
    W[0,0] = zero



    if degree == 0:

        return V,W



    #
    # n=1
    #

    V[1,0] = Z
    W[1,0] = zero


    V[1,1] = X
    W[1,1] = Y



    #
    # Sectorial recursion
    #
    # m=n
    #

    for n in range(2,degree+1):


        V[n,n] = (

            (2*n-1)

            *

            (
                X*V[n-1,n-1]
                -
                Y*W[n-1,n-1]
            )

        )


        W[n,n] = (

            (2*n-1)

            *

            (
                X*W[n-1,n-1]
                +
                Y*V[n-1,n-1]
            )

        )



    #
    # Tesseral recursion
    #
    # 0 <= m < n
    #

    for n in range(2,degree+1):


        for m in range(0,n):



            #
            # first off diagonal
            #
            # n-1,m does not exist
            #

            if (n-2,m) not in V:


                V[n,m] = (

                    (2*n-1)

                    *

                    Z

                    *

                    V[n-1,m]

                )


                W[n,m] = (

                    (2*n-1)

                    *

                    Z

                    *

                    W[n-1,m]

                )



            else:


                V[n,m] = (

                    (
                        (2*n-1)
                        *
                        Z
                        *
                        V[n-1,m]
                    )

                    -

                    (
                        (n+m-1)
                        *
                        V[n-2,m]
                    )

                )

                V[n,m] /= (n-m)



                W[n,m] = (

                    (
                        (2*n-1)
                        *
                        Z
                        *
                        W[n-1,m]
                    )

                    -

                    (
                        (n+m-1)
                        *
                        W[n-2,m]
                    )

                )

                W[n,m] /= (n-m)



    #
    # Restore missing diagonal zero terms
    #

    for n in range(2,degree+1):

        if (n,0) not in V:

            V[n,0] = zero
            W[n,0] = zero



    return V,W