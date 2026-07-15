# ============================================================
# FILE: PinesVW.pyx
#
# Pines (1973) Cartesian harmonic recursion
#
# Returns:
#
#   V,W
#   Vx,Vy,Vz
#   Wx,Wy,Wz
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
    int degree
):


    cdef int n
    cdef int m


    V = {}
    W = {}

    Vx = {}
    Vy = {}
    Vz = {}

    Wx = {}
    Wy = {}
    Wz = {}



    zero = x*0.0
    one = zero + 1.0



    #
    # normalized coordinates
    #

    X = x / radius
    Y = y / radius
    Z = z / radius



    #
    # degree zero
    #

    V[0,0] = one
    W[0,0] = zero



    Vx[0,0] = zero
    Vy[0,0] = zero
    Vz[0,0] = zero

    Wx[0,0] = zero
    Wy[0,0] = zero
    Wz[0,0] = zero



    if degree == 0:

        return (
            V,W,
            Vx,Vy,Vz,
            Wx,Wy,Wz
        )



    #
    # degree one
    #

    V[1,0] = Z
    W[1,0] = zero

    V[1,1] = X
    W[1,1] = Y



    #
    # first derivatives
    #

    Vx[1,0] = zero
    Vy[1,0] = zero
    Vz[1,0] = 1.0/radius


    Wx[1,0] = zero
    Wy[1,0] = zero
    Wz[1,0] = zero



    Vx[1,1] = 1.0/radius
    Vy[1,1] = zero
    Vz[1,1] = zero


    Wx[1,1] = zero
    Wy[1,1] = 1.0/radius
    Wz[1,1] = zero



    #
    # Pines recursion
    #

    for n in range(2,degree+1):


        #
        # sectorial
        #

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
        # derivatives of sectorial terms
        #

        Vx[n,n] = (

            (2*n-1)
            *
            (
                (1/radius)*V[n-1,n-1]
                +
                X*Vx[n-1,n-1]
                -
                Y*Wx[n-1,n-1]
            )

        )


        Vy[n,n] = (

            (2*n-1)
            *
            (
                X*Vy[n-1,n-1]
                -
                (1/radius)*W[n-1,n-1]
                -
                Y*Wy[n-1,n-1]
            )

        )


        Vz[n,n] = (

            (2*n-1)
            *
            (
                X*Vz[n-1,n-1]
                -
                Y*Wz[n-1,n-1]
            )

        )



        Wx[n,n] = (

            (2*n-1)
            *
            (
                (1/radius)*W[n-1,n-1]
                +
                X*Wx[n-1,n-1]
                +
                Y*Vx[n-1,n-1]
            )

        )


        Wy[n,n] = (

            (2*n-1)
            *
            (
                X*Wy[n-1,n-1]
                +
                (1/radius)*V[n-1,n-1]
                +
                Y*Vy[n-1,n-1]
            )

        )


        Wz[n,n] = (

            (2*n-1)
            *
            (
                X*Wz[n-1,n-1]
                +
                Y*Vz[n-1,n-1]
            )

        )



    #
    # tesseral terms
    #

    for n in range(2,degree+1):

        for m in range(0,n):


            if (n-1,m) not in V:
                continue


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



            Vx[n,m] = (
                (2*n-1)
                *
                Z
                *
                Vx[n-1,m]
            )

            Vy[n,m] = (
                (2*n-1)
                *
                Z
                *
                Vy[n-1,m]
            )

            Vz[n,m] = (
                (2*n-1)
                *
                (
                    (1/radius)*V[n-1,m]
                    +
                    Z*Vz[n-1,m]
                )
            )



            Wx[n,m] = (
                (2*n-1)
                *
                Z
                *
                Wx[n-1,m]
            )

            Wy[n,m] = (
                (2*n-1)
                *
                Z
                *
                Wy[n-1,m]
            )

            Wz[n,m] = (
                (2*n-1)
                *
                (
                    (1/radius)*W[n-1,m]
                    +
                    Z*Wz[n-1,m]
                )
            )



    return (
        V,
        W,
        Vx,
        Vy,
        Vz,
        Wx,
        Wy,
        Wz
    )