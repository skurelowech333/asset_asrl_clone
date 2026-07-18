# ============================================================
# FILE: PinesDerivatives.pyx
#
# Pines Cartesian harmonic derivatives
#
# Generates:
#
#   dV/dx,dV/dy,dV/dz
#   dW/dx,dW/dy,dW/dz
#
# Compatible with normalized PinesVW recursion
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


cimport cython



cpdef tuple BuildDerivatives(
    object V,
    object W,
    object x,
    object y,
    object z,
    double radius,
    int degree
):


    cdef int n
    cdef int m


    Vx={}
    Vy={}
    Vz={}

    Wx={}
    Wy={}
    Wz={}



    zero = x*0.0


    invR = 1.0/radius



    #
    # normalized coordinates
    #

    X = x*invR
    Y = y*invR
    Z = z*invR



    #
    # initialize all terms
    #

    for n in range(degree+1):

        for m in range(n+1):

            Vx[n,m]=zero
            Vy[n,m]=zero
            Vz[n,m]=zero

            Wx[n,m]=zero
            Wy[n,m]=zero
            Wz[n,m]=zero



    #
    # degree 0
    #

    if degree == 0:

        return (
            Vx,Vy,Vz,
            Wx,Wy,Wz
        )



    #
    # degree 1
    #

    Vz[1,0]=invR


    Vx[1,1]=invR


    Wy[1,1]=invR



    #
    # sectorial recursion
    #
    # m=n
    #

    for n in range(2,degree+1):


        Vx[n,n] = (

            (2*n-1)

            *

            (
                invR*V[n-1,n-1]

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

                invR*W[n-1,n-1]

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

                invR*W[n-1,n-1]

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

                invR*V[n-1,n-1]

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
    # tesseral recursion
    #
    # m < n
    #

    for n in range(2,degree+1):


        for m in range(n):


            denom = n-m



            #
            # first off diagonal
            #

            if (n-2,m) not in V:


                Vx[n,m] = (

                    (2*n-1)

                    *

                    (

                        Z*Vx[n-1,m]

                    )

                )


                Vy[n,m] = (

                    (2*n-1)

                    *

                    (

                        Z*Vy[n-1,m]

                    )

                )


                Vz[n,m] = (

                    (2*n-1)

                    *

                    (

                        invR*V[n-1,m]

                        +

                        Z*Vz[n-1,m]

                    )

                )



                Wx[n,m] = (

                    (2*n-1)

                    *

                    Z*Wx[n-1,m]

                )


                Wy[n,m] = (

                    (2*n-1)

                    *

                    Z*Wy[n-1,m]

                )


                Wz[n,m] = (

                    (2*n-1)

                    *

                    (

                        invR*W[n-1,m]

                        +

                        Z*Wz[n-1,m]

                    )

                )



            else:


                Vx[n,m] = (

                    (

                        (2*n-1)
                        *
                        Z
                        *
                        Vx[n-1,m]

                    )

                    -

                    (

                        (n+m-1)
                        *
                        Vx[n-2,m]

                    )

                ) / denom



                Vy[n,m] = (

                    (

                        (2*n-1)
                        *
                        Z
                        *
                        Vy[n-1,m]

                    )

                    -

                    (

                        (n+m-1)
                        *
                        Vy[n-2,m]

                    )

                ) / denom



                Vz[n,m] = (

                    (

                        (2*n-1)
                        *
                        (
                            invR*V[n-1,m]
                            +
                            Z*Vz[n-1,m]
                        )

                    )

                    -

                    (

                        (n+m-1)
                        *
                        Vz[n-2,m]

                    )

                ) / denom



                Wx[n,m] = (

                    (

                        (2*n-1)
                        *
                        Z
                        *
                        Wx[n-1,m]

                    )

                    -

                    (

                        (n+m-1)
                        *
                        Wx[n-2,m]

                    )

                ) / denom



                Wy[n,m] = (

                    (

                        (2*n-1)
                        *
                        Z
                        *
                        Wy[n-1,m]

                    )

                    -

                    (

                        (n+m-1)
                        *
                        Wy[n-2,m]

                    )

                ) / denom



                Wz[n,m] = (

                    (

                        (2*n-1)
                        *
                        (
                            invR*W[n-1,m]
                            +
                            Z*Wz[n-1,m]
                        )

                    )

                    -

                    (

                        (n+m-1)
                        *
                        Wz[n-2,m]

                    )

                ) / denom



    return (

        Vx,Vy,Vz,

        Wx,Wy,Wz

    )