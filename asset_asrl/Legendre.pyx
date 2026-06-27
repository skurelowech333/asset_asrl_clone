# ============================================================
# Legendre.pyx
# ============================================================

import asset_asrl as ast

vf = ast.VectorFunctions


def AssociatedLegendre(
        sinphi,
        N
):

    P = {}

    P[0,0] = 1.0

    c = vf.sqrt(
        1.0 - sinphi*sinphi
    )


    if N == 0:
        return P


    # Diagonal Terms
    for n in range(1,N+1):

        P[n,n] = (
            (2*n-1)
            *
            c
            *
            P[n-1,n-1]
        )


    # first off diagonal
    # P(n,n-1)
    for n in range(1,N+1):

        P[n,n-1] = (
            (2*n-1)
            *
            sinphi
            *
            P[n-1,n-1]
        )

    # remaining
    for n in range(2,N+1):

        for m in range(0,n-1):

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