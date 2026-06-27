# ============================================================
# trig recurrence
# ============================================================

import asset_asrl as ast

vf = ast.VectorFunctions



def HarmonicTrig(
        coslam,
        sinlam,
        N
):

    C = {}
    S = {}


    C[0] = 1
    S[0] = 0


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