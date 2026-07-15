# ============================================================
# FILE: test_pines_j2.py
#
# Validate:
#   1. Classical spherical harmonic J2
#   2. Pines Cartesian J2
#   3. Analytical J2 acceleration
#
# ============================================================


import numpy as np
import asset_asrl as ast


from GravitySphericalHarmonic import (
    SphericalHarmonicGravityVF
)

from GravityPinesJ2Only import (
    PinesGravityVF
)



vf = ast.VectorFunctions



# ============================================================
# Constants
# ============================================================

MU = 398600.435507

R = 6378.137



J2 = 1.08262668e-3



# ============================================================
# J2 coefficients
# ============================================================

def j2_coefficients():


    C = {}

    S = {}



    #
    # zero all coefficients
    #

    C[2,0] = -J2 / np.sqrt(5)

    S[2,0] = 0.0



    #
    # required zero terms
    #

    for n in range(3):

        for m in range(n+1):

            if (n,m) not in C:

                C[n,m] = 0.0

            if (n,m) not in S:

                S[n,m] = 0.0



    return C,S



# ============================================================
# Analytical J2 acceleration
# ============================================================

def j2_acceleration(r):


    x,y,z = r


    r2 = np.dot(
        r,r
    )


    rmag = np.sqrt(r2)


    z2 = z*z



    factor = (

        -MU
        /
        rmag**3

    )



    common = (

        1.5
        *
        J2
        *
        (R/rmag)**2

    )



    zx = (

        5*z2/r2
        -
        1

    )


    zz = (

        5*z2/r2
        -
        3

    )



    ax = factor*x*(1-common*zx)

    ay = factor*y*(1-common*zx)

    az = factor*z*(1-common*zz)



    return np.array(
        [
            ax,
            ay,
            az
        ]
    )



# ============================================================
# ASSET evaluator
# ============================================================
def eval_acc(model, state):

    state = np.asarray(
        state,
        dtype=np.float64
    )


    vf = model.vf()


    result = vf.compute(
        state
    )


    return np.asarray(
        result,
        dtype=float
    )


# ============================================================
# Main
# ============================================================

def main():


    degree = 2



    C,S = j2_coefficients()



    print()
    print("Building models...")
    print()



    sh = SphericalHarmonicGravityVF(

        MU,
        R,
        C,
        S,
        degree

    )



    pines = PinesGravityVF(

        MU,
        R,
        C,
        S,
        degree

    )



    #
    # Test states
    #

    states = [

        [
            7000,
            0,
            0,
            0,
            7.5,
            1.0
        ],


        [
            7000,
            1000,
            2000,
            0,
            7.5,
            0
        ],


        [
            8000,
            -2000,
            3000,
            0,
            7,
            1
        ]

    ]



    print("="*60)
    print("J2 VALIDATION")
    print("="*60)



    for state in states:


        r = np.asarray(
            state[:3]
        )


        truth = j2_acceleration(
            r
        )


        sh_acc = eval_acc(
            sh,
            state
        )


        pines_acc = eval_acc(
            pines,
            state
        )



        print()
        print("Position:")
        print(r)


        print()
        print("Analytical J2:")
        print(truth)


        print()
        print("SH:")
        print(sh_acc)


        print()
        print("Pines:")
        print(pines_acc)



        print()
        print("SH error:")
        print(
            np.linalg.norm(
                sh_acc-truth
            )
        )


        print("Pines error:")
        print(
            np.linalg.norm(
                pines_acc-truth
            )
        )


        print("SH-Pines:")
        print(
            np.linalg.norm(
                sh_acc-pines_acc
            )
        )

        print("-"*60)




if __name__ == "__main__":

    main()