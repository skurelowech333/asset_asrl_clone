# ============================================================
# FILE: test_pines_j2.py
#
# Validate:
#
#   1. Analytical J2 acceleration
#   2. Classical spherical harmonic J2
#   3. Pines Cartesian J2
#
# Plots:
#
#   - acceleration magnitude comparison
#   - error comparison
#
# ============================================================


import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast


from GravitySphericalHarmonic import (
    SphericalHarmonicGravityVF
)

from GravityPines import (
    PinesGravityVF
)



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



    C[2,0] = -J2 / np.sqrt(5)

    S[2,0] = 0.0



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
        r,
        r
    )


    rmag = np.sqrt(r2)



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

        5*z*z/r2
        -
        1

    )


    zz = (

        5*z*z/r2
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

def eval_acc(model,state):


    result = model.vf().compute(

        np.asarray(
            state,
            dtype=float
        )

    )


    return np.asarray(
        result,
        dtype=float
    )



# ============================================================
# Acceleration comparison plot
# ============================================================

def plot_acceleration_comparison(
        labels,
        truth,
        sh,
        pines
):


    plt.style.use(
        "dark_background"
    )


    fig,ax = plt.subplots(
        figsize=(11,6)
    )



    x = np.arange(
        len(labels)
    )


    truth_mag = np.linalg.norm(
        truth,
        axis=1
    )


    sh_mag = np.linalg.norm(
        sh,
        axis=1
    )


    pines_mag = np.linalg.norm(
        pines,
        axis=1
    )



    ax.plot(
        x,
        truth_mag,
        marker="o",
        linewidth=3,
        markersize=9,
        label="Analytical J2"
    )


    ax.plot(
        x,
        sh_mag,
        marker="s",
        linewidth=3,
        markersize=9,
        label="Spherical Harmonic"
    )


    ax.plot(
        x,
        pines_mag,
        marker="^",
        linewidth=3,
        markersize=9,
        label="Pines Cartesian"
    )



    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_ylabel(
        "Acceleration Magnitude (km/s²)",
        fontsize=14
    )


    ax.set_xlabel(
        "Test Position",
        fontsize=14
    )


    ax.set_title(
        "J2 Acceleration Validation",
        fontsize=18,
        weight="bold"
    )


    ax.grid(
        True,
        alpha=0.25
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        "j2_acceleration_comparison.png",
        dpi=300
    )


    plt.show()



# ============================================================
# Error plot
# ============================================================

def plot_error_comparison(
        labels,
        sh_error,
        pines_error,
        difference
):


    plt.style.use(
        "dark_background"
    )


    fig,ax = plt.subplots(
        figsize=(11,6)
    )


    x=np.arange(
        len(labels)
    )



    ax.semilogy(
        x,
        sh_error,
        marker="o",
        linewidth=3,
        markersize=9,
        label="SH Error"
    )


    ax.semilogy(
        x,
        pines_error,
        marker="s",
        linewidth=3,
        markersize=9,
        label="Pines Error"
    )


    ax.semilogy(
        x,
        difference,
        marker="^",
        linewidth=3,
        markersize=9,
        label="SH-Pines"
    )



    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        labels
    )


    ax.set_ylabel(
        "Error (km/s²)",
        fontsize=14
    )


    ax.set_xlabel(
        "Test Position",
        fontsize=14
    )


    ax.set_title(
        "J2 Kernel Error Comparison",
        fontsize=18,
        weight="bold"
    )


    ax.grid(
        True,
        which="both",
        alpha=0.25
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        "j2_error_comparison.png",
        dpi=300
    )


    plt.show()



# ============================================================
# Main
# ============================================================

def main():


    degree = 2



    C,S = j2_coefficients()



    print("Building models...")



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



    states = [

        [
            7000,
            0,
            0,
            0,
            7.5,
            1
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



    labels=[]

    truth_values=[]

    sh_values=[]

    pines_values=[]

    sh_errors=[]

    pines_errors=[]

    differences=[]



    print("="*60)
    print("J2 VALIDATION")
    print("="*60)



    for i,state in enumerate(states):


        r=np.asarray(
            state[:3]
        )


        truth=j2_acceleration(
            r
        )


        sh_acc=eval_acc(
            sh,
            state
        )


        pines_acc=eval_acc(
            pines,
            state
        )



        truth_values.append(
            truth
        )


        sh_values.append(
            sh_acc
        )


        pines_values.append(
            pines_acc
        )



        sh_errors.append(
            np.linalg.norm(
                sh_acc-truth
            )
        )


        pines_errors.append(
            np.linalg.norm(
                pines_acc-truth
            )
        )


        differences.append(
            np.linalg.norm(
                sh_acc-pines_acc
            )
        )



        labels.append(
            f"R{i+1}"
        )



        print()
        print("Position:",r)

        print("Truth:",truth)

        print("SH:",sh_acc)

        print("Pines:",pines_acc)



    plot_acceleration_comparison(

        labels,
        np.array(truth_values),
        np.array(sh_values),
        np.array(pines_values)

    )


    plot_error_comparison(

        labels,
        sh_errors,
        pines_errors,
        differences

    )



if __name__=="__main__":

    main()