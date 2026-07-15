# ============================================================
# FILE:
# benchmark_gravity_kernels.py
#
# Fair comparison:
#   - Classical spherical harmonic gravity
#   - Pines gravity kernel
#
# ASSET VectorFunction evaluation
#
# Tests:
#   1. Build time
#   2. Acceleration validation
#   3. Acceleration timing
#   4. Degree scaling
#   5. Propagation timing
#
# ============================================================


import time
import numpy as np
import matplotlib.pyplot as plt


import asset_asrl as ast

from GravitySphericalHarmonic import SphericalHarmonicGravityVF
from GravityPines import PinesGravityVF


# ============================================================
# Constants
# ============================================================


MU = 398600.435507

R = 6378.137


STATE = np.array(
    [
        7000.0,
        0.0,
        0.0,
        0.0,
        7.5,
        1.0
    ],
    dtype=float
)



# ============================================================
# Gravity coefficients
#
# SAME coefficients for both models
# ============================================================


def make_coefficients(N):

    Cnm = np.zeros(
        (N+1,N+1)
    )

    Snm = np.zeros(
        (N+1,N+1)
    )


    # Earth J2
    Cnm[2,0] = -1.08262668e-3


    return Cnm,Snm



# ============================================================
# Builders
# ============================================================


def build_sh(degree):

    Cnm,Snm = make_coefficients(
        degree
    )


    return SphericalHarmonicGravityVF(
        MU,
        R,
        Cnm,
        Snm,
        degree
    )



def build_pines(degree):

    Cnm,Snm = make_coefficients(
        degree
    )


    return PinesGravityVF(
        MU,
        R,
        Cnm,
        Snm,
        degree
    )



# ============================================================
# ASSET evaluation
# ============================================================


def eval_acc(model,state):

    vf = model.vf()


    # ASSET VectorFunction evaluation
    result = vf.compute(
        state
    )


    return np.asarray(
        result,
        dtype=float
    )



# ============================================================
# Build benchmark
# ============================================================


def benchmark_build(builder,degree):

    t0=time.perf_counter()

    model=builder(degree)

    t1=time.perf_counter()


    return model,(t1-t0)*1e6



# ============================================================
# Timing
# ============================================================


def benchmark_eval(model,state,N=1000):


    vf=model.vf()


    # warmup
    vf.compute(state)



    t0=time.perf_counter()


    for _ in range(N):

        vf.compute(
            state
        )


    t1=time.perf_counter()


    return (
        (t1-t0)
        /
        N
        *
        1e6
    )



# ============================================================
# Validation
# ============================================================


def validate(sh,pines):


    a1=eval_acc(
        sh,
        STATE
    )


    a2=eval_acc(
        pines,
        STATE
    )


    error=np.linalg.norm(
        a1-a2
    )


    relative=(
        error
        /
        np.linalg.norm(a1)
    )


    print("\nValidation")
    print("----------------------")


    print("SH acceleration:")
    print(a1)


    print("\nPines acceleration:")
    print(a2)


    print()


    print(
        "Maximum error:"
    )

    print(
        f"{error:.6e}"
    )


    print(
        "Relative:"
    )

    print(
        f"{relative:.6e}"
    )



# ============================================================
# Degree scaling
# ============================================================


def degree_scaling():


    degrees=[
        2,
        4,
        8,
        12,
        16
    ]


    sh_times=[]
    pines_times=[]



    for d in degrees:


        print(
            f"\nDegree {d}"
        )


        sh,_=benchmark_build(
            build_sh,
            d
        )


        pines,_=benchmark_build(
            build_pines,
            d
        )



        sht=benchmark_eval(
            sh,
            STATE
        )


        pint=benchmark_eval(
            pines,
            STATE
        )


        sh_times.append(
            sht
        )


        pines_times.append(
            pint
        )



        print(
            "SH:",
            f"{sht:.3f} us"
        )


        print(
            "Pines:",
            f"{pint:.3f} us"
        )



    plt.figure()

    plt.plot(
        degrees,
        sh_times,
        marker="o",
        label="SH"
    )


    plt.plot(
        degrees,
        pines_times,
        marker="o",
        label="Pines"
    )


    plt.xlabel(
        "Degree"
    )

    plt.ylabel(
        "Evaluation time (us)"
    )


    plt.title(
        "Gravity Kernel Scaling"
    )


    plt.legend()

    plt.grid()

    plt.show()



# ============================================================
# Main
# ============================================================


if __name__=="__main__":


    degree=8


    print(
        "Building kernels..."
    )


    sh,sh_build=benchmark_build(
        build_sh,
        degree
    )


    pines,pines_build=benchmark_build(
        build_pines,
        degree
    )



    print("\nBuild")
    print("----------------------")


    print(
        f"SH:    {sh_build:.3f} us"
    )


    print(
        f"Pines: {pines_build:.3f} us"
    )



    validate(
        sh,
        pines
    )



    print("\nAcceleration Timing")
    print("----------------------")


    sh_time=benchmark_eval(
        sh,
        STATE,
        5000
    )


    pines_time=benchmark_eval(
        pines,
        STATE,
        5000
    )


    print(
        f"SH:    {sh_time:.3f} us"
    )


    print(
        f"Pines: {pines_time:.3f} us"
    )


    print()


    print(
        "Pines speedup:"
    )


    print(
        f"{sh_time/pines_time:.2f}x"
    )



    degree_scaling()