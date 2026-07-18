# ============================================================
# FILE: benchmark_pines_vs_sh.py
#
# Pines vs Classical Spherical Harmonic Benchmark
#
# Measures:
#
#   - Build time
#   - Acceleration evaluation time
#   - Speedup
#   - Absolute error
#   - RMS error
#   - Relative error
#
# ============================================================


import time
import numpy as np
import matplotlib.pyplot as plt


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
# Coefficients
# ============================================================

def generate_coefficients(degree):


    C={}
    S={}



    #
    # J2 coefficient
    #

    C[2,0] = -J2 / np.sqrt(5)

    S[2,0] = 0.0



    #
    # Fill missing coefficients
    #

    for n in range(degree+1):

        for m in range(n+1):

            if (n,m) not in C:

                C[n,m]=0.0


            if (n,m) not in S:

                S[n,m]=0.0



    return C,S



# ============================================================
# Random orbit states
# ============================================================

def generate_states(N):


    np.random.seed(10)



    radius = np.random.uniform(

        6800,

        42000,

        N

    )


    theta = np.random.uniform(

        0,

        np.pi,

        N

    )


    phi = np.random.uniform(

        0,

        2*np.pi,

        N

    )



    x = radius*np.sin(theta)*np.cos(phi)

    y = radius*np.sin(theta)*np.sin(phi)

    z = radius*np.cos(theta)



    position=np.column_stack(

        (
            x,
            y,
            z
        )

    )



    velocity=np.zeros_like(
        position
    )



    return np.hstack(

        (
            position,
            velocity
        )

    )



# ============================================================
# Evaluate kernel
# ============================================================

def evaluate(model,states):


    f=model.vf()


    output=[]


    for s in states:


        a=f.compute(s)


        output.append(

            np.asarray(
                a,
                dtype=float
            )

        )



    return np.asarray(output)



# ============================================================
# Timing
# ============================================================

def benchmark_time(
        model,
        states,
        repeats=10
):


    #
    # warmup
    #

    evaluate(
        model,
        states[:10]
    )



    times=[]



    for _ in range(repeats):


        t0=time.perf_counter()


        evaluate(
            model,
            states
        )


        t1=time.perf_counter()


        times.append(

            (t1-t0)
            /
            len(states)

        )



    return np.mean(times)



# ============================================================
# Error Metrics
# ============================================================

def compute_error(
        sh,
        pines
):


    difference = np.linalg.norm(

        sh-pines,

        axis=1

    )



    magnitude=np.linalg.norm(

        sh,

        axis=1

    )



    max_error=np.max(
        difference
    )


    rms_error=np.sqrt(

        np.mean(
            difference**2
        )

    )



    #
    # avoid tiny acceleration problem
    #

    mask = magnitude > 1e-3



    relative=np.max(

        difference[mask]
        /
        magnitude[mask]

    )



    return (

        max_error,

        rms_error,

        relative

    )



# ============================================================
# Main Benchmark
# ============================================================

def run_benchmark():



    degrees=[

        2,
        4,
        8

    ]



    states=generate_states(
        1000
    )



    results={

        "degree":[],
        "sh":[],
        "pines":[],
        "speedup":[],
        "max_error":[],
        "rms_error":[],
        "relative_error":[]

    }



    for degree in degrees:



        print()

        print("="*70)

        print(
            f"Degree {degree}"
        )

        print("="*70)



        C,S=generate_coefficients(
            degree
        )



        #
        # Build
        #

        t0=time.perf_counter()


        sh=SphericalHarmonicGravityVF(

            MU,
            R,
            C,
            S,
            degree

        )


        sh_build=time.perf_counter()-t0



        t0=time.perf_counter()


        pines=PinesGravityVF(

            MU,
            R,
            C,
            S,
            degree

        )


        pines_build=time.perf_counter()-t0



        #
        # Runtime
        #

        sh_time=benchmark_time(

            sh,
            states

        )


        pines_time=benchmark_time(

            pines,
            states

        )



        #
        # Accuracy
        #

        sh_acc=evaluate(

            sh,
            states

        )


        pines_acc=evaluate(

            pines,
            states

        )



        max_error,rms_error,relative=compute_error(

            sh_acc,

            pines_acc

        )



        speedup=sh_time/pines_time



        print(
            f"SH build:     {sh_build*1000:.3f} ms"
        )

        print(
            f"Pines build:  {pines_build*1000:.3f} ms"
        )


        print()

        print(
            f"SH:           {sh_time*1e6:.3f} us/eval"
        )


        print(
            f"Pines:        {pines_time*1e6:.3f} us/eval"
        )


        print(
            f"Speedup:      {speedup:.2f}x"
        )


        print()

        print(
            f"Max Error:    {max_error:.6e}"
        )


        print(
            f"RMS Error:    {rms_error:.6e}"
        )


        print(
            f"Relative:     {relative:.6e}"
        )



        results["degree"].append(
            degree
        )

        results["sh"].append(
            sh_time
        )

        results["pines"].append(
            pines_time
        )

        results["speedup"].append(
            speedup
        )

        results["max_error"].append(
            max_error
        )

        results["rms_error"].append(
            rms_error
        )

        results["relative_error"].append(
            relative
        )



    return results



# ============================================================
# Plotting
# ============================================================

def make_plots(results):


    plt.style.use(
        "dark_background"
    )



    degree=np.array(
        results["degree"]
    )



    #
    # Runtime
    #

    plt.figure(
        figsize=(9,6)
    )


    plt.loglog(

        degree,

        np.array(results["sh"])*1e6,

        marker="o",

        linewidth=3,

        label="Spherical Harmonic"

    )


    plt.loglog(

        degree,

        np.array(results["pines"])*1e6,

        marker="s",

        linewidth=3,

        label="Pines"

    )


    plt.xlabel(
        "Degree"
    )


    plt.ylabel(
        "microseconds / evaluation"
    )


    plt.title(
        "Gravity Kernel Runtime Scaling"
    )


    plt.grid(
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        "runtime_scaling.png",
        dpi=300
    )



    #
    # Speedup
    #

    plt.figure(
        figsize=(9,6)
    )


    plt.plot(

        degree,

        results["speedup"],

        marker="o",

        linewidth=3

    )


    plt.xlabel(
        "Degree"
    )


    plt.ylabel(
        "Speedup"
    )


    plt.title(
        "Pines Acceleration Speedup"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        "speedup.png",
        dpi=300
    )



    #
    # Error
    #

    plt.figure(
        figsize=(9,6)
    )


    plt.semilogy(

        degree,

        results["rms_error"],

        marker="o",

        linewidth=3,

        label="RMS Error"

    )


    plt.semilogy(

        degree,

        results["max_error"],

        marker="s",

        linewidth=3,

        label="Maximum Error"

    )


    plt.xlabel(
        "Degree"
    )


    plt.ylabel(
        "Acceleration Error"
    )


    plt.title(
        "Pines vs Spherical Harmonic Error"
    )


    plt.grid(
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        "accuracy_error.png",
        dpi=300
    )



# ============================================================
# Run
# ============================================================

if __name__=="__main__":


    results=run_benchmark()


    make_plots(
        results
    )