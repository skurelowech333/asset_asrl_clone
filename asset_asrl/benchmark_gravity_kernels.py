# -*- coding: utf-8 -*-
"""
benchmark_gravity_pines_vs_sh.py

Compare:
    - Classical spherical harmonic gravity
    - Pines gravity kernel

@author: Sarah
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import asset_asrl as ast

from GravitySphericalHarmonic import SphericalHarmonicGravityVF
from GravityPines import PinesGravityVF


oc = ast.OptimalControl


# ============================================================
# Configuration
# ============================================================

MU = 398600.435507
R = 6378.137

DEGREE = 8

N_BUILD = 20
N_EVAL = 10000

MONTE_CARLO = 1000


plt.style.use("dark_background")



# ============================================================
# Coefficients
# ============================================================

def make_coefficients(N):

    C = {}
    S = {}

    for n in range(N+1):

        for m in range(n+1):

            C[n,m] = 1e-6
            S[n,m] = 1e-6


    return C,S



C,S = make_coefficients(DEGREE)



# ============================================================
# Test states
# ============================================================

states = [

    np.array([
        7000,
        0,
        0,
        0,
        7.5,
        0
    ]),

    np.array([
        8000,
        1000,
        500,
        0,
        7,
        1
    ]),

    np.array([
        26560,
        0,
        0,
        0,
        3.87,
        0
    ])

]


states = [
    x.astype(float)
    for x in states
]



# ============================================================
# Build timing
# ============================================================

def benchmark_build(factory):

    times=[]

    obj=None

    for i in range(N_BUILD):

        t0=time.perf_counter()

        obj=factory()

        t1=time.perf_counter()

        times.append(t1-t0)


    return obj,np.array(times)



# ============================================================
# Evaluation timing
# ============================================================

def benchmark_eval(accel,x):

    accel.compute(x)


    times=[]


    for i in range(N_EVAL):

        t0=time.perf_counter()

        accel.compute(x)

        t1=time.perf_counter()

        times.append(t1-t0)


    return np.array(times)



# ============================================================
# Jacobian timing
# ============================================================

def benchmark_jac(accel,x):

    try:

        accel.jacobian(x)

        times=[]


        for i in range(1000):

            t0=time.perf_counter()

            accel.jacobian(x)

            t1=time.perf_counter()

            times.append(t1-t0)


        return np.mean(times)


    except Exception:

        return np.nan




# ============================================================
# Build kernels
# ============================================================


print("Building kernels...")


sh,sh_build = benchmark_build(
    lambda:
        SphericalHarmonicGravityVF(
            MU,
            R,
            C,
            S,
            DEGREE
        )
)


pines,pines_build = benchmark_build(
    lambda:
        PinesGravityVF(
            MU,
            R,
            C,
            S,
            DEGREE
        )
)



sh_accel = sh.build_accel()

pines_accel = pines.build_accel()



# ============================================================
# Accuracy validation
# ============================================================


print("\nValidation")


errors=[]

relative=[]


for x in states:


    ash=np.array(
        sh_accel.compute(x)
    )

    ap=np.array(
        pines_accel.compute(x)
    )


    err=np.linalg.norm(
        ash-ap
    )


    rel=err/np.linalg.norm(ap)


    errors.append(err)
    relative.append(rel)



max_error=max(errors)

max_relative=max(relative)



print(
f"""
Maximum Error:
    {max_error:.3e}

Maximum Relative:
    {max_relative:.3e}
"""
)




# ============================================================
# Runtime
# ============================================================


print("\nTiming acceleration")


sh_times=benchmark_eval(
    sh_accel,
    states[0]
)


pines_times=benchmark_eval(
    pines_accel,
    states[0]
)


sh_mean=np.mean(sh_times)

pines_mean=np.mean(pines_times)



# Jacobians

sh_jac=benchmark_jac(
    sh_accel,
    states[0]
)

pines_jac=benchmark_jac(
    pines_accel,
    states[0]
)



# ============================================================
# Print results
# ============================================================


print(
f"""

====================================================
ACCELERATION PERFORMANCE
====================================================

                 SH             Pines

Build(us)     {np.mean(sh_build)*1e6:.3f}
              {np.mean(pines_build)*1e6:.3f}


Eval(us)      {sh_mean*1e6:.3f}
              {pines_mean*1e6:.3f}


Speedup Pines:
              {sh_mean/pines_mean:.2f}x


Jacobian(us)
              {sh_jac*1e6:.3f}
              {pines_jac*1e6:.3f}


Error:
              {max_error:.3e}

====================================================

"""
)



# ============================================================
# Runtime plot
# ============================================================


plt.figure(figsize=(8,5))


plt.boxplot(
    [
        sh_times*1e6,
        pines_times*1e6
    ],
    labels=[
        "SH",
        "Pines"
    ]
)


plt.ylabel(
    "microseconds"
)

plt.title(
    "Acceleration Runtime Distribution"
)

plt.grid(True)





# ============================================================
# Degree scaling
# ============================================================


degrees=[2,4,8,16,32]


sh_degree=[]

pines_degree=[]


print("\nDegree Scaling")


for deg in degrees:


    Cdeg,Sdeg=make_coefficients(deg)



    shd=SphericalHarmonicGravityVF(
        MU,
        R,
        Cdeg,
        Sdeg,
        deg
    )


    pind=PinesGravityVF(
        MU,
        R,
        Cdeg,
        Sdeg,
        deg
    )


    tsh=np.mean(
        benchmark_eval(
            shd.build_accel(),
            states[0]
        )
    )


    tpin=np.mean(
        benchmark_eval(
            pind.build_accel(),
            states[0]
        )
    )


    sh_degree.append(
        tsh*1e6
    )

    pines_degree.append(
        tpin*1e6
    )


    print(
        deg,
        tsh*1e6,
        tpin*1e6
    )



plt.figure(figsize=(8,5))

plt.loglog(
    degrees,
    sh_degree,
    "o-",
    label="Spherical Harmonic"
)


plt.loglog(
    degrees,
    pines_degree,
    "s-",
    label="Pines"
)


plt.xlabel(
    "Degree"
)

plt.ylabel(
    "microseconds"
)


plt.title(
    "Degree Scaling"
)

plt.legend()
plt.grid(True)



# ============================================================
# Monte Carlo
# ============================================================


print("\nMonte Carlo")


rng=np.random.default_rng(1)


mc=[]


for i in range(MONTE_CARLO):


    r=rng.normal(size=3)

    r*=rng.uniform(
        7000,
        40000
    )/np.linalg.norm(r)



    v=rng.normal(size=3)


    x=np.hstack(
        [
            r,
            v
        ]
    )


    a=np.array(
        sh_accel.compute(x)
    )


    b=np.array(
        pines_accel.compute(x)
    )


    mc.append(
        np.linalg.norm(a-b)
    )



print(
f"""
Samples:
    {MONTE_CARLO}

Max:
    {np.max(mc):.3e}

Mean:
    {np.mean(mc):.3e}

"""
)



plt.figure(figsize=(8,5))

plt.semilogy(
    mc
)

plt.title(
    "Monte Carlo Agreement"
)

plt.xlabel(
    "Sample"
)

plt.ylabel(
    "Acceleration Difference"
)

plt.grid(True)



# ============================================================
# Final summary
# ============================================================


print(
"""
====================================================
FINAL SUMMARY
====================================================

Kernel                 Build(us)    Eval(us)

Spherical Harmonic
"""

)

print(
f"""
                       {np.mean(sh_build)*1e6:.3f}
                       {sh_mean*1e6:.3f}


Pines

                       {np.mean(pines_build)*1e6:.3f}
                       {pines_mean*1e6:.3f}


Pines speedup:
                       {sh_mean/pines_mean:.2f}x


Maximum error:
                       {max_error:.3e}


====================================================
"""
)


plt.show()