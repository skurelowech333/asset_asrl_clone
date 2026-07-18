# ============================================================
# Benchmark old vs optimized spherical harmonic gravity
# Debug version
# ============================================================

import time
import gc
import numpy as np
import asset_asrl as ast
from GravitySphericalHarmonic import (
    SphericalHarmonicGravityVF as OldSH
)

from GravitySphericalHarmonicOptimized import (
    SphericalHarmonicGravityVF as NewSH
)

import functools
print = functools.partial(print, flush=True)

MU_EARTH = 398600.435507
R_EARTH = 6378.137

DEGREE = 8


# ============================================================
# Coefficients
# ============================================================

def make_coefficients(degree):

    print("\nCreating coefficients")
    print("---------------------")


    t0=time.perf_counter()


    C = np.zeros(
        (degree+1, degree+1)
    )

    S = np.zeros(
        (degree+1, degree+1)
    )


    C[2,0] = -0.484165371736e-3


    t1=time.perf_counter()


    print(
        f"Coefficient creation time:"
        f" {t1-t0:.6f} sec"
    )


    print(
        f"Coefficient size:"
        f" {(degree+1)}x{(degree+1)}"
    )


    return C,S



Cnm,Snm = make_coefficients(DEGREE)




# ============================================================
# Build
# ============================================================

def benchmark_build(name, cls):

    print("\n================================")
    print(f"BUILDING {name}")
    print("================================")


    print("Running garbage collection")

    gc.collect()


    print("Calling constructor...")


    t0=time.perf_counter()


    model = cls(
        MU_EARTH,
        R_EARTH,
        Cnm,
        Snm,
        DEGREE
    )


    t1=time.perf_counter()


    print("Constructor returned")


    elapsed=t1-t0


    print(
        f"{name} build:"
        f" {elapsed:.6f} sec"
    )


    print(
        "Model type:",
        type(model)
    )


    return model,elapsed




# ============================================================
# Evaluation
# ============================================================

def benchmark_eval(name, model, N=100):


    print("\n================================")
    print(f"EVALUATION {name}")
    print("================================")



    print("Calling model.vf()")


    t0=time.perf_counter()


    accel=model.vf()


    t1=time.perf_counter()


    print(
        f"VF creation:"
        f" {t1-t0:.6f} sec"
    )



    x0=np.array(
        [
            R_EARTH+500,
            0,
            0,
            0,
            7.6,
            0
        ],
        dtype=float
    )


    print("Starting warmup")


    t0=time.perf_counter()


    result=accel(x0)


    t1=time.perf_counter()


    print(
        f"Warmup:"
        f" {t1-t0:.6f} sec"
    )


    print(
        "Warmup result:"
    )

    print(result)



    print(
        f"Running {N} evaluations"
    )


    t0=time.perf_counter()


    for i in range(N):

        result=accel(x0)


        if i % max(1,N//10)==0:

            print(
                f"  evaluation {i}/{N}"
            )


    t1=time.perf_counter()


    avg=(t1-t0)/N


    print(
        f"Average:"
        f" {avg*1e6:.3f} us"
    )


    return np.array(result),avg


# ============================================================
# Validation
# ============================================================

def validate(old,new):


    print("\n================================")
    print("VALIDATION")
    print("================================")


    print("Creating VFs")

    oldvf=old.vf()

    newvf=new.vf()


    print("VFs created")


    rng=np.random.default_rng(0)


    errors=[]


    for k in range(100):


        print(
            f"Validation {k}/100"
        )


        r=(
            R_EARTH
            +
            rng.uniform(
                300,
                2000
            )
        )


        state=np.array(
            [
                r,
                rng.uniform(-100,100),
                rng.uniform(-100,100),
                0,
                7.5,
                0
            ]
        )


        t0=time.perf_counter()

        a1=np.array(
            oldvf(state)
        )

        t1=time.perf_counter()


        a2=np.array(
            newvf(state)
        )

        t2=time.perf_counter()



        err=np.linalg.norm(
            a1-a2
        )


        errors.append(err)


        print(
            f" old eval:"
            f" {(t1-t0)*1e6:.3f} us"
        )

        print(
            f" new eval:"
            f" {(t2-t1)*1e6:.3f} us"
        )


        print(
            f" error:"
            f" {err:.3e}"
        )


    return max(errors)




# ============================================================
# Main
# ============================================================


print("\n================================")
print("SPHERICAL HARMONIC BENCHMARK")
print("================================")

print(
    f"Degree = {DEGREE}"
)



old,old_build=benchmark_build(
    "OLD SH",
    OldSH
)



new,new_build=benchmark_build(
    "OPTIMIZED SH",
    NewSH
)



print("\nBUILD SUMMARY")

print(
    f"Old:"
    f" {old_build:.6f} sec"
)

print(
    f"New:"
    f" {new_build:.6f} sec"
)



old_a,old_eval=benchmark_eval(
    "OLD SH",
    old,
    100
)



new_a,new_eval=benchmark_eval(
    "OPTIMIZED SH",
    new,
    100
)



print("\nEVALUATION SUMMARY")

print(
    f"Old:"
    f" {old_eval*1e6:.3f} us"
)

print(
    f"New:"
    f" {new_eval*1e6:.3f} us"
)



error=validate(
    old,
    new
)



print("\nFINAL")

print(
f"""
Degree:
{DEGREE}

Build speedup:
{old_build/new_build:.3f}x

Evaluation speedup:
{old_eval/new_eval:.3f}x

Max error:
{error:.3e}
"""
)
    
    # ============================================================
# PROPAGATION BENCHMARK
# ============================================================


import matplotlib.pyplot as plt


oc = ast.OptimalControl
vf = ast.VectorFunctions
Args = vf.Arguments


plt.style.use("dark_background")


# ============================================================
# PROPAGATION BENCHMARK
# ============================================================

import matplotlib.pyplot as plt

oc = ast.OptimalControl
vf = ast.VectorFunctions
Args = vf.Arguments

plt.style.use("dark_background")


print("\n================================")
print("PROPAGATION BENCHMARK")
print("================================")


# ============================================================
# Build fresh gravity models
# Keep validation objects untouched
# ============================================================

print("Creating propagation gravity models")


old_prop = OldSH(
    MU_EARTH,
    R_EARTH,
    Cnm,
    Snm,
    DEGREE
)


new_prop = NewSH(
    MU_EARTH,
    R_EARTH,
    Cnm,
    Snm,
    DEGREE
)


old_accel = old_prop.build_accel()
new_accel = new_prop.build_accel()


print("Acceleration objects ready")
# ============================================================
# PROPAGATION BENCHMARK
# ============================================================

print("\n==============================")
print("PROPAGATION BENCHMARK")
print("==============================")


class OldGravityODE(oc.ODEBase):

    def __init__(self):

        X = Args(6)

        v = X.segment3(3)

        ode = vf.stack(
            v,
            old_accel
        )

        super().__init__(
            ode,
            6,
            0
        )


class NewGravityODE(oc.ODEBase):

    def __init__(self):

        X = Args(6)

        v = X.segment3(3)

        ode = vf.stack(
            v,
            new_accel
        )

        super().__init__(
            ode,
            6,
            0
        )


# ============================================================
# PROPAGATION BENCHMARK
# ============================================================


def benchmark_prop(
        name,
        ode_class,
        x0,
        tf,
        runs=3):


    print("\n--------------------------------")
    print(f"PROPAGATING {name}")
    print("--------------------------------")


    print("Initial state:")
    print(x0)

    print(
        "State shape:",
        x0.shape
    )

    print(
        "State dtype:",
        x0.dtype
    )


    times=[]
    steps=[]
    solutions=[]



    for k in range(runs):

        print("\n==============================")
        print(
            f"Run {k+1}/{runs}"
        )
        print("==============================")


        # --------------------------------------------
        # Build ODE
        # --------------------------------------------

        print("Building ODE")

        t0=time.perf_counter()

        ode = ode_class()

        t1=time.perf_counter()


        print(
            f"ODE creation:"
            f" {(t1-t0):.6f} sec"
        )



        # --------------------------------------------
        # Build integrator
        # --------------------------------------------

        print("Creating integrator")


        t0=time.perf_counter()


        integ = ode.integrator(
            "DOPRI87",
            20
        )


        t1=time.perf_counter()


        print(
            f"Integrator creation:"
            f" {(t1-t0):.6f} sec"
        )


        print(
            "Integrator type:",
            type(integ)
        )



        # --------------------------------------------
        # Integration
        # --------------------------------------------


        print(
            f"Starting integration:"
            f" {tf:.1f} sec"
        )


        t0=time.perf_counter()


        try:

            sol = integ.integrate_dense(
                x0,
                tf
            )

        except Exception as e:

            print("\n!!! INTEGRATION FAILED !!!")
            print(type(e))
            print(e)

            raise


        t1=time.perf_counter()



        print(
            "Integration finished"
        )


        elapsed=t1-t0


        print(
            f"Runtime:"
            f" {elapsed:.6f} sec"
        )



        # --------------------------------------------
        # Process trajectory
        # --------------------------------------------


        print(
            "Converting trajectory"
        )


        traj=np.asarray(sol)



        print(
            "Trajectory shape:",
            traj.shape
        )


        print(
            "First state:"
        )

        print(
            traj[0]
        )


        print(
            "Last state:"
        )

        print(
            traj[-1]
        )


        print(
            "Number of points:",
            len(traj)
        )



        times.append(elapsed)

        steps.append(len(traj))

        solutions.append(traj)



        # --------------------------------------------
        # Position diagnostics
        # --------------------------------------------


        radius = np.linalg.norm(
            traj[:,:3],
            axis=1
        )


        print(
            "Radius check:"
        )


        print(
            f" Min radius:"
            f" {np.min(radius):.3f} km"
        )


        print(
            f" Max radius:"
            f" {np.max(radius):.3f} km"
        )


        print(
            f" Radius change:"
            f" {(radius[-1]-radius[0]):.6e} km"
        )



    return (
        np.mean(times),
        np.mean(steps),
        solutions[-1]
    )



# ============================================================
# Initial condition
# ============================================================


x0 = np.hstack(
    [
        np.array(
            [
                R_EARTH+500.0,
                0.0,
                0.0,

                0.0,
                7.6,
                0.0
            ],
            dtype=np.float64
        ),
        0.0
    ]
).reshape(-1,1)



print("\nInitial propagation vector")
print(x0)
print(x0.shape)



tf = 5400.0



old_time,old_steps,old_traj = benchmark_prop(
    "OLD SH",
    OldGravityODE,
    x0,
    tf
)



new_time,new_steps,new_traj = benchmark_prop(
    "OPTIMIZED SH",
    NewGravityODE,
    x0,
    tf
)



# ============================================================
# Compare trajectories
# ============================================================


n=min(
    len(old_traj),
    len(new_traj)
)


old_traj=old_traj[:n]
new_traj=new_traj[:n]



error=np.linalg.norm(
    old_traj[:,:3]-new_traj[:,:3],
    axis=1
)



print(
f"""
================================
PROP RESULTS
================================

OLD SH

Runtime:
    {old_time:.6f} sec

Steps:
    {old_steps:.0f}


OPTIMIZED SH

Runtime:
    {new_time:.6f} sec

Steps:
    {new_steps:.0f}


Speedup:

    {old_time/new_time:.3f}x


Position Agreement:

Maximum:
    {np.max(error):.3e} km

Mean:
    {np.mean(error):.3e} km

Final:
    {error[-1]:.3e} km

"""
)
    
# ============================================================
# PROPAGATION PLOTS
# ============================================================

print("\nGenerating propagation plots...")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Time vector
# ------------------------------------------------------------

n = len(old_traj)

time_vec = np.linspace(
    0,
    tf,
    n
)


# ------------------------------------------------------------
# Position vectors
# ------------------------------------------------------------

old_pos = old_traj[:, :3]
new_pos = new_traj[:, :3]


old_radius = np.linalg.norm(
    old_pos,
    axis=1
)

new_radius = np.linalg.norm(
    new_pos,
    axis=1
)



# ============================================================
# 1. 3D Orbit Plot
# ============================================================

fig = plt.figure(
    figsize=(8,8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.plot(
    old_pos[:,0],
    old_pos[:,1],
    old_pos[:,2],
    label="Old SH"
)


ax.plot(
    new_pos[:,0],
    new_pos[:,1],
    new_pos[:,2],
    linestyle="--",
    label="Optimized SH"
)


ax.set_title(
    "Orbit Propagation Comparison"
)


ax.set_xlabel("X km")
ax.set_ylabel("Y km")
ax.set_zlabel("Z km")


ax.legend()


plt.tight_layout()

plt.show()



# ============================================================
# 2. Position Error
# ============================================================

plt.figure(
    figsize=(9,5)
)


plt.semilogy(
    time_vec,
    error
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position Error (km)"
)


plt.title(
    "Trajectory Agreement Error"
)


plt.grid(True)

plt.tight_layout()

plt.show()



# ============================================================
# 3. Radius Conservation
# ============================================================

plt.figure(
    figsize=(9,5)
)


plt.plot(
    time_vec,
    old_radius,
    label="Old SH"
)


plt.plot(
    time_vec,
    new_radius,
    "--",
    label="Optimized SH"
)


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Radius (km)"
)


plt.title(
    "Orbital Radius Stability"
)


plt.legend()

plt.grid(True)


plt.tight_layout()

plt.show()



# ============================================================
# 4. Radius Difference
# ============================================================

plt.figure(
    figsize=(9,5)
)


radius_error = (
    new_radius -
    old_radius
)


plt.plot(
    time_vec,
    radius_error
)


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Δ Radius (km)"
)


plt.title(
    "Radius Difference"
)


plt.grid(True)


plt.tight_layout()

plt.show()



# ============================================================
# 5. State Error Components
# ============================================================

state_error = (
    old_traj -
    new_traj
)


labels = [
    "X",
    "Y",
    "Z",
    "Vx",
    "Vy",
    "Vz"
]


plt.figure(
    figsize=(10,6)
)


for i,label in enumerate(labels):

    plt.plot(
        time_vec,
        state_error[:,i],
        label=label
    )


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Difference"
)


plt.title(
    "State Difference Between Kernels"
)


plt.legend()

plt.grid(True)


plt.tight_layout()

plt.show()



# ============================================================
# 6. Runtime Comparison
# ============================================================

plt.figure(
    figsize=(6,5)
)


names=[
    "Old SH",
    "Optimized SH"
]


runtime=[
    old_time,
    new_time
]


plt.bar(
    names,
    runtime
)


plt.ylabel(
    "Runtime (s)"
)


plt.title(
    "Propagation Runtime Comparison"
)


for i,v in enumerate(runtime):

    plt.text(
        i,
        v,
        f"{v:.4f}s",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

plt.show()



# ============================================================
# Summary Statistics
# ============================================================


print(
f"""
================================
PROPAGATION DIAGNOSTICS
================================


Trajectory points:
    {n}


Maximum position error:
    {np.max(error):.6e} km


Mean position error:
    {np.mean(error):.6e} km


Final position error:
    {error[-1]:.6e} km


Maximum radius deviation:

OLD:
    {np.max(abs(old_radius-old_radius[0])):.6e} km

NEW:
    {np.max(abs(new_radius-new_radius[0])):.6e} km


Runtime speedup:

    {old_time/new_time:.3f}x

"""
)
