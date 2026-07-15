# -*- coding: utf-8 -*-
"""
@author: Sarah
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import asset_asrl as ast
from asset_asrl.GravitySphericalHarmonic import SphericalHarmonicGravityVF
from asset_asrl.GravityPines import PinesGravityVF

oc = ast.OptimalControl
vf = ast.VectorFunctions
Args = vf.Arguments

MU = 398600.435507
R  = 6378.137
DEGREE = 8
NSAMPLES = 10000
plt.style.use("dark_background")

def make_coefficients(N):
    """Return dense, unnormalised C/S coefficient matrices.

    The Pines kernel follows the package's unnormalised Legendre convention;
    C[2, 0] = -J2 therefore gives the familiar J2 perturbation.
    """
    C = np.zeros((N + 1, N + 1), dtype=np.float64)
    S = np.zeros((N + 1, N + 1), dtype=np.float64)

    if N >= 2:
        C[2, 0] = -1.08262668e-3

    return C, S

C, S = make_coefficients(DEGREE)

states = [
    np.array(
        [7000.0,
            0.0,
            0.0,
            0.0,
            7.5,
            0.0
        ]
    ),

    np.array(
        [8000.0,
            1000.0,
            500.0,
            0.0,
            7.0,
            1.0
        ]
    ),

    np.array(
        [26560.0,
            0.0,
            0.0,
            0.0,
            3.87,
            0.0
        ]
    )
]

states = [x.astype(np.float64) for x in states]

def build_regular():
    return SphericalHarmonicGravityVF(MU, R, C, S, DEGREE)

def build_pines():
    return PinesGravityVF(MU, R, C, S, DEGREE)

def timed_build(func):
    t0 = time.perf_counter()
    obj = func()
    t1 = time.perf_counter()
    return obj, t1-t0

def timed_eval(accel, x, N):
    accel.compute(x)
    t0 = time.perf_counter()

    for i in range(N):
        accel.compute(x)

    t1 = time.perf_counter()
    return (t1-t0)/N

def timed_jacobian(accel, x, N=1000):
    accel.jacobian(x)   # warmup
    t0=time.perf_counter()

    for _ in range(N):
        accel.jacobian(x)

    return (
        time.perf_counter()-t0
    ) / N

# Build
regular, t_reg_build = timed_build(build_regular)
pines, t_pin_build = timed_build(build_pines)

reg_accel = regular.build_accel()
pin_accel = pines.build_accel()

# Validate
reg_vec,  pin_vec, reg_mag, pin_mag, errors = [], [], [], [], []

for x in states:
    r = np.array(reg_accel.compute(x))
    p = np.array(pin_accel.compute(x))

    reg_vec.append(r)
    pin_vec.append(p)

    reg_mag.append(np.linalg.norm(r))
    pin_mag.append(np.linalg.norm(p))
    errors.append(np.linalg.norm(r-p))

reg_vec = np.array(reg_vec)
pin_vec = np.array(pin_vec)
max_error = max(errors)

t_reg = timed_eval(reg_accel, states[0], NSAMPLES)
t_pin = timed_eval(pin_accel, states[0], NSAMPLES)

speedup = t_reg/t_pin

t_reg_jac = timed_jacobian(reg_accel, states[0])
t_pin_jac = timed_jacobian(pin_accel, states[0])

Jsh = np.array(reg_accel.jacobian(states[0]))
Jpin = np.array(pin_accel.jacobian(states[0]))

print("Jacobian difference:", np.linalg.norm(Jsh-Jpin))

print(
f"""
Jacobian Runtime

SH:
    {t_reg_jac*1e6:.3f} us

Pines:
    {t_pin_jac*1e6:.3f} us

Speedup:
    {t_reg_jac/t_pin_jac:.2f}x
"""
)

print("\n==============================")
print("RESULTS")
print("==============================")

print(
    f"""
Degree:
    {DEGREE}

Build:
    SH     {t_reg_build*1e6:.3f} us
    Pines  {t_pin_build*1e6:.3f} us

Runtime:
    SH     {t_reg*1e6:.3f} us/eval
    Pines  {t_pin*1e6:.3f} us/eval

Speedup:
    {speedup:.2f}x

Max error:
    {max_error:.3e}
"""
)

# Acceleration Magnitude
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(states))

ax.plot(x, reg_mag, marker="o", linewidth=2, label="Spherical Harmonic")
ax.plot(x, pin_mag, marker="s", linewidth=2, label="Pines")

ax.set_title(f"Gravity Magnitude Agreement (Degree {DEGREE})")
ax.set_xlabel("Test Case")
ax.set_ylabel("Acceleration km/s²")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Error
fig, ax = plt.subplots(figsize=(8,5))

if np.allclose(errors, 0):
    ax.plot(x, errors, marker="o", linewidth=2)
    ax.set_ylim(-1e-16, 1e-16)
    ax.axhline(0, color="white", linestyle="--")

else:
    ax.semilogy(x, errors, marker="o")

ax.set_title("Kernel Agreement Error")
ax.set_xlabel("Test Case")
ax.set_ylabel("||aSH - aPines||")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Speed Comparison
fig, ax = plt.subplots(figsize=(8,5))

names = ["Spherical Harmonic", "Pines"]
times = [t_reg*1e6, t_pin*1e6]

bars = ax.bar(names, times)

ax.set_title("Gravity Kernel Performance")
ax.set_ylabel("Microseconds / Evaluation")
ax.grid(True, alpha=0.3)

for b in bars:
    ax.text(
        b.get_x()+b.get_width()/2,
        b.get_height(),
        f"{b.get_height():.2f}",
        ha="center",
        va="bottom")

plt.tight_layout()
plt.show()

# Accel Vector Component Comparison
fig, ax = plt.subplots(figsize=(9,5))
cases = np.arange(len(states))

components = [("Ax", 0), ("Ay", 1), ("Az", 2)]

for name, idx in components:
    ax.plot(
        cases,
        reg_vec[:, idx],
        marker="o",
        linewidth=2,
        label=f"Spherical Harmonics {name}")

    ax.plot(
        cases,
        pin_vec[:, idx],
        marker="x",
        linestyle="--",
        linewidth=2,
        label=f"Pines {name}")

ax.set_title(
    "Acceleration Vector Components\n"
    f"Spherical Harmonic vs Pines (Degree {DEGREE})")

ax.set_xlabel("Test Case")
ax.set_ylabel("Acceleration km/s²")
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

class RegularGravityODE(oc.ODEBase):

    def __init__(self):
        X = Args(6)

        r = X.head3()
        v = X.segment3(3)

        ode = vf.stack(v,reg_accel)

        super().__init__(
            ode,
            6,
            0
        )

class PinesGravityODE(oc.ODEBase):
    def __init__(self):
        X = Args(6)

        r = X.head3()
        v = X.segment3(3)

        ode = vf.stack(v,pin_accel)

        super().__init__(
            ode,
            6,
            0
        )

print("\n==============================")
print("LONG PROPAGATION BENCHMARK")
print("==============================")

def benchmark_prop(ode_class, x0, tf, runs=3):
    times, steps, solutions = [], [], []

    for i in range(runs):
        ode = ode_class()
        integ = ode.integrator("DOPRI87", 20)
        t0 = time.perf_counter()

        sol = integ.integrate_dense(x0, tf)

        t1 = time.perf_counter()
        traj = np.array(sol)

        times.append(t1-t0)
        steps.append(len(traj))
        solutions.append(traj)

    return (
        np.array(times),
        np.array(steps),
        solutions[-1])

Xt0 = np.hstack([states[0], 0.0])

prop_times = [
    5400.0,          # 1.5 hours
    86400.0,         # 1 day
    7*86400.0        # 7 days
]

prop_results = {}

for tf in prop_times:
    print(f"\nPropagation time: {tf/3600:.1f} hours")

    reg_times, reg_steps, reg_traj = benchmark_prop(RegularGravityODE, Xt0, tf)
    pin_times, pin_steps, pin_traj = benchmark_prop(PinesGravityODE, Xt0, tf)

    n = min(len(reg_traj), len(pin_traj))

    reg_traj = reg_traj[:n]
    pin_traj = pin_traj[:n]

    pos_error = np.linalg.norm(reg_traj[:,:3] - pin_traj[:,:3], axis=1)

    print(
        f"""
SH:

    Runtime:
        {np.mean(reg_times):.4f} s

    Steps:
        {np.mean(reg_steps):.0f}


Pines:

    Runtime:
        {np.mean(pin_times):.4f} s

    Steps:
        {np.mean(pin_steps):.0f}


Speedup:
    {np.mean(reg_times)/np.mean(pin_times):.2f}x


Position agreement:

    Max:
        {np.max(pos_error):.3e} km

    Final:
        {pos_error[-1]:.3e} km

    Mean:
        {np.mean(pos_error):.3e} km
"""
    )

    prop_results[tf] = (
        reg_traj,
        pin_traj,
        pos_error
    )

fig, ax = plt.subplots(figsize=(9,5))

for tf, data in prop_results.items():
    error = data[2]
    t = np.linspace(0, tf/3600, len(error))
    ax.semilogy(t, error, linewidth=2, label=f"{tf/86400:.1f} days")

ax.set_title("Long-Term Gravity Kernel Propagation Error")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("||rSH - rPines|| km")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

reg_traj, pin_traj, pos_error = prop_results[7*86400.0]

fig, ax = plt.subplots(figsize=(8,8))

ax.plot(reg_traj[:,0], reg_traj[:,1], linewidth=2, label="Spherical Harmonic")
ax.plot(pin_traj[:,0], pin_traj[:,1], "--", linewidth=2, label="Pines")

ax.set_aspect("equal")
ax.set_title("7 Day Orbit Propagation")

ax.set_xlabel("X km")
ax.set_ylabel("Y km")

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

t = np.linspace(0, 5400, n)

# trajectory
fig, ax = plt.subplots(figsize=(8,8))

ax.plot(reg_traj[:,0], reg_traj[:,1], linewidth=2, label="Spherical Harmonics")
ax.plot(pin_traj[:,0], pin_traj[:,1], "--", linewidth=2, label="Pines")

ax.set_aspect("equal")
ax.set_title("Orbit Propagation Comparison")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Error
fig, ax = plt.subplots(figsize=(8,5))

if np.allclose(pos_error, 0):
    ax.plot(t, pos_error, linewidth=2, marker="o")
    ax.axhline(0, linestyle="--", linewidth=1, label="Zero Error")
    ax.set_ylim(-1e-16, 1e-16)
    ax.legend()

else:
    ax.semilogy(t, pos_error, linewidth=2)

ax.set_title("Propagation Position Error\n""Regular SH vs Pines")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (km)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n==============================")
print("DEGREE SCALING")
print("==============================")

degrees = [2, 4, 8, 16, 32]
sh_degree_times = []
pines_degree_times = []

for deg in degrees:
    Cdeg, Sdeg = make_coefficients(deg)

    sh = SphericalHarmonicGravityVF(MU, R, Cdeg, Sdeg, deg)
    sh_accel = sh.build_accel()

    pines_deg = PinesGravityVF(MU, R, Cdeg, Sdeg, deg)
    pines_accel = pines_deg.build_accel()

    sh_time = timed_eval(sh_accel, states[0], 5000)
    pines_time = timed_eval(pines_accel, states[0], 5000)
    sh_degree_times.append(sh_time * 1e6)
    pines_degree_times.append(pines_time * 1e6)

    print(
        f"Degree {deg}: SH {sh_time*1e6:.3f} us, "
        f"Pines {pines_time*1e6:.3f} us"
    )

fit = np.polyfit(np.log(degrees), np.log(sh_degree_times),1)
order = fit[0]
print(f"\nEstimated SH scaling: O(N^{order:.2f})")

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(degrees, sh_degree_times, marker="o", linewidth=2, label="Spherical Harmonic")
ax.plot(degrees, pines_degree_times, marker="s", linewidth=2, label="Pines")

ax.set_yscale("log")
ax.set_title("Gravity Kernel Scaling With Degree")
ax.set_xlabel("Harmonic Degree")
ax.set_ylabel("Microseconds / Evaluation")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

degree_speedup = np.array(sh_degree_times) / np.array(pines_degree_times)
fig, ax = plt.subplots(figsize=(8,5))

ax.plot(degrees, degree_speedup, marker="o", linewidth=2)

ax.set_title("Pines Speedup vs Harmonic Degree")
ax.set_xlabel("Harmonic Degree")
ax.set_ylabel("Speedup")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n==============================")
print("ENERGY CHECK")
print("==============================")

def specific_energy(traj):
    r = np.linalg.norm(traj[:, :3], axis=1)
    v = np.linalg.norm(traj[:, 3:6], axis=1)
    return 0.5 * v * v - MU / r

E_reg = specific_energy(reg_traj)
energy_error = E_reg - E_reg[0]

print("Maximum energy drift:", np.max(np.abs(energy_error)))

energy_t = np.linspace(0, 5400.0, len(energy_error))

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(energy_t, energy_error, linewidth=2)
ax.axhline(0, linestyle="--", linewidth=1)

ax.set_title("Orbital Energy Drift")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Δ Specific Energy")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n==============================")
print("MONTE CARLO VALIDATION")
print("==============================")

rng = np.random.default_rng(1)
random_errors = []

for i in range(100):
    r = rng.normal(size=3)
    r *= (rng.uniform(7000,40000)/ np.linalg.norm(r))

    v = rng.normal(size=3)
    x = np.hstack([r, v]).astype(np.float64)

    sh_a = np.array(reg_accel.compute(x))
    pin_a = np.array(pin_accel.compute(x))
    
    random_errors.append(np.linalg.norm(sh_a - pin_a))

print("Samples:", len(random_errors))
print("Maximum error:", np.max(random_errors))
print("Mean error:", np.mean(random_errors))

fig, ax = plt.subplots(figsize=(8,5))
ax.semilogy(random_errors, marker="o", linewidth=1)

ax.set_title("Random State Kernel Agreement")
ax.set_xlabel("Sample")
ax.set_ylabel("||aSH-aPines||")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

def summary_row(name,build,runtime,jac,prop,error):
    return [
        name,
        build*1e6,
        runtime*1e6,
        jac*1e6,
        prop,
        error
    ]

table = [
    summary_row(
        "Spherical Harmonic",
        t_reg_build,
        t_reg,
        t_reg_jac,
        np.mean(reg_times),
        max_error),

    summary_row(
        "Pines",
        t_pin_build,
        t_pin,
        t_pin_jac,
        np.mean(pin_times),
        max_error)
]

print(
"""
===========================================================
FINAL PERFORMANCE SUMMARY
===========================================================

Kernel                Build(us)   Eval(us)   Jacob(us)  Prop(s)   Error
-----------------------------------------------------------------------
"""
)

for row in table:

    print(
        f"{row[0]:20s}"
        f"{row[1]:10.3f}"
        f"{row[2]:10.3f}"
        f"{row[3]:10.3f}"
        f"{row[4]:10.5f}"
        f"{row[5]:12.3e}"
    )

print(
"""
===========================================================
"""
)
