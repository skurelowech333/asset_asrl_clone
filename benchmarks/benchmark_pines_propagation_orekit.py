#!/usr/bin/env python3
"""Gravity-only Earth-orbit propagation comparison using EGM96 harmonics.

Implementations: ASSET Pines + native ODEBase/DOPRI54 and Orekit
Holmes-Featherstone + NumericalPropagator. Both use the same EGM96
degree/order, initial orbit, duration, and output epochs. Positions use km
and km/s except inside Orekit (SI). No drag, SRP, third-body forces, or
other perturbations are included.

Example:
  python benchmark_pines_propagation_orekit.py

Edit the configuration constants near the top of the file to change the
degree sweep, duration, output cadence, tolerance, or output path.

Orekit uses GCRF as both its propagation frame and the fixed frame of the
gravity harmonics. This avoids UTC/IERS auxiliary data; only the EGM96 .nor
coefficient file is needed.
"""
from __future__ import annotations

import csv
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MU_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.1363

# Edit these settings directly; this script does not use command-line arguments.
DEFAULT_DEGREES = (2, 4, 8, 12, 20, 50, 100, 120, 180, 250, 300, 360)
COEFFICIENT_FILE = None  # Set to a Path if egm96.nor is outside the searched locations.
DURATION_S = 60000.0
OUTPUT_STEP_S = 10.0
POSITION_TOLERANCE_M = 1.0
MAX_INTERNAL_STEP_S = None  # None uses min(60 s, max(OUTPUT_STEP_S, 1 s)) for both integrators.
TIMING_REPEATS = 5
OUTPUT_FILE = Path(__file__).resolve().parent / "results" / "gravity_propagation_egm96.csv"


def find_coefficients(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else []
    candidates += [
        Path.cwd() / "native/data/egm96.nor",
        Path(__file__).resolve().parent / "native/data/egm96.nor",
        Path(__file__).resolve().parent.parent / "native/data/egm96.nor",
    ]
    for path in candidates:
        if path and path.is_file():
            return path.resolve()
    shown = "\n".join(f"  {x}" for x in candidates)
    raise FileNotFoundError("Cannot find EGM96 coefficient file. Set COEFFICIENT_FILE near the top of this script. Searched:\n" + shown)


def initial_state():
    radius = R_EARTH_KM + 400.0
    speed = math.sqrt(MU_KM3_S2 / radius)
    inclination = math.radians(51.6)
    return np.array([radius, 0.0, 0.0, 0.0, speed*math.cos(inclination), speed*math.sin(inclination), 0.0])


def default_max_internal_step(output_step):
    return min(60.0, max(float(output_step), 1.0))


def asset_propagate(coeff_file, degree, duration, step, position_tolerance_m=1.0, max_internal_step=None):
    try:
        import asset_asrl as ast
        import asset
    except ImportError as exc:
        raise RuntimeError("ASSET is unavailable. Run this script in the Python environment containing asset_asrl/asset.") from exc
    gravity = asset.Extensions.PinesGravity(str(coeff_file), int(degree), MU_KM3_S2, R_EARTH_KM, "fully_normalized")
    vf = ast.VectorFunctions
    Args = vf.Arguments
    X = Args(6)
    v = X.tail3()
    _acc_vf = gravity(X.head3())
    rhs = vf.stack(v, _acc_vf)
    # ODEBase is the Python wrapper in asset_asrl.OptimalControl; the
    # compiled asset.OptimalControl namespace exposes fixed-size ode_6 instead.
    from asset_asrl.OptimalControl import ODEBase
    ode = ODEBase(rhs, 6)
    integ = ode.integrator("DOPRI54", float(step))
    state = initial_state()
    # Match Orekit's Cartesian tolerance construction by default. Raising the
    # position tolerance or allowing larger steps trades accuracy for speed.
    pos_tol_km = float(position_tolerance_m) / 1000.0
    vel_tol_km_s = pos_tol_km * np.linalg.norm(state[3:6]) / np.linalg.norm(state[:3])
    integ.setAbsTols(np.array([pos_tol_km] * 3 + [vel_tol_km_s] * 3))
    integ.setRelTol(0.0)
    # By default, use the same maximum internal step as Orekit.
    max_step = (default_max_internal_step(step) if max_internal_step is None
                else float(max_internal_step))
    if max_step < 1.0e-3:
        raise ValueError("--max-internal-step must be at least 0.001 seconds")
    integ.setStepSizes(min(float(step), max_step), min(1.0e-3, max_step), max_step)
    sample_count = int(round(duration / step)) + 1
    # integrate_dense performs one propagation and returns the entire sampled
    # trajectory, rather than restarting the integrator for every output time.
    arr = np.asarray(integ.integrate_dense(state, float(duration), sample_count), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise RuntimeError(f"Unexpected ASSET trajectory shape {arr.shape}; expected state plus time")
    if arr.shape[0] != sample_count:
        raise RuntimeError(f"ASSET returned {arr.shape[0]} samples; expected {sample_count}")
    return arr[:, :7]


def orekit_propagate(coeff_file, degree, duration, step, position_tolerance_m=1.0, max_internal_step=None):
    try:
        import orekit_jpype
        try:
            orekit_jpype.initVM()
        except Exception as exc:
            # Accept an already-running VM only when Orekit classes are importable.
            try:
                from org.orekit.data import DataContext  # noqa: F401
            except Exception:
                raise RuntimeError("Could not initialize Orekit JVM") from exc
        from java.io import File
        from org.orekit.data import DataContext, DirectoryCrawler
        from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
        from org.orekit.forces.gravity.potential import EGMFormatReader, GravityFieldFactory
        from org.orekit.frames import FramesFactory
        from org.orekit.time import AbsoluteDate
        from org.orekit.orbits import CartesianOrbit, OrbitType
        from org.orekit.propagation import SpacecraftState
        from org.orekit.propagation.numerical import NumericalPropagator
        from org.hipparchus.ode.nonstiff import DormandPrince54Integrator
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.utils import PVCoordinates
    except ImportError as exc:
        raise RuntimeError("Orekit/orekit-jpype classes are unavailable in this Python environment") from exc

    manager = DataContext.getDefault().getDataProvidersManager()
    manager.clearProviders()
    manager.addProvider(DirectoryCrawler(File(str(coeff_file.parent))))
    factory = GravityFieldFactory
    factory.clearPotentialCoefficientsReaders()
    reader = EGMFormatReader("^" + re.escape(coeff_file.name) + "$", False)
    factory.addPotentialCoefficientsReader(reader)
    try:
        provider = factory.getNormalizedProvider(int(degree), int(degree))
    except Exception as exc:
        raise RuntimeError(f"Orekit could not read EGM96 coefficients from {coeff_file}; check that the file is the EGM .nor format supported by this Orekit version") from exc
    date = AbsoluteDate.J2000_EPOCH
    gcrf = FramesFactory.getGCRF()
    gravity_model = HolmesFeatherstoneAttractionModel(gcrf, provider)
    y0 = initial_state()
    pv = PVCoordinates(Vector3D(*(float(v*1000.0) for v in y0[:3])), Vector3D(*(float(v*1000.0) for v in y0[3:6])))
    orbit = CartesianOrbit(pv, gcrf, date, MU_KM3_S2*1e9)
    min_step = 1.0e-3
    max_step = (default_max_internal_step(step) if max_internal_step is None
                else float(max_internal_step))
    tolerances = NumericalPropagator.tolerances(float(position_tolerance_m), orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince54Integrator(min_step, max_step, tolerances[0], tolerances[1])
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(SpacecraftState(orbit))
    propagator.addForceModel(gravity_model)
    # Generate dense ephemeris during one numerical propagation, then sample
    # that stored solution at the same epochs used by ASSET's integrate_dense.
    generator = propagator.getEphemerisGenerator()
    propagator.propagate(date.shiftedBy(float(duration)))
    ephemeris = generator.getGeneratedEphemeris()
    out = []
    for t in np.linspace(0.0, duration, int(round(duration / step)) + 1):
        state = ephemeris.propagate(date.shiftedBy(float(t)))
        p = state.getPVCoordinates(gcrf)
        r, v = p.getPosition(), p.getVelocity()
        out.append([r.getX()/1000, r.getY()/1000, r.getZ()/1000,
                    v.getX()/1000, v.getY()/1000, v.getZ()/1000, float(t)])
    trajectory = np.asarray(out, dtype=np.float64)
    expected_count = int(round(duration / step)) + 1
    if trajectory.shape != (expected_count, 7):
        raise RuntimeError(
            f"Orekit dense ephemeris returned {trajectory.shape[0]} samples; "
            f"expected {expected_count}"
        )
    return trajectory


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "implementation", "time_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for degree, name, trajectory in results:
            for row in trajectory:
                writer.writerow([degree, name, row[6], *row[:6]])


def write_narrative(path, performance):
    asset_wins = [row for row in performance if row["asset_speedup"] > 1.0]
    orekit_wins = [row for row in performance if row["asset_speedup"] < 1.0]
    if asset_wins:
        best = max(asset_wins, key=lambda row: row["asset_speedup"])
        win_degrees = ", ".join(str(row["degree"]) for row in asset_wins)
        lines = [
            f"ASSET was faster at {len(asset_wins)} of {len(performance)} tested gravity degrees.",
            f"ASSET's strongest measured advantage was {best['asset_speedup']:.2f}x at degree/order {best['degree']}/{best['degree']}.",
            f"ASSET-faster degrees: {win_degrees}.",
        ]
    else:
        lines = ["ASSET was not faster at any tested gravity degree in this run."]
    if orekit_wins:
        lines.append("Orekit was faster at degrees/orders: " + ", ".join(
            str(row["degree"]) for row in orekit_wins) + ".")
    lines.append("Speedup is Orekit runtime divided by ASSET runtime; values above 1 mean ASSET was faster.")
    lines.append("Trajectory errors are measured as Orekit minus ASSET over the full sampled trajectory.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def create_plots(output_path, performance):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib. Install it with: python -m pip install matplotlib") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_stem = output_path.with_suffix("")
    degrees = [row["degree"] for row in performance]
    asset_times = [row["asset_time"] for row in performance]
    orekit_times = [row["orekit_time"] for row in performance]
    speedups = [row["asset_speedup"] for row in performance]
    green, gray = "#238636", "#718096"

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = np.arange(len(degrees))
    width = 0.38
    ax.bar(x - width/2, asset_times, width, label="ASSET Pines DOPRI54", color=green)
    ax.bar(x + width/2, orekit_times, width, label="Orekit Holmes-Featherstone", color=gray)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(d) for d in degrees])
    ax.set_xlabel("Gravity degree/order")
    ax.set_ylabel("Median runtime (s, log scale)")
    ax.set_title("ASSET runtime across the EGM96 degree sweep")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(plot_stem.with_name(plot_stem.name + "_runtime_by_degree.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = [green if value > 1.0 else gray for value in speedups]
    ax.bar(x, speedups, color=colors)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2, label="Equal runtime")
    ax.set_xticks(x, [str(d) for d in degrees])
    ax.set_xlabel("Gravity degree/order")
    ax.set_ylabel("ASSET speedup (Orekit time / ASSET time)")
    ax.set_title("Where ASSET is faster (green bars are ASSET wins)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(plot_stem.with_name(plot_stem.name + "_asset_speedup_by_degree.png"), dpi=180)
    plt.close(fig)

    position_errors_m = [max(row["position_error_km"] * 1000.0, 1.0e-15) for row in performance]
    velocity_errors_m_s = [max(row["velocity_error_km_s"] * 1000.0, 1.0e-15) for row in performance]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(degrees, position_errors_m, marker="o", color="#6f42c1")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Max position difference (m)")
    axes[0].grid(True, which="both", alpha=0.22)
    axes[1].plot(degrees, velocity_errors_m_s, marker="o", color="#0969da")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Gravity degree/order")
    axes[1].set_ylabel("Max velocity difference (m/s)")
    axes[1].grid(True, which="both", alpha=0.22)
    fig.suptitle("Trajectory agreement: Orekit versus ASSET")
    fig.savefig(plot_stem.with_name(plot_stem.name + "_trajectory_error_by_degree.png"), dpi=180)
    plt.close(fig)


def main():
    a = SimpleNamespace(
        coeff_file=COEFFICIENT_FILE,
        duration=DURATION_S,
        step=OUTPUT_STEP_S,
        position_tolerance_m=POSITION_TOLERANCE_M,
        max_internal_step=MAX_INTERNAL_STEP_S,
        repeats=TIMING_REPEATS,
        output=OUTPUT_FILE,
    )
    degrees = list(DEFAULT_DEGREES)
    if (not degrees or any(d < 2 for d in degrees) or len(set(degrees)) != len(degrees)
            or a.duration <= 0 or a.step <= 0 or a.repeats < 1
            or a.position_tolerance_m <= 0
            or (a.max_internal_step is not None and a.max_internal_step < 1.0e-3)):
        raise ValueError("degrees must be unique and >=2; duration/step/repeats, position tolerance, and shared max step must be positive")
    if not math.isclose(a.duration / a.step, round(a.duration / a.step), rel_tol=0, abs_tol=1e-9):
        raise ValueError("duration must be an integer multiple of step")
    coeff = find_coefficients(a.coeff_file)
    print(f"EGM96 coefficients: {coeff}; degrees/orders: {', '.join(map(str, degrees))}")
    print("Gravity only; no drag, SRP, or third-body forces.")
    print("Both runs use the same EGM96 degree/order, initial state, duration, and sample cadence.")
    print("ASSET stacks velocity with direct Pines acceleration; Orekit uses EGM96 fixed in GCRF.")
    print("No Orekit UTC/IERS data is needed; the EGM96 .nor coefficient file is loaded directly.")
    shared_max_step = (default_max_internal_step(a.step) if a.max_internal_step is None
                       else a.max_internal_step)
    print(f"Shared maximum internal step: {shared_max_step:g} s for ASSET and Orekit.")
    print(f"Shared Cartesian position tolerance: {a.position_tolerance_m:g} m for ASSET and Orekit.")
    trajectory_rows, summary_rows, performance = [], [], []
    names = ["ASSET Pines DOPRI54", "Orekit Holmes-Featherstone"]
    for degree in degrees:
        print(f"\n--- Degree/order {degree}/{degree} ---")
        functions = {
            names[0]: lambda d=degree: asset_propagate(
                coeff, d, a.duration, a.step, a.position_tolerance_m, shared_max_step),
            names[1]: lambda d=degree: orekit_propagate(
                coeff, d, a.duration, a.step, a.position_tolerance_m, shared_max_step),
        }
        results, timings = {}, {}
        # Check Orekit's reader first so setup errors do not waste ASSET time.
        for name in [names[1], names[0]]:
            run = functions[name]
            run()  # Untimed warm-up for this degree and implementation.
            samples = []
            for _ in range(a.repeats):
                start = time.perf_counter()
                trajectory = run()
                samples.append(time.perf_counter() - start)
            results[name] = trajectory
            timings[name] = float(np.median(samples))
            print(f"{name}: {timings[name]:.6f} s; {len(trajectory)} output points")

        asset, orekit = results[names[0]], results[names[1]]
        if len(asset) != len(orekit) or not np.allclose(asset[:, 6], orekit[:, 6], rtol=0, atol=1e-9):
            raise RuntimeError(f"ASSET and Orekit output epochs do not align at degree {degree}")
        position_error = np.linalg.norm(orekit[:, :3] - asset[:, :3], axis=1)
        velocity_error = np.linalg.norm(orekit[:, 3:6] - asset[:, 3:6], axis=1)
        asset_speedup = timings[names[1]] / timings[names[0]]
        print(f"Degree {degree}: ASSET speedup vs Orekit={asset_speedup:.4f}x "
              f"({'ASSET faster' if asset_speedup > 1 else 'Orekit faster' if asset_speedup < 1 else 'equal runtimes'}); "
              f"max position difference={np.max(position_error):.6e} km; "
              f"max velocity difference={np.max(velocity_error):.6e} km/s")
        trajectory_rows.extend((degree, name, results[name]) for name in names)
        performance.append({
            "degree": degree,
            "asset_time": timings[names[0]],
            "orekit_time": timings[names[1]],
            "asset_speedup": asset_speedup,
            "position_error_km": float(np.max(position_error)),
            "velocity_error_km_s": float(np.max(velocity_error)),
        })
        for name in names:
            summary_rows.append([name, f"{timings[name]:.9g}", f"{asset_speedup:.9g}", degree,
                                 a.duration, a.step, "model construction + propagation + output sampling"])
        summary_rows.append(["Orekit minus ASSET max position error (km)", f"{np.max(position_error):.12g}", "", degree, a.duration, a.step, ""])
        summary_rows.append(["Orekit minus ASSET max velocity error (km/s)", f"{np.max(velocity_error):.12g}", "", degree, a.duration, a.step, ""])

    save_results(a.output, trajectory_rows)
    print(f"Trajectory CSV: {a.output.resolve()}")
    summary_path = a.output.with_name(a.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["implementation", "median_runtime_s", "asset_speedup_vs_orekit", "degree", "duration_s", "output_step_s", "timing_scope"])
        writer.writerows(summary_rows)
    print(f"Benchmark summary CSV: {summary_path.resolve()}")
    report_path = a.output.with_name(a.output.stem + "_report.txt")
    write_narrative(report_path, performance)
    print(f"Narrative report: {report_path.resolve()}")
    create_plots(a.output, performance)
    for suffix in ("runtime_by_degree", "asset_speedup_by_degree", "trajectory_error_by_degree"):
        plot_path = a.output.with_name(a.output.stem + f"_{suffix}.png")
        print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
