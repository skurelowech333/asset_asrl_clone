#pragma once

// ============================================================
// FILE: SphericalHarmonicGravity.h
//
// C++ ASSET VectorFunction node: spherical harmonic gravity
//
// Input  : [x, y, z, vx, vy, vz]  (6D state)
// Output : [ax, ay, az]            (3D acceleration)
//
// Uses AutodiffFwd so ASSET can compute analytic Jacobians
// by propagating Eigen::AutoDiffScalar through compute_impl.
// This avoids building a large ASSET expression graph at
// construction time, which hangs at degree 20-50.
//
// Algorithm:
//   1. Cartesian → spherical  (r, sin φ, cos φ, sin λ, cos λ)
//   2. Associated Legendre recurrence  P_nm(sin φ), dP_nm
//   3. Trig recurrence  cos(m λ), sin(m λ)
//   4. (R/r)^n  radial power sequence
//   5. Accumulate radial / latitude / longitude sums
//   6. Spherical → Cartesian  a_x, a_y, a_z
//
// Reference:
//   Montenbruck & Gill (2000), "Satellite Orbits", §3.2
// ============================================================

#include "VectorFunctions/ASSET_VectorFunctions.h"
#include <cmath>
#include <vector>

namespace ASSET {

struct SphericalHarmonicGravity
    : VectorFunction<SphericalHarmonicGravity, 6, 3, AutodiffFwd, AutodiffFwd> {

  using Base = VectorFunction<SphericalHarmonicGravity, 6, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)

  double mu;
  double radius;
  int degree;
  // Coefficients stored flat: index = n*(degree+1) + m
  std::vector<double> C;
  std::vector<double> S;

  SphericalHarmonicGravity() = default;

  SphericalHarmonicGravity(double mu,
                            double radius,
                            const std::vector<double>& C,
                            const std::vector<double>& S,
                            int degree)
      : mu(mu), radius(radius), degree(degree), C(C), S(S) {}

  // ----------------------------------------------------------
  // compute_impl
  //
  // Templated on Scalar so ASSET can call it with double for
  // evaluation and Eigen::AutoDiffScalar<...> for Jacobians.
  // ----------------------------------------------------------
  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    typedef typename InType::Scalar Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();

    const Scalar px = x[0];
    const Scalar py = x[1];
    const Scalar pz = x[2];

    // --------------------------------------------------------
    // Geometry
    // --------------------------------------------------------
    const Scalar r2 = px * px + py * py + pz * pz;
    const Scalar r  = sqrt(r2);
    const Scalar inv_r = Scalar(1.0) / r;

    const Scalar xy2 = px * px + py * py;
    const Scalar xy  = sqrt(xy2);
    const Scalar inv_xy = Scalar(1.0) / xy;

    const Scalar sinphi = pz * inv_r;
    const Scalar cosphi = xy * inv_r;
    const Scalar coslam = px * inv_xy;
    const Scalar sinlam = py * inv_xy;

    const int N  = degree;
    const int N1 = N + 1;
    const int sz = N1 * N1;

    // --------------------------------------------------------
    // Associated Legendre polynomials P[n,m] and dP[n,m]
    // using the same recurrence as Legendre.pyx
    // --------------------------------------------------------
    std::vector<Scalar> P(sz,  Scalar(0.0));
    std::vector<Scalar> dP(sz, Scalar(0.0));

    P[0]  = Scalar(1.0);
    dP[0] = Scalar(0.0);

    // Diagonal: P[n,n]
    for (int n = 1; n <= N; n++) {
      const int idx  = n * N1 + n;
      const int prev = (n - 1) * N1 + (n - 1);
      P[idx]  = Scalar(2 * n - 1) * cosphi * P[prev];
      dP[idx] = Scalar(-n) * sinphi * P[idx] / cosphi;
    }

    // First off-diagonal: P[n, n-1]
    for (int n = 1; n <= N; n++) {
      const int idx  = n * N1 + (n - 1);
      const int prev = (n - 1) * N1 + (n - 1);
      P[idx]  = Scalar(2 * n - 1) * sinphi * P[prev];
      dP[idx] = (Scalar(n) * sinphi * P[idx]
                 - Scalar(2 * n - 1) * P[prev])
                / cosphi;
    }

    // Remaining: P[n, m] for m < n-1
    for (int n = 2; n <= N; n++) {
      for (int m = 0; m < n - 1; m++) {
        const int idx   = n * N1 + m;
        const int prev1 = (n - 1) * N1 + m;
        const int prev2 = (n - 2) * N1 + m;
        P[idx]  = (Scalar(2 * n - 1) * sinphi * P[prev1]
                   - Scalar(n + m - 1) * P[prev2])
                  / Scalar(n - m);
        dP[idx] = (Scalar(n) * sinphi * P[idx]
                   - Scalar(n + m) * P[prev1])
                  / cosphi;
      }
    }

    // --------------------------------------------------------
    // Trig recurrence: cos(m λ), sin(m λ)
    // --------------------------------------------------------
    std::vector<Scalar> Ctrig(N1);
    std::vector<Scalar> Strig(N1);
    Ctrig[0] = Scalar(1.0);
    Strig[0] = Scalar(0.0);
    if (N >= 1) {
      Ctrig[1] = coslam;
      Strig[1] = sinlam;
    }
    for (int m = 2; m <= N; m++) {
      Ctrig[m] = coslam * Ctrig[m - 1] - sinlam * Strig[m - 1];
      Strig[m] = sinlam * Ctrig[m - 1] + coslam * Strig[m - 1];
    }

    // --------------------------------------------------------
    // Radial power sequence: (R/r)^n
    // --------------------------------------------------------
    std::vector<Scalar> rho_power(N1);
    const Scalar rho = Scalar(radius) * inv_r;
    rho_power[0] = Scalar(1.0);
    for (int n = 1; n <= N; n++) {
      rho_power[n] = rho_power[n - 1] * rho;
    }

    // --------------------------------------------------------
    // Harmonic sums
    // --------------------------------------------------------
    Scalar radial_sum    = Scalar(1.0);
    Scalar latitude_sum  = Scalar(0.0);
    Scalar longitude_sum = Scalar(0.0);

    for (int n = 2; n <= N; n++) {
      for (int m = 0; m <= n; m++) {
        const int idx   = n * N1 + m;
        const double Cn = C[idx];
        const double Sn = S[idx];

        const Scalar Pnm  = P[idx];
        const Scalar dPnm = dP[idx];

        const Scalar trig  = Scalar(Cn) * Ctrig[m] + Scalar(Sn) * Strig[m];
        const Scalar dtrig = Scalar(m) * (Scalar(Sn) * Ctrig[m]
                                          - Scalar(Cn) * Strig[m]);

        const Scalar rho_n = rho_power[n];
        const Scalar term  = rho_n * Pnm * trig;

        radial_sum    += Scalar(n + 1) * term;
        latitude_sum  += rho_n * dPnm * trig;
        longitude_sum += rho_n * Pnm  * dtrig;
      }
    }

    // --------------------------------------------------------
    // Cartesian acceleration
    // --------------------------------------------------------
    const Scalar mu_r2 = Scalar(mu) / r2;

    const Scalar ar   = -mu_r2 * radial_sum;
    const Scalar aph  =  mu_r2 * latitude_sum;
    const Scalar alam =  mu_r2 * longitude_sum / cosphi;

    // Spherical → Cartesian (same as SphericalToCartesian in pines_math.pyx)
    fx[0] = cosphi * coslam * ar - sinphi * coslam * aph - sinlam * alam;
    fx[1] = cosphi * sinlam * ar - sinphi * sinlam * aph + coslam * alam;
    fx[2] = sinphi * ar          + cosphi * aph;
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<SphericalHarmonicGravity>(m, name)
                   .def(py::init<double,
                                 double,
                                 const std::vector<double>&,
                                 const std::vector<double>&,
                                 int>());
    Base::DenseBaseBuild(obj);
  }
};

}  // namespace ASSET
