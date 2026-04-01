#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

// ── Roche geometry utilities for binary mass transfer ─────────────────────
//
// Coordinate convention: "donor-at-origin frame"
//   - Donor  at x = 0
//   - Accretor at x = a
//   - Center of mass at x = M_a * a / (M_d + M_a)
//
// All Lagrange-point solvers work in normalized coordinates (a = 1,
// M_tot = 1, G = 1, Omega^2 = 1). The single free parameter is
// q_d = M_donor / M_accretor. Results are returned as x/a (dimensionless).
//
// In normalized coords:
//   mu_d  = q_d / (1 + q_d)     (donor mass fraction)
//   mu_a  = 1   / (1 + q_d)     (accretor mass fraction)
//   x_com = mu_a = 1 / (1 + q_d)
//
// Lagrange point locations:
//   L1: between donor and accretor, 0 < x/a < 1
//   L2: far side of donor (away from accretor), x/a < 0
//   L3: far side of accretor (away from donor), x/a > 1

// ── Brent's method root solver ────────────────────────────────────────────
// Standard Brent's method combining bisection, secant, and inverse
// quadratic interpolation. Requires f(lo) and f(hi) to have opposite signs.

namespace detail {

template <typename F>
auto brent_solve(F f, double lo, double hi, double tol, int max_iter = 100) -> double
{
    double fa = f(lo);
    double fb = f(hi);

    if (fa * fb > 0.0) {
        throw std::runtime_error(
            "brent_solve: f(lo) and f(hi) must have opposite signs, got f("
            + std::to_string(lo) + ")=" + std::to_string(fa) + ", f("
            + std::to_string(hi) + ")=" + std::to_string(fb));
    }

    // Ensure |f(lo)| >= |f(hi)| so that hi is the better bracket end
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(lo, hi);
        std::swap(fa, fb);
    }

    double c = lo;
    double fc = fa;
    bool mflag = true;
    double d = 0.0; // previous step size

    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(fb) < tol) return hi;
        if (std::abs(hi - lo) < tol) return hi;

        double s;
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = lo * fb * fc / ((fa - fb) * (fa - fc))
              + hi * fa * fc / ((fb - fa) * (fb - fc))
              +  c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = hi - fb * (hi - lo) / (fb - fa);
        }

        // Conditions for rejecting s and using bisection instead
        double mid = 0.5 * (lo + hi);
        bool cond1 = !((s > std::min(mid, hi) - tol) && (s < std::max(mid, hi) + tol));
        bool cond2 = mflag  && std::abs(s - hi) >= 0.5 * std::abs(hi - c);
        bool cond3 = !mflag && std::abs(s - hi) >= 0.5 * std::abs(c - d);
        bool cond4 = mflag  && std::abs(hi - c) < tol;
        bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = mid;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s);
        d = c;
        c = hi;
        fc = fb;

        if (fa * fs < 0.0) {
            hi = s;
            fb = fs;
        } else {
            lo = s;
            fa = fs;
        }

        // Keep |f(lo)| >= |f(hi)|
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(lo, hi);
            std::swap(fa, fb);
        }
    }

    throw std::runtime_error("brent_solve: failed to converge in "
                             + std::to_string(max_iter) + " iterations");
}

} // namespace detail

// ── Eggleton Roche-lobe radius ────────────────────────────────────────────
// Returns R_L / a using the Eggleton (1983) approximation.
// q_d = M_donor / M_accretor (must be positive).
// Accurate to ~1% for all mass ratios.

inline auto eggleton_roche_radius(double q_d) -> double
{
    double q13 = std::cbrt(q_d);
    double q23 = q13 * q13;
    return 0.49 * q23 / (0.6 * q23 + std::log(1.0 + q13));
}

// ── Effective-potential gradient on the binary axis ────────────────────────
// Computes dΦ_eff/dx on the x-axis in the corotating frame, using
// normalized coordinates (a=1, G=1, M_tot=1, Omega^2=1).
//
// F(xi) = mu_d / (xi * |xi|) + mu_a / ((xi-1) * |xi-1|) - (xi - x_com)
//
// The 1/(x·|x|) form is equivalent to x/|x|^3 and correctly handles
// sign changes across x=0 and x=1 without case-splitting:
//   x > 0 → 1/(x·|x|) = +1/x^2  (gradient points away from mass)
//   x < 0 → 1/(x·|x|) = -1/x^2  (gradient points toward mass)
//
// Roots of this function are the collinear Lagrange points L1, L2, L3.

inline auto roche_force(double xi, double q_d) -> double
{
    double mu_d = q_d / (1.0 + q_d);
    double mu_a = 1.0 / (1.0 + q_d);
    double x_com = mu_a;

    double r_d = xi;
    double r_a = xi - 1.0;

    return mu_d / (r_d * std::abs(r_d))
         + mu_a / (r_a * std::abs(r_a))
         - (xi - x_com);
}

// ── Lagrange point L1 ────────────────────────────────────────────────────
// Returns x_L1 / a in the donor-at-origin frame.
// L1 lies between donor and accretor: 0 < x_L1/a < 1.
// q_d = M_donor / M_accretor.

inline auto find_lagrange_L1(double q_d, double tol = 1e-12) -> double
{
    auto f = [q_d](double xi) { return roche_force(xi, q_d); };
    return detail::brent_solve(f, 1e-6, 1.0 - 1e-6, tol);
}

// ── Lagrange point L2 ────────────────────────────────────────────────────
// Returns x_L2 / a in the donor-at-origin frame.
// L2 is on the far side of the DONOR from the accretor: x_L2/a < 0.
// The returned value is negative.

inline auto find_lagrange_L2(double q_d, double tol = 1e-12) -> double
{
    auto f = [q_d](double xi) { return roche_force(xi, q_d); };
    return detail::brent_solve(f, -3.0, -1e-6, tol);
}

// ── Lagrange point L3 ────────────────────────────────────────────────────
// Returns x_L3 / a in the donor-at-origin frame.
// L3 is on the far side of the ACCRETOR from the donor: x_L3/a > 1.

inline auto find_lagrange_L3(double q_d, double tol = 1e-12) -> double
{
    auto f = [q_d](double xi) { return roche_force(xi, q_d); };
    return detail::brent_solve(f, 1.0 + 1e-6, 4.0, tol);
}

// ── Specific angular momentum at L2 ──────────────────────────────────────
// Returns the specific angular momentum of a corotating fluid element at
// L2 in the inertial frame, with G = 1:
//
//   j_L2 = Omega * x_L2_com^2
//
// where Omega = sqrt(M_tot / a^3), and x_L2_com is the distance from the
// center of mass to L2. In the donor-at-origin frame:
//   x_L2_com = xi_L2 * a - M_accretor * a / M_tot
//
// The result is always positive (prograde specific angular momentum).

inline auto j_L2(double M_donor, double M_accretor, double a) -> double
{
    double q_d = M_donor / M_accretor;
    double xi_L2 = find_lagrange_L2(q_d);

    double M_tot = M_donor + M_accretor;
    double x_com = M_accretor * a / M_tot;
    double x_L2_com = xi_L2 * a - x_com;

    double Omega = std::sqrt(M_tot / (a * a * a));
    return Omega * x_L2_com * x_L2_com;
}
