#pragma once

#include "vec3.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

// ── Newton-Raphson root finder ────────────────────────────────────────────
// Solves f(x) = 0 starting from x0.
// Convergence: |f(x)| < 1e-15, max 10 iterations.
template <typename F, typename G>
auto newton_raphson(F f, G fprime, double x) -> double {
    int n = 0;
    while (std::abs(f(x)) > 1e-15) {
        x -= f(x) / fprime(x);
        ++n;
        if (n > 10) {
            throw std::runtime_error("newton_raphson: no solution!");
        }
    }
    return x;
}

// ── Kepler's equation solver ──────────────────────────────────────────────
// Solves E - e*sin(E) = M for eccentric anomaly E.
inline auto eccentric_anomaly(double mean_anomaly, double eccentricity) -> double {
    return newton_raphson(
        [&](double E) { return E - eccentricity * std::sin(E) - mean_anomaly; },
        [&](double E) { return 1.0 - eccentricity * std::cos(E); },
        mean_anomaly);
}

// ── Binary orbital state ─────────────────────────────────────────────────
// Returns {primary_position, secondary_position} with CM at origin.
// Matches the Rust orbital_state() exactly.
inline auto orbital_state(double mass, double q, double a,
                          double ex, double ey, double inclination,
                          double time_since_periapse) -> std::pair<Vec3, Vec3> {
    using std::numbers::pi;

    double e_mag = std::sqrt(ex * ex + ey * ey);

    // Mean angular motion and period
    double omega = std::sqrt(mass / (a * a * a));
    double period = 2.0 * pi / omega;

    // Wrap time into [0, period)
    double t = time_since_periapse - period * std::floor(time_since_periapse / period);
    double mean_anom = omega * t;

    // Solve Kepler's equation
    double ecc_anom = eccentric_anomaly(mean_anom, e_mag);

    // Position in orbital plane
    double x = a * (std::cos(ecc_anom) - e_mag);
    double y = a * std::sqrt(1.0 - e_mag * e_mag) * std::sin(ecc_anom);
    double z = 0.0;

    // Rotate by inclination (i) wrt disk plane
    double ci = std::cos(inclination);
    double si = std::sin(inclination);
    double x_rot_i = x * ci - z * si;
    double y_rot_i = y;
    double z_rot_i = x * si + z * ci;

    // Rotate by argument of periapsis (omega = atan2(ey, ex))
    double arg_peri = std::atan2(ey, ex);
    double cw = std::cos(arg_peri);
    double sw = std::sin(arg_peri);
    double x_rot_w = x_rot_i * cw - y_rot_i * sw;
    double y_rot_w = x_rot_i * sw + y_rot_i * cw;
    double z_rot_w = z_rot_i;

    // Secondary position
    double x2 = -x_rot_w / (1.0 + q);
    double y2 = -y_rot_w / (1.0 + q);
    double z2 =  z_rot_w / (1.0 + q);

    // Primary position: r1 = -q * r2
    double x1 = -x2 * q;
    double y1 = -y2 * q;
    double z1 = -z2 * q;

    return {Vec3(x1, y1, z1), Vec3(x2, y2, z2)};
}

// ── Self-test ─────────────────────────────────────────────────────────────
#ifdef FLEDGE_TEST_ORBITS
#include <cassert>
#include <print>

inline void test_orbits() {
    using std::numbers::pi;
    constexpr double tol = 1e-12;

    // Circular orbit: mass=1, q=1, a=1, e=(0,0), i=0
    // Each body orbits at radius a/(1+q) = 0.5 from CM
    double mass = 1.0, q = 1.0, a = 1.0;
    double ex = 0.0, ey = 0.0, incl = 0.0;
    double r_expected = a / (1.0 + q); // 0.5

    double period = 2.0 * pi / std::sqrt(mass / (a * a * a));

    // Test at several phases
    for (int i = 0; i < 20; ++i) {
        double t = period * i / 20.0;
        auto [r1, r2] = orbital_state(mass, q, a, ex, ey, incl, t);

        // Both bodies should be on a circle of radius r_expected
        double mag1 = r1.mag();
        double mag2 = r2.mag();
        assert(std::abs(mag1 - r_expected) < tol);
        assert(std::abs(mag2 - r_expected) < tol);

        // Center of mass: m1*r1 + m2*r2 = 0
        // m1 = mass/(1+q), m2 = q*mass/(1+q)
        double m1 = mass / (1.0 + q);
        double m2 = q * m1;
        Vec3 cm = r1 * m1 + r2 * m2;
        assert(cm.mag() < tol);

        // Orbit lies in the xy-plane (no inclination)
        assert(std::abs(r1.z) < tol);
        assert(std::abs(r2.z) < tol);
    }

    // Eccentric orbit: verify CM still at origin
    ex = 0.3; ey = 0.4; q = 0.5;
    for (int i = 0; i < 20; ++i) {
        double t = period * i / 20.0;
        auto [r1, r2] = orbital_state(mass, q, a, ex, ey, incl, t);

        double m1 = mass / (1.0 + q);
        double m2 = q * m1;
        Vec3 cm = r1 * m1 + r2 * m2;
        assert(cm.mag() < tol);
    }

    // Inclined eccentric orbit: CM at origin still holds
    incl = 0.5;
    for (int i = 0; i < 20; ++i) {
        double t = period * i / 20.0;
        auto [r1, r2] = orbital_state(mass, q, a, ex, ey, incl, t);

        double m1 = mass / (1.0 + q);
        double m2 = q * m1;
        Vec3 cm = r1 * m1 + r2 * m2;
        assert(cm.mag() < tol);
    }

    std::println("orbits: all tests passed");
}

int main() {
    test_orbits();
    return 0;
}
#endif
