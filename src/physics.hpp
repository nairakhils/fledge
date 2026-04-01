#pragma once

#include "config.hpp"
#include "orbits.hpp"
#include "state.hpp"
#include "vec3.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <tuple>
#include <vector>

// ── Gravitational acceleration on a test particle ────────────────────────
// acc = sum_i  -m_i * (r - r_i) / (|r - r_i|^2 + eps^2)^{3/2}
// Skips zero-separation (self-interaction guard for massive bodies).
inline auto compute_acceleration(Vec3 pos,
                                 const std::vector<Vec3>& mass_positions,
                                 const std::vector<double>& masses,
                                 double softening,
                                 size_t n_masses) -> Vec3 {
    Vec3 acc = Vec3::zero();
    double eps2 = softening * softening;
    for (size_t i = 0; i < n_masses; ++i) {
        Vec3 r_sep = pos - mass_positions[i];
        if (r_sep.mag() == 0.0) continue;
        double r2 = r_sep.dot(r_sep) + eps2;
        acc += -masses[i] / (r2 * std::sqrt(r2)) * r_sep;
    }
    return acc;
}

// ── Initial massive body setup ───────────────────────────────────────────
// Returns {positions, velocities, masses} for the central object(s).
inline auto setup_masses(const Config& cfg)
    -> std::tuple<std::vector<Vec3>, std::vector<Vec3>, std::vector<double>> {

    std::vector<Vec3> mpos;
    std::vector<Vec3> mvel;
    std::vector<double> masses;

    if (cfg.central_object_type == "single") {
        mpos.push_back(Vec3::zero());
        mvel.push_back(Vec3::zero());
        masses.push_back(cfg.mass);

    } else if (cfg.central_object_type == "binary") {
        auto [r1, r2] = orbital_state(cfg.mass, cfg.q1, cfg.a1,
                                      cfg.e1x, cfg.e1y, cfg.inclination,
                                      cfg.tstart);
        double m1 = cfg.mass / (1.0 + cfg.q1);
        double m2 = cfg.q1 * m1;
        mpos.push_back(r1);
        mpos.push_back(r2);
        mvel.push_back(Vec3::zero());
        mvel.push_back(Vec3::zero());
        masses.push_back(m1);
        masses.push_back(m2);

    } else { // "triple"
        double m1 = cfg.mass / (1.0 + cfg.q1) / (1.0 + cfg.q2);
        double m2 = m1 * cfg.q1;
        double m3 = (m1 + m2) * cfg.q2;

        double e1_mag = Vec2{cfg.e1x, cfg.e1y}.mag();
        double e2_mag = Vec2{cfg.e2x, cfg.e2y}.mag();

        // Inner binary at periapsis, tertiary at apoapsis
        Vec3 r1 = cfg.a1 * (1.0 - e1_mag) * cfg.q1 / (1.0 + cfg.q1) * Vec3::xhat();
        Vec3 r2 = -1.0 * r1 / cfg.q1;
        Vec3 r3 = cfg.a2 * (1.0 + e2_mag) / (1.0 + cfg.q2) * Vec3::xhat();

        // Vis-viva velocities
        double r12 = (r2 - r1).mag();
        Vec3 v1 =  std::sqrt(m2 * m2 / cfg.mass * (2.0 / r12 - 1.0 / cfg.a1)) * Vec3::yhat();
        Vec3 v2 = -std::sqrt(m1 * m1 / cfg.mass * (2.0 / r12 - 1.0 / cfg.a1)) * Vec3::yhat();
        Vec3 v3 =  std::sqrt((m1 + m2) * (m1 + m2) / cfg.mass * (2.0 / r3.mag() - 1.0 / cfg.a2)) * Vec3::yhat();

        mpos.push_back(r1);  mpos.push_back(r2);  mpos.push_back(r3);
        mvel.push_back(v1);  mvel.push_back(v2);  mvel.push_back(v3);
        masses.push_back(m1); masses.push_back(m2); masses.push_back(m3);
    }

    return {mpos, mvel, masses};
}

// ── Initial particle setup ───────────────────────────────────────────────
// Sets up test-particle positions and circular Keplerian velocities.
// Handles disk centering (on primary, secondary, or arbitrary point) and
// three layout modes (ring, random_disk, uniform_disk).
inline auto setup_particles(const Config& cfg,
                            const std::vector<Vec3>& mass_positions,
                            [[maybe_unused]] const std::vector<Vec3>& mass_velocities,
                            [[maybe_unused]] const std::vector<double>& masses)
    -> std::pair<std::vector<Vec3>, std::vector<Vec3>> {

    using std::numbers::pi;
    auto n = static_cast<size_t>(cfg.num_particles);

    // Determine disk center shift and effective central mass
    Vec3 rshift = Vec3::zero();
    Vec3 vshift = Vec3::zero();
    double mshift = cfg.mass;

    if (cfg.central_object_type == "binary") {
        Vec3 r1 = mass_positions[0];
        Vec3 r2 = mass_positions[1];
        Vec3 r12 = r2 - r1;
        double theta1 = std::atan2(r1.y, r1.x);
        double theta2 = std::atan2(r2.y, r2.x);

        if (cfg.disk_center == "primary") {
            mshift = cfg.mass / (1.0 + cfg.q1);
            double v1 = std::sqrt(mshift * mshift * cfg.q1 * cfg.q1
                                  / cfg.mass / r12.mag());
            rshift = r1;
            vshift = Vec3(-v1 * std::sin(theta1), v1 * std::cos(theta1), 0.0);
        } else if (cfg.disk_center == "secondary") {
            mshift = cfg.mass * cfg.q1 / (1.0 + cfg.q1);
            double v2 = std::sqrt(mshift * mshift / cfg.q1 / cfg.q1
                                  / cfg.mass / r12.mag());
            rshift = r2;
            vshift = Vec3(-v2 * std::sin(theta2), v2 * std::cos(theta2), 0.0);
        } else { // arbitrary
            rshift = Vec3(cfg.disk_center_x, cfg.disk_center_y, cfg.disk_center_z);
            vshift = Vec3::zero();
            mshift = cfg.mass;
        }
    } else if (cfg.central_object_type == "single") {
        rshift = Vec3::zero();
        vshift = Vec3::zero();
        mshift = cfg.mass;
    } else { // triple
        rshift = Vec3::zero();
        vshift = Vec3::zero();
        mshift = cfg.mass / (1.0 + cfg.q2);
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    positions.reserve(n);
    velocities.reserve(n);

    if (cfg.setup_type == "ring") {
        double radius = cfg.ring_radius;
        for (size_t i = 0; i < n; ++i) {
            double theta = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n);
            double px = radius * std::cos(theta);
            double py = radius * std::sin(theta);
            positions.push_back(Vec3(px, py, 0.0) + rshift);

            double v = std::sqrt(mshift / radius);
            double vx = -v * std::sin(theta);
            double vy =  v * std::cos(theta);
            velocities.push_back(Vec3(vx, vy, 0.0) + vshift);
        }
    } else if (cfg.setup_type == "random_disk") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> r2_dist(
            cfg.inner_radius * cfg.inner_radius,
            cfg.outer_radius * cfg.outer_radius);
        std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * pi);

        for (size_t i = 0; i < n; ++i) {
            double r = std::sqrt(r2_dist(rng));
            double theta = theta_dist(rng);
            double px = r * std::cos(theta);
            double py = r * std::sin(theta);
            positions.push_back(Vec3(px, py, 0.0) + rshift);

            double v = std::sqrt(mshift / r);
            double vx = -v * std::sin(theta);
            double vy =  v * std::cos(theta);
            velocities.push_back(Vec3(vx, vy, 0.0) + vshift);
        }
    } else { // "uniform_disk"
        double phi = (1.0 + std::sqrt(5.0)) / 2.0;
        for (size_t i = 0; i < n; ++i) {
            double frac = static_cast<double>(i) / static_cast<double>(n);
            double r = cfg.inner_radius
                     + (cfg.outer_radius - cfg.inner_radius) * std::sqrt(frac);
            double theta = 2.0 * pi * static_cast<double>(i) * phi;
            double px = r * std::cos(theta);
            double py = r * std::sin(theta);
            positions.push_back(Vec3(px, py, 0.0) + rshift);

            double v = std::sqrt(mshift / r);
            double vx = -v * std::sin(theta);
            double vy =  v * std::cos(theta);
            velocities.push_back(Vec3(vx, vy, 0.0) + vshift);
        }
    }

    return {positions, velocities};
}

// ── Leapfrog kick-drift-kick advance ─────────────────────────────────────
// Single:  mass stays at origin.
// Binary:  mass positions set analytically via orbital_state.
// Triple:  massive bodies integrated with the same leapfrog.
// Then all test particles are advanced.
inline void advance_state(State& state, const Config& cfg,
                          const std::vector<double>& masses, double dt) {
    size_t n  = state.positions.size();
    size_t mn = state.mass_positions.size();
    double eps = cfg.softening;

    // ── Update massive bodies ─────────────────────────────────────────
    if (cfg.central_object_type == "single") {
        state.mass_positions[0] = Vec3::zero();

    } else if (cfg.central_object_type == "binary") {
        auto [r1, r2] = orbital_state(cfg.mass, cfg.q1, cfg.a1,
                                      cfg.e1x, cfg.e1y, cfg.inclination,
                                      state.time);
        state.mass_positions[0] = r1;
        state.mass_positions[1] = r2;

    } else { // triple — leapfrog KDK for massive bodies
        for (size_t i = 0; i < mn; ++i) {
            Vec3 a = compute_acceleration(state.mass_positions[i],
                         state.mass_positions, masses, eps, mn);
            state.mass_velocities[i] += a * (0.5 * dt);
        }
        for (size_t i = 0; i < mn; ++i) {
            state.mass_positions[i] += state.mass_velocities[i] * dt;
        }
        for (size_t i = 0; i < mn; ++i) {
            Vec3 a = compute_acceleration(state.mass_positions[i],
                         state.mass_positions, masses, eps, mn);
            state.mass_velocities[i] += a * (0.5 * dt);
        }
    }

    // ── Leapfrog KDK for test particles ──────────────────────────────
    // Half kick
    for (size_t i = 0; i < n; ++i) {
        Vec3 a = compute_acceleration(state.positions[i],
                     state.mass_positions, masses, eps, mn);
        state.velocities[i] += a * (0.5 * dt);
    }
    // Full drift
    for (size_t i = 0; i < n; ++i) {
        state.positions[i] += state.velocities[i] * dt;
    }
    // Half kick
    for (size_t i = 0; i < n; ++i) {
        Vec3 a = compute_acceleration(state.positions[i],
                     state.mass_positions, masses, eps, mn);
        state.velocities[i] += a * (0.5 * dt);
    }

    state.time += dt;
    state.iteration++;
}
