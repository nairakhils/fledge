#pragma once

#include "mt/binary_evolution.hpp"
#include "mt/closure.hpp"
#include "mt/roche.hpp"
#include "state.hpp"
#include "vec3.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

// ── Tracer parcel launch from L2/L3 ─────────────────────────────────────
//
// When mass is lost from the binary (mdot_loss > 0), tracer parcels are
// spawned near the L2 (and optionally L3) Lagrange point with:
//   - tangential velocity v_phi = jloss / |r_L2_com|
//   - radial velocity v_r = f_v * v_orb (dimensionless fraction mode)
//   - vertical scatter from Gaussian with sigma = tan(opening_angle) * v_r
//
// Each parcel carries mass weight w_i = mdot_loss * dt / N_launch.
//
// Coordinate convention:
//   - L2/L3 positions are computed in the donor-at-origin frame (find_lagrange_L2/L3),
//     converted to COM frame, then rotated by the binary phase to inertial frame.
//   - Launch velocities are computed in the corotating frame basis {r_hat, phi_hat, z_hat}
//     then the corotating frame velocity (Omega × r) is added for full inertial velocity.

// ── Launch parcels from a given Lagrange point ──────────────────────────
// Generic helper for both L2 and L3.

inline auto launch_from_lagrange(
    State& s,
    const BinaryMTState& mt,
    double mdot_loss_share,     // fraction of mdot_loss assigned to this point
    double xi_lagrange,         // x/a of the Lagrange point (donor-at-origin frame)
    double jloss,               // specific AM for tangential velocity
    const MassTransferConfig& cfg,
    double dt,
    std::mt19937& rng) -> int
{
    if (mdot_loss_share <= 0.0 || cfg.parcels_per_step <= 0) return 0;

    double M_tot = mt.donor_mass + mt.accretor_mass;
    double a = mt.separation;
    double Omega = std::sqrt(M_tot / (a * a * a));
    double v_orb = Omega * a;

    // L point position in COM frame (donor-at-origin → COM: shift by -x_com)
    double x_com_frac = mt.accretor_mass / M_tot; // COM position in donor-at-origin frame, as fraction of a
    double r_L_com = (xi_lagrange - x_com_frac) * a; // signed distance from COM (negative for L2)

    // Inertial position: rotate the COM-frame position by phase
    double cp = std::cos(mt.phase);
    double sp = std::sin(mt.phase);

    // In corotating frame, L point is at (r_L_com, 0, 0) relative to COM
    // Rotate to inertial frame
    double abs_r = std::abs(r_L_com);
    Vec3 r_hat_inertial(cp * (r_L_com > 0 ? 1.0 : -1.0),
                        sp * (r_L_com > 0 ? 1.0 : -1.0), 0.0);

    Vec3 r_L_inertial(r_L_com * cp, r_L_com * sp, 0.0);

    // Direction basis in inertial frame
    Vec3 r_hat = r_L_inertial / abs_r; // radial: COM → L point
    Vec3 z_hat = Vec3::zhat();
    Vec3 phi_hat = z_hat.cross(r_hat);  // tangential (prograde)

    // Launch velocity components in the corotating frame
    double v_phi = jloss / abs_r;                     // tangential from specific AM
    double v_r = cfg.launch_vr_fraction * v_orb;      // radial outward

    // Corotating frame velocity at the L point: Omega × r = Omega * r_L_com * phi_hat
    // (already handled by computing inertial velocity = corotating velocity + Omega × r)
    double v_corotation_phi = Omega * abs_r; // magnitude of Omega × r in phi direction

    // Vertical scatter distribution
    double opening_rad = cfg.opening_angle_deg * std::numbers::pi / 180.0;
    double sigma_vz = std::tan(opening_rad) * v_r;
    std::normal_distribution<double> vz_dist(0.0, std::max(sigma_vz, 1e-15));

    // Weight per parcel
    double weight = mdot_loss_share * dt / cfg.parcels_per_step;

    // Small offset along radial direction to avoid exact L point
    Vec3 r_offset = r_hat * (cfg.launch_offset * a);

    int n_launched = 0;
    for (int i = 0; i < cfg.parcels_per_step; ++i) {
        double vz = vz_dist(rng);

        // Position: L point + small radial offset
        Vec3 pos = r_L_inertial + r_offset;

        // Velocity in inertial frame:
        //   v_inertial = (v_r * r_hat + v_phi * phi_hat + vz * z_hat)   [corotating]
        //              + (v_corotation_phi * phi_hat)                     [frame rotation]
        //
        // The sign of phi_hat already encodes prograde direction.
        // For L2 (r_L_com < 0), r_hat points away from COM (toward negative x in corotating),
        // and v_r should push outward (away from COM), which means along r_hat.
        Vec3 vel = v_r * r_hat + (v_phi + v_corotation_phi) * phi_hat + vz * z_hat;

        s.positions.push_back(pos);
        s.velocities.push_back(vel);
        s.tracer_weights.push_back(weight);
        n_launched++;
    }

    return n_launched;
}

// ── Launch L2 parcels ───────────────────────────────────────────────────
// Main entry point for L2 tracer launch. Returns number of parcels launched.

inline auto launch_L2_parcels(State& s, const BinaryMTState& mt,
                              const Closure& closure,
                              const MassTransferConfig& cfg,
                              double dt) -> int
{
    if (closure.mdot_loss <= 0.0) return 0;

    double q_d = mt.donor_mass / mt.accretor_mass;
    double xi_L2 = find_lagrange_L2(q_d);

    // L2 gets (1 - l3_fraction) of the total mass loss
    double mdot_L2 = closure.mdot_loss * (1.0 - cfg.l3_fraction);

    std::mt19937 rng(cfg.launch_seed +
                     static_cast<int>(mt.phase * 1e6)); // vary seed with phase

    return launch_from_lagrange(s, mt, mdot_L2, xi_L2, closure.jloss,
                                cfg, dt, rng);
}

// ── Launch L3 parcels (optional) ────────────────────────────────────────
// Same interface as L2, uses L3 geometry and cfg.l3_fraction of mass loss.

inline auto launch_L3_parcels(State& s, const BinaryMTState& mt,
                              const Closure& closure,
                              const MassTransferConfig& cfg,
                              double dt) -> int
{
    if (closure.mdot_loss <= 0.0 || cfg.l3_fraction <= 0.0) return 0;

    double q_d = mt.donor_mass / mt.accretor_mass;
    double xi_L3 = find_lagrange_L3(q_d);

    double mdot_L3 = closure.mdot_loss * cfg.l3_fraction;

    // Offset seed for L3 to get different random scatter
    std::mt19937 rng(cfg.launch_seed + 7919 +
                     static_cast<int>(mt.phase * 1e6));

    // L3 angular momentum: use jloss scaled by |r_L3_com / r_L2_com| to maintain
    // consistency, or just use jloss directly (caller's choice). We use jloss as-is.
    return launch_from_lagrange(s, mt, mdot_L3, xi_L3, closure.jloss,
                                cfg, dt, rng);
}

// ── Apply sink radii ────────────────────────────────────────────────────
// Remove parcels that have entered the donor or accretor sink radius.
// Uses swap-and-pop for O(1) removal. Returns number of parcels removed.

inline auto apply_sinks(State& s, const BinaryMTState& mt,
                        const MassTransferConfig& cfg) -> int
{
    if (cfg.donor_sink_radius <= 0.0 && cfg.accretor_sink_radius <= 0.0)
        return 0;

    auto [r_d, r_a] = binary_positions(mt);
    double r_sink_d2 = cfg.donor_sink_radius * cfg.donor_sink_radius;
    double r_sink_a2 = cfg.accretor_sink_radius * cfg.accretor_sink_radius;

    int removed = 0;
    size_t i = 0;
    while (i < s.positions.size()) {
        Vec3 dr_d = s.positions[i] - r_d;
        Vec3 dr_a = s.positions[i] - r_a;
        double d2_d = dr_d.dot(dr_d);
        double d2_a = dr_a.dot(dr_a);

        bool in_sink = (cfg.donor_sink_radius > 0.0 && d2_d < r_sink_d2)
                    || (cfg.accretor_sink_radius > 0.0 && d2_a < r_sink_a2);

        if (in_sink) {
            // Swap-and-pop
            size_t last = s.positions.size() - 1;
            if (i < last) {
                std::swap(s.positions[i], s.positions[last]);
                std::swap(s.velocities[i], s.velocities[last]);
                std::swap(s.tracer_weights[i], s.tracer_weights[last]);
            }
            s.positions.pop_back();
            s.velocities.pop_back();
            s.tracer_weights.pop_back();
            removed++;
            // don't increment i — re-check the swapped element
        } else {
            ++i;
        }
    }
    return removed;
}
