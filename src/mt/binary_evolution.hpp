#pragma once

#include "mt/closure.hpp"
#include "mt/ode.hpp"
#include "mt/rates.hpp"
#include "mt/roche.hpp"
#include "mt/subcycle.hpp"
#include "vec3.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

// ── Binary mass-transfer secular evolution ────────────────────────────────
//
// Evolves a circular binary through Roche-lobe overflow mass transfer.
// The state tracks donor/accretor masses, separation, donor radius, and
// orbital phase. The secular ODEs for da/dt and dR_d/dt follow the
// design document (Eqs. 4, 15, Appendix A).
//
// The advance_binary() function:
//   1. Evaluates Roche geometry and overflow
//   2. Computes mdot_tr from the chosen rate law
//   3. Computes beta, jloss from the chosen closure
//   4. Integrates the secular state with RK2, subcycled for safety
//   5. Updates cumulative mass budgets and derived quantities

// ── Mass-transfer configuration ─────────────────────────────────────────
// Kept separate from the main Config for now; will be merged later.

struct MassTransferConfig {
    bool circular_only = true;

    // Initial conditions
    double donor_mass0 = 1.0;
    double accretor_mass0 = 0.5;
    double donor_radius0 = 0.5;
    double separation0 = 1.0;
    double phase0 = 0.0;

    // Donor radius evolution
    std::string donor_radius_mode = "response_law"; // or "prescribed"
    double zeta_star = 0.0;    // d ln R_d / d ln M_d
    double tau_drive = 1e10;   // thermal/nuclear driving timescale

    // Mass-transfer rate
    double Hp_over_R = 1e-4;   // pressure scale height / donor radius
    double mdot0 = 1e-6;       // normalization for Ritter law
    double mdot_cap = 0.0;     // 0 = no cap
    std::string mdot_mode = "ritter"; // "ritter", "capped", "prescribed"
    double mdot_prescribed = 0.0;

    // Non-conservative closure
    std::string beta_mode = "fixed";
    double beta_fixed = 1.0;
    double kappa = 0.34;
    double f_disk_outer = 0.8;
    double logistic_n = 2.0;

    // Angular momentum closure
    std::string jloss_mode = "L2_exact"; // "L2_exact", "scherbak_adiabatic", "fixed_eta"
    double eta_j_fixed = 1.0;

    double max_fractional_change = 0.01; // subcycling threshold
};

// ── Binary mass-transfer state ──────────────────────────────────────────

struct BinaryMTState {
    double donor_mass;
    double accretor_mass;
    double donor_radius;
    double separation;
    double phase;            // orbital phase phi

    double mdot_transfer = 0.0;
    double beta = 1.0;
    double mdot_loss = 0.0;
    double jloss = 0.0;

    double cumulative_transferred = 0.0;
    double cumulative_accreted = 0.0;
    double cumulative_lost = 0.0;

    // Derived quantities (recomputed each step)
    double roche_radius = 0.0;
    double overflow_depth = 0.0;
    double orbital_angular_momentum = 0.0;
    double orbital_period = 0.0;
};

// ── Derived quantity helpers ─────────────────────────────────────────────

inline void recompute_derived(BinaryMTState& mt)
{
    double M_tot = mt.donor_mass + mt.accretor_mass;
    double q_d = mt.donor_mass / mt.accretor_mass;

    mt.roche_radius = eggleton_roche_radius(q_d) * mt.separation;
    mt.overflow_depth = mt.donor_radius - mt.roche_radius;

    // J_orb = M_d * M_a * sqrt(G * a / M_tot), G = 1
    mt.orbital_angular_momentum =
        mt.donor_mass * mt.accretor_mass *
        std::sqrt(mt.separation / M_tot);

    double Omega = std::sqrt(M_tot / (mt.separation * mt.separation * mt.separation));
    mt.orbital_period = 2.0 * std::numbers::pi / Omega;
}

// ── Initialize state from config ────────────────────────────────────────

inline auto init_binary_mt(const MassTransferConfig& cfg) -> BinaryMTState
{
    BinaryMTState mt;
    mt.donor_mass = cfg.donor_mass0;
    mt.accretor_mass = cfg.accretor_mass0;
    mt.donor_radius = cfg.donor_radius0;
    mt.separation = cfg.separation0;
    mt.phase = cfg.phase0;
    mt.beta = cfg.beta_fixed;
    recompute_derived(mt);
    return mt;
}

// ── Star positions in the inertial frame ────────────────────────────────
// COM at origin. Circular orbit in the x-y plane.
//   r_d = -(M_a/M_tot) * a * (cos phi, sin phi, 0)
//   r_a = +(M_d/M_tot) * a * (cos phi, sin phi, 0)

inline auto binary_positions(const BinaryMTState& mt)
    -> std::pair<Vec3, Vec3>
{
    double M_tot = mt.donor_mass + mt.accretor_mass;
    double cp = std::cos(mt.phase);
    double sp = std::sin(mt.phase);

    Vec3 r_hat(cp, sp, 0.0);
    Vec3 r_d = -(mt.accretor_mass / M_tot) * mt.separation * r_hat;
    Vec3 r_a = +(mt.donor_mass / M_tot) * mt.separation * r_hat;
    return {r_d, r_a};
}

// ── Star velocities in the inertial frame ───────────────────────────────
//   v_d = -(M_a/M_tot) * a * Omega * (-sin phi, cos phi, 0)
//   v_a = +(M_d/M_tot) * a * Omega * (-sin phi, cos phi, 0)

inline auto binary_velocities(const BinaryMTState& mt)
    -> std::pair<Vec3, Vec3>
{
    double M_tot = mt.donor_mass + mt.accretor_mass;
    double Omega = std::sqrt(M_tot / (mt.separation * mt.separation * mt.separation));
    double sp = std::sin(mt.phase);
    double cp = std::cos(mt.phase);

    Vec3 phi_hat(-sp, cp, 0.0);
    Vec3 v_d = -(mt.accretor_mass / M_tot) * mt.separation * Omega * phi_hat;
    Vec3 v_a = +(mt.donor_mass / M_tot) * mt.separation * Omega * phi_hat;
    return {v_d, v_a};
}

// ── Core advance function ───────────────────────────────────────────────
// Advances the binary mass-transfer state by one outer timestep dt.
// Uses subcycled RK2 for the secular ODE integration.

inline void advance_binary(BinaryMTState& mt, const MassTransferConfig& cfg, double dt)
{
    // 1. Roche geometry
    recompute_derived(mt);

    // 2. Mass-transfer rate
    double Hp = cfg.Hp_over_R * mt.donor_radius;
    if (cfg.mdot_mode == "prescribed") {
        mt.mdot_transfer = cfg.mdot_prescribed;
    } else if (cfg.mdot_mode == "capped") {
        mt.mdot_transfer = capped_overflow_rate(
            mt.overflow_depth, cfg.mdot0, Hp,
            mt.donor_mass, cfg.tau_drive, cfg.mdot_cap > 0.0 ? cfg.mdot_cap : 0.1);
    } else {
        // "ritter" (default)
        mt.mdot_transfer = ritter_overflow_rate(mt.overflow_depth, cfg.mdot0, Hp);
    }

    // 3. Closure: beta and jloss
    double q = mt.accretor_mass / mt.donor_mass; // q = M_a / M_d
    double jL2_val = 0.0;

    // Compute jL2 only if there is non-conservative loss to avoid
    // unnecessary Lagrange-point solves when beta = 1.
    bool needs_jL2 = (cfg.beta_mode != "fixed" || cfg.beta_fixed < 1.0);
    if (needs_jL2 || cfg.jloss_mode != "fixed_eta") {
        jL2_val = j_L2(mt.donor_mass, mt.accretor_mass, mt.separation);
    }

    // Determine eta_j
    double eta_j;
    if (cfg.jloss_mode == "scherbak_adiabatic") {
        eta_j = scherbak_adiabatic_eta(q);
    } else if (cfg.jloss_mode == "L2_exact") {
        eta_j = 1.0;
    } else {
        eta_j = cfg.eta_j_fixed;
    }

    // Determine beta
    if (cfg.beta_mode == "super_eddington") {
        double q_a = 1.0 / (mt.donor_mass / mt.accretor_mass);
        double rl_a = eggleton_roche_radius(q_a) * mt.separation;
        double r_disk = cfg.f_disk_outer * rl_a;
        auto cl = closure_super_eddington(
            mt.mdot_transfer, mt.accretor_mass, mt.separation,
            cfg.kappa, r_disk, cfg.logistic_n, jL2_val, eta_j);
        mt.beta = cl.beta;
        mt.jloss = cl.jloss;
        mt.mdot_loss = cl.mdot_loss;
    } else {
        // "fixed" (default)
        auto cl = closure_fixed_beta(mt.mdot_transfer, cfg.beta_fixed, jL2_val, eta_j);
        mt.beta = cl.beta;
        mt.jloss = cl.jloss;
        mt.mdot_loss = cl.mdot_loss;
    }

    // 4. If no transfer, just advance phase
    if (mt.mdot_transfer <= 0.0) {
        double M_tot = mt.donor_mass + mt.accretor_mass;
        double Omega = std::sqrt(M_tot / (mt.separation * mt.separation * mt.separation));
        mt.phase += Omega * dt;
        recompute_derived(mt);
        return;
    }

    // 5. Integrate secular ODEs with subcycled RK2.
    // State vector: y = {M_d, M_a, a, R_d, phase}
    // Closure quantities (mdot_tr, beta, jloss) are held constant over
    // the outer timestep; subcycling refines the ODE integration only.

    double mdot_tr = mt.mdot_transfer;
    double beta = mt.beta;
    double jloss_val = mt.jloss;
    bool evolve_radius = (cfg.donor_radius_mode == "response_law");
    double zeta_star = cfg.zeta_star;
    double tau_drive = cfg.tau_drive;

    auto rhs = [&](double /*t*/, const std::vector<double>& y) -> std::vector<double> {
        double M_d = y[0];
        double M_a = y[1];
        double a   = y[2];
        double R_d = y[3];
        // y[4] = phase

        double M_tot = M_d + M_a;

        // J_orb = M_d * M_a * sqrt(a / M_tot)   (G = 1)
        double J_orb = M_d * M_a * std::sqrt(a / M_tot);

        // da/dt = a * [ -2*(1-beta)*mdot_tr*jloss/J_orb
        //              + 2*mdot_tr/M_d
        //              - 2*beta*mdot_tr/M_a
        //              - (1-beta)*mdot_tr/M_tot ]
        double adot = a * (
            -2.0 * (1.0 - beta) * mdot_tr * jloss_val / J_orb
            + 2.0 * mdot_tr / M_d
            - 2.0 * beta * mdot_tr / M_a
            - (1.0 - beta) * mdot_tr / M_tot
        );

        // dR_d/dt = R_d * (zeta_star * (-mdot_tr)/M_d + 1/tau_drive)
        double Rdot = 0.0;
        if (evolve_radius) {
            Rdot = R_d * (zeta_star * (-mdot_tr) / M_d + 1.0 / tau_drive);
        }

        // dphi/dt = Omega = sqrt(M_tot / a^3)
        double Omega = std::sqrt(M_tot / (a * a * a));

        return {-mdot_tr, beta * mdot_tr, adot, Rdot, Omega};
    };

    // Subcycled integration
    SecularODE ode;
    ode.y = {mt.donor_mass, mt.accretor_mass, mt.separation,
             mt.donor_radius, mt.phase};

    subcycle(dt, cfg.max_fractional_change, [&](double dt_sub) -> double {
        if (dt_sub > 0.0) {
            ode.step_rk2(0.0, dt_sub, rhs);
        }
        // Return rate scale: max(|dM_d/dt|/M_d, |da/dt|/a)
        double M_d = ode.y[0];
        double a = ode.y[2];
        double rate_Md = std::abs(mdot_tr) / std::abs(M_d);
        auto dydt = rhs(0.0, ode.y);
        double rate_a = std::abs(dydt[2]) / std::abs(a);
        return std::max(rate_Md, rate_a);
    });

    // 6. Unpack and update cumulative budgets
    double dm_transferred = mdot_tr * dt;
    double dm_accreted = beta * mdot_tr * dt;
    double dm_lost = (1.0 - beta) * mdot_tr * dt;

    mt.donor_mass = ode.y[0];
    mt.accretor_mass = ode.y[1];
    mt.separation = ode.y[2];
    mt.donor_radius = ode.y[3];
    mt.phase = ode.y[4];

    mt.cumulative_transferred += dm_transferred;
    mt.cumulative_accreted += dm_accreted;
    mt.cumulative_lost += dm_lost;

    // 7. Recompute derived quantities
    recompute_derived(mt);
}
