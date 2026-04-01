#pragma once

#include "mt/roche.hpp"

#include <algorithm>
#include <cmath>

// ── Mass-transfer rate computation ──────────────────────────────────────
//
// Computes the Roche-lobe overflow state and the mass-transfer rate
// mdot_tr (mass leaving the donor per unit time, always >= 0).
//
// Three rate prescriptions from the design document:
//   Mode 1: directly prescribed mdot_tr(t)
//   Mode 2: Ritter-like exponential overflow (Eq. 8)
//   Mode 3: capped overflow (Eq. 9)

// ── Overflow state ──────────────────────────────────────────────────────

struct OverflowState {
    double roche_radius;      // R_L (dimensional, same units as a)
    double overflow_depth;    // Delta_R = R_donor - R_L (negative = underfilling)
    double mdot_transfer;     // mass leaving donor per unit time (>= 0)
};

// ── Compute Roche radius and overflow depth ─────────────────────────────
// Uses the Eggleton formula for R_L. The overflow depth Delta_R is
// positive when the donor overfills its Roche lobe.

inline auto compute_overflow(double M_donor, double M_accretor, double a,
                             double R_donor) -> OverflowState
{
    double q_d = M_donor / M_accretor;
    double rl_over_a = eggleton_roche_radius(q_d);
    double rl = rl_over_a * a;
    double delta_r = R_donor - rl;
    return {rl, delta_r, 0.0};
}

// ── Mode 2: Ritter-like exponential overflow law ────────────────────────
// mdot_tr = mdot0 * exp(Delta_R / Hp)   if Delta_R > 0
//         = 0                            if Delta_R <= 0
//
// mdot0: normalization rate
// Hp:    donor photospheric pressure scale height (dimensional)

inline auto ritter_overflow_rate(double overflow_depth, double mdot0,
                                 double Hp) -> double
{
    if (overflow_depth <= 0.0) return 0.0;
    return mdot0 * std::exp(overflow_depth / Hp);
}

// ── Mode 3: capped overflow law ─────────────────────────────────────────
// mdot_tr = min(mdot0 * exp(Delta_R / Hp),  f_sat * M_donor / tau_drive)
//
// The cap prevents the rate from exceeding a fraction f_sat of the donor
// mass per driving timescale, avoiding spurious common-envelope behavior.

inline auto capped_overflow_rate(double overflow_depth, double mdot0, double Hp,
                                 double M_donor, double tau_drive,
                                 double f_sat = 0.1) -> double
{
    double ritter = ritter_overflow_rate(overflow_depth, mdot0, Hp);
    double cap = f_sat * M_donor / tau_drive;
    return std::min(ritter, cap);
}

// ── Mode 1: prescribed rate ─────────────────────────────────────────────
// Simply returns the user-specified value. Exists for API completeness
// so all three modes have a callable function.

inline auto prescribed_rate(double mdot_prescribed) -> double
{
    return mdot_prescribed;
}
