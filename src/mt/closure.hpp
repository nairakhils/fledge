#pragma once

#include "mt/roche.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <utility>

// ── Non-conservative mass-transfer closure ──────────────────────────────
//
// The two most important closure choices in the entire mass-transfer model:
//
//   1. beta  — the fraction of transferred mass that is accreted (vs lost)
//   2. jloss — the specific angular momentum carried away by the lost gas
//
// These determine whether the orbit shrinks or widens under mass transfer,
// which is the central dynamical question the secular model must answer.
//
// Beta modes:
//   A: fixed beta (regression tests, simple experiments)
//   B: Lu-2023 / L2massloss lookup table (deferred to later phase)
//   C: super-Eddington logistic threshold
//
// jloss modes:
//   1: cooled-default (eta_j = 1, jloss = jL2)
//   2: adiabatic calibration (eta_j from Scherbak piecewise-linear fit)
//   3: user-specified eta_j

// ── Closure result ──────────────────────────────────────────────────────

struct Closure {
    double beta;              // accreted fraction, 0 <= beta <= 1
    double mdot_loss;         // (1 - beta) * mdot_tr
    double jloss;             // specific angular momentum of lost gas
    double eta_j;             // jloss / jL2 ratio actually used
};

// ── Mode A: fixed beta ──────────────────────────────────────────────────
// The simplest closure. beta is a user-specified constant.
// jloss = eta_j * jL2.

inline auto closure_fixed_beta(double mdot_tr, double beta_fixed,
                               double jL2, double eta_j) -> Closure
{
    double mdot_loss = (1.0 - beta_fixed) * mdot_tr;
    double jloss = eta_j * jL2;
    return {beta_fixed, mdot_loss, jloss, eta_j};
}

// ── Mode C: super-Eddington logistic ────────────────────────────────────
// Transition from conservative to non-conservative transfer when the
// outer accretion disk becomes super-Eddington.
//
// mdot_edd = 4 * pi * c * r_disk / kappa           (Eq. 10)
// 1 - beta = 1 / (1 + (mdot_edd / mdot_tr)^n)      (Eq. 11)
//
// r_disk = f_disk_outer * R_L_accretor
// R_L_accretor uses the Eggleton formula with q_a = M_acc / M_don = 1/q_d.
//
// Mode B (Lu-2023 / L2massloss lookup table) is the recommended production
// closure but requires external table data. It is deferred to a later phase.

inline auto closure_super_eddington(double mdot_tr, double M_accretor, double a,
                                    double kappa, double f_disk_outer,
                                    double logistic_n, double jL2,
                                    double eta_j) -> Closure
{
    // r_disk is the pre-computed outer disk radius in code units.
    // The caller should set f_disk_outer = f_d * R_L_accretor where R_L_accretor
    // is the accretor's Roche-lobe radius and f_d ~ 0.8 is the default.
    //
    // The speed of light is absorbed into kappa: the caller passes kappa
    // such that 4*pi/kappa has the right units for mdot_edd.
    double r_disk = f_disk_outer;
    double mdot_edd = 4.0 * std::numbers::pi * r_disk / kappa;

    double ratio = mdot_edd / mdot_tr;
    double ratio_n = std::pow(ratio, logistic_n);
    double one_minus_beta = 1.0 / (1.0 + ratio_n);
    double beta = 1.0 - one_minus_beta;

    double mdot_loss = one_minus_beta * mdot_tr;
    double jloss = eta_j * jL2;
    return {beta, mdot_loss, jloss, eta_j};
}

// ── Scherbak adiabatic calibration for eta_j ────────────────────────────
// Piecewise-linear interpolation through the adiabatic paper's values:
//   q = M_acc/M_don = 0.25 → eta_j = 0.95
//   q = 0.50 → eta_j = 0.90
//   q = 1.00 → eta_j = 0.80
//   q = 2.00 → eta_j = 0.65
//
// Clamped to the endpoint values outside [0.25, 2.0].

inline auto scherbak_adiabatic_eta(double q) -> double
{
    // Calibration nodes
    constexpr double qs[]   = {0.25, 0.50, 1.00, 2.00};
    constexpr double etas[] = {0.95, 0.90, 0.80, 0.65};
    constexpr int n = 4;

    if (q <= qs[0]) return etas[0];
    if (q >= qs[n - 1]) return etas[n - 1];

    // Find the bracketing interval
    for (int i = 0; i < n - 1; ++i) {
        if (q <= qs[i + 1]) {
            double t = (q - qs[i]) / (qs[i + 1] - qs[i]);
            return etas[i] + t * (etas[i + 1] - etas[i]);
        }
    }
    return etas[n - 1]; // unreachable
}

// ── Compute j_loss from mode selection ──────────────────────────────────
// Returns {jloss, eta_j_used}.
//
// mode = "cooled"    → eta_j = 1.0 (Scherbak cooled default)
// mode = "adiabatic" → eta_j from scherbak_adiabatic_eta(q)
// mode = "fixed"     → eta_j = eta_j_fixed (user-specified)

inline auto compute_jloss(const std::string& mode, double jL2, double eta_j_fixed,
                          double q) -> std::pair<double, double>
{
    double eta;
    if (mode == "cooled") {
        eta = 1.0;
    } else if (mode == "adiabatic") {
        eta = scherbak_adiabatic_eta(q);
    } else {
        // "fixed" or any other mode: use the user-specified value
        eta = eta_j_fixed;
    }
    return {eta * jL2, eta};
}
