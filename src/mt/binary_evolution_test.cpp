// Binary evolution test suite
// Tests the secular mass-transfer ODE integration against analytic solutions.

#include "mt/binary_evolution.hpp"

#include <cmath>
#include <format>
#include <numbers>
#include <print>
#include <string>

// ── Test harness ─────────────────────────────────────────────────────────

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, int test_num, const std::string& desc,
                  const std::string& detail) {
    if (ok) {
        g_pass++;
        std::println("[PASS] Test {}: {} ({})", test_num, desc, detail);
    } else {
        g_fail++;
        std::println("[FAIL] Test {}: {} ({})", test_num, desc, detail);
    }
}

// ── Test 01: Conservative transfer analytic solution ────────────────────
// For beta=1 (fully conservative), M_tot is constant and the exact
// analytic solution for the separation is:
//   a / a0 = (M_d0 * M_a0 / (M_d * M_a))^2
// We use prescribed mdot_tr for a controlled test.

static void test_01_conservative_analytic() {
    MassTransferConfig cfg;
    cfg.donor_mass0 = 2.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0; // large so overflow is always active
    cfg.phase0 = 0.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 1.0;
    cfg.donor_radius_mode = "prescribed"; // hold radius constant
    cfg.max_fractional_change = 0.001;    // tight subcycling

    auto mt = init_binary_mt(cfg);

    double M_d0 = cfg.donor_mass0;
    double M_a0 = cfg.accretor_mass0;
    double M_tot0 = M_d0 + M_a0;

    // Transfer 0.1 solar masses: time = 0.1 / 0.001 = 100
    double dt = 0.1;
    int n_steps = 1000;
    for (int i = 0; i < n_steps; ++i) {
        advance_binary(mt, cfg, dt);
    }

    double M_d = mt.donor_mass;
    double M_a = mt.accretor_mass;
    double M_tot = M_d + M_a;

    // Check M_tot conservation
    double mtot_err = std::abs(M_tot - M_tot0) / M_tot0;
    check(mtot_err < 1e-10, 1,
          "conservative M_tot conservation",
          std::format("|dM_tot/M_tot| = {:.2e}", mtot_err));

    // Check analytic separation
    double a_analytic = cfg.separation0 * std::pow(M_d0 * M_a0 / (M_d * M_a), 2.0);
    double a_err = std::abs(mt.separation - a_analytic) / a_analytic;
    check(a_err < 1e-4, 1,
          "conservative a(t) vs analytic",
          std::format("a = {:.8f}, a_analytic = {:.8f}, rel_err = {:.2e}",
                      mt.separation, a_analytic, a_err));
}

// ── Test 02: Non-conservative total mass loss ───────────────────────────
// With beta=0.5, the total mass lost should be (1-beta)*mdot_tr*dt_total.

static void test_02_nonconservative_mass_loss() {
    MassTransferConfig cfg;
    cfg.donor_mass0 = 2.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0;
    cfg.phase0 = 0.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 0.5;
    cfg.jloss_mode = "L2_exact";
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);
    double M_tot0 = mt.donor_mass + mt.accretor_mass;

    double dt = 0.1;
    int n_steps = 100;
    double dt_total = dt * n_steps;
    for (int i = 0; i < n_steps; ++i) {
        advance_binary(mt, cfg, dt);
    }

    double M_tot = mt.donor_mass + mt.accretor_mass;
    double expected_loss = (1.0 - cfg.beta_fixed) * cfg.mdot_prescribed * dt_total;
    double actual_loss = M_tot0 - M_tot;
    double err = std::abs(actual_loss - expected_loss) / expected_loss;
    check(err < 1e-6, 2,
          "non-conservative total mass loss",
          std::format("lost = {:.10f}, expected = {:.10f}, rel_err = {:.2e}",
                      actual_loss, expected_loss, err));
}

// ── Test 03: Phase advance matches orbital period ───────────────────────
// No mass transfer (donor underfills). Phase should advance by 2*pi
// over one orbital period T = 2*pi / Omega.

static void test_03_phase_advance() {
    MassTransferConfig cfg;
    cfg.donor_mass0 = 1.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 0.1; // well inside Roche lobe → no overflow
    cfg.phase0 = 0.0;
    cfg.mdot_mode = "ritter";
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 1.0;
    cfg.donor_radius_mode = "prescribed";

    auto mt = init_binary_mt(cfg);

    double M_tot = mt.donor_mass + mt.accretor_mass;
    double Omega = std::sqrt(M_tot / (mt.separation * mt.separation * mt.separation));
    double period = 2.0 * std::numbers::pi / Omega;

    // Advance for exactly one period
    int n_steps = 1000;
    double dt = period / n_steps;
    for (int i = 0; i < n_steps; ++i) {
        advance_binary(mt, cfg, dt);
    }

    double phase_err = std::abs(mt.phase - 2.0 * std::numbers::pi);
    check(phase_err < 1e-8, 3,
          "phase advance over one period",
          std::format("phase = {:.10f}, expected = {:.10f}, err = {:.2e}",
                      mt.phase, 2.0 * std::numbers::pi, phase_err));
}

// ── Test 04: RK2 convergence order ──────────────────────────────────────
// Conservative transfer at dt, dt/2, dt/4. Measure error vs analytic.
// Expect convergence rate between 1.8 and 2.2 (confirming 2nd order).

static void test_04_convergence_order() {
    auto run_conservative = [](int n_steps) -> double {
        MassTransferConfig cfg;
        cfg.donor_mass0 = 2.0;
        cfg.accretor_mass0 = 1.0;
        cfg.separation0 = 1.0;
        cfg.donor_radius0 = 10.0;
        cfg.phase0 = 0.0;
        cfg.mdot_mode = "prescribed";
        cfg.mdot_prescribed = 0.001;
        cfg.beta_mode = "fixed";
        cfg.beta_fixed = 1.0;
        cfg.donor_radius_mode = "prescribed";
        cfg.max_fractional_change = 1e10; // disable subcycling

        auto mt = init_binary_mt(cfg);
        double t_end = 50.0; // transfer 0.05 M_sun
        double dt = t_end / n_steps;

        for (int i = 0; i < n_steps; ++i) {
            advance_binary(mt, cfg, dt);
        }

        // Analytic
        double M_d0 = cfg.donor_mass0, M_a0 = cfg.accretor_mass0;
        double M_d = mt.donor_mass, M_a = mt.accretor_mass;
        double a_analytic = cfg.separation0 * std::pow(M_d0 * M_a0 / (M_d * M_a), 2.0);
        return std::abs(mt.separation - a_analytic);
    };

    double err1 = run_conservative(100);
    double err2 = run_conservative(200);
    double err3 = run_conservative(400);

    double order_12 = std::log2(err1 / err2);
    double order_23 = std::log2(err2 / err3);

    check(order_12 > 1.8 && order_12 < 2.2, 4,
          "RK2 convergence order (100→200 steps)",
          std::format("order = {:.3f}, err1 = {:.2e}, err2 = {:.2e}", order_12, err1, err2));
    check(order_23 > 1.8 && order_23 < 2.2, 4,
          "RK2 convergence order (200→400 steps)",
          std::format("order = {:.3f}, err2 = {:.2e}, err3 = {:.2e}", order_23, err2, err3));
}

// ── Test 05: Separation shrinks for massive donor + non-conservative ────
// M_d > M_a, beta < 1: orbit should shrink (standard result for
// non-conservative mass transfer from the more massive star).

static void test_05_separation_shrinks() {
    MassTransferConfig cfg;
    cfg.donor_mass0 = 3.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0;
    cfg.phase0 = 0.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 0.5;
    cfg.jloss_mode = "L2_exact";
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);
    double a0 = mt.separation;

    double dt = 1.0;
    for (int i = 0; i < 100; ++i) {
        advance_binary(mt, cfg, dt);
    }

    check(mt.separation < a0, 5,
          "separation shrinks for M_d > M_a, beta < 1",
          std::format("a0 = {:.6f}, a = {:.6f}, da/a = {:.4e}",
                      a0, mt.separation, (mt.separation - a0) / a0));
}

// ── Test 06: Separation grows for conservative transfer from lighter donor ──
// M_d < M_a, beta=1: orbit widens (angular momentum conservation forces
// the orbit to expand when mass flows from less to more massive star).

static void test_06_separation_grows() {
    MassTransferConfig cfg;
    cfg.donor_mass0 = 0.5;
    cfg.accretor_mass0 = 2.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0;
    cfg.phase0 = 0.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 1.0;
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);
    double a0 = mt.separation;

    double dt = 1.0;
    for (int i = 0; i < 100; ++i) {
        advance_binary(mt, cfg, dt);
    }

    check(mt.separation > a0, 6,
          "separation grows for M_d < M_a, beta=1",
          std::format("a0 = {:.6f}, a = {:.6f}, da/a = {:.4e}",
                      a0, mt.separation, (mt.separation - a0) / a0));
}

// ── Test 07: Binary positions and velocities ────────────────────────────
// At phase=0 the donor should be at negative x, accretor at positive x.
// Velocities should be tangential (y-direction for phase=0).

static void test_07_positions_velocities() {
    BinaryMTState mt;
    mt.donor_mass = 1.0;
    mt.accretor_mass = 1.0;
    mt.separation = 2.0;
    mt.phase = 0.0;

    auto [r_d, r_a] = binary_positions(mt);
    // Equal mass: each at distance a/2 from COM
    check(std::abs(r_d.x - (-1.0)) < 1e-12, 7,
          "donor position x at phase=0",
          std::format("r_d.x = {:.10f}", r_d.x));
    check(std::abs(r_a.x - 1.0) < 1e-12, 7,
          "accretor position x at phase=0",
          std::format("r_a.x = {:.10f}", r_a.x));

    auto [v_d, v_a] = binary_velocities(mt);
    // At phase=0, velocity is in y-direction
    check(std::abs(v_d.x) < 1e-12, 7,
          "donor velocity x = 0 at phase=0",
          std::format("v_d.x = {:.2e}", v_d.x));
    check(v_d.y < 0.0, 7,
          "donor velocity y < 0 at phase=0",
          std::format("v_d.y = {:.6f}", v_d.y));

    // COM velocity should be zero
    Vec3 v_com = v_d * mt.donor_mass + v_a * mt.accretor_mass;
    double v_com_mag = v_com.mag();
    check(v_com_mag < 1e-12, 7,
          "COM velocity = 0",
          std::format("|v_com| = {:.2e}", v_com_mag));
}

// ── Test 08: Cumulative mass budgets ────────────────────────────────────
// Verify: transferred = accreted + lost, and donor mass deficit matches.

static void test_08_mass_budgets() {
    MassTransferConfig cfg;
    cfg.donor_mass0 = 2.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 0.7;
    cfg.jloss_mode = "L2_exact";
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);

    double dt = 0.5;
    for (int i = 0; i < 200; ++i) {
        advance_binary(mt, cfg, dt);
    }

    // transferred = accreted + lost
    double budget_err = std::abs(mt.cumulative_transferred
                                 - mt.cumulative_accreted
                                 - mt.cumulative_lost);
    check(budget_err < 1e-10, 8,
          "transferred = accreted + lost",
          std::format("transferred={:.8f}, accreted={:.8f}, lost={:.8f}, err={:.2e}",
                      mt.cumulative_transferred, mt.cumulative_accreted,
                      mt.cumulative_lost, budget_err));

    // Donor mass deficit should match cumulative transferred
    double donor_deficit = cfg.donor_mass0 - mt.donor_mass;
    double deficit_err = std::abs(donor_deficit - mt.cumulative_transferred) / donor_deficit;
    check(deficit_err < 1e-6, 8,
          "donor deficit matches cumulative transferred",
          std::format("deficit = {:.8f}, cum_tr = {:.8f}, rel_err = {:.2e}",
                      donor_deficit, mt.cumulative_transferred, deficit_err));
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::println("=== Binary evolution tests ===\n");

    test_01_conservative_analytic();
    test_02_nonconservative_mass_loss();
    test_03_phase_advance();
    test_04_convergence_order();
    test_05_separation_shrinks();
    test_06_separation_grows();
    test_07_positions_velocities();
    test_08_mass_budgets();

    int total = g_pass + g_fail;
    std::println("\n=== {}/{} tests passed ===", g_pass, total);
    return g_fail > 0 ? 1 : 0;
}
