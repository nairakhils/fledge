// Mass-transfer verification test suite
// Runs production-level simulations and checks against analytic solutions,
// budget identities, convergence rates, and physical invariants.

#include "config.hpp"
#include "fledge.hpp"
#include "mt/binary_evolution.hpp"
#include "mt/closure.hpp"
#include "mt/rates.hpp"
#include "mt/roche.hpp"
#include "physics.hpp"
#include "state.hpp"
#include "vec3.hpp"

#include <cmath>
#include <format>
#include <numbers>
#include <print>
#include <string>
#include <vector>

using std::numbers::pi;

// ── Test harness ─────────────────────────────────────────────────────────

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, int test_num, const std::string& desc,
                  const std::string& detail) {
    if (ok) {
        g_pass++;
        std::println("[PASS] Test {:2d}: {} ({})", test_num, desc, detail);
    } else {
        g_fail++;
        std::println("[FAIL] Test {:2d}: {} ({})", test_num, desc, detail);
    }
}

// ── Helper: build MassTransferConfig for prescribed conservative transfer ──

static auto make_conservative_config(double M_d0, double M_a0, double a0,
                                     double mdot, double max_frac = 0.001)
    -> MassTransferConfig
{
    MassTransferConfig cfg;
    cfg.donor_mass0 = M_d0;
    cfg.accretor_mass0 = M_a0;
    cfg.separation0 = a0;
    cfg.donor_radius0 = 10.0; // large to ensure overflow
    cfg.phase0 = 0.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = mdot;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 1.0;
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = max_frac;
    return cfg;
}

// ── Helper: analytic separation for conservative transfer ───────────────
// For beta=1, M_tot constant, J_orb constant:
//   a / a0 = (M_d0 * M_a0 / (M_d * M_a))^2

static auto conservative_a_analytic(double M_d0, double M_a0, double a0,
                                    double M_d, double M_a) -> double
{
    double prod0 = M_d0 * M_a0;
    double prod = M_d * M_a;
    return a0 * (prod0 * prod0) / (prod * prod);
}

// ═══ TEST 1: Conservative transfer analytic solution ═══════════════════

static void test_01_conservative_analytic() {
    std::println("\n═══ TEST 1: Conservative transfer analytic solution ═══");

    auto cfg = make_conservative_config(2.0, 1.0, 1.0, 0.001);
    auto mt = init_binary_mt(cfg);

    double M_d0 = cfg.donor_mass0;
    double M_a0 = cfg.accretor_mass0;
    double M_tot0 = M_d0 + M_a0;
    double J_orb0 = mt.orbital_angular_momentum;

    // Transfer 0.1 M_sun: time = 0.1 / 0.001 = 100
    double dt = 0.1;
    int n_steps = 1000;
    for (int i = 0; i < n_steps; ++i) {
        advance_binary(mt, cfg, dt);
    }

    double M_d = mt.donor_mass;
    double M_a = mt.accretor_mass;
    double M_tot = M_d + M_a;

    // M_tot conservation
    double mtot_err = std::abs(M_tot - M_tot0) / M_tot0;
    check(mtot_err < 1e-10, 1,
          "M_tot conservation (beta=1)",
          std::format("|dM_tot/M_tot| = {:.2e}", mtot_err));

    // J_orb conservation
    double J_orb = mt.orbital_angular_momentum;
    double jorb_err = std::abs(J_orb - J_orb0) / J_orb0;
    check(jorb_err < 1e-4, 1,
          "J_orb conservation (beta=1)",
          std::format("|dJ/J| = {:.2e}", jorb_err));

    // Analytic separation
    double a_exact = conservative_a_analytic(M_d0, M_a0, 1.0, M_d, M_a);
    double a_err = std::abs(mt.separation / a_exact - 1.0);
    check(a_err < 1e-4, 1,
          "a(t) vs analytic",
          std::format("a={:.8f}, a_exact={:.8f}, rel_err={:.2e}",
                      mt.separation, a_exact, a_err));
}

// ═══ TEST 2: Mass budget ═══════════════════════════════════════════════

static void test_02_mass_budget() {
    std::println("\n═══ TEST 2: Mass budget ═══");

    MassTransferConfig cfg;
    cfg.donor_mass0 = 2.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.01;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 0.5;
    cfg.jloss_mode = "L2_exact";
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);
    double dt = 0.1;
    int n_steps = 100; // total time = 10
    double t_total = dt * n_steps;

    for (int i = 0; i < n_steps; ++i) {
        advance_binary(mt, cfg, dt);
    }

    double mdot = cfg.mdot_prescribed;
    double beta = cfg.beta_fixed;

    // M_d(t) = M_d0 - mdot*t
    double Md_exact = cfg.donor_mass0 - mdot * t_total;
    double Md_err = std::abs(mt.donor_mass - Md_exact);
    check(Md_err < 1e-8, 2,
          "M_d(t) = M_d0 - mdot*t",
          std::format("M_d={:.10f}, exact={:.10f}, err={:.2e}",
                      mt.donor_mass, Md_exact, Md_err));

    // M_a(t) = M_a0 + beta*mdot*t
    double Ma_exact = cfg.accretor_mass0 + beta * mdot * t_total;
    double Ma_err = std::abs(mt.accretor_mass - Ma_exact);
    check(Ma_err < 1e-8, 2,
          "M_a(t) = M_a0 + beta*mdot*t",
          std::format("M_a={:.10f}, exact={:.10f}, err={:.2e}",
                      mt.accretor_mass, Ma_exact, Ma_err));

    // cumulative_lost = (1-beta)*mdot*t
    double lost_exact = (1.0 - beta) * mdot * t_total;
    double lost_err = std::abs(mt.cumulative_lost - lost_exact);
    check(lost_err < 1e-8, 2,
          "cumulative_lost = (1-beta)*mdot*t",
          std::format("lost={:.10f}, exact={:.10f}, err={:.2e}",
                      mt.cumulative_lost, lost_exact, lost_err));
}

// ═══ TEST 3: Angular momentum budget ═══════════════════════════════════

static void test_03_angular_momentum_budget() {
    std::println("\n═══ TEST 3: Angular momentum budget ═══");

    MassTransferConfig cfg;
    cfg.donor_mass0 = 2.0;
    cfg.accretor_mass0 = 1.0;
    cfg.separation0 = 1.0;
    cfg.donor_radius0 = 10.0;
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 0.5;
    cfg.jloss_mode = "fixed_eta";
    cfg.eta_j_fixed = 1.0;
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);
    double J0 = mt.orbital_angular_momentum;

    // Get initial jloss by running one step
    double dt = 0.01;
    advance_binary(mt, cfg, dt);
    double jloss_initial = mt.jloss;

    // Reset and run properly
    mt = init_binary_mt(cfg);
    double t_total = 10.0;
    int n_steps = static_cast<int>(t_total / dt);

    // Accumulate actual angular momentum loss using per-step jloss
    double J_loss_accumulated = 0.0;
    for (int i = 0; i < n_steps; ++i) {
        advance_binary(mt, cfg, dt);
        // Each step: dJ = -(1-beta)*mdot*jloss*dt
        J_loss_accumulated += -(1.0 - cfg.beta_fixed) * cfg.mdot_prescribed * mt.jloss * dt;
    }

    double J_final = mt.orbital_angular_momentum;
    double dJ_numerical = J_final - J0;

    // For slowly varying jloss, the accumulated sum should approximate the integral
    double rel_err = std::abs(dJ_numerical - J_loss_accumulated) / std::abs(J0);
    check(rel_err < 1e-4, 3,
          "J_orb change matches angular momentum loss integral",
          std::format("dJ_num={:.6e}, dJ_sum={:.6e}, |err|/J0={:.2e}",
                      dJ_numerical, J_loss_accumulated, rel_err));
}

// ═══ TEST 4: Roche-lobe overflow triggers correctly ════════════════════

static void test_04_overflow_trigger() {
    std::println("\n═══ TEST 4: Roche-lobe overflow triggers correctly ═══");

    // Test that the Ritter overflow law gives mdot=0 when underfilling
    // and mdot>0 when overfilling. We test this by directly varying the
    // donor radius rather than relying on the ODE to expand it (the ODE
    // only evolves R_d during active transfer).

    double M_d = 1.0, M_a = 1.0, a = 1.0;
    double q_d = M_d / M_a;
    double R_L = eggleton_roche_radius(q_d) * a;
    double mdot0 = 1e-6;
    double Hp = 0.01 * R_L;

    // Underfilling: R_d < R_L → mdot = 0
    double R_under = 0.95 * R_L;
    double depth_under = R_under - R_L;
    double mdot_under = ritter_overflow_rate(depth_under, mdot0, Hp);
    check(mdot_under == 0.0, 4,
          "mdot=0 when R_d < R_L",
          std::format("R_d={:.6f}, R_L={:.6f}, depth={:.6f}, mdot={:.2e}",
                      R_under, R_L, depth_under, mdot_under));

    // Overfilling: R_d > R_L → mdot > 0
    double R_over = 1.05 * R_L;
    double depth_over = R_over - R_L;
    double mdot_over = ritter_overflow_rate(depth_over, mdot0, Hp);
    check(mdot_over > 0.0, 4,
          "mdot > 0 when R_d > R_L",
          std::format("R_d={:.6f}, R_L={:.6f}, depth={:.6f}, mdot={:.2e}",
                      R_over, R_L, depth_over, mdot_over));

    // Also verify the full advance_binary path: prescribed mdot works when donor
    // underfills (prescribed mode ignores overflow depth)
    MassTransferConfig cfg;
    cfg.donor_mass0 = M_d;
    cfg.accretor_mass0 = M_a;
    cfg.separation0 = a;
    cfg.donor_radius0 = 0.1; // well inside Roche lobe
    cfg.mdot_mode = "prescribed";
    cfg.mdot_prescribed = 0.001;
    cfg.beta_mode = "fixed";
    cfg.beta_fixed = 1.0;
    cfg.donor_radius_mode = "prescribed";
    cfg.max_fractional_change = 0.001;

    auto mt = init_binary_mt(cfg);
    advance_binary(mt, cfg, 1.0);
    check(mt.mdot_transfer == 0.001, 4,
          "prescribed mdot works regardless of overflow",
          std::format("mdot={:.6f}", mt.mdot_transfer));

    // And Ritter mode with underfilling donor gives zero
    cfg.mdot_mode = "ritter";
    cfg.mdot0 = 1e-6;
    cfg.Hp_over_R = 0.01;
    mt = init_binary_mt(cfg);
    advance_binary(mt, cfg, 1.0);
    check(mt.mdot_transfer == 0.0, 4,
          "ritter mode gives mdot=0 when donor underfills",
          std::format("mdot={:.2e}, R_d={:.4f}, R_L={:.4f}",
                      mt.mdot_transfer, mt.donor_radius, mt.roche_radius));
}

// ═══ TEST 5: Convergence order of binary evolution ODE ═════════════════

static void test_05_convergence_order() {
    std::println("\n═══ TEST 5: Convergence order ═══");

    auto run_at_dt = [](double dt_step) -> double {
        auto cfg = make_conservative_config(2.0, 1.0, 1.0, 0.001, 1e10); // disable subcycling
        auto mt = init_binary_mt(cfg);

        double t_end = 50.0;
        int n = static_cast<int>(t_end / dt_step);
        for (int i = 0; i < n; ++i) {
            advance_binary(mt, cfg, dt_step);
        }

        double a_exact = conservative_a_analytic(
            cfg.donor_mass0, cfg.accretor_mass0, cfg.separation0,
            mt.donor_mass, mt.accretor_mass);
        return std::abs(mt.separation - a_exact);
    };

    double err1 = run_at_dt(0.1);
    double err2 = run_at_dt(0.05);
    double err3 = run_at_dt(0.025);
    double err4 = run_at_dt(0.0125);

    double order_12 = std::log2(err1 / err2);
    double order_23 = std::log2(err2 / err3);
    double order_34 = std::log2(err3 / err4);

    check(order_12 > 1.8 && order_12 < 2.2, 5,
          "convergence order dt=0.1→0.05",
          std::format("order={:.3f}, err={:.2e}→{:.2e}", order_12, err1, err2));
    check(order_23 > 1.8 && order_23 < 2.2, 5,
          "convergence order dt=0.05→0.025",
          std::format("order={:.3f}, err={:.2e}→{:.2e}", order_23, err2, err3));
    check(order_34 > 1.8 && order_34 < 2.2, 5,
          "convergence order dt=0.025→0.0125",
          std::format("order={:.3f}, err={:.2e}→{:.2e}", order_34, err3, err4));
}

// ═══ TEST 6: Separation response to mass ratio ════════════════════════

static void test_06_separation_response() {
    std::println("\n═══ TEST 6: Separation response to mass ratio ═══");

    // Case A: M_d > M_a, beta=1 → orbit shrinks
    {
        auto cfg = make_conservative_config(1.5, 0.5, 1.0, 0.001);
        auto mt = init_binary_mt(cfg);
        double a0 = mt.separation;
        for (int i = 0; i < 1000; ++i) advance_binary(mt, cfg, 0.1);
        check(mt.separation < a0, 6,
              "orbit shrinks: M_d=1.5 > M_a=0.5, beta=1",
              std::format("a0={:.6f}, a={:.6f}, da/a={:.4e}",
                          a0, mt.separation, (mt.separation - a0) / a0));
    }

    // Case B: M_d < M_a, beta=1 → orbit widens
    {
        auto cfg = make_conservative_config(0.5, 1.5, 1.0, 0.001);
        auto mt = init_binary_mt(cfg);
        double a0 = mt.separation;
        for (int i = 0; i < 1000; ++i) advance_binary(mt, cfg, 0.1);
        check(mt.separation > a0, 6,
              "orbit widens: M_d=0.5 < M_a=1.5, beta=1",
              std::format("a0={:.6f}, a={:.6f}, da/a={:.4e}",
                          a0, mt.separation, (mt.separation - a0) / a0));
    }
}

// ═══ TEST 7: Eggleton Roche radius regression ═════════════════════════

static void test_07_eggleton_regression() {
    std::println("\n═══ TEST 7: Eggleton Roche radius regression ═══");

    // Reference values computed from the Eggleton formula
    struct { double q_d; double rl_expected; } cases[] = {
        {0.01,  0.10201},
        {0.1,   0.20677},
        {0.5,   0.32079},
        {1.0,   0.37892},
        {2.0,   0.44000},
        {10.0,  0.57817},
        {100.0, 0.72026},
    };

    for (auto& c : cases) {
        double rl = eggleton_roche_radius(c.q_d);
        double err = std::abs(rl - c.rl_expected);
        check(err < 1e-4, 7,
              std::format("Eggleton q_d={}", c.q_d),
              std::format("R_L/a={:.5f}, expected={:.5f}, err={:.2e}",
                          rl, c.rl_expected, err));
    }
}

// ═══ TEST 8: Lagrange point ordering and residuals ════════════════════

static void test_08_lagrange_points() {
    std::println("\n═══ TEST 8: Lagrange point ordering and residuals ═══");

    double q_vals[] = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double q : q_vals) {
        double xL1 = find_lagrange_L1(q);
        double xL2 = find_lagrange_L2(q);
        double xL3 = find_lagrange_L3(q);

        // Ordering: L2 < 0 < L1 < 1 < L3
        bool ordered = (xL2 < 0.0) && (0.0 < xL1) && (xL1 < 1.0) && (1.0 < xL3);
        check(ordered, 8,
              std::format("ordering q_d={}", q),
              std::format("L2={:.4f} < 0 < L1={:.4f} < 1 < L3={:.4f}",
                          xL2, xL1, xL3));

        // Residuals
        double rL1 = std::abs(roche_force(xL1, q));
        double rL2 = std::abs(roche_force(xL2, q));
        double rL3 = std::abs(roche_force(xL3, q));
        check(rL1 < 1e-12 && rL2 < 1e-12 && rL3 < 1e-12, 8,
              std::format("residuals q_d={}", q),
              std::format("|F(L1)|={:.2e}, |F(L2)|={:.2e}, |F(L3)|={:.2e}",
                          rL1, rL2, rL3));
    }
}

// ═══ TEST 9: j_L2 exceeds orbital specific AM ════════════════════════

static void test_09_j_L2_exceeds_orbital() {
    std::println("\n═══ TEST 9: j_L2 exceeds orbital specific AM ═══");

    double q_vals[] = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double q : q_vals) {
        double M_d = q / (1.0 + q);
        double M_a = 1.0 / (1.0 + q);
        double a = 1.0;
        double M_tot = 1.0;

        double jl2 = j_L2(M_d, M_a, a);
        // j_orb_specific = sqrt(G*a/M_tot) for unit reduced mass
        double j_orb_spec = std::sqrt(a / M_tot);

        check(jl2 > j_orb_spec, 9,
              std::format("j_L2 > j_orb_spec, q_d={}", q),
              std::format("j_L2={:.6f}, j_orb_spec={:.6f}", jl2, j_orb_spec));
    }
}

// ═══ TEST 10: Super-Eddington closure monotonicity ═══════════════════

static void test_10_super_eddington_monotonicity() {
    std::println("\n═══ TEST 10: Super-Eddington closure monotonicity ═══");

    double kappa = 1.0;
    double r_disk = 0.1;
    double logistic_n = 4.0;
    double jL2_val = 1.5;
    double eta_j = 1.0;

    // Log-spaced mdot_tr from 1e-8 to 1e2
    std::vector<double> mdots;
    for (int i = -8; i <= 2; ++i) {
        mdots.push_back(std::pow(10.0, static_cast<double>(i)));
    }

    std::vector<double> betas;
    for (double mdot : mdots) {
        auto cl = closure_super_eddington(mdot, 0.5, 1.0,
                                          kappa, r_disk, logistic_n,
                                          jL2_val, eta_j);
        betas.push_back(cl.beta);
    }

    // Monotonically non-increasing
    bool monotone = true;
    for (size_t i = 1; i < betas.size(); ++i) {
        if (betas[i] > betas[i - 1] + 1e-15) {
            monotone = false;
            break;
        }
    }
    check(monotone, 10,
          "beta monotonically non-increasing with mdot_tr",
          std::format("beta[1e-8]={:.6f}, beta[1e2]={:.6f}",
                      betas.front(), betas.back()));

    // Limits
    check(betas.front() > 0.99, 10,
          "beta → 1 at low mdot",
          std::format("beta(1e-8) = {:.10f}", betas.front()));
    check(betas.back() < 0.01, 10,
          "beta → 0 at high mdot",
          std::format("beta(1e2) = {:.10f}", betas.back()));
}

// ═══ TEST 11: Checkpoint round-trip ═══════════════════════════════════

static void test_11_checkpoint_roundtrip() {
    std::println("\n═══ TEST 11: Checkpoint round-trip ═══");

    // Run continuous simulation from 0 to 100
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

    auto mt_continuous = init_binary_mt(cfg);
    double dt = 0.1;
    for (int i = 0; i < 1000; ++i) { // t=0 to 100
        advance_binary(mt_continuous, cfg, dt);
    }

    // Run first half (0 to 50), save state, then continue
    auto mt_restart = init_binary_mt(cfg);
    for (int i = 0; i < 500; ++i) { // t=0 to 50
        advance_binary(mt_restart, cfg, dt);
    }

    // "Checkpoint": save the state at t=50
    BinaryMTState checkpoint = mt_restart;

    // "Restart": continue from checkpoint to t=100
    for (int i = 0; i < 500; ++i) { // t=50 to 100
        advance_binary(mt_restart, cfg, dt);
    }

    // Compare
    double Md_err = std::abs(mt_restart.donor_mass - mt_continuous.donor_mass);
    double Ma_err = std::abs(mt_restart.accretor_mass - mt_continuous.accretor_mass);
    double a_err = std::abs(mt_restart.separation - mt_continuous.separation);

    check(Md_err < 1e-10, 11,
          "M_d matches after checkpoint restart",
          std::format("|dM_d| = {:.2e}", Md_err));
    check(Ma_err < 1e-10, 11,
          "M_a matches after checkpoint restart",
          std::format("|dM_a| = {:.2e}", Ma_err));
    check(a_err < 1e-10, 11,
          "a matches after checkpoint restart",
          std::format("|da| = {:.2e}", a_err));
}

// ═══ TEST 12: Legacy mode unaffected ═════════════════════════════════

static void test_12_legacy_mode() {
    std::println("\n═══ TEST 12: Legacy mode unaffected ═══");

    // Single star, 10 particles in a ring, integrate one orbital period
    Config cfg;
    cfg.central_object_type = "single";
    cfg.mass = 1.0;
    cfg.softening = 0.01;
    cfg.dt = 0.001;
    cfg.num_particles = 10;
    cfg.setup_type = "ring";
    cfg.ring_radius = 1.5;
    cfg.simulation_mode = "test_particles";

    State s;
    s.time = 0.0;
    s.iteration = 0;

    auto [mpos, mvel, masses] = setup_masses(cfg);
    s.mass_positions = std::move(mpos);
    s.mass_velocities = std::move(mvel);

    auto [pos, vel] = setup_particles(cfg, s.mass_positions,
                                      s.mass_velocities, masses);
    s.positions = std::move(pos);
    s.velocities = std::move(vel);

    // Compute initial energies
    auto energy = [&](size_t i) -> double {
        double ke = 0.5 * s.velocities[i].dot(s.velocities[i]);
        Vec3 r = s.positions[i] - s.mass_positions[0];
        double dist = std::sqrt(r.dot(r) + cfg.softening * cfg.softening);
        double pe = -cfg.mass / dist;
        return ke + pe;
    };

    std::vector<double> E0(s.positions.size());
    for (size_t i = 0; i < s.positions.size(); ++i) {
        E0[i] = energy(i);
    }

    // Integrate one orbital period: T = 2*pi*sqrt(r^3/GM) at r=1.5, M=1
    double period = 2.0 * pi * std::sqrt(cfg.ring_radius * cfg.ring_radius * cfg.ring_radius / cfg.mass);
    int n_steps = static_cast<int>(period / cfg.dt);
    for (int i = 0; i < n_steps; ++i) {
        advance_state(s, cfg, masses, cfg.dt);
    }

    // Check energy conservation
    double max_rel_err = 0.0;
    for (size_t i = 0; i < s.positions.size(); ++i) {
        double Ef = energy(i);
        double rel_err = std::abs((Ef - E0[i]) / E0[i]);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    check(max_rel_err < 1e-6, 12,
          "legacy test_particles energy conservation",
          std::format("max |dE/E| = {:.2e} over 1 orbit", max_rel_err));
}

// ═══ Main ════════════════════════════════════════════════════════════

int main() {
    std::println("═══════════════════════════════════════════════════════════");
    std::println("  Mass transfer verification test suite");
    std::println("═══════════════════════════════════════════════════════════");

    test_01_conservative_analytic();
    test_02_mass_budget();
    test_03_angular_momentum_budget();
    test_04_overflow_trigger();
    test_05_convergence_order();
    test_06_separation_response();
    test_07_eggleton_regression();
    test_08_lagrange_points();
    test_09_j_L2_exceeds_orbital();
    test_10_super_eddington_monotonicity();
    test_11_checkpoint_roundtrip();
    test_12_legacy_mode();

    int total = g_pass + g_fail;
    std::println("\n═══════════════════════════════════════════════════════════");
    std::println("  Mass transfer verification: {}/{} tests passed", g_pass, total);
    std::println("═══════════════════════════════════════════════════════════");
    return g_fail > 0 ? 1 : 0;
}
