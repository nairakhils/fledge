// Closure and rates test suite
// Tests mass-transfer rate prescriptions and non-conservative closure models.

#include "mt/closure.hpp"
#include "mt/rates.hpp"

#include <cmath>
#include <format>
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

// ── Test 01: Ritter overflow, no overflow ───────────────────────────────

static void test_01_ritter_no_overflow() {
    double mdot = ritter_overflow_rate(-0.1, 1e-8, 0.01);
    check(mdot == 0.0, 1,
          "Ritter with negative overflow depth",
          std::format("mdot = {:.2e}, expected 0", mdot));
}

// ── Test 02: Ritter overflow, positive depth ────────────────────────────

static void test_02_ritter_positive() {
    double Hp = 0.01;
    double mdot0 = 1e-8;
    double depth = 0.05;
    double mdot = ritter_overflow_rate(depth, mdot0, Hp);
    double expected = mdot0 * std::exp(depth / Hp);
    check(std::abs(mdot - expected) / expected < 1e-12, 2,
          "Ritter exponential overflow",
          std::format("mdot = {:.4e}, expected = {:.4e}", mdot, expected));
}

// ── Test 03: Capped overflow clamps ─────────────────────────────────────

static void test_03_capped_overflow() {
    double Hp = 0.001;
    double mdot0 = 1e-8;
    double depth = 0.1; // huge overflow → exp(100) → capped
    double M_donor = 1.0;
    double tau_drive = 1e6;
    double f_sat = 0.1;
    double mdot = capped_overflow_rate(depth, mdot0, Hp, M_donor, tau_drive, f_sat);
    double cap = f_sat * M_donor / tau_drive;
    check(std::abs(mdot - cap) / cap < 1e-12, 3,
          "capped overflow at saturation",
          std::format("mdot = {:.4e}, cap = {:.4e}", mdot, cap));
}

// ── Test 04: compute_overflow fills state correctly ─────────────────────

static void test_04_compute_overflow() {
    double M_d = 1.0, M_a = 1.0, a = 1.0;
    double R_L_expected = eggleton_roche_radius(1.0) * a;

    // Underfilling case
    double R_donor = 0.3;
    auto ov = compute_overflow(M_d, M_a, a, R_donor);
    check(std::abs(ov.roche_radius - R_L_expected) < 1e-10, 4,
          "compute_overflow Roche radius",
          std::format("R_L = {:.6f}, expected = {:.6f}", ov.roche_radius, R_L_expected));
    check(ov.overflow_depth < 0.0, 4,
          "compute_overflow underfilling",
          std::format("Delta_R = {:.6f} < 0", ov.overflow_depth));

    // Overfilling case
    R_donor = 0.5;
    ov = compute_overflow(M_d, M_a, a, R_donor);
    check(ov.overflow_depth > 0.0, 4,
          "compute_overflow overfilling",
          std::format("Delta_R = {:.6f} > 0", ov.overflow_depth));
}

// ── Test 05: fixed beta=1, fully conservative ──────────────────────────

static void test_05_fixed_beta_conservative() {
    double mdot_tr = 1e-6;
    double jL2_val = 1.5;
    auto cl = closure_fixed_beta(mdot_tr, 1.0, jL2_val, 1.0);
    check(cl.beta == 1.0, 5,
          "fixed beta=1 conservative",
          std::format("beta = {}", cl.beta));
    check(cl.mdot_loss == 0.0, 5,
          "fixed beta=1 no mass loss",
          std::format("mdot_loss = {:.2e}", cl.mdot_loss));
}

// ── Test 06: fixed beta=0.5 ────────────────────────────────────────────

static void test_06_fixed_beta_half() {
    double mdot_tr = 1e-6;
    double jL2_val = 1.5;
    auto cl = closure_fixed_beta(mdot_tr, 0.5, jL2_val, 0.9);
    double expected_loss = 0.5 * mdot_tr;
    check(std::abs(cl.mdot_loss - expected_loss) < 1e-20, 6,
          "fixed beta=0.5 mass loss",
          std::format("mdot_loss = {:.4e}, expected = {:.4e}", cl.mdot_loss, expected_loss));
    check(std::abs(cl.jloss - 0.9 * jL2_val) < 1e-15, 6,
          "fixed beta=0.5 jloss",
          std::format("jloss = {:.6f}, expected = {:.6f}", cl.jloss, 0.9 * jL2_val));
}

// ── Test 07: super-Eddington, mdot >> mdot_edd → beta → 0 ─────────────

static void test_07_super_edd_high_mdot() {
    double mdot_tr = 1e3;      // much larger than mdot_edd
    double kappa = 1.0;
    double r_disk = 0.1;       // mdot_edd = 4*pi*0.1/1 ≈ 1.257
    double logistic_n = 4.0;
    double jL2_val = 1.5;

    auto cl = closure_super_eddington(mdot_tr, 0.5, 1.0,
                                      kappa, r_disk, logistic_n,
                                      jL2_val, 1.0);
    check(cl.beta < 0.01, 7,
          "super-Eddington high mdot → beta ≈ 0",
          std::format("beta = {:.6f}", cl.beta));
}

// ── Test 08: super-Eddington, mdot << mdot_edd → beta → 1 ─────────────

static void test_08_super_edd_low_mdot() {
    double mdot_tr = 1e-6;     // much smaller than mdot_edd
    double kappa = 1.0;
    double r_disk = 0.1;       // mdot_edd ≈ 1.257
    double logistic_n = 4.0;
    double jL2_val = 1.5;

    auto cl = closure_super_eddington(mdot_tr, 0.5, 1.0,
                                      kappa, r_disk, logistic_n,
                                      jL2_val, 1.0);
    check(cl.beta > 0.99, 8,
          "super-Eddington low mdot → beta ≈ 1",
          std::format("beta = {:.10f}", cl.beta));
}

// ── Test 09: Scherbak adiabatic eta at calibration nodes ────────────────

static void test_09_scherbak_nodes() {
    check(std::abs(scherbak_adiabatic_eta(0.25) - 0.95) < 1e-12, 9,
          "Scherbak eta(0.25) = 0.95",
          std::format("eta = {:.6f}", scherbak_adiabatic_eta(0.25)));
    check(std::abs(scherbak_adiabatic_eta(0.50) - 0.90) < 1e-12, 9,
          "Scherbak eta(0.50) = 0.90",
          std::format("eta = {:.6f}", scherbak_adiabatic_eta(0.50)));
    check(std::abs(scherbak_adiabatic_eta(1.00) - 0.80) < 1e-12, 9,
          "Scherbak eta(1.00) = 0.80",
          std::format("eta = {:.6f}", scherbak_adiabatic_eta(1.00)));
    check(std::abs(scherbak_adiabatic_eta(2.00) - 0.65) < 1e-12, 9,
          "Scherbak eta(2.00) = 0.65",
          std::format("eta = {:.6f}", scherbak_adiabatic_eta(2.00)));
}

// ── Test 10: Scherbak adiabatic eta interpolation ───────────────────────

static void test_10_scherbak_interp() {
    double eta = scherbak_adiabatic_eta(0.75);
    // q=0.75 is between (0.5, 0.9) and (1.0, 0.8)
    // Linear interp: t = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
    // eta = 0.9 + 0.5 * (0.8 - 0.9) = 0.85
    check(std::abs(eta - 0.85) < 1e-12, 10,
          "Scherbak eta(0.75) interpolation",
          std::format("eta = {:.6f}, expected 0.85", eta));
    // Verify it's between the bounding node values
    check(eta > 0.80 && eta < 0.90, 10,
          "Scherbak eta(0.75) in range [0.80, 0.90]",
          std::format("eta = {:.6f}", eta));
}

// ── Test 11: Scherbak adiabatic eta clamping ────────────────────────────

static void test_11_scherbak_clamp() {
    double eta_low = scherbak_adiabatic_eta(0.01);
    double eta_high = scherbak_adiabatic_eta(100.0);
    check(std::abs(eta_low - 0.95) < 1e-12, 11,
          "Scherbak eta clamped below q=0.25",
          std::format("eta(0.01) = {:.6f}", eta_low));
    check(std::abs(eta_high - 0.65) < 1e-12, 11,
          "Scherbak eta clamped above q=2.0",
          std::format("eta(100) = {:.6f}", eta_high));
}

// ── Test 12: compute_jloss cooled mode → eta=1 → jloss=jL2 ─────────────

static void test_12_jloss_cooled() {
    double jL2_val = 1.4362;
    auto [jloss, eta] = compute_jloss("cooled", jL2_val, 0.5, 1.0);
    check(std::abs(jloss - jL2_val) < 1e-12, 12,
          "cooled mode jloss = jL2",
          std::format("jloss = {:.6f}, jL2 = {:.6f}", jloss, jL2_val));
    check(std::abs(eta - 1.0) < 1e-12, 12,
          "cooled mode eta_j = 1",
          std::format("eta = {:.6f}", eta));
}

// ── Test 13: compute_jloss adiabatic mode ───────────────────────────────

static void test_13_jloss_adiabatic() {
    double jL2_val = 1.5;
    double q = 1.0;
    auto [jloss, eta] = compute_jloss("adiabatic", jL2_val, 0.99, q);
    check(std::abs(eta - 0.80) < 1e-12, 13,
          "adiabatic mode eta_j(q=1) = 0.80",
          std::format("eta = {:.6f}", eta));
    check(std::abs(jloss - 0.80 * jL2_val) < 1e-12, 13,
          "adiabatic mode jloss = 0.80 * jL2",
          std::format("jloss = {:.6f}, expected = {:.6f}", jloss, 0.80 * jL2_val));
}

// ── Test 14: compute_jloss fixed mode ───────────────────────────────────

static void test_14_jloss_fixed() {
    double jL2_val = 2.0;
    double eta_fixed = 0.73;
    auto [jloss, eta] = compute_jloss("fixed", jL2_val, eta_fixed, 1.0);
    check(std::abs(eta - eta_fixed) < 1e-12, 14,
          "fixed mode uses user eta_j",
          std::format("eta = {:.6f}", eta));
    check(std::abs(jloss - eta_fixed * jL2_val) < 1e-12, 14,
          "fixed mode jloss = eta_j * jL2",
          std::format("jloss = {:.6f}, expected = {:.6f}", jloss, eta_fixed * jL2_val));
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::println("=== Closure and rates tests ===\n");

    test_01_ritter_no_overflow();
    test_02_ritter_positive();
    test_03_capped_overflow();
    test_04_compute_overflow();
    test_05_fixed_beta_conservative();
    test_06_fixed_beta_half();
    test_07_super_edd_high_mdot();
    test_08_super_edd_low_mdot();
    test_09_scherbak_nodes();
    test_10_scherbak_interp();
    test_11_scherbak_clamp();
    test_12_jloss_cooled();
    test_13_jloss_adiabatic();
    test_14_jloss_fixed();

    int total = g_pass + g_fail;
    std::println("\n=== {}/{} tests passed ===", g_pass, total);
    return g_fail > 0 ? 1 : 0;
}
