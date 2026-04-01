// Roche geometry test suite
// Tests Eggleton Roche radius, Lagrange point solvers, and j_L2 computation.

#include "mt/roche.hpp"

#include <cmath>
#include <format>
#include <print>
#include <string>
#include <vector>

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

// ── Test 01: Eggleton q_d = 1 ───────────────────────────────────────────

static void test_01_eggleton_equal_mass() {
    double rl = eggleton_roche_radius(1.0);
    // Known value: R_L/a ≈ 0.37892 for q_d = 1
    check(std::abs(rl - 0.37892) < 1e-4, 1,
          "Eggleton q_d=1",
          std::format("R_L/a = {:.6f}, expected ~0.37892", rl));
}

// ── Test 02: Eggleton q_d = 0.1 ─────────────────────────────────────────

static void test_02_eggleton_small_q() {
    double rl = eggleton_roche_radius(0.1);
    // Eggleton formula: 0.49 * 0.1^(2/3) / (0.6 * 0.1^(2/3) + ln(1 + 0.1^(1/3)))
    check(std::abs(rl - 0.20677) < 1e-4, 2,
          "Eggleton q_d=0.1",
          std::format("R_L/a = {:.6f}, expected ~0.20677", rl));
}

// ── Test 03: Eggleton q_d = 10 ──────────────────────────────────────────

static void test_03_eggleton_large_q() {
    double rl = eggleton_roche_radius(10.0);
    // Eggleton formula: 0.49 * 10^(2/3) / (0.6 * 10^(2/3) + ln(1 + 10^(1/3)))
    check(std::abs(rl - 0.57817) < 1e-4, 3,
          "Eggleton q_d=10",
          std::format("R_L/a = {:.6f}, expected ~0.57817", rl));
}

// ── Test 04: L1 equal mass symmetry ─────────────────────────────────────

static void test_04_L1_equal_mass() {
    double x_L1 = find_lagrange_L1(1.0);
    // By symmetry, L1 must be at the midpoint for equal masses
    check(std::abs(x_L1 - 0.5) < 1e-10, 4,
          "L1 equal mass at midpoint",
          std::format("x_L1/a = {:.12f}, expected 0.5", x_L1));
}

// ── Test 05: L1 residuals ───────────────────────────────────────────────

static void test_05_L1_residuals() {
    std::vector<double> q_vals = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double q : q_vals) {
        double x_L1 = find_lagrange_L1(q);
        double residual = std::abs(roche_force(x_L1, q));
        check(residual < 1e-11, 5,
              std::format("L1 residual q_d={}", q),
              std::format("|F(x_L1)| = {:.2e}", residual));
    }
}

// ── Test 06: L2 residuals ───────────────────────────────────────────────

static void test_06_L2_residuals() {
    std::vector<double> q_vals = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double q : q_vals) {
        double x_L2 = find_lagrange_L2(q);
        double residual = std::abs(roche_force(x_L2, q));
        check(residual < 1e-11, 6,
              std::format("L2 residual q_d={}", q),
              std::format("|F(x_L2)| = {:.2e}, x_L2/a = {:.6f}", residual, x_L2));
    }
}

// ── Test 07: L3 residuals ───────────────────────────────────────────────

static void test_07_L3_residuals() {
    std::vector<double> q_vals = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double q : q_vals) {
        double x_L3 = find_lagrange_L3(q);
        double residual = std::abs(roche_force(x_L3, q));
        check(residual < 1e-11, 7,
              std::format("L3 residual q_d={}", q),
              std::format("|F(x_L3)| = {:.2e}, x_L3/a = {:.6f}", residual, x_L3));
    }
}

// ── Test 08: Lagrange point ordering ────────────────────────────────────

static void test_08_ordering() {
    std::vector<double> q_vals = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double q : q_vals) {
        double x_L1 = find_lagrange_L1(q);
        double x_L2 = find_lagrange_L2(q);
        double x_L3 = find_lagrange_L3(q);
        bool ordered = (x_L2 < 0.0) && (0.0 < x_L1) && (x_L1 < 1.0) && (1.0 < x_L3);
        check(ordered, 8,
              std::format("ordering q_d={}", q),
              std::format("L2={:.4f} < 0 < L1={:.4f} < 1 < L3={:.4f}", x_L2, x_L1, x_L3));
    }
}

// ── Test 09: Equal mass L2/L3 symmetry ──────────────────────────────────

static void test_09_equal_mass_symmetry() {
    double x_L2 = find_lagrange_L2(1.0);
    double x_L3 = find_lagrange_L3(1.0);
    // For q_d=1: |x_L2| should equal x_L3 - 1 by the donor↔accretor symmetry
    double diff = std::abs(std::abs(x_L2) - (x_L3 - 1.0));
    check(diff < 1e-10, 9,
          "equal mass L2/L3 symmetry",
          std::format("|x_L2| = {:.10f}, x_L3-1 = {:.10f}, diff = {:.2e}",
                      std::abs(x_L2), x_L3 - 1.0, diff));
}

// ── Test 10: j_L2 for equal mass ────────────────────────────────────────

static void test_10_j_L2_value() {
    // q_d=1, M_d=0.5, M_a=0.5, a=1 → M_tot=1, Omega=1, x_com=0.5
    // j_L2 = Omega * (xi_L2 * a - x_com)^2 = (xi_L2 - 0.5)^2
    double xi_L2 = find_lagrange_L2(1.0);
    double expected = (xi_L2 - 0.5) * (xi_L2 - 0.5);
    double computed = j_L2(0.5, 0.5, 1.0);
    double rel_err = std::abs(computed - expected) / expected;
    check(rel_err < 1e-10, 10,
          "j_L2 equal mass value",
          std::format("j_L2 = {:.10f}, expected = {:.10f}, rel_err = {:.2e}",
                      computed, expected, rel_err));
}

// ── Test 11: j_L2 > specific orbital angular momentum ──────────────────

static void test_11_j_L2_exceeds_orbital() {
    // L2 is farther from COM than either star, so a corotating element
    // there has more specific angular momentum than the reduced-mass
    // specific orbital angular momentum j_orb = sqrt(G * M_tot * a) * mu/M_tot
    // Actually j_orb_specific = J_orb / M_reduced where
    //   J_orb = mu * sqrt(M_tot * a), mu = M_d*M_a/M_tot
    //   j_orb_specific = sqrt(M_tot * a)
    // j_L2 should exceed the specific orbital AM of either star.
    std::vector<double> q_vals = {0.1, 0.5, 1.0, 2.0, 10.0};
    double a = 1.0;
    for (double q : q_vals) {
        double M_d = q / (1.0 + q);
        double M_a = 1.0 / (1.0 + q);
        double M_tot = 1.0;
        double j_orb_specific = std::sqrt(M_tot * a); // = sqrt(a) for M_tot=1
        double jl2 = j_L2(M_d, M_a, a);
        check(jl2 > j_orb_specific, 11,
              std::format("j_L2 > j_orb_specific q_d={}", q),
              std::format("j_L2 = {:.6f}, j_orb_spec = {:.6f}", jl2, j_orb_specific));
    }
}

// ── Test 12: j_L2 scaling with separation ───────────────────────────────

static void test_12_j_L2_scaling() {
    // j_L2 = Omega * x_L2_com^2 = sqrt(M_tot/a^3) * (xi_L2 * a - x_com_frac * a)^2
    //       = sqrt(M_tot) * a^{1/2} * (xi_L2 - x_com_frac)^2
    // So j_L2 ∝ sqrt(a) for fixed masses. Doubling a → ratio = sqrt(2).
    double M_d = 0.5, M_a = 0.5;
    double j1 = j_L2(M_d, M_a, 1.0);
    double j2 = j_L2(M_d, M_a, 2.0);
    double ratio = j2 / j1;
    double expected_ratio = std::sqrt(2.0);
    double rel_err = std::abs(ratio - expected_ratio) / expected_ratio;
    check(rel_err < 1e-10, 12,
          "j_L2 scales as sqrt(a)",
          std::format("j_L2(a=2)/j_L2(a=1) = {:.10f}, expected sqrt(2) = {:.10f}",
                      ratio, expected_ratio));
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::println("=== Roche geometry tests ===\n");

    test_01_eggleton_equal_mass();
    test_02_eggleton_small_q();
    test_03_eggleton_large_q();
    test_04_L1_equal_mass();
    test_05_L1_residuals();
    test_06_L2_residuals();
    test_07_L3_residuals();
    test_08_ordering();
    test_09_equal_mass_symmetry();
    test_10_j_L2_value();
    test_11_j_L2_exceeds_orbital();
    test_12_j_L2_scaling();

    int total = g_pass + g_fail;
    std::println("\n=== {}/{} tests passed ===", g_pass, total);
    return g_fail > 0 ? 1 : 0;
}
