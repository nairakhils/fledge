// Tracer launch test suite
// Tests parcel spawning from L2/L3, weight conservation, inertial velocity,
// sink removal, and ballistic energy conservation.

#include "mt/tracer_launch.hpp"
#include "physics.hpp"

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
        std::println("[PASS] Test {}: {} ({})", test_num, desc, detail);
    } else {
        g_fail++;
        std::println("[FAIL] Test {}: {} ({})", test_num, desc, detail);
    }
}

// ── Helper: build a standard binary MT state at phase=0 ─────────────────

static auto make_test_binary(double M_d = 1.0, double M_a = 1.0, double a = 1.0,
                             double phase = 0.0) -> BinaryMTState
{
    MassTransferConfig cfg;
    cfg.donor_mass0 = M_d;
    cfg.accretor_mass0 = M_a;
    cfg.separation0 = a;
    cfg.phase0 = phase;
    cfg.donor_radius0 = 10.0;
    return init_binary_mt(cfg);
}

// ── Test 01: Weight conservation ────────────────────────────────────────

static void test_01_weight_conservation() {
    State s;
    auto mt = make_test_binary();

    Closure closure{0.5, 0.01, 1.5, 1.0}; // beta=0.5, mdot_loss=0.01
    MassTransferConfig cfg;
    cfg.parcels_per_step = 20;
    cfg.launch_vr_fraction = 0.1;
    cfg.opening_angle_deg = 15.0;
    cfg.launch_offset = 1e-3;
    cfg.launch_seed = 42;
    cfg.l3_fraction = 0.0;
    double dt = 0.1;

    int n = launch_L2_parcels(s, mt, closure, cfg, dt);

    check(n == 20, 1,
          "correct number of parcels launched",
          std::format("n = {}", n));

    double total_weight = 0.0;
    for (double w : s.tracer_weights) total_weight += w;
    double expected_weight = closure.mdot_loss * dt;
    double err = std::abs(total_weight - expected_weight) / expected_weight;
    check(err < 1e-12, 1,
          "sum of weights = mdot_loss * dt",
          std::format("total={:.10e}, expected={:.10e}, rel_err={:.2e}",
                      total_weight, expected_weight, err));
}

// ── Test 02: No launch when mdot_loss = 0 ───────────────────────────────

static void test_02_no_launch_zero_loss() {
    State s;
    auto mt = make_test_binary();

    Closure closure{1.0, 0.0, 0.0, 1.0}; // beta=1, mdot_loss=0
    MassTransferConfig cfg;
    cfg.parcels_per_step = 10;
    double dt = 0.1;

    int n = launch_L2_parcels(s, mt, closure, cfg, dt);

    check(n == 0, 2,
          "no parcels when mdot_loss=0",
          std::format("n = {}", n));
    check(s.positions.empty(), 2,
          "state unchanged",
          std::format("positions.size() = {}", s.positions.size()));
}

// ── Test 03: Launch position at correct distance from COM ───────────────

static void test_03_launch_position() {
    State s;
    auto mt = make_test_binary(1.0, 1.0, 2.0, 0.0); // a=2, phase=0

    Closure closure{0.5, 0.01, 1.5, 1.0};
    MassTransferConfig cfg;
    cfg.parcels_per_step = 1;
    cfg.launch_vr_fraction = 0.1;
    cfg.opening_angle_deg = 0.0; // no vertical scatter for clean position test
    cfg.launch_offset = 0.0;     // no offset for exact position test
    cfg.launch_seed = 42;
    cfg.l3_fraction = 0.0;
    double dt = 0.1;

    launch_L2_parcels(s, mt, closure, cfg, dt);

    // L2 position: in donor-at-origin frame, xi_L2 * a.
    // In COM frame: x_L2_com = (xi_L2 - M_a/M_tot) * a
    // At phase=0: inertial = COM frame position on x-axis
    double q_d = mt.donor_mass / mt.accretor_mass;
    double xi_L2 = find_lagrange_L2(q_d);
    double M_tot = mt.donor_mass + mt.accretor_mass;
    double x_com_frac = mt.accretor_mass / M_tot;
    double r_L2_com = (xi_L2 - x_com_frac) * mt.separation;

    double r_actual = s.positions[0].mag();
    double r_expected = std::abs(r_L2_com);
    double err = std::abs(r_actual - r_expected) / r_expected;
    check(err < 1e-10, 3,
          "launch position at correct distance from COM",
          std::format("r={:.8f}, expected={:.8f}, rel_err={:.2e}",
                      r_actual, r_expected, err));

    // At phase=0, L2 is on the negative x-axis (behind donor)
    check(s.positions[0].x < 0.0, 3,
          "L2 parcel on negative x-axis at phase=0",
          std::format("x = {:.6f}", s.positions[0].x));
}

// ── Test 04: Inertial velocity includes corotating frame contribution ──

static void test_04_inertial_velocity() {
    State s;
    auto mt = make_test_binary(1.0, 1.0, 2.0, 0.0); // equal mass, a=2, phase=0

    Closure closure{0.5, 0.01, 1.5, 1.0};
    MassTransferConfig cfg;
    cfg.parcels_per_step = 1;
    cfg.launch_vr_fraction = 0.0; // no radial velocity
    cfg.opening_angle_deg = 0.0;  // no vertical scatter
    cfg.launch_offset = 0.0;
    cfg.launch_seed = 42;
    cfg.l3_fraction = 0.0;
    double dt = 0.1;

    launch_L2_parcels(s, mt, closure, cfg, dt);

    // With v_r=0 and no vertical scatter, velocity should be purely tangential.
    // At phase=0, L2 is on the negative x-axis, so phi_hat = (0, -1, 0)
    // (z_hat cross (-x_hat) = -y_hat)
    // Actually let me compute more carefully.
    // r_hat points from COM to L2 = negative x direction = (-1, 0, 0)
    // phi_hat = z_hat cross r_hat = (0,0,1) cross (-1,0,0) = (0, -1, 0)...
    // Wait: (0,0,1)×(-1,0,0) = (0*0 - 1*0, 1*(-1) - 0*0, 0*0 - 0*(-1)) = (0, -1, 0)
    // So phi_hat = (0, -1, 0)
    // v_phi = jloss / |r_L2_com|, v_corotation = Omega * |r_L2_com|
    // total phi component = (v_phi + v_corotation) * phi_hat
    // This means v_y should be negative and v_x should be ~0

    check(std::abs(s.velocities[0].x) < 1e-10, 4,
          "v_x ≈ 0 with no radial velocity at phase=0",
          std::format("v_x = {:.6e}", s.velocities[0].x));

    // The y-component should be negative (prograde in this geometry)
    check(s.velocities[0].y < 0.0, 4,
          "v_y < 0 (prograde tangential at phase=0)",
          std::format("v_y = {:.6f}", s.velocities[0].y));

    // v_z should be 0 (no vertical scatter)
    check(std::abs(s.velocities[0].z) < 1e-15, 4,
          "v_z = 0 with no vertical scatter",
          std::format("v_z = {:.6e}", s.velocities[0].z));

    // Check magnitude: v_y should be -(v_phi + v_corotation)
    double M_tot = mt.donor_mass + mt.accretor_mass;
    double Omega = std::sqrt(M_tot / (mt.separation * mt.separation * mt.separation));
    double q_d = mt.donor_mass / mt.accretor_mass;
    double xi_L2 = find_lagrange_L2(q_d);
    double x_com_frac = mt.accretor_mass / M_tot;
    double abs_r = std::abs((xi_L2 - x_com_frac) * mt.separation);
    double v_phi = closure.jloss / abs_r;
    double v_corot = Omega * abs_r;
    double expected_vy = -(v_phi + v_corot);
    double err = std::abs(s.velocities[0].y - expected_vy) / std::abs(expected_vy);
    check(err < 1e-10, 4,
          "v_y magnitude matches v_phi + v_corotation",
          std::format("v_y={:.8f}, expected={:.8f}, rel_err={:.2e}",
                      s.velocities[0].y, expected_vy, err));
}

// ── Test 05: Sink removal ───────────────────────────────────────────────

static void test_05_sink_removal() {
    State s;
    auto mt = make_test_binary(1.0, 1.0, 2.0, 0.0);

    // Place parcels: one at the donor, one at the accretor, one far away
    auto [r_d, r_a] = binary_positions(mt);
    s.positions = {r_d, r_a, Vec3(5.0, 0.0, 0.0)};
    s.velocities = {Vec3::zero(), Vec3::zero(), Vec3::zero()};
    s.tracer_weights = {1.0, 1.0, 1.0};

    MassTransferConfig cfg;
    cfg.donor_sink_radius = 0.1;
    cfg.accretor_sink_radius = 0.1;

    int removed = apply_sinks(s, mt, cfg);

    check(removed == 2, 5,
          "two parcels removed (one at donor, one at accretor)",
          std::format("removed = {}", removed));
    check(s.positions.size() == 1, 5,
          "one parcel survives",
          std::format("remaining = {}", s.positions.size()));
    // The surviving parcel should be the one at (5, 0, 0)
    check(std::abs(s.positions[0].x - 5.0) < 1e-10, 5,
          "surviving parcel is the distant one",
          std::format("x = {:.6f}", s.positions[0].x));
}

// ── Test 06: No sink removal when radii are zero ────────────────────────

static void test_06_no_sink_zero_radius() {
    State s;
    auto mt = make_test_binary(1.0, 1.0, 2.0, 0.0);
    auto [r_d, r_a] = binary_positions(mt);

    s.positions = {r_d, r_a};
    s.velocities = {Vec3::zero(), Vec3::zero()};
    s.tracer_weights = {1.0, 1.0};

    MassTransferConfig cfg;
    cfg.donor_sink_radius = 0.0;
    cfg.accretor_sink_radius = 0.0;

    int removed = apply_sinks(s, mt, cfg);

    check(removed == 0, 6,
          "no removal with zero sink radii",
          std::format("removed = {}, remaining = {}", removed, s.positions.size()));
}

// ── Test 07: Ballistic energy conservation ──────────────────────────────

static void test_07_ballistic_energy() {
    // Launch one parcel from L2 in a static binary, advance for many steps,
    // verify specific energy E = 0.5*v^2 - sum(GM_i/|r-r_i|) is conserved.
    auto mt = make_test_binary(1.0, 1.0, 2.0, 0.0);

    State s;
    Closure closure{0.5, 0.01, 1.5, 1.0};
    MassTransferConfig cfg;
    cfg.parcels_per_step = 1;
    cfg.launch_vr_fraction = 0.15;
    cfg.opening_angle_deg = 0.0; // no scatter for clean energy test
    cfg.launch_offset = 1e-3;
    cfg.launch_seed = 42;
    cfg.l3_fraction = 0.0;
    double dt = 0.001;

    launch_L2_parcels(s, mt, closure, cfg, dt);

    // Set up the static binary as mass sources
    auto [r_d, r_a] = binary_positions(mt);
    s.mass_positions = {r_d, r_a};
    s.mass_velocities = {Vec3::zero(), Vec3::zero()};
    std::vector<double> masses = {mt.donor_mass, mt.accretor_mass};

    auto energy = [&](size_t i) -> double {
        double ke = 0.5 * s.velocities[i].dot(s.velocities[i]);
        double pe = 0.0;
        double eps = 0.001; // small softening
        for (size_t j = 0; j < masses.size(); ++j) {
            Vec3 dr = s.positions[i] - s.mass_positions[j];
            double dist = std::sqrt(dr.dot(dr) + eps * eps);
            pe -= masses[j] / dist;
        }
        return ke + pe;
    };

    double E0 = energy(0);

    // Advance with leapfrog for 5000 steps (static binary, no mass body update)
    double softening = 0.001;
    for (int step = 0; step < 5000; ++step) {
        // KDK leapfrog for the single parcel
        Vec3 acc = compute_acceleration(s.positions[0], s.mass_positions,
                                        masses, softening, 2);
        s.velocities[0] += acc * (0.5 * dt);
        s.positions[0] += s.velocities[0] * dt;
        acc = compute_acceleration(s.positions[0], s.mass_positions,
                                   masses, softening, 2);
        s.velocities[0] += acc * (0.5 * dt);
    }

    double E_final = energy(0);
    double rel_err = std::abs((E_final - E0) / E0);
    check(rel_err < 1e-6, 7,
          "ballistic energy conservation",
          std::format("E0={:.8e}, E_f={:.8e}, |dE/E|={:.2e}", E0, E_final, rel_err));
}

// ── Test 08: L3 launch with l3_fraction ─────────────────────────────────

static void test_08_L3_launch() {
    State s;
    auto mt = make_test_binary(1.0, 1.0, 2.0, 0.0);

    Closure closure{0.5, 0.01, 1.5, 1.0};
    MassTransferConfig cfg;
    cfg.parcels_per_step = 5;
    cfg.launch_vr_fraction = 0.1;
    cfg.opening_angle_deg = 0.0;
    cfg.launch_offset = 0.0;
    cfg.launch_seed = 42;
    cfg.l3_fraction = 0.3;
    double dt = 0.1;

    int n_L2 = launch_L2_parcels(s, mt, closure, cfg, dt);
    int n_L3 = launch_L3_parcels(s, mt, closure, cfg, dt);

    check(n_L2 == 5 && n_L3 == 5, 8,
          "both L2 and L3 launch parcels",
          std::format("n_L2={}, n_L3={}", n_L2, n_L3));

    // Total weight should equal mdot_loss * dt
    double total_weight = 0.0;
    for (double w : s.tracer_weights) total_weight += w;
    double expected = closure.mdot_loss * dt;
    double err = std::abs(total_weight - expected) / expected;
    check(err < 1e-12, 8,
          "L2+L3 total weight = mdot_loss * dt",
          std::format("total={:.10e}, expected={:.10e}, rel_err={:.2e}",
                      total_weight, expected, err));

    // L3 parcels should be on positive x (behind accretor at phase=0)
    // The first 5 are L2 (negative x), the next 5 are L3 (positive x)
    check(s.positions[0].x < 0.0, 8,
          "L2 parcels on negative x-axis",
          std::format("x_L2 = {:.4f}", s.positions[0].x));
    check(s.positions[5].x > 0.0, 8,
          "L3 parcels on positive x-axis",
          std::format("x_L3 = {:.4f}", s.positions[5].x));
}

// ── Test 09: Launch at nonzero phase ────────────────────────────────────

static void test_09_nonzero_phase() {
    // At phase = pi/2, L2 should be on the negative y-axis
    State s;
    auto mt = make_test_binary(1.0, 1.0, 2.0, pi / 2.0);

    Closure closure{0.5, 0.01, 1.5, 1.0};
    MassTransferConfig cfg;
    cfg.parcels_per_step = 1;
    cfg.launch_vr_fraction = 0.1;
    cfg.opening_angle_deg = 0.0;
    cfg.launch_offset = 0.0;
    cfg.launch_seed = 42;
    cfg.l3_fraction = 0.0;
    double dt = 0.1;

    launch_L2_parcels(s, mt, closure, cfg, dt);

    // At phase=pi/2, the binary has rotated 90 degrees. The L2 point
    // (which is behind the donor on the corotating x-axis) should now
    // be on the negative y-axis.
    check(std::abs(s.positions[0].x) < 1e-10, 9,
          "L2 x ≈ 0 at phase=pi/2",
          std::format("x = {:.6e}", s.positions[0].x));
    check(s.positions[0].y < 0.0, 9,
          "L2 on negative y-axis at phase=pi/2",
          std::format("y = {:.6f}", s.positions[0].y));
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::println("=== Tracer launch tests ===\n");

    test_01_weight_conservation();
    test_02_no_launch_zero_loss();
    test_03_launch_position();
    test_04_inertial_velocity();
    test_05_sink_removal();
    test_06_no_sink_zero_radius();
    test_07_ballistic_energy();
    test_08_L3_launch();
    test_09_nonzero_phase();

    int total = g_pass + g_fail;
    std::println("\n=== {}/{} tests passed ===", g_pass, total);
    return g_fail > 0 ? 1 : 0;
}
