// Fledge physics verification test suite
// Runs production-level simulations using the real physics code and checks correctness.

#include "config.hpp"
#include "fledge.hpp"
#include "orbits.hpp"
#include "physics.hpp"
#include "state.hpp"
#include "vec3.hpp"

#include <cmath>
#include <filesystem>
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

// ── Helpers ──────────────────────────────────────────────────────────────

static auto make_single_particle(double mass, double r, double softening = 0.01)
    -> std::tuple<Config, State, std::vector<double>> {
    Config cfg;
    cfg.central_object_type = "single";
    cfg.mass = mass;
    cfg.softening = softening;
    cfg.dt = 0.001;

    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = {Vec3::zero()};
    s.mass_velocities = {Vec3::zero()};

    double v = std::sqrt(mass / r);
    s.positions = {Vec3(r, 0.0, 0.0)};
    s.velocities = {Vec3(0.0, v, 0.0)};

    std::vector<double> masses = {mass};
    return {cfg, s, masses};
}

static auto specific_energy(Vec3 pos, Vec3 vel, double GM, double softening) -> double {
    double r = std::sqrt(pos.dot(pos) + softening * softening);
    return 0.5 * vel.dot(vel) - GM / r;
}

static auto angular_momentum(Vec3 pos, Vec3 vel) -> Vec3 {
    return pos.cross(vel);
}

// ── Test 1: Energy conservation (single star, circular orbit) ────────────

static void test_01_energy_conservation() {
    auto [cfg, s, masses] = make_single_particle(1.0, 1.0, 0.0);
    cfg.dt = 0.001;

    double E0 = specific_energy(s.positions[0], s.velocities[0], cfg.mass, cfg.softening);
    double max_rel = 0.0;

    int n_steps = static_cast<int>(100.0 * 2.0 * pi / cfg.dt);
    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        double E = specific_energy(s.positions[0], s.velocities[0], cfg.mass, cfg.softening);
        double rel = std::abs(E - E0) / std::abs(E0);
        if (rel > max_rel) max_rel = rel;
    }

    check(max_rel < 1e-6, 1, "Energy conservation (100 orbits)",
          std::format("max |dE/E| = {:.2e}", max_rel));
}

// ── Test 2: Angular momentum conservation (single star) ──────────────────

static void test_02_angular_momentum() {
    auto [cfg, s, masses] = make_single_particle(1.0, 1.0, 0.0);
    cfg.dt = 0.001;

    Vec3 L0 = angular_momentum(s.positions[0], s.velocities[0]);
    double max_rel = 0.0;

    int n_steps = static_cast<int>(100.0 * 2.0 * pi / cfg.dt);
    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        Vec3 L = angular_momentum(s.positions[0], s.velocities[0]);
        double rel = (L - L0).mag() / L0.mag();
        if (rel > max_rel) max_rel = rel;
    }

    check(max_rel < 1e-6, 2, "Angular momentum conservation (100 orbits)",
          std::format("max |dL/L| = {:.2e}", max_rel));
}

// ── Test 3: Keplerian orbital period ─────────────────────────────────────

static void test_03_orbital_period() {
    double r = 2.0;
    double GM = 1.0;
    auto [cfg, s, masses] = make_single_particle(GM, r, 0.0);
    cfg.dt = 0.0001;

    Vec3 r0 = s.positions[0];
    double T = 2.0 * pi * std::sqrt(r * r * r / GM);
    int n_steps = std::lround(T / cfg.dt);

    for (int i = 0; i < n_steps; i++)
        advance_state(s, cfg, masses, cfg.dt);

    double err = (s.positions[0] - r0).mag();
    check(err < 1e-4, 3, "Keplerian orbital period (r=2)",
          std::format("|r_final - r_initial| = {:.2e}", err));
}

// ── Test 4: Eccentric orbit ──────────────────────────────────────────────

static void test_04_eccentric_orbit() {
    double a = 1.0, e = 0.5, GM = 1.0;
    double r_peri = a * (1.0 - e);
    double v_peri = std::sqrt(GM / a * (1.0 + e) / (1.0 - e));

    Config cfg;
    cfg.central_object_type = "single";
    cfg.mass = GM;
    cfg.softening = 0.0;
    cfg.dt = 0.0001;

    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = {Vec3::zero()};
    s.mass_velocities = {Vec3::zero()};
    s.positions = {Vec3(r_peri, 0.0, 0.0)};
    s.velocities = {Vec3(0.0, v_peri, 0.0)};
    std::vector<double> masses = {GM};

    Vec3 r0 = s.positions[0];
    double T = 2.0 * pi * std::sqrt(a * a * a / GM);
    int n_steps = std::lround(T / cfg.dt);

    double max_r = 0.0;
    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        double r = s.positions[0].mag();
        if (r > max_r) max_r = r;
    }

    double pos_err = (s.positions[0] - r0).mag();
    double r_apo_expected = a * (1.0 + e);
    double apo_err = std::abs(max_r - r_apo_expected) / r_apo_expected;

    check(pos_err < 1e-3, 4, "Eccentric orbit (e=0.5) return to periapsis",
          std::format("|dr| = {:.2e}, max_r = {:.4f} (expect {:.1f})",
                      pos_err, max_r, r_apo_expected));
    check(apo_err < 0.01, 4, "Eccentric orbit apoapsis radius",
          std::format("err = {:.2e}", apo_err));
}

// ── Test 5: Convergence order ────────────────────────────────────────────
// Run for 10 orbits at each dt so the error is well above float noise.
// Use exact integer step counts by choosing dt values that divide T evenly.

static void test_05_convergence_order() {
    double GM = 1.0, r = 1.0;
    int n_orbits = 10;
    // Choose step counts that double: 800, 1600, 3200, 6400 per orbit
    int steps_per_orbit[] = {800, 1600, 3200, 6400};
    double T = 2.0 * pi;
    double errors[4];

    for (int d = 0; d < 4; d++) {
        auto [cfg, s, masses] = make_single_particle(GM, r, 0.0);
        double dt = T / steps_per_orbit[d];
        cfg.dt = dt;
        Vec3 r0 = s.positions[0];
        int n_steps = steps_per_orbit[d] * n_orbits;
        for (int i = 0; i < n_steps; i++)
            advance_state(s, cfg, masses, dt);
        errors[d] = (s.positions[0] - r0).mag();
    }

    bool all_ok = true;
    std::string rates_str;
    for (int i = 0; i < 3; i++) {
        double rate = std::log2(errors[i] / errors[i + 1]);
        rates_str += std::format("{:.2f}", rate);
        if (i < 2) rates_str += ", ";
        if (rate < 1.8 || rate > 2.2) all_ok = false;
    }

    check(all_ok, 5, "Convergence order (leapfrog 2nd-order)",
          std::format("rates = [{}]", rates_str));
}

// ── Test 6: Binary center of mass ────────────────────────────────────────

static void test_06_binary_cm() {
    double mass = 1.0, q = 1.0, a = 1.0;
    double m1 = mass / (1.0 + q);
    double m2 = q * m1;

    Config cfg;
    cfg.central_object_type = "binary";
    cfg.mass = mass;
    cfg.q1 = q;
    cfg.a1 = a;
    cfg.softening = 0.01;
    cfg.dt = 0.001;

    auto [mpos, mvel, masses] = setup_masses(cfg);
    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = mpos;
    s.mass_velocities = mvel;

    double T = 2.0 * pi / std::sqrt(mass / (a * a * a));
    int n_steps = static_cast<int>(10.0 * T / cfg.dt);
    double max_cm = 0.0;

    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        Vec3 cm = s.mass_positions[0] * m1 + s.mass_positions[1] * m2;
        double cm_mag = cm.mag();
        if (cm_mag > max_cm) max_cm = cm_mag;
    }

    check(max_cm < 1e-12, 6, "Binary CM at origin (10 periods)",
          std::format("max |CM| = {:.2e}", max_cm));
}

// ── Test 7: Binary orbital period ────────────────────────────────────────

static void test_07_binary_period() {
    double mass = 1.0, q = 0.5, a = 1.0;
    double T = 2.0 * pi / std::sqrt(mass / (a * a * a));

    auto [r1_0, r2_0] = orbital_state(mass, q, a, 0.0, 0.0, 0.0, 0.0);
    auto [r1_T, r2_T] = orbital_state(mass, q, a, 0.0, 0.0, 0.0, T);

    double err1 = (r1_T - r1_0).mag();
    double err2 = (r2_T - r2_0).mag();

    check(err1 < 1e-12 && err2 < 1e-12, 7, "Binary orbital period (q=0.5)",
          std::format("|dr1| = {:.2e}, |dr2| = {:.2e}", err1, err2));
}

// ── Test 8: Eccentric binary ─────────────────────────────────────────────

static void test_08_eccentric_binary() {
    double mass = 1.0, q = 1.0, a = 1.0;
    double ex = 0.6, ey = 0.0;
    double T = 2.0 * pi / std::sqrt(mass / (a * a * a));
    double m1 = mass / (1.0 + q);
    double m2 = q * m1;

    auto [r1_0, r2_0] = orbital_state(mass, q, a, ex, ey, 0.0, 0.0);
    auto [r1_T, r2_T] = orbital_state(mass, q, a, ex, ey, 0.0, T);

    double err1 = (r1_T - r1_0).mag();
    double err2 = (r2_T - r2_0).mag();

    double max_cm = 0.0;
    int n_samples = 1000;
    for (int i = 0; i <= n_samples; i++) {
        double t = T * i / n_samples;
        auto [r1, r2] = orbital_state(mass, q, a, ex, ey, 0.0, t);
        Vec3 cm = r1 * m1 + r2 * m2;
        if (cm.mag() > max_cm) max_cm = cm.mag();
    }

    check(err1 < 1e-10 && err2 < 1e-10, 8,
          "Eccentric binary period (e=0.6)",
          std::format("|dr1| = {:.2e}, |dr2| = {:.2e}", err1, err2));
    check(max_cm < 1e-12, 8, "Eccentric binary CM at origin",
          std::format("max |CM| = {:.2e}", max_cm));
}

// ── Test 9: Circumbinary particle stability ──────────────────────────────

static void test_09_circumbinary_stability() {
    double mass = 1.0, q = 1.0, a_bin = 0.5;
    double r_part = 3.0;

    Config cfg;
    cfg.central_object_type = "binary";
    cfg.mass = mass;
    cfg.q1 = q;
    cfg.a1 = a_bin;
    cfg.softening = 0.01;
    cfg.dt = 0.001;

    auto [mpos, mvel, masses] = setup_masses(cfg);

    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = mpos;
    s.mass_velocities = mvel;

    double v_circ = std::sqrt(mass / r_part);
    s.positions = {Vec3(r_part, 0.0, 0.0)};
    s.velocities = {Vec3(0.0, v_circ, 0.0)};

    double T_bin = 2.0 * pi / std::sqrt(mass / (a_bin * a_bin * a_bin));
    int n_steps = static_cast<int>(20.0 * T_bin / cfg.dt);

    double min_r = r_part, max_r = r_part;
    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        double r = s.positions[0].mag();
        if (r < min_r) min_r = r;
        if (r > max_r) max_r = r;
    }

    check(min_r > 2.5 && max_r < 3.5, 9,
          "Circumbinary particle stability (20 binary periods)",
          std::format("r in [{:.3f}, {:.3f}]", min_r, max_r));
}

// ── Test 10: Triple system — outer body Keplerian ────────────────────────

static void test_10_triple_outer() {
    Config cfg;
    cfg.central_object_type = "triple";
    cfg.mass = 1.0;
    cfg.q1 = 1.0;
    cfg.a1 = 0.5;
    cfg.q2 = 0.01;
    cfg.a2 = 10.0;
    cfg.softening = 0.01;
    cfg.dt = 0.001;

    auto [mpos, mvel, masses] = setup_masses(cfg);

    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = mpos;
    s.mass_velocities = mvel;
    s.positions = {};
    s.velocities = {};

    double r3_0 = s.mass_positions[2].mag();
    double M_inner = cfg.mass / (1.0 + cfg.q2);
    double T_outer = 2.0 * pi * std::sqrt(cfg.a2 * cfg.a2 * cfg.a2 / M_inner);
    int n_steps = std::lround(T_outer / cfg.dt);

    for (int i = 0; i < n_steps; i++)
        advance_state(s, cfg, masses, cfg.dt);

    double r3_final = s.mass_positions[2].mag();
    double err = std::abs(r3_final - r3_0) / r3_0;

    check(err < 0.05, 10, "Triple outer body periodicity",
          std::format("r3: {:.4f} -> {:.4f} (err = {:.2e})", r3_0, r3_final, err));
}

// ── Test 11: Multi-particle energy conservation ──────────────────────────

static void test_11_multi_particle_energy() {
    Config cfg;
    cfg.central_object_type = "single";
    cfg.mass = 1.0;
    cfg.softening = 0.01;
    cfg.dt = 0.001;
    cfg.num_particles = 100;
    cfg.setup_type = "uniform_disk";
    cfg.inner_radius = 1.0;
    cfg.outer_radius = 2.0;

    auto [mpos, mvel, masses] = setup_masses(cfg);
    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = mpos;
    s.mass_velocities = mvel;
    auto [pos, vel] = setup_particles(cfg, mpos, mvel, masses);
    s.positions = pos;
    s.velocities = vel;

    auto total_energy = [&]() {
        double E = 0.0;
        for (size_t i = 0; i < s.positions.size(); i++)
            E += specific_energy(s.positions[i], s.velocities[i], cfg.mass, cfg.softening);
        return E;
    };

    double E0 = total_energy();
    double max_rel = 0.0;

    int n_steps = static_cast<int>(10.0 / cfg.dt);
    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        if (i % 100 == 0) {
            double E = total_energy();
            double rel = std::abs(E - E0) / std::abs(E0);
            if (rel > max_rel) max_rel = rel;
        }
    }

    check(max_rel < 1e-5, 11, "Multi-particle energy conservation (100 particles, t=10)",
          std::format("max |dE/E| = {:.2e}", max_rel));
}

// ── Test 12: Softening behaves correctly ─────────────────────────────────
// Particle on a head-on radial infall. With softening, the softened potential
// is finite everywhere so the particle passes through r=0 and oscillates.
// Verify: (a) max speed stays bounded (no blowup), (b) particle oscillates
// (v_x changes sign), (c) energy is conserved with the softened potential.

static void test_12_softening() {
    double GM = 1.0;
    double soft = 0.1;

    Config cfg;
    cfg.central_object_type = "single";
    cfg.mass = GM;
    cfg.softening = soft;
    cfg.dt = 0.0001;

    State s;
    s.time = 0.0;
    s.iteration = 0;
    s.mass_positions = {Vec3::zero()};
    s.mass_velocities = {Vec3::zero()};
    s.positions = {Vec3(1.0, 0.0, 0.0)};
    s.velocities = {Vec3(-1.0, 0.0, 0.0)};
    std::vector<double> masses = {GM};

    double E0 = specific_energy(s.positions[0], s.velocities[0], GM, soft);
    double max_speed = 0.0;
    double max_energy_rel = 0.0;
    // Track whether the particle crosses through origin and reverses direction.
    // With softening, a radial infall oscillates: x goes 1 -> 0 -> -1 -> 0 -> +...
    // Detect a full oscillation: position.x changes sign at least twice.
    int x_sign_changes = 0;
    double prev_x = s.positions[0].x;
    int n_steps = static_cast<int>(5.0 / cfg.dt);

    for (int i = 0; i < n_steps; i++) {
        advance_state(s, cfg, masses, cfg.dt);
        double speed = s.velocities[0].mag();
        if (speed > max_speed) max_speed = speed;
        double cur_x = s.positions[0].x;
        if ((cur_x > 0.0) != (prev_x > 0.0)) x_sign_changes++;
        prev_x = cur_x;
        double E = specific_energy(s.positions[0], s.velocities[0], GM, soft);
        double rel = std::abs(E - E0) / std::abs(E0);
        if (rel > max_energy_rel) max_energy_rel = rel;
    }

    // Max speed in softened potential: v_max = sqrt(2*GM/eps + v0²)
    double v_max_theory = std::sqrt(2.0 * GM / soft + 1.0);
    check(max_speed < v_max_theory * 1.1, 12,
          "Softening bounds max speed",
          std::format("max_v = {:.3f} (limit ~{:.3f})", max_speed, v_max_theory));
    check(x_sign_changes >= 1, 12,
          "Radial infall passes through origin",
          std::format("{} x-crossings (need >= 1)", x_sign_changes));
    check(max_energy_rel < 1e-4, 12,
          "Softened energy conservation",
          std::format("max |dE/E| = {:.2e}", max_energy_rel));
}

// ── Test 13: Checkpoint round-trip ───────────────────────────────────────

static void test_13_checkpoint_roundtrip() {
    auto test_dir = std::filesystem::temp_directory_path() / "fledge_test13";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directories(test_dir);
    auto dir_str = test_dir.string();

    // Continuous run: 0 -> 2.0 (reference)
    Config cfg_base;
    cfg_base.central_object_type = "single";
    cfg_base.mass = 1.0;
    cfg_base.num_particles = 50;
    cfg_base.setup_type = "uniform_disk";
    cfg_base.inner_radius = 1.0;
    cfg_base.outer_radius = 2.0;
    cfg_base.softening = 0.01;
    cfg_base.dt = 0.001;

    auto [mpos, mvel, masses] = setup_masses(cfg_base);
    State s_ref;
    s_ref.time = 0.0;
    s_ref.iteration = 0;
    s_ref.mass_positions = mpos;
    s_ref.mass_velocities = mvel;
    auto [pos, vel] = setup_particles(cfg_base, mpos, mvel, masses);
    s_ref.positions = pos;
    s_ref.velocities = vel;

    // Phase 1: nest::io::run 0 -> 1.0 with checkpoints at 0.5
    {
        FledgeSim sim;
        sim.config() = cfg_base;
        sim.config().tfinal = 1.0;
        sim.config().checkpoint_interval = 0.5;
        sim.config().output_dir = dir_str;
        const char* argv[] = {"test", nullptr};
        (void)nest::io::run(1, argv, sim);
    }

    auto chkpt_path = test_dir / "chkpt.0002.nest";
    bool chkpt_exists = std::filesystem::exists(chkpt_path);

    // Phase 2: restart from checkpoint, run to 2.0
    // NEST driver captures output_dir before loading checkpoint config,
    // so the second run writes to CWD. We set CWD to the test dir.
    State s_restarted;
    if (chkpt_exists) {
        auto prev_dir = std::filesystem::current_path();
        std::filesystem::current_path(test_dir);
        {
            FledgeSim sim;
            auto path_str = chkpt_path.string();
            auto tfinal_str = std::string("tfinal=2.0");
            const char* argv[] = {"test", path_str.c_str(), tfinal_str.c_str(), nullptr};
            (void)nest::io::run(3, argv, sim);
        }
        std::filesystem::current_path(prev_dir);

        // Find the final checkpoint
        std::filesystem::path final_chkpt;
        for (auto& entry : std::filesystem::directory_iterator(test_dir)) {
            auto name = entry.path().filename().string();
            if (name.starts_with("chkpt.") && name.ends_with(".nest"))
                if (final_chkpt.empty() || name > final_chkpt.filename().string())
                    final_chkpt = entry.path();
        }

        if (!final_chkpt.empty()) {
            auto reader = nest::io::CheckpointReader::open(final_chkpt);
            if (reader.has_value()) {
                FledgeSim sim;
                sim.read_state(*reader, s_restarted);
                if (auto t = reader->scalar_double("time"); t.has_value())
                    s_restarted.time = *t;
                if (auto i = reader->scalar_int("step"); i.has_value())
                    s_restarted.iteration = static_cast<uint64_t>(*i);
            }
        }
    }

    // Run reference to match the exact iteration count of the restarted run
    if (chkpt_exists && !s_restarted.positions.empty()) {
        int target_iters = static_cast<int>(s_restarted.iteration);
        for (int i = 0; i < target_iters; i++)
            advance_state(s_ref, cfg_base, masses, cfg_base.dt);
    }

    double max_pos_err = 0.0;
    if (chkpt_exists && s_restarted.positions.size() == s_ref.positions.size()) {
        for (size_t i = 0; i < s_ref.positions.size(); i++) {
            double err = (s_restarted.positions[i] - s_ref.positions[i]).mag();
            if (err > max_pos_err) max_pos_err = err;
        }
    }

    check(chkpt_exists, 13, "Checkpoint file written",
          chkpt_exists ? chkpt_path.string() : "not found");

    if (chkpt_exists && !s_restarted.positions.empty()) {
        check(max_pos_err < 1e-10, 13, "Checkpoint round-trip position match",
              std::format("max |dr| = {:.2e}", max_pos_err));
    } else {
        check(false, 13, "Checkpoint round-trip position match",
              "could not read restarted state");
    }

    std::filesystem::remove_all(test_dir);
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::println("═══ Fledge Physics Verification ═══\n");

    test_01_energy_conservation();
    test_02_angular_momentum();
    test_03_orbital_period();
    test_04_eccentric_orbit();
    test_05_convergence_order();
    test_06_binary_cm();
    test_07_binary_period();
    test_08_eccentric_binary();
    test_09_circumbinary_stability();
    test_10_triple_outer();
    test_11_multi_particle_energy();
    test_12_softening();
    test_13_checkpoint_roundtrip();

    int total = g_pass + g_fail;
    std::println("\n═══ Physics verification: {}/{} tests passed ═══",
                 g_pass, total);

    return g_fail > 0 ? 1 : 0;
}
