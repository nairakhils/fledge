#pragma once

#include <nest/io/serializable.hpp>

#include <cstdint>
#include <string>

struct Config {
    // Physics
    double tstart = 0.0;
    double tfinal = 10.0;
    double dt = 0.001;
    double softening = 0.01;

    // Central object type: "single", "binary", "triple"
    std::string central_object_type = "single";

    // Orbital parameters (superset of Single/Binary/Triple)
    double mass = 1.0;
    double q1 = 1.0;
    double a1 = 0.5;
    double e1x = 0.0;
    double e1y = 0.0;
    double q2 = 0.001;
    double a2 = 10.0;
    double e2x = 0.0;
    double e2y = 0.0;
    double inclination = 0.0;

    // Initial conditions
    uint64_t num_particles = 1000;
    std::string setup_type = "random_disk"; // "ring", "random_disk", "uniform_disk"
    double ring_radius = 1.5;
    double inner_radius = 1.0;
    double outer_radius = 2.0;
    std::string disk_center = "arbitrary"; // "primary", "secondary", "arbitrary"
    double disk_center_x = 0.0;
    double disk_center_y = 0.0;
    double disk_center_z = 0.0;

    // Driver
    double checkpoint_interval = 1.0;
    std::string output_dir = ".";

    // Simulation mode: "test_particles" or "mass_transfer"
    std::string simulation_mode = "test_particles";

    // Mass-transfer parameters (used when simulation_mode == "mass_transfer")
    double mt_donor_mass0 = 1.0;
    double mt_accretor_mass0 = 0.5;
    double mt_donor_radius0 = 0.5;
    double mt_separation0 = 1.0;
    double mt_phase0 = 0.0;
    std::string mt_donor_radius_mode = "response_law";
    double mt_zeta_star = 0.0;
    double mt_tau_drive = 1e10;
    double mt_Hp_over_R = 1e-4;
    double mt_mdot0 = 1e-6;
    double mt_mdot_cap = 0.0;
    std::string mt_mdot_mode = "ritter";
    double mt_mdot_prescribed = 0.0;
    std::string mt_beta_mode = "fixed";
    double mt_beta_fixed = 1.0;
    double mt_kappa = 0.34;
    double mt_f_disk_outer = 0.8;
    double mt_logistic_n = 2.0;
    std::string mt_jloss_mode = "L2_exact";
    double mt_eta_j_fixed = 1.0;
    double mt_max_fractional_change = 0.01;
};

NEST_SERIALIZABLE(Config,
    tstart, tfinal, dt, softening,
    central_object_type,
    mass, q1, a1, e1x, e1y,
    q2, a2, e2x, e2y, inclination,
    num_particles, setup_type, ring_radius,
    inner_radius, outer_radius,
    disk_center, disk_center_x, disk_center_y, disk_center_z,
    checkpoint_interval, output_dir,
    simulation_mode,
    mt_donor_mass0, mt_accretor_mass0, mt_donor_radius0,
    mt_separation0, mt_phase0,
    mt_donor_radius_mode, mt_zeta_star, mt_tau_drive,
    mt_Hp_over_R, mt_mdot0, mt_mdot_cap,
    mt_mdot_mode, mt_mdot_prescribed,
    mt_beta_mode, mt_beta_fixed,
    mt_kappa, mt_f_disk_outer, mt_logistic_n,
    mt_jloss_mode, mt_eta_j_fixed, mt_max_fractional_change)
