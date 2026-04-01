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
};

NEST_SERIALIZABLE(Config,
    tstart, tfinal, dt, softening,
    central_object_type,
    mass, q1, a1, e1x, e1y,
    q2, a2, e2x, e2y, inclination,
    num_particles, setup_type, ring_radius,
    inner_radius, outer_radius,
    disk_center, disk_center_x, disk_center_y, disk_center_z,
    checkpoint_interval, output_dir)
