#pragma once

#include "mt/binary_evolution.hpp"
#include "vec3.hpp"

#include <cstdint>
#include <vector>

struct State {
    double time = 0.0;
    uint64_t iteration = 0;
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<Vec3> mass_positions;
    std::vector<Vec3> mass_velocities;

    // Mass-transfer state (used when simulation_mode == "mass_transfer")
    BinaryMTState mt;
    std::vector<double> tracer_weights;
};
