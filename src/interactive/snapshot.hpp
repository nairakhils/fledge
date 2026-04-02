#pragma once

#include "command.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fledge {

struct MassTransferSnapshot {
    bool active = false;
    double donor_mass = 0.0;
    double accretor_mass = 0.0;
    double separation = 0.0;
    double donor_radius = 0.0;
    double roche_radius = 0.0;
    double overflow_depth = 0.0;
    double mdot_transfer = 0.0;
    double beta = 1.0;
    double mdot_loss = 0.0;
    double jloss = 0.0;
    double phase = 0.0;
    double orbital_period = 0.0;
    double orbital_angular_momentum = 0.0;
    double cumulative_transferred = 0.0;
    double cumulative_lost = 0.0;
    int num_tracers = 0;
};

struct RuntimeSnapshot {
    DriverMode mode = DriverMode::Idle;
    bool has_state = false;
    int64_t iteration = 0;
    double time = 0.0;
    std::string status_text;
    std::map<std::string, std::vector<double>> linear;
    MassTransferSnapshot mt;
};

} // namespace fledge
