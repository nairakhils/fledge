#pragma once

#include "command.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fledge {

struct RuntimeSnapshot {
    DriverMode mode = DriverMode::Idle;
    bool has_state = false;
    int64_t iteration = 0;
    double time = 0.0;
    std::string status_text;
    std::map<std::string, std::vector<double>> linear;
};

} // namespace fledge
