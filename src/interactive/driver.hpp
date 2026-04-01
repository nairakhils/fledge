#pragma once

#include "command.hpp"
#include "snapshot.hpp"
#include "watch.hpp"
#include "../config.hpp"
#include "../physics.hpp"
#include "../state.hpp"

#include <chrono>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace fledge {

class InteractiveDriver {
public:
    explicit InteractiveDriver(Config cfg) : cfg_(std::move(cfg)) {}

    auto accept(const Command& cmd) -> std::vector<Event> {
        return std::visit([this](const auto& c) { return handle(c); }, cmd);
    }

    auto is_running() const -> bool {
        return mode_ == DriverMode::Running;
    }

    void write_snapshot(Watch<RuntimeSnapshot>& snap) const {
        RuntimeSnapshot s;
        s.mode = mode_;
        s.has_state = state_.has_value();
        if (state_) {
            s.iteration = static_cast<int64_t>(state_->iteration);
            s.time = state_->time;
            s.status_text = status_text_;

            auto& xs = s.linear["x"];
            auto& ys = s.linear["y"];
            xs.reserve(state_->positions.size());
            ys.reserve(state_->positions.size());
            for (const auto& p : state_->positions) {
                xs.push_back(p.x);
                ys.push_back(p.y);
            }

            auto& mxs = s.linear["mass_x"];
            auto& mys = s.linear["mass_y"];
            mxs.reserve(state_->mass_positions.size());
            mys.reserve(state_->mass_positions.size());
            for (const auto& mp : state_->mass_positions) {
                mxs.push_back(mp.x);
                mys.push_back(mp.y);
            }
        }
        snap.write(std::move(s));
    }

private:
    Config cfg_;
    std::optional<State> state_;
    std::vector<double> masses_;
    DriverMode mode_ = DriverMode::Idle;
    std::string status_text_;

    void step() {
        if (!state_) return;
        auto t1 = std::chrono::steady_clock::now();
        advance_state(*state_, cfg_, masses_, cfg_.dt);
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t1).count();
        status_text_ = std::format(
            "[{:06d}] t={:.6e} {:.3f} sec/update n={}",
            state_->iteration, state_->time, elapsed,
            state_->positions.size());
    }

    auto handle(const CmdRun&) -> std::vector<Event> {
        if (!state_) return {EvError{"no state"}};
        mode_ = DriverMode::Running;
        step();
        if (state_->time >= cfg_.tfinal) {
            mode_ = DriverMode::Idle;
            return {EvSimulationDone{}};
        }
        return {};
    }

    auto handle(const CmdPause&) -> std::vector<Event> {
        mode_ = DriverMode::Idle;
        return {};
    }

    auto handle(const CmdStep&) -> std::vector<Event> {
        if (!state_) return {EvError{"no state"}};
        step();
        if (state_->time >= cfg_.tfinal) {
            return {EvSimulationDone{}};
        }
        return {};
    }

    auto handle(const CmdCreateState&) -> std::vector<Event> {
        State s;
        s.time = cfg_.tstart;
        s.iteration = 0;
        auto [mpos, mvel, m] = setup_masses(cfg_);
        s.mass_positions = std::move(mpos);
        s.mass_velocities = std::move(mvel);
        masses_ = std::move(m);
        auto [pos, vel] = setup_particles(cfg_, s.mass_positions,
                                          s.mass_velocities, masses_);
        s.positions = std::move(pos);
        s.velocities = std::move(vel);
        state_ = std::move(s);
        status_text_ = std::format("[{:06d}] t={:.6e} ready n={}",
                                   state_->iteration, state_->time,
                                   state_->positions.size());
        return {EvStateCreated{}};
    }

    auto handle(const CmdDestroyState&) -> std::vector<Event> {
        state_.reset();
        masses_.clear();
        mode_ = DriverMode::Idle;
        status_text_.clear();
        return {EvStateDestroyed{}};
    }

    auto handle(const CmdCheckpoint&) -> std::vector<Event> {
        return {EvCheckpointWritten{"checkpoint.nest"}};
    }

    auto handle(const CmdQuit&) -> std::vector<Event> {
        mode_ = DriverMode::Idle;
        return {EvFinished{}};
    }
};

} // namespace fledge
