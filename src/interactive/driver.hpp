#pragma once

#include "command.hpp"
#include "snapshot.hpp"
#include "watch.hpp"
#include "../config.hpp"
#include "../mt/binary_evolution.hpp"
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

            if (cfg_.simulation_mode == "mass_transfer") {
                s.mt.active = true;
                s.mt.donor_mass = state_->mt.donor_mass;
                s.mt.accretor_mass = state_->mt.accretor_mass;
                s.mt.separation = state_->mt.separation;
                s.mt.donor_radius = state_->mt.donor_radius;
                s.mt.roche_radius = state_->mt.roche_radius;
                s.mt.overflow_depth = state_->mt.overflow_depth;
                s.mt.mdot_transfer = state_->mt.mdot_transfer;
                s.mt.beta = state_->mt.beta;
                s.mt.mdot_loss = state_->mt.mdot_loss;
                s.mt.jloss = state_->mt.jloss;
                s.mt.phase = state_->mt.phase;
                s.mt.orbital_period = state_->mt.orbital_period;
                s.mt.orbital_angular_momentum = state_->mt.orbital_angular_momentum;
                s.mt.cumulative_transferred = state_->mt.cumulative_transferred;
                s.mt.cumulative_lost = state_->mt.cumulative_lost;
                s.mt.num_tracers = static_cast<int>(state_->positions.size());
            }
        }
        snap.write(std::move(s));
    }

private:
    Config cfg_;
    std::optional<State> state_;
    std::vector<double> masses_;
    MassTransferConfig mt_config_;
    DriverMode mode_ = DriverMode::Idle;
    std::string status_text_;

    void build_mt_config() {
        mt_config_.donor_mass0 = cfg_.mt_donor_mass0;
        mt_config_.accretor_mass0 = cfg_.mt_accretor_mass0;
        mt_config_.donor_radius0 = cfg_.mt_donor_radius0;
        mt_config_.separation0 = cfg_.mt_separation0;
        mt_config_.phase0 = cfg_.mt_phase0;
        mt_config_.donor_radius_mode = cfg_.mt_donor_radius_mode;
        mt_config_.zeta_star = cfg_.mt_zeta_star;
        mt_config_.tau_drive = cfg_.mt_tau_drive;
        mt_config_.Hp_over_R = cfg_.mt_Hp_over_R;
        mt_config_.mdot0 = cfg_.mt_mdot0;
        mt_config_.mdot_cap = cfg_.mt_mdot_cap;
        mt_config_.mdot_mode = cfg_.mt_mdot_mode;
        mt_config_.mdot_prescribed = cfg_.mt_mdot_prescribed;
        mt_config_.beta_mode = cfg_.mt_beta_mode;
        mt_config_.beta_fixed = cfg_.mt_beta_fixed;
        mt_config_.kappa = cfg_.mt_kappa;
        mt_config_.f_disk_outer = cfg_.mt_f_disk_outer;
        mt_config_.logistic_n = cfg_.mt_logistic_n;
        mt_config_.jloss_mode = cfg_.mt_jloss_mode;
        mt_config_.eta_j_fixed = cfg_.mt_eta_j_fixed;
        mt_config_.max_fractional_change = cfg_.mt_max_fractional_change;
    }

    void step() {
        if (!state_) return;
        auto t1 = std::chrono::steady_clock::now();

        if (cfg_.simulation_mode == "mass_transfer") {
            advance_binary(state_->mt, mt_config_, cfg_.dt);

            auto [r_d, r_a] = binary_positions(state_->mt);
            auto [v_d, v_a] = binary_velocities(state_->mt);
            state_->mass_positions[0] = r_d;
            state_->mass_positions[1] = r_a;
            state_->mass_velocities[0] = v_d;
            state_->mass_velocities[1] = v_a;
            masses_[0] = state_->mt.donor_mass;
            masses_[1] = state_->mt.accretor_mass;

            // Leapfrog KDK for test particles / tracers
            size_t n = state_->positions.size();
            if (n > 0) {
                double eps = cfg_.softening;
                size_t mn = 2;
                double dt = cfg_.dt;
                for (size_t i = 0; i < n; ++i) {
                    Vec3 a = compute_acceleration(state_->positions[i],
                                 state_->mass_positions, masses_, eps, mn);
                    state_->velocities[i] += a * (0.5 * dt);
                }
                for (size_t i = 0; i < n; ++i) {
                    state_->positions[i] += state_->velocities[i] * dt;
                }
                for (size_t i = 0; i < n; ++i) {
                    Vec3 a = compute_acceleration(state_->positions[i],
                                 state_->mass_positions, masses_, eps, mn);
                    state_->velocities[i] += a * (0.5 * dt);
                }
            }

            state_->time += cfg_.dt;
            state_->iteration++;

            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t1).count();
            status_text_ = std::format(
                "[{:06d}] t={:.6e} {:.3f} sec/update "
                "Md={:.4f} Ma={:.4f} a={:.4f} mdot={:.2e}",
                state_->iteration, state_->time, elapsed,
                state_->mt.donor_mass, state_->mt.accretor_mass,
                state_->mt.separation, state_->mt.mdot_transfer);
        } else {
            advance_state(*state_, cfg_, masses_, cfg_.dt);
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t1).count();
            status_text_ = std::format(
                "[{:06d}] t={:.6e} {:.3f} sec/update n={}",
                state_->iteration, state_->time, elapsed,
                state_->positions.size());
        }
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

        if (cfg_.simulation_mode == "mass_transfer") {
            build_mt_config();
            s.mt = init_binary_mt(mt_config_);

            auto [r_d, r_a] = binary_positions(s.mt);
            auto [v_d, v_a] = binary_velocities(s.mt);
            s.mass_positions = {r_d, r_a};
            s.mass_velocities = {v_d, v_a};
            masses_ = {s.mt.donor_mass, s.mt.accretor_mass};

            if (cfg_.num_particles > 0) {
                auto [pos, vel] = setup_particles(cfg_, s.mass_positions,
                                                  s.mass_velocities, masses_);
                s.positions = std::move(pos);
                s.velocities = std::move(vel);
            }
        } else {
            auto [mpos, mvel, m] = setup_masses(cfg_);
            s.mass_positions = std::move(mpos);
            s.mass_velocities = std::move(mvel);
            masses_ = std::move(m);
            auto [pos, vel] = setup_particles(cfg_, s.mass_positions,
                                              s.mass_velocities, masses_);
            s.positions = std::move(pos);
            s.velocities = std::move(vel);
        }

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
