#pragma once

// config.hpp must come first — NEST_SERIALIZABLE(Config, ...) must be
// visible before driver.hpp instantiates nest_visit_fields on Config.
#include "config.hpp"
#include "physics.hpp"
#include "state.hpp"

#include <nest/io/driver.hpp>

#include <cstring>
#include <print>
#include <span>
#include <string>
#include <vector>

static_assert(sizeof(Vec3) == 3 * sizeof(double),
              "Vec3 must be tightly packed for checkpoint reinterpret_cast");

class FledgeSim : public nest::io::Simulation<Config, State> {
public:
    // ── Identity ──────────────────────────────────────────────────────
    auto name() const -> const char* override { return "fledge"; }
    auto output_directory() const -> const char* override {
        return config().output_dir.c_str();
    }

    // ── State queries ─────────────────────────────────────────────────
    auto get_time(const State& s) const -> double override {
        return s.time;
    }
    auto get_iteration(const State& s) const -> uint64_t override {
        return s.iteration;
    }
    auto should_continue(const State& s) const -> bool override {
        return s.time < config().tfinal;
    }

    // ── Scheduling ────────────────────────────────────────────────────
    auto checkpoint_interval() const -> double override {
        return config().checkpoint_interval;
    }
    auto timeseries_interval() const -> double override {
        return config().dt * 100.0;
    }
    auto product_interval() const -> double override {
        return config().dt * 1000.0;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────
    void initial_state(State& s) const override {
        s.time = config().tstart;
        s.iteration = 0;
        auto [mpos, mvel, m] = setup_masses(config());
        s.mass_positions = std::move(mpos);
        s.mass_velocities = std::move(mvel);
        masses_ = std::move(m);
        auto [pos, vel] = setup_particles(config(), s.mass_positions,
                                          s.mass_velocities, masses_);
        s.positions = std::move(pos);
        s.velocities = std::move(vel);
    }

    void update(State& s) const override {
        advance_state(s, config(), masses_, config().dt);
    }

    // ── Timeseries ────────────────────────────────────────────────────
    auto timeseries_columns() const -> std::vector<std::string> override {
        return {"time", "num_particles", "max_radius"};
    }

    auto compute_timeseries(const State& s) const
        -> std::vector<double> override {
        double max_r = 0.0;
        for (const auto& p : s.positions) {
            double r = p.mag();
            if (r > max_r) max_r = r;
        }
        return {s.time,
                static_cast<double>(s.positions.size()),
                max_r};
    }

    // ── Products (snapshots) ──────────────────────────────────────────
    auto product_grid(const State& s) const -> nest::io::GridSpec override {
        return {"uniform", {s.positions.size()}, {0.0}, {1.0}};
    }

    auto product_fields(const State& s) const
        -> std::vector<std::pair<std::string, std::vector<double>>> override {
        std::vector<double> xs, ys, zs;
        xs.reserve(s.positions.size());
        ys.reserve(s.positions.size());
        zs.reserve(s.positions.size());
        for (const auto& p : s.positions) {
            xs.push_back(p.x);
            ys.push_back(p.y);
            zs.push_back(p.z);
        }
        return {{"x", std::move(xs)},
                {"y", std::move(ys)},
                {"z", std::move(zs)}};
    }

    // ── Checkpointing ─────────────────────────────────────────────────
    void write_state(nest::io::CheckpointWriter& w,
                     const State& s) const override {
        w.set_scalar("time", s.time);
        w.set_scalar("step", static_cast<int64_t>(s.iteration));

        auto add_vec3 = [&](std::string_view field_name,
                            const std::vector<Vec3>& vecs) {
            std::size_t count = vecs.size() * 3;
            std::size_t shape[] = {count};
            int halo[] = {0};
            w.add_field<double>(field_name,
                std::span<const std::size_t>(shape),
                std::span<const int>(halo),
                std::span<const double>(
                    reinterpret_cast<const double*>(vecs.data()), count));
        };
        add_vec3("positions", s.positions);
        add_vec3("velocities", s.velocities);
        add_vec3("mass_positions", s.mass_positions);
        add_vec3("mass_velocities", s.mass_velocities);
    }

    void read_state(nest::io::CheckpointReader& r,
                    State& s) const override {
        if (auto t = r.scalar_double("time"); t.has_value())
            s.time = *t;
        if (auto i = r.scalar_int("step"); i.has_value())
            s.iteration = static_cast<uint64_t>(*i);

        auto read_vec3 = [&](std::string_view field_name,
                             std::vector<Vec3>& vecs) {
            auto data = r.read_field<double>(field_name);
            if (data.has_value()) {
                vecs.resize(data->size() / 3);
                std::memcpy(vecs.data(), data->data(),
                            data->size() * sizeof(double));
            }
        };
        read_vec3("positions", s.positions);
        read_vec3("velocities", s.velocities);
        read_vec3("mass_positions", s.mass_positions);
        read_vec3("mass_velocities", s.mass_velocities);

        // Recompute mass values from config (constant, not checkpointed)
        auto [mpos, mvel, m] = setup_masses(config());
        masses_ = std::move(m);
    }

    // ── Status ────────────────────────────────────────────────────────
    void print_status(const State& s, double secs_per_update) const override {
        std::println("[{:06d}] t={:.6e}  {:.3f} sec/update  n={}",
                     get_iteration(s), get_time(s), secs_per_update,
                     s.positions.size());
    }

private:
    mutable std::vector<double> masses_;
};
