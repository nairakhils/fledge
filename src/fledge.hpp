#pragma once

// config.hpp must come first — NEST_SERIALIZABLE(Config, ...) must be
// visible before driver.hpp instantiates nest_visit_fields on Config.
#include "config.hpp"
#include "mt/binary_evolution.hpp"
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

        if (config().simulation_mode == "mass_transfer") {
            const auto& mtcfg = ensure_mt_config();
            s.mt = init_binary_mt(mtcfg);

            auto [r_d, r_a] = binary_positions(s.mt);
            auto [v_d, v_a] = binary_velocities(s.mt);
            s.mass_positions = {r_d, r_a};
            s.mass_velocities = {v_d, v_a};
            masses_ = {s.mt.donor_mass, s.mt.accretor_mass};

            if (config().num_particles > 0) {
                auto [pos, vel] = setup_particles(config(), s.mass_positions,
                                                  s.mass_velocities, masses_);
                s.positions = std::move(pos);
                s.velocities = std::move(vel);
            }
        } else {
            auto [mpos, mvel, m] = setup_masses(config());
            s.mass_positions = std::move(mpos);
            s.mass_velocities = std::move(mvel);
            masses_ = std::move(m);
            auto [pos, vel] = setup_particles(config(), s.mass_positions,
                                              s.mass_velocities, masses_);
            s.positions = std::move(pos);
            s.velocities = std::move(vel);
        }
    }

    void update(State& s) const override {
        if (config().simulation_mode == "mass_transfer") {
            const auto& mtcfg = ensure_mt_config();

            advance_binary(s.mt, mtcfg, config().dt);

            auto [r_d, r_a] = binary_positions(s.mt);
            auto [v_d, v_a] = binary_velocities(s.mt);
            s.mass_positions[0] = r_d;
            s.mass_positions[1] = r_a;
            s.mass_velocities[0] = v_d;
            s.mass_velocities[1] = v_a;
            masses_[0] = s.mt.donor_mass;
            masses_[1] = s.mt.accretor_mass;

            // Leapfrog KDK for test particles
            size_t n = s.positions.size();
            if (n > 0) {
                double eps = config().softening;
                size_t mn = 2;
                double dt = config().dt;

                for (size_t i = 0; i < n; ++i) {
                    Vec3 a = compute_acceleration(s.positions[i],
                                 s.mass_positions, masses_, eps, mn);
                    s.velocities[i] += a * (0.5 * dt);
                }
                for (size_t i = 0; i < n; ++i) {
                    s.positions[i] += s.velocities[i] * dt;
                }
                for (size_t i = 0; i < n; ++i) {
                    Vec3 a = compute_acceleration(s.positions[i],
                                 s.mass_positions, masses_, eps, mn);
                    s.velocities[i] += a * (0.5 * dt);
                }
            }

            s.time += config().dt;
            s.iteration++;
        } else {
            advance_state(s, config(), masses_, config().dt);
        }
    }

    // ── Timeseries ────────────────────────────────────────────────────
    auto timeseries_columns() const -> std::vector<std::string> override {
        if (config().simulation_mode == "mass_transfer") {
            return {"time", "num_particles", "max_radius",
                    "donor_mass", "accretor_mass", "separation",
                    "mdot_transfer", "beta", "orbital_period",
                    "roche_radius", "overflow_depth"};
        }
        return {"time", "num_particles", "max_radius"};
    }

    auto compute_timeseries(const State& s) const
        -> std::vector<double> override {
        double max_r = 0.0;
        for (const auto& p : s.positions) {
            double r = p.mag();
            if (r > max_r) max_r = r;
        }

        if (config().simulation_mode == "mass_transfer") {
            return {s.time,
                    static_cast<double>(s.positions.size()),
                    max_r,
                    s.mt.donor_mass,
                    s.mt.accretor_mass,
                    s.mt.separation,
                    s.mt.mdot_transfer,
                    s.mt.beta,
                    s.mt.orbital_period,
                    s.mt.roche_radius,
                    s.mt.overflow_depth};
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

        if (config().simulation_mode == "mass_transfer") {
            w.set_scalar("mt_donor_mass", s.mt.donor_mass);
            w.set_scalar("mt_accretor_mass", s.mt.accretor_mass);
            w.set_scalar("mt_donor_radius", s.mt.donor_radius);
            w.set_scalar("mt_separation", s.mt.separation);
            w.set_scalar("mt_phase", s.mt.phase);
            w.set_scalar("mt_mdot_transfer", s.mt.mdot_transfer);
            w.set_scalar("mt_beta", s.mt.beta);
            w.set_scalar("mt_mdot_loss", s.mt.mdot_loss);
            w.set_scalar("mt_jloss", s.mt.jloss);
            w.set_scalar("mt_cumulative_transferred", s.mt.cumulative_transferred);
            w.set_scalar("mt_cumulative_accreted", s.mt.cumulative_accreted);
            w.set_scalar("mt_cumulative_lost", s.mt.cumulative_lost);
        }
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

        if (config().simulation_mode == "mass_transfer") {
            if (auto v = r.scalar_double("mt_donor_mass")) s.mt.donor_mass = *v;
            if (auto v = r.scalar_double("mt_accretor_mass")) s.mt.accretor_mass = *v;
            if (auto v = r.scalar_double("mt_donor_radius")) s.mt.donor_radius = *v;
            if (auto v = r.scalar_double("mt_separation")) s.mt.separation = *v;
            if (auto v = r.scalar_double("mt_phase")) s.mt.phase = *v;
            if (auto v = r.scalar_double("mt_mdot_transfer")) s.mt.mdot_transfer = *v;
            if (auto v = r.scalar_double("mt_beta")) s.mt.beta = *v;
            if (auto v = r.scalar_double("mt_mdot_loss")) s.mt.mdot_loss = *v;
            if (auto v = r.scalar_double("mt_jloss")) s.mt.jloss = *v;
            if (auto v = r.scalar_double("mt_cumulative_transferred")) s.mt.cumulative_transferred = *v;
            if (auto v = r.scalar_double("mt_cumulative_accreted")) s.mt.cumulative_accreted = *v;
            if (auto v = r.scalar_double("mt_cumulative_lost")) s.mt.cumulative_lost = *v;

            recompute_derived(s.mt);
            masses_ = {s.mt.donor_mass, s.mt.accretor_mass};
        } else {
            auto [mpos, mvel, m] = setup_masses(config());
            masses_ = std::move(m);
        }
    }

    // ── Status ────────────────────────────────────────────────────────
    void print_status(const State& s, double secs_per_update) const override {
        if (config().simulation_mode == "mass_transfer") {
            std::println("[{:06d}] t={:.6e}  {:.3f} sec/update  n={}  "
                         "M_d={:.6f}  M_a={:.6f}  a={:.6f}  mdot={:.4e}",
                         get_iteration(s), get_time(s), secs_per_update,
                         s.positions.size(),
                         s.mt.donor_mass, s.mt.accretor_mass,
                         s.mt.separation, s.mt.mdot_transfer);
        } else {
            std::println("[{:06d}] t={:.6e}  {:.3f} sec/update  n={}",
                         get_iteration(s), get_time(s), secs_per_update,
                         s.positions.size());
        }
    }

private:
    mutable std::vector<double> masses_;
    mutable MassTransferConfig mt_config_;
    mutable bool mt_config_built_ = false;

    auto ensure_mt_config() const -> const MassTransferConfig& {
        if (!mt_config_built_) {
            mt_config_.donor_mass0 = config().mt_donor_mass0;
            mt_config_.accretor_mass0 = config().mt_accretor_mass0;
            mt_config_.donor_radius0 = config().mt_donor_radius0;
            mt_config_.separation0 = config().mt_separation0;
            mt_config_.phase0 = config().mt_phase0;
            mt_config_.donor_radius_mode = config().mt_donor_radius_mode;
            mt_config_.zeta_star = config().mt_zeta_star;
            mt_config_.tau_drive = config().mt_tau_drive;
            mt_config_.Hp_over_R = config().mt_Hp_over_R;
            mt_config_.mdot0 = config().mt_mdot0;
            mt_config_.mdot_cap = config().mt_mdot_cap;
            mt_config_.mdot_mode = config().mt_mdot_mode;
            mt_config_.mdot_prescribed = config().mt_mdot_prescribed;
            mt_config_.beta_mode = config().mt_beta_mode;
            mt_config_.beta_fixed = config().mt_beta_fixed;
            mt_config_.kappa = config().mt_kappa;
            mt_config_.f_disk_outer = config().mt_f_disk_outer;
            mt_config_.logistic_n = config().mt_logistic_n;
            mt_config_.jloss_mode = config().mt_jloss_mode;
            mt_config_.eta_j_fixed = config().mt_eta_j_fixed;
            mt_config_.max_fractional_change = config().mt_max_fractional_change;
            mt_config_built_ = true;
        }
        return mt_config_;
    }
};
