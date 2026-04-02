#define SDL_MAIN_HANDLED
#define GL_SILENCE_DEPRECATION
#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#if defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#include "interactive/worker.hpp"
#include "config.hpp"
#include "mt/roche.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

// ── Helpers ──────────────────────────────────────────────────────────────

enum class LogLevel { Info, Error, Iteration };

struct LogEntry {
    float timestamp;
    LogLevel level;
    std::string message;
};

static auto event_log_level(const fledge::Event& ev) -> LogLevel {
    if (std::holds_alternative<fledge::EvError>(ev)) return LogLevel::Error;
    return LogLevel::Info;
}

static auto format_event(const fledge::Event& ev) -> std::string {
    return std::visit([](const auto& e) -> std::string {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, fledge::EvStateCreated>)
            return "State created";
        else if constexpr (std::is_same_v<T, fledge::EvStateDestroyed>)
            return "State destroyed";
        else if constexpr (std::is_same_v<T, fledge::EvSimulationDone>)
            return "Simulation done";
        else if constexpr (std::is_same_v<T, fledge::EvCheckpointWritten>)
            return "Checkpoint: " + e.path;
        else if constexpr (std::is_same_v<T, fledge::EvError>)
            return "Error: " + e.message;
        else if constexpr (std::is_same_v<T, fledge::EvFinished>)
            return "Finished";
        else
            return "Unknown";
    }, ev);
}

static auto combo_index(const std::string& val, const char* const* items, int count) -> int {
    for (int i = 0; i < count; i++)
        if (val == items[i]) return i;
    return 0;
}

// ── Rolling buffer for time-history plots ────────────────────────────────

struct RollingBuffer {
    std::vector<double> time;
    std::vector<double> data;
    size_t max_size = 5000;

    void push(double t, double val) {
        time.push_back(t);
        data.push_back(val);
        if (time.size() > max_size) {
            time.erase(time.begin());
            data.erase(data.begin());
        }
    }
    void clear() { time.clear(); data.clear(); }
    auto size() const -> int { return static_cast<int>(time.size()); }
    auto empty() const -> bool { return time.empty(); }
};

// ── Marching squares contour extraction ──────────────────────────────────

struct LineSegment { double x1, y1, x2, y2; };

static auto marching_squares(const std::vector<double>& grid, int nx, int ny,
                             double x0, double y0, double dx, double dy,
                             double level) -> std::vector<LineSegment>
{
    std::vector<LineSegment> segs;
    auto idx = [nx](int ix, int iy) { return iy * nx + ix; };

    auto lerp_x = [&](int ix, int iy0, int iy1) -> double {
        (void)iy1;
        return x0 + ix * dx;
    };
    (void)lerp_x;

    for (int iy = 0; iy < ny - 1; ++iy) {
        for (int ix = 0; ix < nx - 1; ++ix) {
            double v00 = grid[idx(ix,   iy)]   - level;
            double v10 = grid[idx(ix+1, iy)]   - level;
            double v11 = grid[idx(ix+1, iy+1)] - level;
            double v01 = grid[idx(ix,   iy+1)] - level;

            int config_bits = 0;
            if (v00 > 0) config_bits |= 1;
            if (v10 > 0) config_bits |= 2;
            if (v11 > 0) config_bits |= 4;
            if (v01 > 0) config_bits |= 8;

            if (config_bits == 0 || config_bits == 15) continue;

            // Edge midpoints with linear interpolation
            double cx = x0 + ix * dx;
            double cy = y0 + iy * dy;

            // Bottom edge: (ix,iy)-(ix+1,iy)
            double bx = cx + dx * (-v00) / (v10 - v00);
            double by = cy;
            // Right edge: (ix+1,iy)-(ix+1,iy+1)
            double rx = cx + dx;
            double ry = cy + dy * (-v10) / (v11 - v10);
            // Top edge: (ix,iy+1)-(ix+1,iy+1)
            double tx = cx + dx * (-v01) / (v11 - v01);
            double ty = cy + dy;
            // Left edge: (ix,iy)-(ix,iy+1)
            double lx = cx;
            double ly = cy + dy * (-v00) / (v01 - v00);

            auto add = [&](double ax, double ay, double bxx, double byy) {
                segs.push_back({ax, ay, bxx, byy});
            };

            switch (config_bits) {
                case  1: case 14: add(bx,by, lx,ly); break;
                case  2: case 13: add(bx,by, rx,ry); break;
                case  3: case 12: add(lx,ly, rx,ry); break;
                case  4: case 11: add(rx,ry, tx,ty); break;
                case  5: add(bx,by, rx,ry); add(lx,ly, tx,ty); break;
                case  6: case  9: add(bx,by, tx,ty); break;
                case  7: case  8: add(lx,ly, tx,ty); break;
                case 10: add(bx,by, lx,ly); add(rx,ry, tx,ty); break;
            }
        }
    }
    return segs;
}

// ── Roche contour cache ─────────────────────────────────────────────────

struct RocheCache {
    double cached_q = -1.0;
    double cached_a = -1.0;
    std::vector<LineSegment> contour;

    // Recompute if q or a changed significantly
    void update(double M_d, double M_a, double a) {
        double q = M_d / M_a;
        if (cached_a > 0 && std::abs(q - cached_q) / q < 0.001 &&
            std::abs(a - cached_a) / cached_a < 0.001)
            return;

        cached_q = q;
        cached_a = a;

        double M_tot = M_d + M_a;
        double x_d = -M_a / M_tot * a;
        double x_a =  M_d / M_tot * a;
        double Omega2 = M_tot / (a * a * a);

        // Phi_eff at L1
        double xi_L1 = find_lagrange_L1(q);
        double xL1_com = (xi_L1 - M_a / M_tot) * a;
        double rd_L1 = std::abs(xL1_com - x_d);
        double ra_L1 = std::abs(xL1_com - x_a);
        double phi_L1 = -M_d / rd_L1 - M_a / ra_L1
                       - 0.5 * Omega2 * xL1_com * xL1_com;

        // Build potential grid
        constexpr int N = 200;
        double extent = 2.5 * a;
        double gx0 = -extent, gy0 = -extent;
        double gdx = 2.0 * extent / (N - 1);
        double gdy = 2.0 * extent / (N - 1);

        std::vector<double> grid(N * N);
        for (int iy = 0; iy < N; ++iy) {
            double y = gy0 + iy * gdy;
            for (int ix = 0; ix < N; ++ix) {
                double x = gx0 + ix * gdx;
                double rd = std::sqrt((x - x_d) * (x - x_d) + y * y);
                double ra = std::sqrt((x - x_a) * (x - x_a) + y * y);
                if (rd < 1e-6 * a) rd = 1e-6 * a;
                if (ra < 1e-6 * a) ra = 1e-6 * a;
                grid[iy * N + ix] = -M_d / rd - M_a / ra
                                   - 0.5 * Omega2 * (x * x + y * y);
            }
        }

        contour = marching_squares(grid, N, N, gx0, gy0, gdx, gdy, phi_L1);
    }
};

// ── Application state ────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;

struct App {
    Config config;
    std::optional<fledge::DriverHandle> handle;
    fledge::RuntimeSnapshot snapshot;

    bool show_log = false;
    bool auto_fit = false;
    float left_panel_width = 400.0f;

    std::vector<LogEntry> log_entries;
    std::string last_event_msg;

    Clock::time_point app_start = Clock::now();

    // Speed tracking
    int64_t prev_iteration = -1;
    int64_t speed_base_iter = 0;
    double speed_base_wall = 0.0;
    double iter_per_sec = 0.0;

    // Mass-transfer visualization state
    bool mt_corotating = true;
    RocheCache roche_cache;
    RollingBuffer rb_donor_mass;
    RollingBuffer rb_accretor_mass;
    RollingBuffer rb_separation;
    RollingBuffer rb_roche_radius;
    RollingBuffer rb_overflow;
    RollingBuffer rb_mdot;
    RollingBuffer rb_beta;
    int64_t mt_last_push_iter = -1;

    void clear_mt_buffers() {
        rb_donor_mass.clear();
        rb_accretor_mass.clear();
        rb_separation.clear();
        rb_roche_radius.clear();
        rb_overflow.clear();
        rb_mdot.clear();
        rb_beta.clear();
        mt_last_push_iter = -1;
        roche_cache.cached_q = -1.0;
    }
};

static auto elapsed(const App& app) -> float {
    return std::chrono::duration<float>(Clock::now() - app.app_start).count();
}

static void recreate_driver(App& app) {
    app.handle.reset();
    app.handle.emplace(fledge::spawn(fledge::InteractiveDriver(app.config)));
}

static void apply_config(App& app) {
    recreate_driver(app);
    app.handle->send(fledge::CmdCreateState{});
    app.log_entries.push_back({elapsed(app), LogLevel::Info, "Config applied"});
}

static void drain_events(App& app) {
    if (!app.handle) return;

    float ts = elapsed(app);

    while (auto ev = app.handle->try_recv()) {
        auto msg = format_event(*ev);
        auto lvl = event_log_level(*ev);
        app.log_entries.push_back({ts, lvl, msg});
        app.last_event_msg = msg;
        if (std::holds_alternative<fledge::EvStateCreated>(*ev)) {
            app.auto_fit = true;
            app.clear_mt_buffers();
        }
        if (std::holds_alternative<fledge::EvStateDestroyed>(*ev))
            app.clear_mt_buffers();
    }

    // Log iteration status when iteration advances
    auto& snap = app.snapshot;
    if (snap.has_state && snap.iteration != app.prev_iteration) {
        if (app.prev_iteration >= 0 && !snap.status_text.empty()) {
            app.log_entries.push_back({ts, LogLevel::Iteration, snap.status_text});
        }
        app.prev_iteration = snap.iteration;
    }

    // Speed computation: iter/sec over the last ~1 second
    double wall = static_cast<double>(ts);
    double dt = wall - app.speed_base_wall;
    if (dt >= 1.0) {
        int64_t di = snap.iteration - app.speed_base_iter;
        app.iter_per_sec = (dt > 0.0) ? static_cast<double>(di) / dt : 0.0;
        app.speed_base_iter = snap.iteration;
        app.speed_base_wall = wall;
    }

    // Push MT rolling buffer data when iteration advances
    if (snap.mt.active && snap.iteration != app.mt_last_push_iter) {
        app.mt_last_push_iter = snap.iteration;
        double t = snap.time;
        app.rb_donor_mass.push(t, snap.mt.donor_mass);
        app.rb_accretor_mass.push(t, snap.mt.accretor_mass);
        app.rb_separation.push(t, snap.mt.separation);
        app.rb_roche_radius.push(t, snap.mt.roche_radius);
        app.rb_overflow.push(t, snap.mt.overflow_depth);
        app.rb_mdot.push(t, snap.mt.mdot_transfer);
        app.rb_beta.push(t, snap.mt.beta);
    }
}

// ── Config editor (left panel) ───────────────────────────────────────────

static void render_config_panel(App& app) {
    auto& c = app.config;
    bool has_state = app.snapshot.has_state;

    // Simulation mode selector
    {
        static const char* modes[] = {"test_particles", "mass_transfer"};
        int mi = combo_index(c.simulation_mode, modes, 2);
        if (ImGui::Combo("Simulation Mode", &mi, modes, 2)) {
            c.simulation_mode = modes[mi];
            if (has_state && app.handle)
                app.handle->send(fledge::CmdDestroyState{});
        }
        ImGui::Separator();
    }

    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputDouble("tstart", &c.tstart, 0, 0, "%.4g");
        ImGui::InputDouble("tfinal", &c.tfinal, 0, 0, "%.4g");
        ImGui::InputDouble("dt", &c.dt, 0, 0, "%.6g");
        ImGui::InputDouble("softening", &c.softening, 0, 0, "%.4g");
    }

    if (c.simulation_mode != "mass_transfer" &&
        ImGui::CollapsingHeader("Central Object", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* obj_types[] = {"single", "binary", "triple"};
        int idx = combo_index(c.central_object_type, obj_types, 3);
        if (ImGui::Combo("Type", &idx, obj_types, 3))
            c.central_object_type = obj_types[idx];

        ImGui::InputDouble("mass", &c.mass, 0, 0, "%.6g");

        bool show_binary = (c.central_object_type != "single");
        bool show_triple = (c.central_object_type == "triple");

        if (show_binary) {
            ImGui::InputDouble("q1", &c.q1, 0, 0, "%.6g");
            ImGui::InputDouble("a1", &c.a1, 0, 0, "%.6g");
            ImGui::InputDouble("e1x", &c.e1x, 0, 0, "%.6g");
            ImGui::InputDouble("e1y", &c.e1y, 0, 0, "%.6g");
        }
        if (show_triple) {
            ImGui::InputDouble("q2", &c.q2, 0, 0, "%.6g");
            ImGui::InputDouble("a2", &c.a2, 0, 0, "%.6g");
            ImGui::InputDouble("e2x", &c.e2x, 0, 0, "%.6g");
            ImGui::InputDouble("e2y", &c.e2y, 0, 0, "%.6g");
        }
        if (show_binary) {
            ImGui::InputDouble("inclination", &c.inclination, 0, 0, "%.6g");
        }
    }

    // Disable initial conditions while state exists
    if (has_state) ImGui::BeginDisabled();

    if (ImGui::CollapsingHeader("Initial Conditions", ImGuiTreeNodeFlags_DefaultOpen)) {
        int np = static_cast<int>(c.num_particles);
        if (ImGui::InputInt("num_particles", &np, 10, 100))
            c.num_particles = static_cast<uint64_t>(std::max(1, np));

        static const char* setup_types[] = {"ring", "random_disk", "uniform_disk"};
        int si = combo_index(c.setup_type, setup_types, 3);
        if (ImGui::Combo("setup_type", &si, setup_types, 3))
            c.setup_type = setup_types[si];

        if (c.setup_type == "ring") {
            ImGui::InputDouble("ring_radius", &c.ring_radius, 0, 0, "%.4g");
        } else {
            ImGui::InputDouble("inner_radius", &c.inner_radius, 0, 0, "%.4g");
            ImGui::InputDouble("outer_radius", &c.outer_radius, 0, 0, "%.4g");
        }

        if (c.central_object_type == "binary") {
            static const char* centers[] = {"primary", "secondary", "arbitrary"};
            int di = combo_index(c.disk_center, centers, 3);
            if (ImGui::Combo("disk_center", &di, centers, 3))
                c.disk_center = centers[di];
        }
        if (c.disk_center == "arbitrary") {
            ImGui::InputDouble("disk_center_x", &c.disk_center_x, 0, 0, "%.4g");
            ImGui::InputDouble("disk_center_y", &c.disk_center_y, 0, 0, "%.4g");
            ImGui::InputDouble("disk_center_z", &c.disk_center_z, 0, 0, "%.4g");
        }
    }

    if (has_state) ImGui::EndDisabled();

    if (ImGui::CollapsingHeader("Output")) {
        ImGui::InputDouble("checkpoint_interval", &c.checkpoint_interval, 0, 0, "%.4g");
        char buf[256];
        std::strncpy(buf, c.output_dir.c_str(), sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        if (ImGui::InputText("output_dir", buf, sizeof(buf)))
            c.output_dir = buf;
    }

    // Mass transfer config (only shown in mass_transfer mode)
    if (c.simulation_mode == "mass_transfer") {
        if (has_state) ImGui::BeginDisabled();

        if (ImGui::CollapsingHeader("Mass Transfer", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragScalar("donor_mass0", ImGuiDataType_Double,
                              &c.mt_donor_mass0, 0.01f, nullptr, nullptr, "%.4f");
            ImGui::DragScalar("accretor_mass0", ImGuiDataType_Double,
                              &c.mt_accretor_mass0, 0.01f, nullptr, nullptr, "%.4f");
            ImGui::DragScalar("separation0", ImGuiDataType_Double,
                              &c.mt_separation0, 0.01f, nullptr, nullptr, "%.4f");
            ImGui::DragScalar("donor_radius0", ImGuiDataType_Double,
                              &c.mt_donor_radius0, 0.01f, nullptr, nullptr, "%.4f");

            ImGui::Separator();

            static const char* mdot_modes[] = {"ritter", "capped", "prescribed"};
            int mdi = combo_index(c.mt_mdot_mode, mdot_modes, 3);
            if (ImGui::Combo("mdot_mode", &mdi, mdot_modes, 3))
                c.mt_mdot_mode = mdot_modes[mdi];

            ImGui::InputDouble("Hp_over_R", &c.mt_Hp_over_R, 0, 0, "%.2e");
            ImGui::InputDouble("mdot0", &c.mt_mdot0, 0, 0, "%.2e");
            if (c.mt_mdot_mode == "prescribed")
                ImGui::InputDouble("mdot_prescribed", &c.mt_mdot_prescribed, 0, 0, "%.4e");

            ImGui::Separator();

            static const char* beta_modes[] = {"fixed", "super_eddington"};
            int bi = combo_index(c.mt_beta_mode, beta_modes, 2);
            if (ImGui::Combo("beta_mode", &bi, beta_modes, 2))
                c.mt_beta_mode = beta_modes[bi];
            if (c.mt_beta_mode == "fixed") {
                double lo = 0.0, hi = 1.0;
                ImGui::SliderScalar("beta_fixed", ImGuiDataType_Double,
                                    &c.mt_beta_fixed, &lo, &hi, "%.3f");
            }

            static const char* jloss_modes[] = {"L2_exact", "scherbak_adiabatic", "fixed_eta"};
            int ji = combo_index(c.mt_jloss_mode, jloss_modes, 3);
            if (ImGui::Combo("jloss_mode", &ji, jloss_modes, 3))
                c.mt_jloss_mode = jloss_modes[ji];
            if (c.mt_jloss_mode == "fixed_eta") {
                double lo = 0.0, hi = 2.0;
                ImGui::SliderScalar("eta_j_fixed", ImGuiDataType_Double,
                                    &c.mt_eta_j_fixed, &lo, &hi, "%.3f");
            }

            ImGui::Separator();
            ImGui::InputDouble("zeta_star", &c.mt_zeta_star, 0, 0, "%.4f");
            ImGui::InputDouble("tau_drive", &c.mt_tau_drive, 0, 0, "%.2e");
        }

        if (has_state) ImGui::EndDisabled();
    }
}

// ── Mass-transfer binary system view ─────────────────────────────────────

static void render_mt_binary_view(App& app) {
    auto& snap = app.snapshot;
    auto& m = snap.mt;
    if (!m.active) return;

    double a = m.separation;
    double M_d = m.donor_mass;
    double M_a = m.accretor_mass;
    double M_tot = M_d + M_a;
    double q_d = M_d / M_a;
    double phase = m.phase;
    double cp = std::cos(phase);
    double sp = std::sin(phase);

    // Star positions in COM frame (inertial)
    double x_d_com = -(M_a / M_tot) * a;
    double x_a_com =  (M_d / M_tot) * a;

    // Frame toggle
    ImGui::Checkbox("Corotating frame", &app.mt_corotating);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                       "(uncheck for inertial / spiral view)");

    // Update Roche contour cache
    app.roche_cache.update(M_d, M_a, a);

    constexpr ImVec4 col_donor   = {1.0f, 0.53f, 0.27f, 0.8f};
    constexpr ImVec4 col_accretor = {0.4f, 0.67f, 1.0f, 0.8f};
    constexpr ImVec4 col_tracer  = {1.0f, 1.0f, 1.0f, 0.5f};
    constexpr ImVec4 col_roche   = {0.6f, 0.6f, 0.6f, 0.6f};
    constexpr ImVec4 col_lpoint  = {0.9f, 0.9f, 0.2f, 0.8f};

    ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0.06f, 0.06f, 0.08f, 1.0f));

    double extent = 2.5 * a;
    if (ImPlot::BeginPlot("##binary_mt", ImVec2(-1, -1),
                          ImPlotFlags_Equal | ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("x", "y");
        if (app.auto_fit) {
            ImPlot::SetupAxesLimits(-extent, extent, -extent, extent,
                                    ImPlotCond_Always);
            app.auto_fit = false;
        }

        // Roche contour (corotating frame only)
        if (app.mt_corotating) {
            for (auto& seg : app.roche_cache.contour) {
                double xs[] = {seg.x1, seg.x2};
                double ys[] = {seg.y1, seg.y2};
                ImPlot::SetNextLineStyle(col_roche, 1.0f);
                ImPlot::PlotLine("##roche", xs, ys, 2);
            }
        }

        // Determine display positions
        double dx, dy, ax_disp, ay_disp;
        if (app.mt_corotating) {
            dx = x_d_com; dy = 0.0;
            ax_disp = x_a_com; ay_disp = 0.0;
        } else {
            dx = x_d_com * cp; dy = x_d_com * sp;
            ax_disp = x_a_com * cp; ay_disp = x_a_com * sp;
        }

        // Stars as scatter points
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f, col_donor, 0);
        ImPlot::PlotScatter("donor", &dx, &dy, 1);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, col_accretor, 0);
        ImPlot::PlotScatter("accretor", &ax_disp, &ay_disp, 1);

        // Draw filled circles for stellar radii using ImDrawList
        auto* draw_list = ImPlot::GetPlotDrawList();
        {
            auto p_d = ImPlot::PlotToPixels(dx, dy);
            double r_d_plot = m.donor_radius;
            auto p_d_edge = ImPlot::PlotToPixels(dx + r_d_plot, dy);
            float r_d_px = std::max(6.0f, std::abs(p_d_edge.x - p_d.x));
            draw_list->AddCircleFilled(p_d, r_d_px,
                ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 0.53f, 0.27f, 0.25f)));
        }

        // COM marker
        {
            double ox = 0.0, oy = 0.0;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Plus, 4.0f,
                                       ImVec4(0.5f, 0.5f, 0.5f, 0.7f), 1.0f);
            ImPlot::PlotScatter("COM", &ox, &oy, 1);
        }

        // Lagrange points (corotating frame only)
        if (app.mt_corotating) {
            double x_com_frac = M_a / M_tot;
            double xi_L1 = find_lagrange_L1(q_d);
            double xi_L2 = find_lagrange_L2(q_d);
            double xi_L3 = find_lagrange_L3(q_d);
            double xL1 = (xi_L1 - x_com_frac) * a;
            double xL2 = (xi_L2 - x_com_frac) * a;
            double xL3 = (xi_L3 - x_com_frac) * a;
            double yL = 0.0;

            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 4.0f, col_lpoint, 1.0f);
            ImPlot::PlotScatter("##L1", &xL1, &yL, 1);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 4.0f, col_lpoint, 1.0f);
            ImPlot::PlotScatter("##L2", &xL2, &yL, 1);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 4.0f, col_lpoint, 1.0f);
            ImPlot::PlotScatter("##L3", &xL3, &yL, 1);

            ImPlot::Annotation(xL1, yL, col_lpoint, ImVec2(5, -10), false, "L1");
            ImPlot::Annotation(xL2, yL, col_lpoint, ImVec2(5, -10), false, "L2");
            ImPlot::Annotation(xL3, yL, col_lpoint, ImVec2(5, -10), false, "L3");
        }

        // Tracer particles
        if (snap.linear.contains("x") && snap.linear.contains("y")) {
            auto& xs_in = snap.linear.at("x");
            auto& ys_in = snap.linear.at("y");
            if (!xs_in.empty()) {
                if (app.mt_corotating) {
                    // Transform from inertial to corotating
                    std::vector<double> cx(xs_in.size()), cy(xs_in.size());
                    for (size_t i = 0; i < xs_in.size(); ++i) {
                        cx[i] =  xs_in[i] * cp + ys_in[i] * sp;
                        cy[i] = -xs_in[i] * sp + ys_in[i] * cp;
                    }
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.5f, col_tracer, 0);
                    ImPlot::PlotScatter("tracers", cx.data(), cy.data(),
                                        static_cast<int>(cx.size()));
                } else {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.5f, col_tracer, 0);
                    ImPlot::PlotScatter("tracers", xs_in.data(), ys_in.data(),
                                        static_cast<int>(xs_in.size()));
                }
            }
        }

        // Info overlay
        {
            auto tl = ImPlot::GetPlotPos();
            auto sz = ImPlot::GetPlotSize();
            float ox = tl.x + sz.x - 10.0f;
            float oy = tl.y + 10.0f;
            auto txt = std::format(
                "q = {:.3f}\na = {:.4f}\nR_L = {:.4f}\ndR = {:.2e}\nmdot = {:.2e}\nbeta = {:.3f}",
                M_a / M_d, a, m.roche_radius, m.overflow_depth,
                m.mdot_transfer, m.beta);
            draw_list->AddText(ImVec2(ox - 130.0f, oy),
                               ImGui::ColorConvertFloat4ToU32(ImVec4(0.7f, 0.7f, 0.7f, 0.8f)),
                               txt.c_str());
        }

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleColor();
}

// ── Mass-transfer evolution strip charts ─────────────────────────────────

static void render_mt_evolution(App& app) {
    if (app.rb_donor_mass.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                           "No evolution data yet. Run the simulation.");
        return;
    }

    constexpr ImVec4 col_orange = {1.0f, 0.6f, 0.2f, 1.0f};
    constexpr ImVec4 col_blue   = {0.4f, 0.67f, 1.0f, 1.0f};
    constexpr ImVec4 col_white  = {1.0f, 1.0f, 1.0f, 0.9f};
    constexpr ImVec4 col_grey   = {0.5f, 0.5f, 0.5f, 0.7f};
    constexpr ImVec4 col_green  = {0.3f, 0.9f, 0.3f, 0.9f};

    if (ImPlot::BeginSubplots("##evolution", 5, 1, ImVec2(-1, -1),
                               ImPlotSubplotFlags_LinkAllX)) {

        // Row 1: Masses
        if (ImPlot::BeginPlot("Masses", ImVec2(-1, 0))) {
            ImPlot::SetupAxes("", "M", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetNextLineStyle(col_orange, 1.5f);
            ImPlot::PlotLine("M_donor", app.rb_donor_mass.time.data(),
                             app.rb_donor_mass.data.data(), app.rb_donor_mass.size());
            ImPlot::SetNextLineStyle(col_blue, 1.5f);
            ImPlot::PlotLine("M_accretor", app.rb_accretor_mass.time.data(),
                             app.rb_accretor_mass.data.data(), app.rb_accretor_mass.size());
            ImPlot::EndPlot();
        }

        // Row 2: Orbit — separation and Roche radius
        if (ImPlot::BeginPlot("Orbit", ImVec2(-1, 0))) {
            ImPlot::SetupAxes("", "a, R_L", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetNextLineStyle(col_white, 1.5f);
            ImPlot::PlotLine("separation", app.rb_separation.time.data(),
                             app.rb_separation.data.data(), app.rb_separation.size());
            ImPlot::SetNextLineStyle(col_grey, 1.0f);
            ImPlot::PlotLine("R_L", app.rb_roche_radius.time.data(),
                             app.rb_roche_radius.data.data(), app.rb_roche_radius.size());
            ImPlot::EndPlot();
        }

        // Row 3: Overflow depth
        if (ImPlot::BeginPlot("Overflow", ImVec2(-1, 0))) {
            ImPlot::SetupAxes("", "dR", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetNextLineStyle(col_green, 1.5f);
            ImPlot::PlotLine("overflow_depth", app.rb_overflow.time.data(),
                             app.rb_overflow.data.data(), app.rb_overflow.size());
            double zero_x[] = {app.rb_overflow.time.front(), app.rb_overflow.time.back()};
            double zero_y[] = {0.0, 0.0};
            ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 0.4f), 1.0f);
            ImPlot::PlotLine("##zero", zero_x, zero_y, 2);
            ImPlot::EndPlot();
        }

        // Row 4: Transfer rate
        if (ImPlot::BeginPlot("Transfer Rate", ImVec2(-1, 0))) {
            ImPlot::SetupAxes("", "mdot", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetNextLineStyle(col_orange, 1.5f);
            ImPlot::PlotLine("mdot_tr", app.rb_mdot.time.data(),
                             app.rb_mdot.data.data(), app.rb_mdot.size());
            ImPlot::EndPlot();
        }

        // Row 5: Beta
        if (ImPlot::BeginPlot("Beta", ImVec2(-1, 0))) {
            ImPlot::SetupAxes("time", "beta",
                              ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetNextLineStyle(col_blue, 1.5f);
            ImPlot::PlotLine("beta", app.rb_beta.time.data(),
                             app.rb_beta.data.data(), app.rb_beta.size());
            ImPlot::EndPlot();
        }

        ImPlot::EndSubplots();
    }
}

// ── Scatter plot (right panel) ───────────────────────────────────────────

static void render_plot(App& app) {
    auto& snap = app.snapshot;

    // Show centered hint when no state exists
    if (!snap.has_state) {
        auto avail = ImGui::GetContentRegionAvail();
        const char* hint = "Press [N] to create initial state";
        auto sz = ImGui::CalcTextSize(hint);
        ImGui::SetCursorPos(ImVec2(
            ImGui::GetCursorPosX() + (avail.x - sz.x) * 0.5f,
            ImGui::GetCursorPosY() + (avail.y - sz.y) * 0.5f));
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s", hint);
        return;
    }

    constexpr ImVec4 col_particle  = {1.0f, 1.0f, 1.0f, 0.4f};
    constexpr ImVec4 col_primary   = {1.0f, 0.863f, 0.471f, 1.0f};
    constexpr ImVec4 col_secondary = {1.0f, 0.588f, 0.471f, 1.0f};
    constexpr ImVec4 col_tertiary  = {0.471f, 0.784f, 1.0f, 1.0f};

    ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f));

    if (ImPlot::BeginPlot("##particles", ImVec2(-1, -1),
                          ImPlotFlags_Equal | ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("x", "y");

        if (app.auto_fit && snap.linear.contains("x") &&
            !snap.linear.at("x").empty()) {
            auto& xs = snap.linear.at("x");
            auto& ys = snap.linear.at("y");
            auto [xlo, xhi] = std::minmax_element(xs.begin(), xs.end());
            auto [ylo, yhi] = std::minmax_element(ys.begin(), ys.end());
            double pad = std::max(0.5, std::max(*xhi - *xlo, *yhi - *ylo) * 0.1);
            ImPlot::SetupAxesLimits(*xlo - pad, *xhi + pad,
                                    *ylo - pad, *yhi + pad,
                                    ImPlotCond_Always);
            app.auto_fit = false;
        }

        if (snap.linear.contains("x") && snap.linear.contains("y")) {
            auto& xs = snap.linear.at("x");
            auto& ys = snap.linear.at("y");
            if (!xs.empty()) {
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.5f, col_particle, 0);
                ImPlot::PlotScatter("particles", xs.data(), ys.data(),
                                    static_cast<int>(xs.size()));
            }
        }

        if (snap.linear.contains("mass_x") && snap.linear.contains("mass_y")) {
            auto& mx = snap.linear.at("mass_x");
            auto& my = snap.linear.at("mass_y");
            const ImVec4 colors[] = {col_primary, col_secondary, col_tertiary};
            const float sizes[] = {6.0f, 5.0f, 3.5f};
            const char* labels[] = {"primary", "secondary", "tertiary"};
            for (int i = 0; i < static_cast<int>(mx.size()) && i < 3; i++) {
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, sizes[i],
                                           colors[i], 0);
                ImPlot::PlotScatter(labels[i], &mx[i], &my[i], 1);
            }
        }

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleColor();
}

// ── Log view (right panel alternate) ─────────────────────────────────────

static void render_log(App& app) {
    constexpr ImVec4 col_info = {0.4f, 0.9f, 0.4f, 1.0f};
    constexpr ImVec4 col_error = {1.0f, 0.4f, 0.4f, 1.0f};
    constexpr ImVec4 col_iter = {0.45f, 0.45f, 0.45f, 1.0f};
    constexpr ImVec4 col_ts = {0.35f, 0.35f, 0.35f, 1.0f};

    ImGui::BeginChild("##log_scroll", ImVec2(-1, -1), ImGuiChildFlags_None);

    for (auto& entry : app.log_entries) {
        ImGui::TextColored(col_ts, "[%7.1fs]", static_cast<double>(entry.timestamp));
        ImGui::SameLine();
        ImVec4 color;
        switch (entry.level) {
            case LogLevel::Info:      color = col_info; break;
            case LogLevel::Error:     color = col_error; break;
            case LogLevel::Iteration: color = col_iter; break;
        }
        ImGui::TextColored(color, "%s", entry.message.c_str());
    }

    // Auto-scroll to bottom
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f)
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
}

// ── Footer bar ───────────────────────────────────────────────────────────

static auto footer_hint(const char* key, const char* desc) -> bool {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.5f));
    auto label = std::format("[{}] {}", key, desc);
    bool clicked = ImGui::SmallButton(label.c_str());
    ImGui::PopStyleColor(3);
    return clicked;
}

static void render_footer(App& app) {
    bool running = (app.snapshot.mode == fledge::DriverMode::Running);
    bool has_state = app.snapshot.has_state;

    ImGui::Separator();

    if (running) {
        if (footer_hint("P", "pause") && app.handle)
            app.handle->send(fledge::CmdPause{});
    } else if (has_state) {
        if (footer_hint("P", "play") && app.handle)
            app.handle->send(fledge::CmdRun{});
    }

    if (has_state && !running) {
        ImGui::SameLine();
        if (footer_hint("S", "step") && app.handle)
            app.handle->send(fledge::CmdStep{});
    }

    if (!has_state) {
        ImGui::SameLine();
        if (footer_hint("N", "new"))
            apply_config(app);
    }

    if (has_state && !running) {
        ImGui::SameLine();
        if (footer_hint("D", "destroy") && app.handle)
            app.handle->send(fledge::CmdDestroyState{});
        ImGui::SameLine();
        if (footer_hint("C", "checkpoint") && app.handle)
            app.handle->send(fledge::CmdCheckpoint{});
    }

    ImGui::SameLine();
    if (footer_hint("L", app.show_log ? "plot" : "log"))
        app.show_log = !app.show_log;

    // Right side: speed + last event + iteration counter
    float right_width = 600.0f;
    float avail = ImGui::GetContentRegionAvail().x;
    if (avail > right_width)
        ImGui::SameLine(avail - right_width + ImGui::GetCursorPosX());
    else
        ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    if (!app.last_event_msg.empty()) {
        ImGui::Text("%s", app.last_event_msg.c_str());
        ImGui::SameLine();
    }
    if (running && app.iter_per_sec > 0.0) {
        ImGui::Text("%.0f iter/s", app.iter_per_sec);
        ImGui::SameLine();
    }
    ImGui::PopStyleColor();

    if (has_state) {
        if (app.snapshot.mt.active) {
            auto& m = app.snapshot.mt;
            ImGui::Text("[%06lld] t=%.4f  Md=%.4f  Ma=%.4f  a=%.4f  mdot=%.2e",
                        static_cast<long long>(app.snapshot.iteration),
                        app.snapshot.time,
                        m.donor_mass, m.accretor_mass, m.separation,
                        m.mdot_transfer);
        } else {
            ImGui::Text("[%06lld] t=%.6e",
                        static_cast<long long>(app.snapshot.iteration),
                        app.snapshot.time);
        }
    }
}

// ── Keyboard shortcuts ───────────────────────────────────────────────────

static void handle_keyboard(App& app) {
    auto& io = ImGui::GetIO();

    // Cmd+S / Ctrl+S: apply config (works even when editing text fields)
#ifdef __APPLE__
    bool mod = io.KeySuper;
#else
    bool mod = io.KeyCtrl;
#endif
    if (mod && ImGui::IsKeyPressed(ImGuiKey_S)) {
        apply_config(app);
        return;
    }

    // Other shortcuts only when not editing text
    if (io.WantCaptureKeyboard) return;

    bool running = (app.snapshot.mode == fledge::DriverMode::Running);
    bool has_state = app.snapshot.has_state;

    if (ImGui::IsKeyPressed(ImGuiKey_P)) {
        if (has_state && app.handle)
            app.handle->send(running ? fledge::Command{fledge::CmdPause{}}
                                     : fledge::Command{fledge::CmdRun{}});
    }
    if (ImGui::IsKeyPressed(ImGuiKey_S)) {
        if (has_state && !running && app.handle)
            app.handle->send(fledge::CmdStep{});
    }
    if (ImGui::IsKeyPressed(ImGuiKey_N)) {
        if (!has_state)
            apply_config(app);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_D)) {
        if (has_state && !running && app.handle)
            app.handle->send(fledge::CmdDestroyState{});
    }
    if (ImGui::IsKeyPressed(ImGuiKey_C)) {
        if (has_state && !running && app.handle)
            app.handle->send(fledge::CmdCheckpoint{});
    }
    if (ImGui::IsKeyPressed(ImGuiKey_L)) {
        app.show_log = !app.show_log;
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

int main(int, char**) {
    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) return 1;

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    auto* window = SDL_CreateWindow(
        "Fledge", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) { SDL_Quit(); return 1; }

    auto gl_ctx = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_ctx);
    SDL_GL_SetSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplSDL2_InitForOpenGL(window, gl_ctx);
    ImGui_ImplOpenGL3_Init("#version 150");

    App app;
    recreate_driver(app);

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE)
                running = false;
        }

        if (app.handle)
            app.snapshot = app.handle->read_snapshot();
        drain_events(app);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        auto& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("##main", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        float footer_h = 30.0f;
        float content_h = io.DisplaySize.y - footer_h;

        // Left panel
        ImGui::BeginChild("##left", ImVec2(app.left_panel_width, content_h),
                          ImGuiChildFlags_Border);
        render_config_panel(app);
        ImGui::EndChild();

        // Splitter
        ImGui::SameLine();
        ImGui::InvisibleButton("##splitter", ImVec2(6.0f, content_h));
        if (ImGui::IsItemActive())
            app.left_panel_width += io.MouseDelta.x;
        app.left_panel_width = std::clamp(app.left_panel_width, 150.0f, 800.0f);
        if (ImGui::IsItemHovered() || ImGui::IsItemActive())
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

        // Right panel
        ImGui::SameLine();
        ImGui::BeginChild("##right", ImVec2(0, content_h));
        if (app.show_log) {
            render_log(app);
        } else if (app.snapshot.mt.active) {
            if (ImGui::BeginTabBar("##mt_tabs")) {
                if (ImGui::BeginTabItem("Binary")) {
                    render_mt_binary_view(app);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Evolution")) {
                    render_mt_evolution(app);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Particles")) {
                    render_plot(app);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            render_plot(app);
        }
        ImGui::EndChild();

        // Footer
        render_footer(app);

        handle_keyboard(app);

        ImGui::End();

        ImGui::Render();
        int w, h;
        SDL_GL_GetDrawableSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.06f, 0.06f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    app.handle.reset();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
