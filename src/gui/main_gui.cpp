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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
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
        if (std::holds_alternative<fledge::EvStateCreated>(*ev))
            app.auto_fit = true;
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
}

// ── Config editor (left panel) ───────────────────────────────────────────

static void render_config_panel(App& app) {
    auto& c = app.config;
    bool has_state = app.snapshot.has_state;

    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputDouble("tstart", &c.tstart, 0, 0, "%.4g");
        ImGui::InputDouble("tfinal", &c.tfinal, 0, 0, "%.4g");
        ImGui::InputDouble("dt", &c.dt, 0, 0, "%.6g");
        ImGui::InputDouble("softening", &c.softening, 0, 0, "%.4g");
    }

    if (ImGui::CollapsingHeader("Central Object", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        ImGui::Text("[%06lld] t=%.6e",
                    static_cast<long long>(app.snapshot.iteration),
                    app.snapshot.time);
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
        if (app.show_log)
            render_log(app);
        else
            render_plot(app);
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
