cmake_minimum_required(VERSION 3.28)
project(fledge VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)
cmake_policy(SET CMP0169 OLD) # allow FetchContent_Populate for header-only deps

# ── NEST submodule ─────────────────────────────────────────────────────
set(NEST_BACKEND "CPU" CACHE STRING "" FORCE)
set(NEST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/nest)

# ── Batch executable ───────────────────────────────────────────────────
add_executable(fledge src/main.cpp)
target_link_libraries(fledge PRIVATE nest)
target_include_directories(fledge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nest/src
)

# ── GUI dependencies ──────────────────────────────────────────────────

# SDL2 (static only, no tests)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    SDL2
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        release-2.30.12
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(SDL2)

# Dear ImGui (no CMakeLists.txt — fetch sources only)
FetchContent_Declare(
    imgui_src
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(imgui_src)
if(NOT imgui_src_POPULATED)
    FetchContent_Populate(imgui_src)
endif()

# ImPlot (no CMakeLists.txt — fetch sources only)
FetchContent_Declare(
    implot_src
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG        v0.16
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(implot_src)
if(NOT implot_src_POPULATED)
    FetchContent_Populate(implot_src)
endif()

# OpenGL
find_package(OpenGL REQUIRED)

# ImGui static library (core + SDL2/OpenGL3 backends + ImPlot)
add_library(imgui STATIC
    ${imgui_src_SOURCE_DIR}/imgui.cpp
    ${imgui_src_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_src_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_src_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_src_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_src_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_src_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_src_SOURCE_DIR}/implot.cpp
    ${implot_src_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_src_SOURCE_DIR}
    ${imgui_src_SOURCE_DIR}/backends
    ${implot_src_SOURCE_DIR}
)
target_link_libraries(imgui PUBLIC SDL2::SDL2-static OpenGL::GL)
target_compile_options(imgui PRIVATE -w)

# ── GUI executable ─────────────────────────────────────────────────────
add_executable(fledge_gui src/gui/main_gui.cpp)
target_link_libraries(fledge_gui PRIVATE imgui nest)
target_include_directories(fledge_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nest/src
)
