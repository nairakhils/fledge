#pragma once

#include <string>
#include <variant>

namespace fledge {

// ── Commands (GUI → driver) ──────────────────────────────────────────────

struct CmdRun {};
struct CmdPause {};
struct CmdStep {};
struct CmdCreateState {};
struct CmdDestroyState {};
struct CmdCheckpoint {};
struct CmdQuit {};

using Command = std::variant<CmdRun, CmdPause, CmdStep, CmdCreateState,
                             CmdDestroyState, CmdCheckpoint, CmdQuit>;

// ── Events (driver → GUI) ────────────────────────────────────────────────

struct EvSimulationDone {};
struct EvStateCreated {};
struct EvStateDestroyed {};
struct EvCheckpointWritten { std::string path; };
struct EvError { std::string message; };
struct EvFinished {};

using Event = std::variant<EvSimulationDone, EvStateCreated, EvStateDestroyed,
                           EvCheckpointWritten, EvError, EvFinished>;

// ── Driver mode ──────────────────────────────────────────────────────────

enum class DriverMode { Idle, Running };

} // namespace fledge
