#pragma once

#include "channel.hpp"
#include "command.hpp"
#include "driver.hpp"
#include "snapshot.hpp"
#include "watch.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <variant>

namespace fledge {

// Handle returned by spawn(). Provides the GUI-side interface to the worker.
// Non-copyable; closing the command channel and joining the thread on destruction.
class DriverHandle {
public:
    DriverHandle(std::shared_ptr<Channel<Command>> cmd_tx,
                 std::shared_ptr<Channel<Event>> event_rx,
                 Watch<RuntimeSnapshot> snapshot,
                 std::thread thread)
        : cmd_tx_(std::move(cmd_tx))
        , event_rx_(std::move(event_rx))
        , snapshot_(std::move(snapshot))
        , thread_(std::move(thread)) {}

    ~DriverHandle() {
        if (cmd_tx_) {
            cmd_tx_->send(CmdQuit{});
            cmd_tx_->close();
        }
        if (thread_.joinable()) thread_.join();
    }

    DriverHandle(const DriverHandle&) = delete;
    auto operator=(const DriverHandle&) -> DriverHandle& = delete;
    DriverHandle(DriverHandle&&) = default;
    auto operator=(DriverHandle&&) -> DriverHandle& = default;

    void send(Command cmd) {
        cmd_tx_->send(std::move(cmd));
    }

    auto try_recv() -> std::optional<Event> {
        return event_rx_->try_recv();
    }

    auto read_snapshot() const -> RuntimeSnapshot {
        return snapshot_.read();
    }

private:
    std::shared_ptr<Channel<Command>> cmd_tx_;
    std::shared_ptr<Channel<Event>> event_rx_;
    Watch<RuntimeSnapshot> snapshot_;
    std::thread thread_;
};

// Spawn a background worker thread running the interactive driver loop.
// Returns a DriverHandle for the GUI to communicate with.
inline auto spawn(InteractiveDriver driver) -> DriverHandle {
    auto cmd_ch = std::make_shared<Channel<Command>>();
    auto evt_ch = std::make_shared<Channel<Event>>();
    Watch<RuntimeSnapshot> snapshot;

    auto worker = std::thread([d = std::move(driver),
                               cmd_rx = cmd_ch,
                               evt_tx = evt_ch,
                               snap = snapshot]() mutable {
        using clock = std::chrono::steady_clock;
        constexpr auto batch_budget = std::chrono::milliseconds(30);

        auto emit = [&](std::vector<Event> events) {
            for (auto& e : events) {
                evt_tx->send(std::move(e));
            }
        };

        // Drain all pending commands. Returns true if Quit was received.
        auto drain_commands = [&]() -> bool {
            while (auto cmd = cmd_rx->try_recv()) {
                auto events = d.accept(*cmd);
                emit(std::move(events));
                if (std::holds_alternative<CmdQuit>(*cmd)) return true;
            }
            return false;
        };

        for (;;) {
            if (d.is_running()) {
                if (drain_commands()) break;

                // Batch-step for up to 30ms
                auto deadline = clock::now() + batch_budget;
                while (d.is_running() && clock::now() < deadline) {
                    auto events = d.accept(CmdRun{});
                    emit(std::move(events));
                }
                d.write_snapshot(snap);

            } else {
                // Idle — block until a command arrives
                auto cmd = cmd_rx->recv();
                if (!cmd) break; // channel closed
                auto events = d.accept(*cmd);
                emit(std::move(events));
                if (std::holds_alternative<CmdQuit>(*cmd)) break;
                d.write_snapshot(snap);
            }
        }
    });

    return DriverHandle(std::move(cmd_ch), std::move(evt_ch),
                        std::move(snapshot), std::move(worker));
}

} // namespace fledge
