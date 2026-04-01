#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace fledge {

// Thread-safe MPSC queue with blocking and non-blocking receive.
template <typename T>
class Channel {
public:
    void send(T value) {
        {
            std::lock_guard lock(mtx_);
            if (closed_) return;
            queue_.push_back(std::move(value));
        }
        cv_.notify_one();
    }

    // Blocking receive. Returns nullopt when closed and empty.
    auto recv() -> std::optional<T> {
        std::unique_lock lock(mtx_);
        cv_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return std::nullopt;
        T val = std::move(queue_.front());
        queue_.pop_front();
        return val;
    }

    // Non-blocking receive.
    auto try_recv() -> std::optional<T> {
        std::lock_guard lock(mtx_);
        if (queue_.empty()) return std::nullopt;
        T val = std::move(queue_.front());
        queue_.pop_front();
        return val;
    }

    void close() {
        {
            std::lock_guard lock(mtx_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    bool closed_ = false;
};

} // namespace fledge
