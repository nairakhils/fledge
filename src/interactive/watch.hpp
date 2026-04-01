#pragma once

#include <memory>
#include <mutex>
#include <utility>

namespace fledge {

// Thread-safe single-value channel (like Arc<Mutex<T>> in Rust).
// Copyable — all copies share the same underlying value.
template <typename T>
class Watch {
public:
    Watch() : inner_(std::make_shared<Inner>()) {}
    explicit Watch(T initial)
        : inner_(std::make_shared<Inner>(std::move(initial))) {}

    void write(T value) {
        std::lock_guard lock(inner_->mtx);
        inner_->value = std::move(value);
    }

    auto read() const -> T {
        std::lock_guard lock(inner_->mtx);
        return inner_->value;
    }

private:
    struct Inner {
        mutable std::mutex mtx;
        T value{};

        Inner() = default;
        explicit Inner(T v) : value(std::move(v)) {}
    };

    std::shared_ptr<Inner> inner_;
};

} // namespace fledge
