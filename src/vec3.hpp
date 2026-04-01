#pragma once

#include <cmath>

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    constexpr auto mag() const -> double {
        return std::sqrt(x * x + y * y);
    }
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    constexpr Vec3() = default;
    constexpr Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    // Arithmetic operators
    constexpr auto operator+(Vec3 v) const -> Vec3 {
        return {x + v.x, y + v.y, z + v.z};
    }

    constexpr auto operator-(Vec3 v) const -> Vec3 {
        return {x - v.x, y - v.y, z - v.z};
    }

    constexpr auto operator*(double c) const -> Vec3 {
        return {x * c, y * c, z * c};
    }

    constexpr auto operator/(double c) const -> Vec3 {
        return {x / c, y / c, z / c};
    }

    constexpr auto operator+=(Vec3 v) -> Vec3& {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    // Methods
    constexpr auto dot(Vec3 v) const -> double {
        return x * v.x + y * v.y + z * v.z;
    }

    constexpr auto mag() const -> double {
        return std::sqrt(dot(*this));
    }

    constexpr auto normalize() const -> Vec3 {
        return *this / mag();
    }

    constexpr auto cross(Vec3 v) const -> Vec3 {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x,
        };
    }

    // Static helpers
    static constexpr auto zero() -> Vec3 { return {0.0, 0.0, 0.0}; }
    static constexpr auto xhat() -> Vec3 { return {1.0, 0.0, 0.0}; }
    static constexpr auto yhat() -> Vec3 { return {0.0, 1.0, 0.0}; }
    static constexpr auto zhat() -> Vec3 { return {0.0, 0.0, 1.0}; }
};

// double * Vec3
constexpr auto operator*(double c, Vec3 v) -> Vec3 {
    return {c * v.x, c * v.y, c * v.z};
}
