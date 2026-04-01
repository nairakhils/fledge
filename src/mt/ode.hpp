#pragma once

#include <cstddef>
#include <functional>
#include <vector>

// ── Simple ODE integrator for secular binary evolution ─────────────────────
// State vector of doubles advanced by Euler or RK2 (midpoint).
// Intended for the mass-transfer secular system:
//   y = {M_donor, M_accretor, separation, donor_radius, phase, J_orb}
// with a user-supplied RHS computing derivatives.

struct SecularODE {
    std::vector<double> y;

    using RHS = std::function<std::vector<double>(double t,
                                                   const std::vector<double>& y)>;

    // Forward Euler: y_{n+1} = y_n + dt * f(t, y_n)
    void step_euler(double t, double dt, RHS rhs)
    {
        auto dydt = rhs(t, y);
        for (size_t i = 0; i < y.size(); ++i) {
            y[i] += dt * dydt[i];
        }
    }

    // RK2 midpoint: k1 = f(t, y_n)
    //               y_mid = y_n + 0.5*dt*k1
    //               y_{n+1} = y_n + dt * f(t + 0.5*dt, y_mid)
    void step_rk2(double t, double dt, RHS rhs)
    {
        auto k1 = rhs(t, y);

        std::vector<double> y_mid(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            y_mid[i] = y[i] + 0.5 * dt * k1[i];
        }

        auto k2 = rhs(t + 0.5 * dt, y_mid);
        for (size_t i = 0; i < y.size(); ++i) {
            y[i] += dt * k2[i];
        }
    }
};
