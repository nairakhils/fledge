#pragma once

#include <algorithm>
#include <cmath>
#include <print>

// ── Adaptive subcycling for secular binary evolution ───────────────────────
// Divides an outer timestep dt_outer into N equal substeps such that no
// single substep changes any tracked quantity by more than
// max_fractional_change. The callable step_fn(dt_sub) advances the secular
// state by dt_sub and returns the instantaneous "rate scale": the largest
// |dX/dt| / |X| among the secular variables. subcycle uses that rate scale
// (evaluated once before the loop) to choose N.

template <typename F>
void subcycle(double dt_outer, double max_fractional_change, F&& step_fn)
{
    // Probe the rate scale with a zero-width step (no state mutation).
    double rate_scale = step_fn(0.0);

    // Number of substeps: ceil(rate_scale * dt_outer / max_fractional_change),
    // clamped to [1, 10000].
    int n_sub = 1;
    if (rate_scale > 0.0 && max_fractional_change > 0.0) {
        double n_raw = std::ceil(rate_scale * dt_outer / max_fractional_change);
        n_sub = std::clamp(static_cast<int>(n_raw), 1, 10000);
        if (n_raw > 10000.0) {
            std::println("subcycle: clamped to 10000 substeps "
                         "(requested {:.0f}, rate_scale={:.4e}, dt={:.4e})",
                         n_raw, rate_scale, dt_outer);
        }
    }

    double dt_sub = dt_outer / n_sub;
    for (int i = 0; i < n_sub; ++i) {
        step_fn(dt_sub);
    }
}
