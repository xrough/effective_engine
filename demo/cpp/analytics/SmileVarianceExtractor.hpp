#pragma once
#include <memory>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

// ============================================================
// File: SmileVarianceExtractor.hpp  (demo/cpp/analytics/)
// Role: subscribes to SmileSnapshotEvent and exposes the latest
//       VIX variance swap rate + SSVI parameters to signals.
//
// Signals call vix_variance() / ssvi_*() each tick.
// Has_vix() / has_ssvi() allow clean fallback to rough vol expansion.
// ============================================================

namespace omm::demo {

class SmileVarianceExtractor {
public:
    explicit SmileVarianceExtractor(std::shared_ptr<events::EventBus> bus)
        : bus_(std::move(bus)) {}

    void register_handlers() {
        bus_->subscribe<events::SmileSnapshotEvent>(
            [this](const events::SmileSnapshotEvent& s) { last_ = s; }
        );
    }

    double vix_variance()  const { return last_.vix_varswap; }
    double ssvi_rho()      const { return last_.ssvi_rho; }
    double ssvi_phi()      const { return last_.ssvi_phi; }
    double ssvi_theta()    const { return last_.ssvi_theta; }
    double skew_25d()      const { return last_.rr25_iv; }
    double curvature_25d() const { return last_.bf25_iv; }
    bool   has_vix()       const { return last_.has_vix; }
    bool   has_ssvi()      const { return last_.has_ssvi; }

private:
    std::shared_ptr<events::EventBus> bus_;
    events::SmileSnapshotEvent        last_ = {};
};

} // namespace omm::demo
