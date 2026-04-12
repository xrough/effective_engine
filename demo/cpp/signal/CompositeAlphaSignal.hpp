#pragma once
#include <memory>
#include <vector>
#include <numeric>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "modules/buyer/IAlphaSignal.hpp"

// ============================================================
// File: CompositeAlphaSignal.hpp  (demo/cpp/signal/)
// Role: Weighted multi-signal blender.
//
// Aggregates N IAlphaSignal instances, each with a scalar weight,
// into a single merged SignalSnapshotEvent per tick.
//
// Architecture (inner bus isolation):
//   Sub-signals are constructed with the INNER bus (get via inner_bus()).
//   They publish SignalSnapshotEvents to the inner bus.
//   The outer bus (main pipeline) only ever sees the merged composite event.
//   This prevents the StrategyController from receiving raw sub-signal events
//   which would corrupt bars_in_state_ counters and trigger premature exits.
//
// Merge mechanism:
//   - Outer bus OptionMidQuoteEvent subscriber 1: forward to inner bus (sub-signals process)
//   - Inner bus SignalSnapshotEvent subscriber: collect into pending_ buffer
//   - Outer bus OptionMidQuoteEvent subscriber 2: merge pending_ if N snapshots received
//   Subscriber 1 and 2 are registered sequentially, so 2 always runs after 1.
//   If a sub-signal skips a tick (early-return without publishing), merge is skipped
//   for that tick — graceful degradation during warmup.
//
// Merge rule:
//   zscore_composite = Σ (w_i * z_i) / Σ w_i
//   valid            = ALL contributing sub-signals have valid=true
//   raw_spread       = weighted average of raw_spread_i
//   other fields     = taken from the most-recently-received snapshot (last sub-signal)
//
// Usage:
//   auto composite = std::make_shared<omm::buyer::CompositeAlphaSignal>(outer_bus);
//   auto inner_bus = composite->inner_bus();  // pass to sub-signal constructors
//   composite->add(vrp_signal,  0.50);
//   composite->add(har_signal,  0.30);
//   composite->add(skew_signal, 0.20);
//   // Register AFTER BuyerModule::install() so extractor is on bus first
//   composite->register_handlers();
// ============================================================

namespace omm::buyer {

class CompositeAlphaSignal : public IAlphaSignal {
public:
    explicit CompositeAlphaSignal(std::shared_ptr<events::EventBus> outer_bus)
        : outer_bus_(std::move(outer_bus))
        , inner_bus_(std::make_shared<events::EventBus>()) {}

    // Returns the isolated inner bus — pass this to sub-signal constructors
    // so their SignalSnapshotEvent publications stay off the outer bus.
    std::shared_ptr<events::EventBus> inner_bus() const { return inner_bus_; }

    // Add a sub-signal with a positive weight (unnormalized — normalised internally).
    void add(std::shared_ptr<IAlphaSignal> signal, double weight) {
        entries_.push_back({std::move(signal), weight});
    }

    void register_handlers() override {
        // 1. Sub-signals register to the inner bus (market data forwarded below)
        for (auto& entry : entries_) {
            entry.signal->register_handlers();
        }

        // 2. Collect sub-signal snapshots from inner bus into pending_ buffer
        inner_bus_->subscribe<events::SignalSnapshotEvent>(
            [this](const events::SignalSnapshotEvent& s) { pending_.push_back(s); }
        );

        // 3a. Forward MarketDataEvent: outer → inner bus (for sub-signal ring buffers)
        outer_bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { inner_bus_->publish(e); }
        );

        // 3b. Forward OptionMidQuoteEvent outer → inner (subscriber 1, runs first)
        //     Then merge pending_ (subscriber 2, registered second, runs after sub-signals)
        outer_bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { inner_bus_->publish(e); }
        );
        outer_bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& /*e*/) { try_merge(); }
        );
    }

    // Not used when inner bus is active; kept for interface compliance.
    void on_market_data(const events::MarketDataEvent& event) override {
        for (auto& entry : entries_)
            entry.signal->on_market_data(event);
    }

private:
    struct Entry {
        std::shared_ptr<IAlphaSignal> signal;
        double                        weight;
    };

    void try_merge() {
        // Only merge when all N sub-signals published this tick
        if (pending_.size() != entries_.size()) { pending_.clear(); return; }

        double weight_sum  = 0.0;
        double zscore_sum  = 0.0;
        double spread_sum  = 0.0;
        bool   all_valid   = true;

        for (std::size_t i = 0; i < entries_.size(); ++i) {
            double w   = entries_[i].weight;
            weight_sum += w;
            zscore_sum += w * pending_[i].zscore;
            spread_sum += w * pending_[i].raw_spread;
            if (!pending_[i].valid) all_valid = false;
        }

        // Use the last sub-snapshot as the template for non-blended fields
        const events::SignalSnapshotEvent& ref = pending_[entries_.size() - 1];

        events::SignalSnapshotEvent merged;
        merged.ts                      = ref.ts;
        merged.valid                   = all_valid;
        merged.atm_implied_variance    = ref.atm_implied_variance;
        merged.rough_forecast_variance = ref.rough_forecast_variance;
        merged.raw_spread              = (weight_sum > 1e-15) ? spread_sum  / weight_sum : 0.0;
        merged.zscore                  = (weight_sum > 1e-15) ? zscore_sum  / weight_sum : 0.0;
        merged.calibration_ok          = ref.calibration_ok;
        merged.atm_vega                = ref.atm_vega;
        merged.atm_spot                = ref.atm_spot;

        // Publish merged event to OUTER bus only — StrategyController sees one event/tick
        outer_bus_->publish(merged);
        pending_.clear();
    }

    std::shared_ptr<events::EventBus>          outer_bus_;
    std::shared_ptr<events::EventBus>          inner_bus_;
    std::vector<Entry>                         entries_;
    std::vector<events::SignalSnapshotEvent>   pending_;
};

} // namespace omm::buyer
