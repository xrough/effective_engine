#pragma once
#include "../domain/Instrument.hpp"

// ============================================================
// File：PricingEngine.hpp
// Role: Realize IPricingEngine strategy interface
//
//  SimplePricingEngine is a naive implementation for MVP demonstration, with intrinsic value only.
//
//   To implement new models: 1) create new class inheriting IPricingEngine; 2) implement price() method; 3) inject new strategy in main.cpp.
// ============================================================

namespace omm::domain {

// ============================================================
// pricing output struct
// ============================================================
struct PriceResult {
    double theo;   // theoretical price
    double delta;  
};

// ============================================================
// baseline strategy interface for pricing engines
// ============================================================

class IPricingEngine {
public:
    virtual ~IPricingEngine() = default; // deconstructor
    // price() — core pricing method
    virtual PriceResult price(
        const Option& option,
        double        underlying_price
    ) const = 0;
};

// ============================================================
// SimplePricingEngine — simplified pricing engine (MVP specific strategy)
//
// Pricing logic (intrinsic value only, no time value)
// ============================================================
class SimplePricingEngine final : public IPricingEngine {
public:
    PriceResult price(
        const Option& option,
        double        underlying_price
    ) const override;
};

// ============================================================
// BlackScholesPricingEngine
// ============================================================
class BlackScholesPricingEngine final : public IPricingEngine {
public:
    explicit BlackScholesPricingEngine(double vol = 0.20, double r = 0.05);

    void set_vol(double vol);

    double get_vol() const;

    PriceResult price(
        const Option& option,
        double        underlying_price
    ) const override;

private:
    double vol_; 
    double r_; 
};

} // namespace omm::domain
