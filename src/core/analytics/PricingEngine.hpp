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
    double theo;          // theoretical price
    double delta;
    double gamma = 0.0;   // N'(d1) / (S σ √T)
    double vega  = 0.0;   // S √T N'(d1)  （每单位σ变动的价值变化）
    double theta = 0.0;   // −(S σ N'(d1))/(2√T) ∓ r·K·e^{−rT}·N(∓d2)
};

// ============================================================
// baseline strategy interface for pricing engines
// ============================================================

class IPricingEngine {
public:
    virtual ~IPricingEngine() = default; // deconstructor
    // price() — core pricing method (uses internal T from system clock + model sigma)
    virtual PriceResult price(
        const Option& option,
        double        underlying_price
    ) const = 0;

    // price_at_iv() — pricing with explicit historical T and market implied vol.
    // Used by DeltaHedger in historical replay so delta is computed with the
    // correct simulated T_sim and the market-observed sigma rather than the
    // rough-vol model forecast.
    virtual PriceResult price_at_iv(
        double S, double K, double T_sim,
        double sigma_market, bool is_call
    ) const = 0;

    // price_with_rough_delta() — minimum-variance delta with Bergomi-Guyon smile correction.
    // Δ_rough = Δ_BS(σ_K) + Vega(σ_K) · (∂σ_K/∂S)
    // where ∂σ_K/∂S = −(ψ(T) + χ(T)·k) / S   [k = log(K/S)]
    // Default: falls back to price_at_iv() (plain BS delta).
    // Override in stochastic-vol engines to activate the smile-slope correction.
    virtual PriceResult price_with_rough_delta(
        double S, double K, double T_sim,
        double sigma_atm, bool is_call
    ) const {
        return price_at_iv(S, K, T_sim, sigma_atm, is_call);
    }
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

    PriceResult price_at_iv(
        double S, double K, double T_sim,
        double sigma_market, bool is_call
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

    PriceResult price_at_iv(
        double S, double K, double T_sim,
        double sigma_market, bool is_call
    ) const override;

private:
    double vol_;
    double r_;
};

} // namespace omm::domain
