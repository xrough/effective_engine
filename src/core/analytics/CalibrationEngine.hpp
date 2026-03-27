#pragma once
#include <vector>
#include <functional>
#include <utility>

// ============================================================
// File: CalibrationEngine.hpp
// Role: Calibration engine for adjusting model parameters based on market observations.
//
// Calibration Objective (from Risk_Calibration.md §2.6):
//   Minimize Mean Squared Error (MSE) loss function:      
//     L(θ) = mean[(price_model(θ) - price_market)²]
//
// Golden-Section Search
//
// Pipeline:
//   1. observe(market_price, model_price)  - tick-frequent
//   2. solve(vol_lo, vol_hi, loss_fn)      — global optimization
// ============================================================

namespace omm::domain {

// Observation — tick-level price pair: (market_price, model_price)
struct Observation {
    double market_price; 
    double model_price; 
};

class CalibrationEngine {
public:
    CalibrationEngine() = default;
    
    // In each tick, observe the market price and model price to accumulate data for calibration.
    //     engine.observe(market_engine.price(...).theo,
    //                    model_engine.price(...).theo)
    void observe(double market_price, double model_price);
    double solve(
        double lo,
        double hi,
        std::function<double(double)> loss_fn, // loss function
        double tol = 1e-6
    ) const;

    double mse() const;

    int observation_count() const;

private:
    std::vector<Observation> observations_; // historical price observations
};

} // namespace omm::domain
