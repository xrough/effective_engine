#include "CalibrationEngine.hpp"
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================
// CalibrationEngine realization
// Pipeline: 1. observe() collects market/model price pairs; 2. solve() uses golden section search to find the parameter minimizing the loss function (e.g., MSE).
// ============================================================

namespace omm::domain {

void CalibrationEngine::observe(double market_price, double model_price) {
    observations_.push_back(Observation{market_price, model_price}); // append new observation
}

double CalibrationEngine::solve(
    double lo,
    double hi,
    std::function<double(double)> loss_fn,
    double tol) const {

    if (observations_.empty()) {
        std::cout << "[Calibration] Warning: No observations available, returning initial parameter midpoint\n";
        return (lo + hi) / 2.0;
    }

    std::cout << "[Calibration] Starting golden section search, parameter range ["
              << std::fixed << std::setprecision(4) << lo
              << ", " << hi << "]，number of observations: " << observations_.size() << "\n";

    // golden ratio
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;

    // initial probe points
    double c = hi - phi * (hi - lo); 
    double d = lo + phi * (hi - lo);

    int iterations = 0;

    while ((hi - lo) > tol) {
        ++iterations;
        double fc = loss_fn(c); 
        double fd = loss_fn(d); 

        if (fc < fd) {
            hi = d;
            d  = c;
            c  = hi - phi * (hi - lo);
        } else {
            lo = c;
            c  = d;
            d  = lo + phi * (hi - lo);
        }
    }

    double best_param = (lo + hi) / 2.0;

    std::cout << "[Calibration] Search completed, iterations: " << iterations
              << "  Best parameter: " << std::fixed << std::setprecision(6)
              << best_param
              << "  Final loss (MSE): " << std::setprecision(8) << loss_fn(best_param)
              << "\n";

    return best_param;
}

double CalibrationEngine::mse() const {
    if (observations_.empty()) return 0.0;

    // MSE = mean[(model_price - market_price)²]
    double sum_sq = 0.0;
    for (const auto& obs : observations_) {
        double err = obs.model_price - obs.market_price;
        sum_sq += err * err;
    }
    return sum_sq / static_cast<double>(observations_.size());
}

int CalibrationEngine::observation_count() const {
    return static_cast<int>(observations_.size());
}

} // namespace omm::domain
