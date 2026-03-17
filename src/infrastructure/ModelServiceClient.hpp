#pragma once
/**
 * ModelServiceClient — synchronous C++ gRPC client for RoughPricingService.
 *
 * Wraps the generated stub behind a PIMPL boundary so that callers only need
 * to include this header; the heavy grpcpp and protobuf headers are confined
 * to ModelServiceClient.cpp.
 *
 * Usage:
 *   auto client = omm::infrastructure::ModelServiceClient{"localhost:50051"};
 *   double p    = client.bs_price(100, 100, 1.0, 0.05, 0.0, 0.2, true);
 *   auto calib  = client.calibrate_bs(100.0, quotes);
 *
 * All methods are synchronous (blocking).  Do NOT call from the live
 * event-loop tick path — use the C++ CalibrationEngine for per-tick needs.
 * Reserve this client for Phase-2 batch/calibration work.
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace omm {
namespace infrastructure {

// ── Plain data structs (no gRPC headers needed here) ───────────────────────

struct MCPriceResult {
    double price;
    double stderr;
    double ci95_lo;
    double ci95_hi;
    int    n_paths;
    int    n_steps;
};

struct CalibResult {
    std::string                          model_name;
    std::unordered_map<std::string, double> params;
    double                               mse;
    double                               elapsed_s;
    std::vector<double>                  per_option_ivols;  // BS only
};

struct OptionQuote {
    double strike;
    double maturity_years;
    bool   is_call;
    double market_price;
};

// ── Client ──────────────────────────────────────────────────────────────────

class ModelServiceClient {
public:
    /**
     * @param server_address  e.g. "localhost:50051"
     */
    explicit ModelServiceClient(const std::string& server_address = "localhost:50051");
    ~ModelServiceClient();

    // Non-copyable, movable
    ModelServiceClient(const ModelServiceClient&)            = delete;
    ModelServiceClient& operator=(const ModelServiceClient&) = delete;
    ModelServiceClient(ModelServiceClient&&)                 = default;
    ModelServiceClient& operator=(ModelServiceClient&&)      = default;

    // ── Black-Scholes closed-form ──────────────────────────────────────────

    double bs_price(
        double spot, double strike, double maturity,
        double rate, double div, double vol, bool is_call
    );

    double implied_vol(
        double price, double spot, double strike,
        double maturity, double rate, double div, bool is_call
    );

    // ── Monte Carlo pricing ────────────────────────────────────────────────

    MCPriceResult mc_price_vanilla_gbm(
        double sigma,
        double strike, double maturity, bool is_call,
        double spot, double rate, double div_yield,
        int n_paths = 20000, int n_steps = 50,
        int seed = 42, bool antithetic = true
    );

    MCPriceResult mc_price_vanilla_heston(
        double kappa, double theta, double xi, double rho, double v0,
        double strike, double maturity, bool is_call,
        double spot, double rate, double div_yield,
        int n_paths = 20000, int n_steps = 50,
        int seed = 42, bool antithetic = true
    );

    // ── Calibration ────────────────────────────────────────────────────────

    CalibResult calibrate_bs(
        double spot,
        const std::vector<OptionQuote>& quotes,
        double rate = 0.05, double div = 0.0
    );

    CalibResult calibrate_gbm(
        double spot,
        const std::vector<OptionQuote>& quotes,
        double rate = 0.05, double div = 0.0,
        double x0_sigma = 0.20,
        int n_paths = 20000, int n_steps = 50, int seed = 42
    );

    CalibResult calibrate_heston(
        double spot,
        const std::vector<OptionQuote>& quotes,
        double rate = 0.05, double div = 0.0,
        double x0_sigma = 0.20,
        int n_paths = 20000, int n_steps = 50, int seed = 42
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infrastructure
} // namespace omm
