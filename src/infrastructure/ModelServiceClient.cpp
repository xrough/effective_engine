/**
 * ModelServiceClient.cpp — C++ gRPC client implementation.
 *
 * Heavy gRPC and protobuf headers are included only here to keep
 * build times fast for files that only need ModelServiceClient.hpp.
 *
 * Build prerequisite:
 *   cmake -DBUILD_GRPC_CLIENT=ON ..
 *   (requires grpc and protobuf, e.g. `brew install grpc protobuf`)
 *   (requires proto-cpp stub generation: `make proto-cpp` from MVP root)
 */

#include "ModelServiceClient.hpp"

#include <grpcpp/grpcpp.h>

// Generated protobuf/gRPC stubs (produced by `make proto-cpp`)
#include "../../generated/cpp/rough_pricing.pb.h"
#include "../../generated/cpp/rough_pricing.grpc.pb.h"

namespace omm {
namespace infrastructure {

// ── PIMPL body ──────────────────────────────────────────────────────────────

struct ModelServiceClient::Impl {
    std::unique_ptr<roughvol::RoughPricingService::Stub> stub;
};

// ── Constructor / destructor ────────────────────────────────────────────────

ModelServiceClient::ModelServiceClient(const std::string& server_address)
    : impl_(std::make_unique<Impl>())
{
    auto channel = grpc::CreateChannel(
        server_address, grpc::InsecureChannelCredentials()
    );
    impl_->stub = roughvol::RoughPricingService::NewStub(channel);
}

ModelServiceClient::~ModelServiceClient() = default;

// ── Helper: check gRPC status ───────────────────────────────────────────────

static void check_status(const grpc::Status& status, const char* rpc_name) {
    if (!status.ok()) {
        throw std::runtime_error(
            std::string(rpc_name) + " RPC failed: " + status.error_message()
        );
    }
}

// ── Helper: build CalibrateRequest ─────────────────────────────────────────

static roughvol::CalibrateRequest make_calibrate_request(
    roughvol::ModelType model_type,
    double spot,
    const std::vector<OptionQuote>& quotes,
    double rate, double div,
    const std::vector<double>& x0,
    int n_paths, int n_steps, int seed
) {
    roughvol::CalibrateRequest req;
    req.set_model_type(model_type);
    req.set_spot(spot);
    req.set_rate(rate);
    req.set_div(div);
    req.set_n_paths(n_paths);
    req.set_n_steps(n_steps);
    req.set_seed(seed);
    req.set_antithetic(true);
    for (double v : x0) req.add_x0(v);
    for (const auto& q : quotes) {
        auto* pq = req.add_quotes();
        pq->set_strike(q.strike);
        pq->set_maturity_years(q.maturity_years);
        pq->set_is_call(q.is_call);
        pq->set_market_price(q.market_price);
    }
    return req;
}

// ── Helper: unpack CalibrateResponse ───────────────────────────────────────

static CalibResult unpack_calib(const roughvol::CalibrateResponse& resp) {
    CalibResult result;
    result.model_name = resp.model_name();
    result.mse        = resp.mse();
    result.elapsed_s  = resp.elapsed_s();
    for (const auto& kv : resp.params()) {
        result.params[kv.first] = kv.second;
    }
    for (double iv : resp.per_option_ivols()) {
        result.per_option_ivols.push_back(iv);
    }
    return result;
}

// ── BSPrice ─────────────────────────────────────────────────────────────────

double ModelServiceClient::bs_price(
    double spot, double strike, double maturity,
    double rate, double div, double vol, bool is_call
) {
    roughvol::BSPriceRequest req;
    req.set_spot(spot);
    req.set_strike(strike);
    req.set_maturity(maturity);
    req.set_rate(rate);
    req.set_div(div);
    req.set_vol(vol);
    req.set_is_call(is_call);

    roughvol::BSPriceResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->BSPrice(&ctx, req, &resp), "BSPrice");
    return resp.price();
}

// ── ImpliedVol ──────────────────────────────────────────────────────────────

double ModelServiceClient::implied_vol(
    double price, double spot, double strike,
    double maturity, double rate, double div, bool is_call
) {
    roughvol::ImpliedVolRequest req;
    req.set_price(price);
    req.set_spot(spot);
    req.set_strike(strike);
    req.set_maturity(maturity);
    req.set_rate(rate);
    req.set_div(div);
    req.set_is_call(is_call);

    roughvol::ImpliedVolResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->ImpliedVol(&ctx, req, &resp), "ImpliedVol");
    return resp.vol();
}

// ── MCPrice (GBM vanilla) ───────────────────────────────────────────────────

MCPriceResult ModelServiceClient::mc_price_vanilla_gbm(
    double sigma,
    double strike, double maturity, bool is_call,
    double spot, double rate, double div_yield,
    int n_paths, int n_steps, int seed, bool antithetic
) {
    roughvol::MCPriceRequest req;
    req.mutable_gbm()->set_sigma(sigma);
    req.mutable_vanilla()->set_strike(strike);
    req.mutable_vanilla()->set_maturity(maturity);
    req.mutable_vanilla()->set_is_call(is_call);
    req.mutable_market()->set_spot(spot);
    req.mutable_market()->set_rate(rate);
    req.mutable_market()->set_div_yield(div_yield);
    req.set_n_paths(n_paths);
    req.set_n_steps(n_steps);
    req.set_seed(seed);
    req.set_antithetic(antithetic);

    roughvol::MCPriceResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->MCPrice(&ctx, req, &resp), "MCPrice(GBM)");
    return {resp.price(), resp.stderr(), resp.ci95_lo(), resp.ci95_hi(),
            resp.n_paths(), resp.n_steps()};
}

// ── MCPrice (Heston vanilla) ────────────────────────────────────────────────

MCPriceResult ModelServiceClient::mc_price_vanilla_heston(
    double kappa, double theta, double xi, double rho, double v0,
    double strike, double maturity, bool is_call,
    double spot, double rate, double div_yield,
    int n_paths, int n_steps, int seed, bool antithetic
) {
    roughvol::MCPriceRequest req;
    req.mutable_heston()->set_kappa(kappa);
    req.mutable_heston()->set_theta(theta);
    req.mutable_heston()->set_xi(xi);
    req.mutable_heston()->set_rho(rho);
    req.mutable_heston()->set_v0(v0);
    req.mutable_vanilla()->set_strike(strike);
    req.mutable_vanilla()->set_maturity(maturity);
    req.mutable_vanilla()->set_is_call(is_call);
    req.mutable_market()->set_spot(spot);
    req.mutable_market()->set_rate(rate);
    req.mutable_market()->set_div_yield(div_yield);
    req.set_n_paths(n_paths);
    req.set_n_steps(n_steps);
    req.set_seed(seed);
    req.set_antithetic(antithetic);

    roughvol::MCPriceResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->MCPrice(&ctx, req, &resp), "MCPrice(Heston)");
    return {resp.price(), resp.stderr(), resp.ci95_lo(), resp.ci95_hi(),
            resp.n_paths(), resp.n_steps()};
}

// ── Calibrate BS ────────────────────────────────────────────────────────────

CalibResult ModelServiceClient::calibrate_bs(
    double spot,
    const std::vector<OptionQuote>& quotes,
    double rate, double div
) {
    auto req = make_calibrate_request(
        roughvol::ModelType::BS, spot, quotes, rate, div, {}, 0, 0, 0
    );
    roughvol::CalibrateResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->Calibrate(&ctx, req, &resp), "Calibrate(BS)");
    return unpack_calib(resp);
}

// ── Calibrate GBM-MC ────────────────────────────────────────────────────────

CalibResult ModelServiceClient::calibrate_gbm(
    double spot,
    const std::vector<OptionQuote>& quotes,
    double rate, double div,
    double x0_sigma,
    int n_paths, int n_steps, int seed
) {
    auto req = make_calibrate_request(
        roughvol::ModelType::GBM_MC, spot, quotes, rate, div,
        {x0_sigma}, n_paths, n_steps, seed
    );
    roughvol::CalibrateResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->Calibrate(&ctx, req, &resp), "Calibrate(GBM_MC)");
    return unpack_calib(resp);
}

// ── Calibrate Heston ────────────────────────────────────────────────────────

CalibResult ModelServiceClient::calibrate_heston(
    double spot,
    const std::vector<OptionQuote>& quotes,
    double rate, double div,
    double x0_sigma,
    int n_paths, int n_steps, int seed
) {
    auto req = make_calibrate_request(
        roughvol::ModelType::HESTON, spot, quotes, rate, div,
        {x0_sigma}, n_paths, n_steps, seed
    );
    roughvol::CalibrateResponse resp;
    grpc::ClientContext ctx;
    check_status(impl_->stub->Calibrate(&ctx, req, &resp), "Calibrate(Heston)");
    return unpack_calib(resp);
}

} // namespace infrastructure
} // namespace omm
