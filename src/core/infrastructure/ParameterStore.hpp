#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"

// ============================================================
// File: ParameterStore.hpp
// Role: Parameter repository — subscribes to ParamUpdateEvent,
//       persists model parameters, and provides a versioned
//       query interface (latest params by model ID).
//
// Position in the calibration feedback loop:
//   BacktestCalibrationApp → ParamUpdateEvent → ParameterStore
//                                                     ↓
//                                   live engine queries get_params(model_id)
//
// Interface:
//   subscribe_handlers()        — register ParamUpdateEvent handler on EventBus
//   get_params(model_id) → map  — retrieve latest params for a given model
//   print_all()                 — print all stored params (used for end-of-sim logging)
// ============================================================

namespace omm::infrastructure {

// VersionedParams — a parameter snapshot with a timestamp (supports history)
struct VersionedParams {
    std::unordered_map<std::string, double> params;     // key-value parameter map
    events::Timestamp                       updated_at; // when this version was stored
};

class ParameterStore {
public:
    explicit ParameterStore(std::shared_ptr<events::EventBus> bus);

    // subscribe_handlers() — subscribe to ParamUpdateEvent on the bus
    void subscribe_handlers();

    // get_params() — returns the latest parameters for model_id.
    //   Returns an empty map if the model has not been registered yet.
    std::unordered_map<std::string, double> get_params(
        const std::string& model_id
    ) const;

    // print_all() — print the latest params for all registered models
    void print_all() const;

private:
    // on_param_update() — handler for ParamUpdateEvent
    //   Appends a new version to the history; all versions are retained.
    void on_param_update(const events::ParamUpdateEvent& event);

    std::shared_ptr<events::EventBus> bus_;

    // history: model_id → list of versioned snapshots (chronological order)
    std::unordered_map<std::string, std::vector<VersionedParams>> history_;
};

} // namespace omm::infrastructure
