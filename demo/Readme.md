# Lifted Rough Heston + Deep BSDE Demo

Self-contained pipeline for training and benchmarking a neural BSDE hedger on lifted rough Heston paths. Standalone from the main engine — `demo/` can be built and run independently.

## What it does

1. **C++ path generator** — simulates lifted rough Heston (H=0.1, m=4 Markovian factors), writes training paths and normalization stats to `artifacts/`
2. **Python trainer** — loads paths, trains a shared-weight MLP offline to solve the BSDE hedging problem (driver f=0), exports frozen model to ONNX
3. **C++ inference benchmark** — loads the ONNX artifact, replays out-of-sample paths, compares three hedgers

## Quick start

```bash
# 1. Build path generator
mkdir -p demo/build && cd demo/build
cmake .. && make
./demo_runner          # writes artifacts/

# 2. Validate on Black-Scholes (Gate 1)
cd demo
python python/trainer.py --config python/configs/bs_validation.yaml

# 3. Train on LRH paths (Gate 2)
python python/trainer.py --config python/configs/lifted_rough_heston.yaml

# 4. Export to ONNX
python python/export.py --checkpoint artifacts/checkpoints/best.pt

# 5. Rebuild with inference and benchmark
cd demo/build
cmake .. -DBUILD_ONNX_DEMO=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime && make
./demo_runner
```

## Results (seed=42, H=0.1, m=4, 2000 OOS paths)

PnL measured in BM space: `PnL = Σ Z_i · dW1_i − payoff`.

| Hedger | Std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

Neural BSDE reduces hedging variance by ~20% vs analytic BS delta. FD delta beats analytic BS because it uses instantaneous V_t rather than fixed σ=√V₀. All three converge to the same mean (≈ −Y0 = −9.78, confirming correct PnL accounting).

## Key design notes

- **Z is a BM integrand, not a delta.** Z[0] ≈ N(d1) · σ · S in the BS limit (~12.74). Promoted to "hedge signal" only after Gate 1 empirical validation.
- **1D hedge by design.** Lifted rough Heston has two BMs (spot dW1, vol dW2). Demo v1 hedges dW1 only; dW2 is intentionally unhedged. This explains the high terminal loss floor (~10.7) — it reflects an incomplete market, not a training failure.
- **No BatchNorm.** Removed because it mixes batch statistics across different τ values within each training batch, corrupting Z convergence. Plain Tanh activations used instead.
- **Two separate optimizers.** `opt_net` (Adam lr=1e-3, cosine decay, grad clipped) and `opt_Y0` (Adam lr=0.1, constant, no clipping). Required because Y0's gradient is dominated by the network's ~9k parameters under a shared optimizer.
- **Normalization written once by C++.** `artifacts/normalization.json` is written by the path generator and read verbatim by both the Python trainer and C++ inference — never recomputed.

## Artifacts

| File | Written by | Read by |
|---|---|---|
| `artifacts/training_states.npy` | C++ | Python |
| `artifacts/training_dW1.npy` | C++ | Python |
| `artifacts/training_payoff.npy` | C++ | Python |
| `artifacts/normalization.json` | C++ | Python + C++ |
| `artifacts/neural_bsde.onnx` | Python | C++ |
| `artifacts/checkpoints/best.pt` | Python | Python (export) |
