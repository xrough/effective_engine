# Demo — Buyer Alpha Pipeline + Deep BSDE Hedging

Two standalone demos built on top of the engine in `src/`. Each has its own `CMakeLists.txt` target and can be run independently.

---

## Demo 1 — Variance Alpha Pipeline (`alpha_runner`)

End-to-end buyer-side pipeline: synthetic option feed → implied variance extraction → rolling z-score signal → straddle entry/exit → delta hedge → PnL attribution.

### Quick start

```bash
cd demo
mkdir -p build && cd build && cmake .. && make alpha_runner
cd ..
./build/alpha_runner
```

### What runs

```
MarketDataEvent (CSV or fallback 20-tick data)
  → SyntheticOptionFeed    — prices ATM straddle at each tick, publishes OptionMidQuoteEvent
  → ImpliedVarianceExtractor — BS IV bisection → σ²_atm, σ²T
  → VarianceAlphaSignal    — rolling z-score: (σ²_atm − xi0·T − mean) / std
  → StrategyController     — Flat → Live → Cooldown state machine
                             enters on |zscore| > 1.5, exits on |zscore| < 0.5
  → SimpleExecSim          — converts OrderSubmittedEvent → FillEvent at market price
  → DeltaHedger            — neutralises delta after each fill
  → AlphaPnLTracker        — per-instrument MTM + delta hedge PnL + transaction cost
```

### Output (example)

```
[StrategyController] Flat → Live  交易类型=LongFrontVariance  zscore=-1.80
[Delta对冲] 组合 Delta = 0.000  (阈值 ±0.300)

Strategy PnL Attribution
  Option MTM (unrealized):   $...
  Delta hedge PnL (realized): $0.00
  Transaction cost:          -$60.00
  Total PnL:                  $...
```

### Key files

| File | Role |
|---|---|
| [cpp/alpha_main.cpp](cpp/alpha_main.cpp) | Composition root — wires all components |
| [cpp/VarianceAlphaSignal.hpp](cpp/VarianceAlphaSignal.hpp) | Concrete `IAlphaSignal`: rolling z-score vs rough forecast |
| [cpp/StrategyController.hpp](cpp/StrategyController.hpp) | Concrete `IEntryPolicy`: Flat/Live/Cooldown FSM |
| [cpp/SyntheticOptionFeed.hpp](cpp/SyntheticOptionFeed.hpp) | Simulation adapter: `MarketDataEvent` → `OptionMidQuoteEvent` |
| [cpp/SimpleExecSim.hpp](cpp/SimpleExecSim.hpp) | Simulation adapter: `OrderSubmittedEvent` → `FillEvent` |
| [cpp/AlphaPnLTracker.hpp](cpp/AlphaPnLTracker.hpp) | PnL attribution (event-semantic only, no feed assumptions) |

### Design notes

- **Adapter isolation.** `SyntheticOptionFeed` and `SimpleExecSim` are simulation shims. `BuyerModule`, `DeltaHedger`, and `AlphaPnLTracker` depend only on stable event types — replacing either adapter with a real feed or OMS requires no engine changes.
- **Injection pattern.** `BuyerModule::install()` accepts injected `IAlphaSignal` and `IEntryPolicy` implementations. The demo creates `VarianceAlphaSignal` and `StrategyController` and passes them in — a different strategy just swaps the impls.
- **Entry once per signal.** `StrategyController` submits the straddle order exactly once on `Flat → Live` transition, not on every tick.
- **Per-instrument PnL.** `AlphaPnLTracker` maintains separate position and cost basis per `instrument_id`, so call and put legs are tracked independently.

---

## Demo 2 — Deep BSDE Neural Hedging (`demo_runner`)

Generates lifted rough Heston paths in C++, trains a shared-weight MLP offline in Python to solve the BSDE hedging problem, exports to ONNX, and benchmarks against analytic and FD BS delta.

### Quick start

```bash
# 1. Generate paths
cd demo && mkdir -p build && cd build && cmake .. && make demo_runner
cd .. && ./build/demo_runner     # writes artifacts/

# 2. Validate on Black-Scholes (Gate 1)
python python/trainer.py --config python/configs/bs_validation.yaml

# 3. Train on LRH paths (Gate 2)
python python/trainer.py --config python/configs/lifted_rough_heston.yaml

# 4. Export to ONNX
python python/export.py --checkpoint artifacts/checkpoints/best.pt

# 5. Rebuild with ONNX inference and benchmark
cd build && cmake .. -DBUILD_ONNX_DEMO=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime && make
cd .. && ./build/demo_runner
```

### Results (seed=42, H=0.1, m=4, 2000 OOS paths)

PnL measured in BM space: `PnL = Σ Z_i · dW1_i − payoff`.

| Hedger | PnL std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

Neural BSDE reduces hedging variance ~20% vs analytic BS delta. FD beats analytic because it uses instantaneous V_t rather than fixed σ=√V₀.

### Design notes

- **Z is a BM integrand, not a delta.** Z[0] ≈ N(d1)·σ·S in the BS limit (~12.74). Promoted to hedge signal only after Gate 1 empirical validation.
- **1D hedge by design.** Lifted rough Heston has two BMs (spot dW1, vol dW2). Demo v1 hedges dW1 only; dW2 is intentionally unhedged — the terminal loss floor reflects an incomplete market, not a training failure.
- **No BatchNorm.** Removed because it mixes batch statistics across τ values within each training batch, corrupting Z convergence.
- **Two separate optimizers.** `opt_net` (Adam lr=1e-3, cosine decay) and `opt_Y0` (Adam lr=0.1, constant). Required because Y0's gradient is dominated by the network's ~9k parameters under a shared optimizer.
- **Normalization written once by C++.** `artifacts/normalization.json` is written by the path generator and read verbatim by both the Python trainer and C++ inference — never recomputed.

### Artifacts

| File | Written by | Read by |
|---|---|---|
| `artifacts/training_states.npy` | C++ | Python |
| `artifacts/training_dW1.npy` | C++ | Python |
| `artifacts/training_payoff.npy` | C++ | Python |
| `artifacts/normalization.json` | C++ | Python + C++ |
| `artifacts/neural_bsde.onnx` | Python | C++ |
| `artifacts/checkpoints/best.pt` | Python | Python (export) |
