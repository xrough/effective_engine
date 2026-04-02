Lifted Rough Heston + Deep BSDE Demo Plan

Objective
Build a first demo on top of the existing C++ event-driven option engine in which:
1. the engine simulates lifted rough Heston paths,
2. a deep BSDE model is trained offline,
3. the trained model is embedded back into the C++ engine for low-latency online hedging.

Demo Scope
Keep the first milestone narrow and operationally clean:
- Product: European call
- Model: lifted rough Heston
- Runtime output: price estimate Y0 and one hedge signal
- Preferred first hedge signal: spot delta
- Optional second hedge signal only if the engine already supports a clean volatility hedge execution path
- Inference location: inside the current C++ engine
- Training location: offline, outside the runtime engine

Core Design Principle
The current C++ engine remains the owner of:
- simulation,
- state evolution,
- event handling,
- market replay,
- hedge execution logic,
- latency measurement.

The deep-learning layer is added as:
- an offline trainer for model discovery,
- a frozen online inference component for pricing/hedging decisions.

Target Architecture
1. C++ engine
   - lifted rough Heston simulator
   - deterministic replay mode
   - batched training-path API
   - online inference call on hedge decision events
   - hedge execution and PnL accounting

2. Python training module
   - consumes batched paths from the engine
   - trains a deep BSDE model
   - validates price and hedge quality
   - exports frozen inference artifact

3. C++ inference module
   - loads frozen model artifact via LibTorch or ONNX Runtime
   - evaluates model in-process with low latency
   - returns executable hedge signal to the engine

First Demo Deliverable
A reproducible end-to-end pipeline showing:
- the C++ engine generates lifted rough Heston training paths,
- the deep BSDE model is trained offline,
- the frozen model is loaded back into the engine,
- the engine uses the model online to generate hedge decisions,
- the resulting price and hedge quality are benchmarked against existing engine baselines.

Minimal Runtime Feature Vector
Use a compact fixed-size state vector with aggressive normalization.
Recommended starting features:
- time to maturity tau
- log spot or log-moneyness
- lifted factors U^1, ..., U^m
- instantaneous variance V_t
- optional simple context variables only if already available in the engine

Avoid adding extra runtime context until the first demo is stable.

Minimal Runtime Output Vector
Start with:
- hedge signal for the underlying (spot delta)

Optionally later:
- one additional volatility hedge coordinate

Do not start with a large hedge basis.

Training Formulation
Start with the simplest numerically stable setup:
- frictionless pricing/hedging objective
- European payoff
- BSDE driver f = 0
- small factor count first, e.g. m = 4 or 8
- fixed time grid for training
- shared-weight network with time embedding
- trainable scalar Y0
- state normalization
- gradient clipping
- antithetic sampling if easy to support
- deterministic seeding for reproducibility

Training Objective
Primary objective:
- terminal mismatch loss between propagated Y_T and payoff

Recommended stabilizers:
- Z-regularization
- variance control on terminal residual
- monitoring of gradient norms and Z norms

Validation Benchmarks
The first validation targets should be:
1. Black-Scholes via the same data interface
2. classical Heston via the same data interface
3. lifted rough Heston with small factor count

Benchmarks for the demo:
- price versus engine Monte Carlo price
- hedge versus coarse finite-difference delta benchmark
- hedge PnL/risk on replay versus existing baseline hedge

Implementation Sequence
Step 1
Define the exact feature contract and output contract for the runtime model.

Step 2
Add a batched lifted rough Heston training API to the C++ engine returning:
- state tensor of shape [batch, time, state_dim]
- Brownian increments
- terminal payoff
- optional path auxiliaries if needed later

Step 3
Add deterministic replay mode and fixed seeds.

Step 4
Build the offline deep BSDE trainer with:
- shared MLP
- time embedding
- Y0 parameter
- BSDE forward propagation
- logging and checkpointing

Step 5
Validate the whole learning stack first on Black-Scholes or classical Heston.

Step 6
Switch to lifted rough Heston with small factor count.

Step 7
Export the trained model to a frozen inference artifact.

Step 8
Load the artifact back into the C++ engine and benchmark in-process inference latency.

Step 9
Run end-to-end replay with online hedge generation and compare:
- price quality
- hedge quality
- PnL distribution
- risk metrics
- latency metrics

Latency Plan
Since the model is meant to be an actual online hedging component:
- inference must run in-process in C++
- avoid Python in production inference
- use a small fixed-shape MLP
- evaluate only on hedge decision events, not necessarily every market tick
- measure p50 and p99 latency, not just averages
- optimize only after obtaining a stable baseline

Main Risks
1. Misalignment between event-driven runtime and fixed-grid BSDE training
2. Misinterpretation of BSDE Z as directly executable hedge in an incomplete market
3. Poor numerical scaling of lifted state variables
4. Too many factors too early
5. Network instability before validation on simpler models
6. Latency creep from oversized models or oversized feature sets

Practical Defaults
- use log-price or log-moneyness rather than raw spot
- include explicit variance V_t
- start with m = 4 or 8 factors
- use a small shared-weight MLP
- normalize all state inputs
- train offline in Python first
- deploy frozen inference in C++
- begin with spot-only hedge output

Success Criteria for the Demo
The demo is successful if:
- the learned price is close to the engine Monte Carlo benchmark,
- the hedge is stable and competitive with the existing baseline in replay,
- the inference artifact runs inside the C++ engine,
- p50 and p99 inference latency are within target,
- the full train-export-load-replay loop is reproducible.

Immediate Next Task
Write down the exact runtime interface:
Input:
- normalized state vector
- optional market context needed at hedge time

Output:
- price estimate or continuation value if needed
- executable hedge signal(s)

This interface should be finalized before implementation begins.
