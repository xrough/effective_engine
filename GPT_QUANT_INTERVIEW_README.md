# Effective Engine: Quant Interview Research Context

**Status:** Research and engineering prototype  
**Repository review date:** 20 August 2026  
**Underlying:** SPY options  
**Primary topics:** volatility risk premium, option-smile dynamics, rough-volatility hypotheses, neural hedging, causal model evaluation, and research-to-C++ deployment

## Purpose of this document

This document is the interview-safe source of truth for discussing Effective Engine with an interviewer or another GPT. It separates implemented functionality from empirical evidence and separates provisional findings from invalidated strategy-performance claims.

The project should be described as an **integrated volatility research and model-governance platform**, not as a proven profitable trading strategy. Its strongest contributions are the formulation of the hedge objective, the use of simple baselines and falsification tests, the Python-to-ONNX-to-C++ model lifecycle, and the willingness to retain negative results.

When this document is supplied to a GPT, it should follow four rules:

1. Preserve the evidence labels and do not upgrade an exploratory result into validated alpha.
2. Never cite the current historical dollar PnL, Sharpe, or hedge ranking as economically valid performance.
3. Distinguish the 127-day multi-expiry research archive, the five-day registered replay panel, and the synthetic hedge datasets.
4. Treat limitations as part of the interview story and use the proposed repair-and-retest plan when discussing next steps.

## Executive summary

Effective Engine asks one central economic question:

> Which risks should a hedge remove, and which residual risks should remain because they are the intended source of return?

The system connects four layers that are often evaluated separately:

1. volatility and option-smile research;
2. strategy signals and position construction;
3. analytic, rough-volatility, and neural hedge policies;
4. execution, accounting, walk-forward evaluation, and model promotion.

The empirical research does **not** establish production alpha. The most defensible findings are:

- SPY smile skew exhibits a strong cross-sectional power-law fit, but the fitted exponent does not confirm the original rough-volatility prior.
- A raw rough-structure forecast fails to beat naive carry and is explicitly rejected.
- A carry-conditioned rough-inspired correction, mostly using 25-delta butterfly curvature, improves forecast RMSE across many tested cells; this remains exploratory because selection and evaluation use the same historical sample.
- Project prose reports lower neural hedge-error RMSE and tail dispersion on synthetic Lifted Rough Heston tests. The underlying immutable result reports are not present in the current tree, so this is a historical model-level claim to reproduce, not current evidence of trading profitability.
- The current historical strategy PnL and Sharpe figures are not economically valid because the replay ledger and contract lifecycle are incomplete. They must not be presented as performance evidence.

## 1. Research question and economic thesis

An options strategy can have several simultaneous sources of exposure:

- directional spot delta;
- convexity and realized-variance exposure through gamma;
- implied-volatility and surface exposure through vega and higher-order Greeks;
- financing, spread, turnover, and discrete-hedging costs;
- residual model risk.

The research thesis is that hedge design is part of alpha design. A hedge optimized to minimize the full option-payoff error may suppress both unwanted directional risk and residual volatility exposure. A delta-only, variance-optimal, or cost-aware hedge can instead neutralize a narrower risk component and leave a deliberate residual.

This is currently a **research hypothesis and objective-design insight**, not a conclusion established by the historical PnL. In a stochastic-volatility market with untraded volatility risk, exact replication is generally unavailable; the economically correct interpretation is an incomplete-market or minimum-mean-square hedge.

## 2. System scope

The implemented research-to-replay flow is:

```text
SPY option-chain observations
  -> data validation and causal date windows
  -> IV, smile, model-free variance, and realized-volatility features
  -> volatility and smile hypotheses
  -> signal combination and ATM option positioning
  -> simulated order lifecycle and bid/ask execution
  -> Black-Scholes, rough-volatility, or neural hedge policy
  -> position, Greek, cost, and replay diagnostics
  -> experiment manifests, gates, artifact hashes, and promotion decisions
```

The C++ layer provides deterministic event-driven replay, signal and strategy components, order/fill events, execution frictions, hedgers, and reporting. The Python layer provides market-data preparation, statistical research, Lifted Rough Heston simulation, PyTorch training, ONNX export, validation, and walk-forward orchestration.

Execution simulation includes bid/ask pricing, an underlying half-spread, latency, partial fills, order provenance, and lifecycle events. These are useful infrastructure features, although the present strategy-level accounting remains incomplete.

## 3. Dataset and evidence lineage

Three distinct evidence sets must not be conflated.

### 3.1 Multi-expiry SPY research archive

- **Source:** OPRA-derived SPY option-chain data.
- **Period:** 7 August 2025 through 6 February 2026.
- **Trading days:** 127 total.
- **Calendar split:** 102 dates before 1 January 2026 and 25 dates in 2026.
- **Skew-scaling sample:** 44,483 unique timestamps, with a mean of approximately 11.5 expiries per timestamp.
- **Use:** cross-sectional skew scaling and rough-inspired temporal/conditional research.

This is the source of the 127-day skew-scaling result. It is not the same object as the small registered replay panel.

### 3.2 Registered local replay panel

- **Dataset ID:** `spy_chain_panel_local_v1`.
- **Period:** 7 August 2025 through 13 August 2025.
- **Trading days:** 5.
- **Rows:** 2,029.
- **Fields:** underlying price, ATM strike and expiry, call/put quotes, ATM IV, 25-delta call/put points, risk reversal, butterfly, model-free variance proxy, SSVI parameters, and annualized five-minute realized volatility.
- **Use:** local strategy replay and walk-forward plumbing.

The market-data directory is intentionally gitignored. The registry therefore records the expected schema and lineage but does not by itself make the raw dataset publicly reproducible.

### 3.3 Synthetic hedge data

The neural hedge is trained predominantly on simulated Lifted Rough Heston paths calibrated to market-level parameters. Stored metadata describes **9,560 paths: 9,500 synthetic paths plus 60 trajectories derived from pre-split quotes, with 50 steps per path**. It should not be described as training on 102 real market sessions. The real-derived path construction is itself provisional: missing IV values are backward-filled, one path is constructed per expiry, and the terminal payoff uses the last observed bar rather than an actual expiry settlement.

Project prose also describes validation on fresh simulated paths, including an independent Bayer-Breneis-style simulator. The associated machine-readable result artifacts are not committed, so those numerical results are retained below as reported historical diagnostics that require reproduction.

This evidence tests whether the network can reduce hedge error under the assumed stochastic model. It does not test whether a tradeable market signal exists.

### Dataset claims that should not be used

Earlier project prose refers to 154 total days, 127 in-sample days, and 27 out-of-sample days. Those counts are inconsistent with the available 127 daily archive files and the 25 stored 2026 result rows. The interview-safe counts are the ones stated above, with the five-day replay panel identified separately.

## 4. Features and models

### 4.1 Smile and volatility features

The research layer reconstructs forwards from put-call parity where the richer chain is available and solves implied volatility numerically. The principal smile variables are:

```text
RR25 = IV_25-delta-call - IV_25-delta-put
BF25 = (IV_25-delta-call + IV_25-delta-put) / 2 - IV_ATM
```

It also uses:

- ATM implied volatility and annualized implied variance;
- realized-volatility measures;
- a single-expiry model-free variance proxy;
- SSVI-style surface parameters;
- variance-risk-premium, HAR-inspired, skew, and curvature signals.

The demonstration composite assigns fixed weights of **50% variance risk premium, 30% HAR-inspired realized volatility, and 20% skew/curvature**. These weights are heuristic implementation choices, not coefficients estimated or validated on an untouched sample. The strategy controller translates the composite score into ATM straddle orders sized from signal strength and estimated vega.

The variance proxy should be called a **single-expiry model-free variance proxy**, not VIX. It omits the full strike, rate, zero-bid, `K0`, and two-maturity 30-day interpolation conventions. The SSVI fit is parsimonious and based on the ATM point and the same two 25-delta wings later used by the signal, so it is a feature representation rather than an independent surface-mispricing test.

### 4.2 Forecast baselines

Naive carry is the key benchmark: the next smile observation is forecast by the current observation. Depending on the experiment, additional autoregressive or conditioned regressions are compared with the raw rough-inspired forecast.

This baseline is economically important. A sophisticated forecast that cannot beat carry does not justify additional model risk or turnover.

### 4.3 Hedge-policy ladder

The platform implements four relevant hedge concepts:

1. **Black-Scholes delta:** the transparent analytic baseline.
2. **Rough-volatility delta:** Black-Scholes delta plus a vega-times-smile-response correction.
3. **Payoff-targeted neural BSDE:** trained to minimize discounted terminal-payoff error under simulated Lifted Rough Heston dynamics.
4. **Delta-targeted neural model:** trained against the pathwise Black-Scholes delta-hedge integral.

The neural state is:

```text
[time to maturity, log(S/K), variance, U1, U2, U3, U4]
```

The four `U` factors form a finite-dimensional Markovian approximation to a rough-volatility kernel. The delta-targeted model currently learns a surrogate of a known analytic quantity. Its present novelty is the objective and deployment framework, not proof that a neural model is superior to Black-Scholes in market data.

## 5. Research hypotheses and tests

| ID | Hypothesis | Test | Baseline or decision rule |
|---|---|---|---|
| H1 | Short-end SPY skew follows a stable maturity power law. | At each timestamp, regress `log(abs(RR25(T)))` on `log(T)` across expiries; summarize slope and fit stability. | Compare fitted slope with the exponent implied by the original `H = 0.10` rough prior. |
| H2 | Raw rough structural forecasts improve one-step smile prediction. | Sweep `H` over `{0.03, 0.05, 0.07, 0.10, 0.15, 0.20}` and bars over `{1, 5, 15, 30, 60}` minutes. | Out-of-sample-style forecast RMSE versus naive carry. |
| H3 | Raw rough information becomes useful after large market moves. | Score forecast errors separately in active and quiet regimes across move thresholds and horizons. | Improvement over carry within the same scored observations. |
| H4 | Rough geometry is more useful as a correction to carry than as a replacement for carry. | Compare a carry-conditioned rough residual and a recency-weighted rough forecast across the same `H` and frequency grid. | Internal PASS requires a material RMSE improvement over carry. |
| H5 | The H4 improvement is concentrated in stressed regimes. | Compare active-regime and quiet-regime improvements. | Desired pattern is positive active improvement without broad quiet-regime dependence. |
| H6 | A state-dependent neural hedge can reduce discrete hedge error under rough stochastic-volatility dynamics. | Evaluate held-out simulated paths using hedge-error RMSE, MAE, standard deviation, and CVaR. | Black-Scholes delta and finite-difference delta controls. |
| H7 | Model artifacts can be trained and deployed without future-date leakage or cross-language drift. | Date-exclusive walk-forward training, PyTorch/ONNX parity, schema and hash checks, sanity gates, replay, and explicit promotion. | All required artifact and replay gates must pass before promotion. |

The research forecasts generally use causal expanding or prior-date windows. However, the same historical research sample was used to compare many `H`, frequency, feature, method, and regime choices. Internal PASS labels are therefore engineering research gates, **not statistical significance statements**.

## 6. Current empirical results

### H1: Skew maturity scaling — weak or partial support

Across 127 trading days and 44,483 timestamps:

- median fitted slope `beta`: **+0.2123**;
- mean cross-sectional `R-squared`: **0.8606**;
- median cross-sectional `R-squared`: **0.9068**;
- slope coefficient of variation: **0.4210**.

The high `R-squared` values indicate a stable power-law description of the observed term structure. They do not validate the original rough prior. Under the stated scaling relation, `H = 0.10` implies an exponent near `-0.40`, which differs materially in sign and magnitude from the median empirical slope.

**Verdict:** a power-law shape is present, but the original rough-parameter interpretation is not confirmed.

### H2: Raw rough temporal forecast — rejected

The robustness grid tests six values of `H` and five bar frequencies, producing 30 evaluable cells.

- PASS cells: **0/30**;
- raw rough forecasts lose to carry at the shortest horizons;
- the gap narrows at some coarser horizons but does not establish aggregate superiority.

**Verdict:** reject the raw rough forecast and retain it as a negative control.

### H3: Conditional raw rough forecast — narrow support

On a 90-day conditional study, **6/42** cells pass at the top-10% move definition, **6/42** pass at top-20%, and **0/42** pass at top-30%. All surviving cells use a daily-like 390-minute risk-reversal signal.

**Verdict:** too narrow for an unconditional alpha claim; at most a candidate sizing overlay.

### H4: Incremental rough correction to carry — exploratory pass

The conditioned sweep finds:

- PASS cells: **24/30**, or 80%;
- broad internal passes at 1, 5, 15, and 60 minutes;
- 30-minute cells are marginal;
- the winning feature is overwhelmingly `BF25` curvature rather than `RR25` skew;
- carry-conditioned rough residuals lead at short horizons;
- recency-weighted rough estimates lead at 60 minutes;
- reported cell-level RMSE improvements are generally approximately 3% to 8%.

**Verdict:** the data support a hypothesis that rough-inspired curvature can incrementally improve carry. Because feature, method, `H`, horizon, and regime selection were performed on the same sample without an untouched outer holdout or multiplicity correction, this is hypothesis-generating rather than confirmed tradeable edge.

### H5: Stress concentration — weak support

Only **6/42** tested stress-regime cells pass under the reported 30% move threshold, all involving five-minute `BF25` carry-conditioned forecasts.

**Verdict:** do not claim that the H4 effect is broadly concentrated in stress periods.

### H6: Reported synthetic neural hedge diagnostics — reproduction required

The project documentation reports the following controlled simulated results, but the current tree does not contain immutable machine-readable reports tied to these tables:

| Test | Black-Scholes delta RMSE | Neural RMSE | Relative improvement |
|---|---:|---:|---:|
| Stored held-out rough paths | 6.273 | 4.495 | 28.3% |
| Independent simulator | 4.542 | 3.391 | 25.3% |

In a separate 2,000-path, 50-step synthetic benchmark:

| Hedger | Hedge-PnL standard deviation | CVaR 95% |
|---|---:|---:|
| Analytic Black-Scholes delta | 4.24 | 22.57 |
| Neural hedge | 3.40 | 19.35 |

Median in-process inference is reported at approximately **3.4 microseconds**.

**Verdict:** treat these numbers as a historical result to reproduce. If reproduced, they would support only the claim that the network reduces hedge error under the specified simulated model. They would not establish superiority on historical SPY data, and the C++ live conversion requires correction before cross-language performance can be trusted.

### Archived historical replay totals — accounting outputs, not PnL

The archived post-split CSVs contain 25 rows from 2 January through 6 February 2026. They reproduce the large totals quoted in the older project README:

| Policy | Engine-reported total | Recorded fills |
|---|---:|---:|
| Black-Scholes delta | 1,060,036.23 | 1,022 |
| Rough-volatility delta | 700,200.86 | 1,019 |
| Full-payoff BSDE | 284,847.38 | 9,903 |
| Delta-targeted BSDE | 885,949.75 | 9,961 |

These values are included only to reconcile the archived files with earlier documentation. They are **not PnL, returns, alpha, or a valid hedge ranking**. The reported total adds option mark-to-market to stock hedge fill cash flows and subtracts costs without marking or closing the corresponding stock inventory. The large fill-count difference also shows that model choice is confounded with rebalance policy.

### H7: Walk-forward and deployment lifecycle — strong engineering evidence

For each deployment date, the orchestrator is designed to:

1. train only on earlier dates;
2. estimate market parameters;
3. generate calibrated synthetic paths;
4. retrain or warm-start the model;
5. export ONNX;
6. test PyTorch/ONNX parity and delta sanity;
7. replay both neural and Black-Scholes policies;
8. record metrics, logs, manifests, and hashes;
9. promote only after explicit gates and archive the previous artifact.

The latest stored experiment is a **two-window smoke profile**, not a full walk-forward backtest. Its training windows contain only three dates/1,217 rows and four dates/1,623 rows. Both windows passed plumbing checks, neither promoted a model, and the recorded maximum ONNX `Z` parity errors were approximately `1.14e-5` and `1.81e-5`.

**Verdict:** this is one of the project's strongest engineering points. It demonstrates implemented causal orchestration, reproducibility controls, cross-language parity checking, and model-governance thinking at smoke-test scale. It does not demonstrate production walk-forward performance or validate the replay's economic metric.

## 7. Evidence hierarchy

### Supported claims

- An event-driven C++ and Python volatility research platform was implemented.
- The system contains analytic, rough-inspired, and neural hedge benchmarks.
- The research archive shows a strong descriptive power-law fit in smile maturity geometry.
- The raw rough forecast fails the carry benchmark.
- A carry-conditioned `BF25` correction is a promising exploratory feature.
- Project documentation records synthetic neural hedge diagnostics that are explicitly separated from market alpha claims.
- The project implements and smoke-tests date-exclusive walk-forward scheduling, ONNX parity, manifests, hashes, gates, and artifact archival.

### Provisional claims

- Rough-inspired curvature contains incremental forecast information beyond carry.
- State-dependent neural hedging may reduce real-market hedge error.
- Conditional smile effects may help size volatility exposure in selected regimes.
- The documented synthetic neural hedge improvements will persist when reproduced from versioned artifacts.

These require a fresh outer holdout, controlled execution policy, block-bootstrap uncertainty, and corrected accounting.

### Claims that must not be made

- The strategy has demonstrated a million-dollar profit.
- The reported strategy Sharpe is economically valid.
- The full neural BSDE exactly replicates the option in a complete market.
- Neural hedging has beaten Black-Scholes on a controlled historical SPY replay.
- The data confirm `H = 0.10` for SPY.
- Rough volatility has been proven to generate tradeable alpha.
- The research used 154 clean trading days with 127 IS and 27 OOS days.

## 8. Current validity limitations

These limitations should be volunteered in a serious quant interview.

### 8.1 The strategy PnL is not self-financing

Underlying hedge fills are recorded as cash flow, but the remaining stock position is not marked to market or liquidated in total PnL. For example, short-sale proceeds can appear as positive hedge PnL while the corresponding short inventory remains open. Consequently, headline dollar PnL and derived Sharpe figures are not economically interpretable.

The economic units also require a consistent option contract multiplier. The current position aggregation does not reliably apply the standard 100-share equity-option multiplier across delta and PnL calculations.

### 8.2 Exits and option lifecycle are incomplete

The strategy exit changes internal state without submitting closing trades. ATM call and put identifiers are also reused while strike and expiry change. Positions from distinct contracts can therefore be aggregated and revalued as if they were one instrument. Expiry, roll, and terminal liquidation need contract-aware implementation.

### 8.3 Hedge comparisons are not controlled

The analytic delta hedger and neural hedger rebalance under different event schedules. Differences in turnover, spread cost, and hedge error therefore combine model choice with scheduling choice. All hedge policies need identical timestamps, thresholds, lot rounding, and execution rules.

### 8.4 Cost treatment needs reconciliation

Execution prices already reflect bid/ask spread, while the PnL tracker separately subtracts spread-like costs. This can double-count part of the friction. Commission, spread, slippage, borrow, financing, and market impact should be separate ledger terms.

### 8.5 Rough-volatility units are inconsistent

Annualized implied variance, total variance, and the initial variance-curve parameter are not used consistently between extraction, calibration, signal construction, and pricing. The rough-delta underperformance cannot be economically interpreted until these units are corrected.

### 8.6 Neural artifact and delta scaling need correction

Walk-forward normalization writes the strike scale under one key while the C++ runtime reads another and can silently default to 100. In addition, training defines `Z` proportional to `sigma * S * delta`, while C++ converts it to delta using `sigma * K_train`; Python validation correctly divides by `sigma * S`. Current deployed deltas can therefore be mis-scaled.

### 8.7 The full BSDE interpretation is incomplete-market hedging

The Lifted Rough Heston simulation contains spot and volatility Brownian risk, while the BSDE recursion integrates only the spot shock. Volatility risk is unhedged. The correct description is an L2 or variance-optimal projection using the traded underlying, not exact full replication of a seven-dimensional claim.

### 8.8 Statistical selection remains exploratory

Many combinations of `H`, frequency, feature, method, and regime were examined on the same sample. Some regime thresholds also depend on full-sample distributions. There is no outer untouched period or formal multiple-testing adjustment for the selected Gate 4 specification.

Neural checkpoint selection is primarily based on training loss rather than a separate validation loader. Some documented multi-seed and tolerance requirements are not enforced by the executable gates, and the existing promotion tolerance is too permissive to act as an economic performance test.

### 8.9 Feature definitions and event timing need tightening

The lightweight preprocessing path simplifies rates, dividends, forward recovery, and time-to-expiry treatment. Its model-free variance estimate omits several conventions required for the official VIX calculation. The HAR-inspired signal uses 1-, 5-, and 22-bar windows even when the bars are minutes, with fixed coefficients rather than SPY-fitted daily, weekly, and monthly components.

In the replay feed, call and put quote events are published before the current smile snapshot. Signals can therefore update more than once per source row while using a preceding smile state. Daily boundary handling also assigns the first new-day bar before closing the preceding day, and realized-volatility buffers are not consistently session-reset. These points affect the interpretation of holding periods, observations, and daily metrics.

### 8.10 Risk controls are implemented separately from the main replay

The repository contains loss-limit, blocking, reduce-only, delta-limit, and drawdown-policy infrastructure. It is not fully wired into the alpha replay that generated the reported strategy results. It should be described as implemented risk infrastructure, not as evidence that those controls governed the historical experiment.

## 9. Recommended interview narrative

### One-minute version

> I built a research-to-replay platform for SPY volatility around one economic question: what should a hedge neutralize, and what residual should remain as alpha? The pipeline extracts implied-volatility and smile features, tests variance-premium, HAR, and rough-smile hypotheses against simple causal baselines, compares Black-Scholes, rough-volatility, and neural hedge policies, and exports PyTorch models through ONNX into an event-driven C++ simulator. The strongest research result was not a large PnL number. Raw rough forecasts failed carry and were rejected, while a carry-conditioned butterfly-curvature feature showed provisional three-to-eight-percent RMSE improvements. The project documentation also reports lower neural hedge error on synthetic paths, which I would reproduce from versioned artifacts before presenting as validated evidence. The project demonstrates falsification and model-governance discipline, but I would not claim strategy returns until the self-financing ledger, contract lifecycle, hedge cadence, and scaling issues are corrected.

### Five contributions to emphasize

1. **Economic formulation:** hedge objective and alpha objective must be designed jointly.
2. **Benchmark discipline:** simple carry and Black-Scholes baselines remain the burden of proof for complex models.
3. **Falsification:** the raw rough hypothesis was rejected rather than hidden.
4. **Model lifecycle:** Python research is connected to ONNX validation and C++ inference through explicit artifacts and walk-forward gates.
5. **Model-risk awareness:** limitations are identified with a concrete repair and revalidation plan.

### Interview questions to expect

**Why use a neural network if the target is Black-Scholes delta?**  
The delta-only experiment first validates the training and deployment scaffold against a known target. It is not itself evidence of new alpha. A meaningful next target would use non-Black-Scholes state information or directly optimize residual variance and cost under a controlled policy.

**Can the full BSDE exactly replicate the option?**  
Not with untraded volatility risk and only the underlying as hedge instrument. It should be described as a minimum-error projection under an incomplete stochastic-volatility model.

**How was lookahead prevented?**  
The walk-forward orchestrator uses observations strictly before each deployment date, and the forecasting code uses causal or expanding estimates. However, research-model selection used the same broader sample, so a new outer holdout is still required.

**Why did rough delta underperform?**  
The historical comparison is not interpretable yet because variance units, contract identity, rebalance cadence, PnL, and neural scaling are inconsistent. Separately, the forecast research provides a genuine negative result: raw rough structure did not beat carry.

**What is the correct evaluation metric?**  
For hedge quality: terminal hedge-error RMSE, bias, standard deviation, CVaR, turnover, and cost under identical schedules. For a strategy: self-financing returns on a defined capital base, drawdown, tail risk, exposure attribution, capacity, and uncertainty intervals.

**What would you fix first?**  
First build a contract-aware, self-financing cash and inventory ledger with expiry and liquidation. Then standardize hedge schedules and cost treatment, correct units and neural scaling, and rerun a fresh purged outer walk-forward evaluation.

## 10. Next research plan

The recommended order is:

1. Represent each option by immutable underlying, expiry, strike, type, and multiplier.
2. Implement cash, option inventory, stock inventory, financing, borrow, commission, and terminal liquidation in a self-financing ledger.
3. Correct daily boundaries, expiry handling, rolls, and position close-outs.
4. Give every hedger identical observation times, thresholds, lot rounding, and execution rules.
5. Reconcile annualized variance, total variance, and rough-model parameters.
6. Unify the Python and C++ artifact schema and convert neural `Z` using `sigma * S`.
7. Freeze the selected conditioned-curvature model before observing new dates.
8. Evaluate on a new chronological outer holdout with block-bootstrap confidence intervals.
9. Report hedge-error RMSE, bias, standard deviation, CVaR, turnover, spread cost, drawdown, and capital-normalized returns.
10. Compare added complexity against carry, AR(1), Black-Scholes delta, and a cost-aware linear hedge.

## 11. Bottom line

Effective Engine is strongest as evidence of integrated quantitative thinking: translating an economic hypothesis into features, baselines, falsification tests, hedge objectives, execution semantics, and deployment controls. The current project supports a negative result for raw rough forecasting and an exploratory result for carry-conditioned smile curvature. It also documents positive synthetic neural hedge diagnostics that should be reproduced from versioned artifacts before being treated as validated model evidence.

It does **not** currently support a claim of profitable historical strategy performance. Presenting that distinction clearly is part of the project's value: it demonstrates the ability to distinguish an interesting model, a statistically promising feature, a correctly controlled hedge experiment, and an economically valid trading result.

## 12. Evidence map

- Dataset registry: `lab/registry/datasets.json`
- Research-gate registry: `lab/registry/research_gates.json`
- Skew-scaling benchmark: `demo/python/research/skew_scaling/`
- Temporal rough forecast tests: `demo/python/research/roughtemporal_intraday/`
- Conditional regime tests: `demo/python/research/conditional_dynamics/`
- Neural model definitions: `demo/python/bsde/model.py`
- Synthetic hedge validation: `demo/python/validation/bsde_hedge_validation.py`
- Walk-forward lifecycle: `demo/python/bsde/walk_forward_pipeline.py`
- ONNX export and parity: `demo/python/bsde/export.py`
- C++ replay entry point: `demo/cpp/alpha_main.cpp`
- Execution simulator: `demo/cpp/execution/SimpleExecSim.hpp`
- Current PnL implementation: `demo/cpp/pnl/ExtendedPnLTracker.hpp`
