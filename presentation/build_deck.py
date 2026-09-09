#!/usr/bin/env python3
"""
Build presentation/effective_engine.pptx.

    python3 build_deck.py

Part 1 mirrors talk.tex slide for slide -- same argument, same numbers, same
midnight-blue/pine-green identity. Part 2 is a component library: blank,
reusable versions of every layout, for data-analysis talks and quant
interviews. Edit a Part-2 slide, or copy its four lines of code, and you have
a new slide.

Figures live in presentation/fig/ (recovered from git history).
"""
import os
from slidekit import *          # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")


def fig(name):
    p = os.path.join(FIG, name)
    return p if os.path.exists(p) else None


d = Deck()
SEC = 6          # number of sections, for the divider progress motif


# ===========================================================================
# PART 1 -- THE TALK
# ===========================================================================

# --- 1. Title -------------------------------------------------------------
s = title_slide(
    d, "Effective Engine MVP",
    "Variance alpha, neural BSDE hedging, and execution-aware\n"
    "walk-forward retraining",
    meta="Xiaohao Ji   ·   independent project\n"
         "C++ options trading engine + Python research pipeline   ·   2026")
s.notes_slide.notes_text_frame.text = (
    "30-45 minute project report. The spine: choose the right hedge "
    "objective, replay honestly, promote only after walk-forward gates.")

# --- 2. Map ---------------------------------------------------------------
s = d.slide()
y = title(s, "Thirty-Five Minute Map")
y = table(s, M, 1.42, CW, [
    ["Part", "Main message", "Time"],
    ["Motivation", "Options alpha is signal, hedge, execution, and validation.", "5 min"],
    ["State of art", "From Black–Scholes to rough vol and deep hedging; deployment remains hard.", "7 min"],
    ["Core idea", "Use BSDE learning as a hedge operator, not an alpha destroyer.", "10 min"],
    ["System", "C++ replay plus Python walk-forward retraining makes the idea testable.", "8 min"],
    ["Results", "Delta-only BSDE preserves most of BS Delta's OOS VRP capture.", "7 min"],
    ["Outlook", "Mature pieces, research-grade pieces, and next steps.", "3 min"],
], col_w=[0.19, 0.68, 0.13], align=["l", "l", "r"], row_h=0.52)
note_bottom(s, "Talk spine",
            "A research prototype for equity-index volatility trading: choose "
            "the right hedge objective, replay honestly, and promote only "
            "after walk-forward gates.", h=1.00, accent=PINE)
footer(s, "Effective Engine MVP", "1")

# --- 3. One picture -------------------------------------------------------
s = d.slide()
title(s, "The Project In One Picture")
bw, gap = 3.2, 1.245
x1, x2, x3 = M, M + bw + gap, M + 2 * (bw + gap)
r1, r2, bh = 1.78, 3.66, 1.02

flow_row(s, M, r1, CW, bh, [
    ("Option panel", "SPY spot, ATM, 25Δ, SSVI"),
    ("Variance alpha", "VRP + HAR-RV + skew / curvature"),
    ("Hedging models", "BS, rough delta, neural BSDE"),
], gap=gap, accent=NAVY, fill=TINT_NAVY)

flow_box(s, x3, r2, bw, bh, "C++ replay engine", "fills, PnL, lifecycle",
         fill=TINT_PINE, accent=PINE)
flow_box(s, x2, r2, bw, bh, "Walk-forward retrain", "train, export ONNX, replay",
         fill=TINT_PINE, accent=PINE)
flow_box(s, x1, r2, bw, bh, "Promote or reject", "balanced gates",
         fill=TINT_PINE, accent=PINE)
arrow(s, x3 - 0.07, r2 + bh / 2, x2 + bw + 0.07, r2 + bh / 2, color=SLATE)
arrow(s, x2 - 0.07, r2 + bh / 2, x1 + bw + 0.07, r2 + bh / 2, color=SLATE)

# models -> replay, down the right-hand quarter of the column
dn = x3 + bw * 0.74
arrow(s, dn, r1 + bh, dn, r2 - 0.05, color=SLATE)
# promote -> models, back up the left-hand quarter through the clear lane
up, lane = x3 + bw * 0.26, 3.06
line(s, x1 + bw / 2, r2 - 0.02, x1 + bw / 2, lane, color=SLATE)
line(s, x1 + bw / 2, lane, up, lane, color=SLATE)
arrow(s, up, lane, up, r1 + bh + 0.03, color=SLATE)
text(s, x1 + bw / 2 + 0.16, lane + 0.07, 3.6, 0.26,
     "retrain · re-export · re-gate", size=9.5, color=SLATE, italic=True)

note_bottom(s, "Main design principle",
            "The project treats alpha, hedge, execution, and model lifecycle "
            "as one coupled system — not four separate research problems.",
            h=1.00, accent=PINE)
footer(s, "Effective Engine MVP", "2")

# --- 4. Section: Market Problem -------------------------------------------
section_slide(d, 1, "Market Problem",
              "What the trade is, and where the premium is supposed to come "
              "from.", total=SEC)

# --- 5. Trading question --------------------------------------------------
s = d.slide()
title(s, "What Is The Trading Question?")
cw = (CW - 0.40) / 2
formula(s, M, 1.50, cw, "VRP  ≈  σ²(implied)  −  "
                        "σ²(realized)", size=17, h=0.86)
bullets(s, M, 2.62, cw, 2.2, [
    "If implied variance is rich, short front variance can earn carry.",
    "If realized volatility dominates, long gamma can profit from "
    "rebalancing.",
    "Both legs are quoted, so the spread you cross is the first thing that "
    "eats the edge.",
    "Which of the two regimes you are in changes daily — the signal has "
    "to say which.",
], size=13, gap=10)
card(s, M + cw + 0.40, 1.50, cw, 2.98, "Project implementation", [
    "SPY intraday option-chain replay.",
    "Model-free VIX-style variance proxy, with an ATM fallback.",
    "SSVI smile features from ATM and 25Δ quotes.",
    "Greek attribution and daily stability reports.",
], accent=NAVY, size=13)
note_bottom(s, "The constraint that decides everything",
            "The signal is only useful if hedging and transaction costs do "
            "not erase it. That is why the hedge objective — not the "
            "pricing model — is the strategic choice in this project.",
            h=1.06, accent=CLAY)
footer(s, "Market Problem", "3")

# --- 6. State of the art --------------------------------------------------
s = d.slide()
title(s, "State Of Art: Volatility Trading Stack")
y = table(s, M, 1.44, CW, [
    ["Layer", "Classical / current tools", "Why it matters here"],
    ["Pricing", "Black–Scholes, local vol, Heston, stochastic vol",
     "Gives deltas, vegas, and sanity checks."],
    ["Surface", "SVI / SSVI, skew, curvature, variance swaps",
     "Alpha lives in surface dislocations, not only ATM IV."],
    ["Dynamics", "Rough volatility and Markovian lifts",
     "Short-dated skew has rough-style scaling."],
    ["Learning", "Deep BSDE and deep hedging",
     "High-dimensional hedging without finite-difference grids."],
    ["Deployment", "Walk-forward validation, execution replay, risk gates",
     "Research signal must survive data drift and fills."],
], col_w=[0.16, 0.42, 0.42], align=["l", "l", "l"], row_h=0.60)
note_bottom(s, "Where this project sits",
            "The first four layers are well covered by the literature. The "
            "fifth — deployment — is where the prototype spends most "
            "of its engineering.", h=0.94, accent=PINE)
footer(s, "Market Problem", "4")

# --- 7. Data and signal ---------------------------------------------------
s = d.slide()
title(s, "Data And Signal Construction")
cw = (CW - 0.40) / 2
card(s, M, 1.50, cw, 2.68, "Available panel", [
    "SPY spot recovered from the option-chain panel.",
    "ATM call / put quotes with bid and ask.",
    "25Δ call / put quotes for skew and butterfly.",
    "SSVI parameters (ρ, φ) and realized-vol features.",
], accent=NAVY, size=13)
text(s, M + cw + 0.40, 1.50, cw, 0.30, "Composite alpha", size=13, bold=True,
     color=PINE, font=HEAD)
table(s, M + cw + 0.40, 1.94, cw, [
    ["Signal", "Weight"],
    ["VRP", "50%"],
    ["HAR-RV", "30%"],
    ["Skew / curvature", "20%"],
], col_w=[0.66, 0.34], align=["l", "r"], head_fill=PINE, row_h=0.62)
note_bottom(s, "Important choice",
            "Use the wings and the shape of the surface. ATM variance alone "
            "misses where index-option premium often hides — the 25Δ "
            "quotes are doing real work here.", h=1.00, accent=CLAY)
footer(s, "Market Problem", "5")

# --- 8. Market context figure ---------------------------------------------
s = d.slide()
title(s, "Market Context In The Current Panel")
plot_holder(s, M, 1.42, CW, 5.02, "Market context dashboard",
            "spot · ATM IV · realized vol · model-free VIX "
            "variance · 25Δ smile · SSVI",
            image=fig("fig2_market_context.png"))
caption(s, 6.54, "The dashboard tracks spot, ATM IV, realized vol, model-free "
                 "VIX variance, 25Δ smile structure, and SSVI dynamics.")
footer(s, "Market Problem", "6")

# --- 9. Section: Hedging Models -------------------------------------------
section_slide(d, 2, "Hedging Models",
              "Which risks the model should remove — and which residuals "
              "are the alpha.", total=SEC)

# --- 10. PnL anatomy ------------------------------------------------------
s = d.slide()
title(s, "PnL Anatomy: Alpha Versus Hedge",
      "Delta-hedged option PnL, schematically")
formula(s, M, 1.62, CW,
        "dΠ  ≈  Δ dS  +  ½ Γ d⟨S⟩  +  "
        "ν dσ  +  Θ dt  −  costs", size=20, h=0.90)
terms = [("Δ dS", "spot", SLATE), ("½ Γ d⟨S⟩",
          "gamma / realized variance", PINE),
         ("ν dσ", "vega / surface move", PINE),
         ("Θ dt", "decay", SLATE), ("costs", "execution", CLAY)]
tw = (CW - 4 * 0.20) / 5
for i, (sym, lab, col) in enumerate(terms):
    tx = M + i * (tw + 0.20)
    tnt = {PINE: TINT_PINE, CLAY: TINT_CLAY}.get(col, TINT_GREY)
    rect(s, tx, 2.76, tw, 0.92, fill=tnt, radius=0.07)
    text(s, tx, 2.88, tw, 0.32, sym, size=15, bold=True, color=col,
         align=PP_ALIGN.CENTER, font=HEAD)
    text(s, tx + 0.06, 3.24, tw - 0.12, 0.36, lab, size=9.5, color=SLATE,
         align=PP_ALIGN.CENTER, line_spacing=0.95)
bullets(s, M, 4.00, CW, 1.10, [
    "The hedge should remove unwanted spot exposure.",
    [{"t": "It should "}, {"t": "not automatically remove", "b": True,
                           "c": CLAY},
     {"t": " the gamma / vega residual that carries the volatility premium."}],
], size=13)
note(s, M, 5.34, CW, 0.86, "Consequence",
     "The hedging objective is a strategic choice, not a modelling detail.",
     accent=PINE)
footer(s, "Hedging Models", "7")

# --- 11. Four hedgers -----------------------------------------------------
s = d.slide()
title(s, "Four Hedgers Tested")
y = table(s, M, 1.46, CW, [
    ["Hedger", "Delta computation"],
    ["BSDelta", "N(d₁) at market ATM IV and historical time-to-expiry."],
    ["RoughVolDelta", "BS delta plus a Bergomi–Guyon smile-slope "
                      "correction."],
    ["NeuralBSDE-Full", "ONNX network trained to replicate the full "
                        "discounted payoff."],
    ["NeuralBSDE-Delta", "ONNX network trained to replicate only the BS delta "
                         "hedge PnL."],
], col_w=[0.26, 0.74], align=["l", "l"], row_h=0.72)
note_bottom(s, "Hypothesis",
            "A neural hedge can be useful only if its training target matches "
            "the trading objective. Everything that follows is a test of that "
            "one sentence.", h=1.06, accent=CLAY)
footer(s, "Hedging Models", "8")

# --- 12. Rough volatility -------------------------------------------------
s = d.slide()
title(s, "Rough Volatility: Why It Enters")
lw_, rw_ = 6.60, CW - 6.60 - 0.40
formula(s, M, 1.46, lw_,
        "Xₜ = ( τ, log(S/K), Vₜ, U₁, U₂, U₃, "
        "U₄ )", size=16, h=0.70)
bullets(s, M, 2.36, lw_, 2.2, [
    "Rough-vol models capture short-time skew scaling.",
    "Four OU factors approximate the rough kernel in Markovian form.",
    "The network input is 7-dimensional — already beyond comfortable "
    "finite-difference grids.",
], size=13)
formula(s, M + lw_ + 0.40, 1.46, rw_,
        "Δ(rough) = Δ(BS) + Vega · ∂σₖ/∂S",
        size=15, h=0.70, tint=TINT_PINE, color=PINE)
card(s, M + lw_ + 0.40, 2.36, rw_, 2.06, None, [
    "A more complete hedge than the BS delta.",
    "But it can consume carry in a positive-VRP regime.",
], tint=TINT_PINE, accent=PINE, size=13)
note_bottom(s, "Why it is only a stepping stone",
            "The rough correction is still a closed-form patch on a BS delta. "
            "The BSDE replaces the patch with a learned hedge operator on the "
            "full 7-D state.", h=1.00, accent=NAVY)
footer(s, "Hedging Models", "9")

# --- 13. Deep BSDE --------------------------------------------------------
s = d.slide()
title(s, "Deep BSDE Hedging: The Modelling Bridge")
formula(s, M, 1.46, CW,
        "Y(t+Δt)  =  Y(t)  +  Z(t) · ΔW(t),"
        "         Y(T) = Φ(X(T))", size=18, h=0.82)
bullets(s, M, 2.48, CW, 1.20, [
    "By Feynman–Kac, Y(t) = u(t, X(t)) and Z(t) is the Brownian hedge "
    "exposure.",
    "A shared-weight MLP maps each normalized state to (Y, Z).",
    "Export to ONNX makes the hedge available inside the C++ replay loop.",
], size=12.5)
y = table(s, M, 3.84, CW, [
    ["Training mode", "Economic meaning"],
    ["lrh", "Full payoff replication — removes nearly all residual "
            "alpha."],
    ["lrh_delta", "Delta-only target — leaves gamma + vega as VRP "
                  "exposure."],
], col_w=[0.24, 0.76], align=["l", "l"], row_h=0.62)
note_bottom(s, "One line of code, two different strategies",
            "Same network, same state, same export path. Only the training "
            "target changes.", h=0.90, accent=PINE)
footer(s, "Hedging Models", "10")

# --- 14. Key discovery ----------------------------------------------------
s = d.slide()
title(s, "The Key Discovery: Full Replication Can Kill Alpha")
cw = (CW - 0.40) / 2
formula(s, M, 1.46, cw,
        "Y₀ + Σ ZᵢΔWᵢ  ≈  "
        "e⁻ʳᵀ(S(T) − K)⁺", size=15, h=0.72,
        tint=TINT_GREY, color=SLATE)
card(s, M, 2.34, cw, 2.36, "Full replication target", [
    "The model learns to neutralize payoff risk.",
    "That includes the gamma / vega residual the variance strategy wants.",
], tint=TINT_GREY, accent=SLATE, size=12.5)
formula(s, M + cw + 0.40, 1.46, cw,
        "Φ(Δ)  =  Σ N(d₁,ᵢ) σᵢ Sᵢ "
        "ΔWᵢ", size=15, h=0.72, tint=TINT_PINE, color=PINE)
card(s, M + cw + 0.40, 2.34, cw, 2.36, "Delta-only target", [
    "The model learns the hedge component only.",
    "The residual Γ + ν remains available as alpha.",
], tint=TINT_PINE, accent=PINE, size=12.5)
note_bottom(s, "The result this talk is built around",
            "A theoretically cleaner hedge was economically worse. Full "
            "replication kept 26.9% of BS Delta's PnL; the delta-only target "
            "kept 83.6%.", h=1.00, accent=CLAY)
footer(s, "Hedging Models", "11")

# --- 15. Section: System Architecture -------------------------------------
section_slide(d, 3, "System Architecture",
              "Replay, execution, and the retraining loop that makes the "
              "claim testable.", total=SEC)

# --- 16. C++ architecture -------------------------------------------------
s = d.slide()
title(s, "C++ Event-Driven Replay Architecture")
bw3, g3 = 3.2, 1.245
xa, xb, xc = M, M + bw3 + g3, M + 2 * (bw3 + g3)
ra, rb, rc, bh3 = 1.72, 3.16, 4.60, 0.96
flow_row(s, M, ra, CW, bh3, [
    ("Historical feed", "market + option quotes"),
    ("IV + smile extractors", "ATM, VIX, SSVI"),
    ("Composite signal", "VRP / HAR / skew"),
], gap=g3, accent=NAVY, fill=TINT_NAVY)
arrow(s, xc + bw3 / 2, ra + bh3, xc + bw3 / 2, rb - 0.05, color=SLATE)
flow_box(s, xc, rb, bw3, bh3, "StrategyController", "straddle orders",
         fill=TINT_PINE, accent=PINE)
flow_box(s, xb, rb, bw3, bh3, "SimpleExecSim", "order lifecycle",
         fill=TINT_PINE, accent=PINE)
flow_box(s, xa, rb, bw3, bh3, "Delta / neural BSDE", "hedge orders",
         fill=TINT_PINE, accent=PINE)
arrow(s, xc - 0.07, rb + bh3 / 2, xb + bw3 + 0.07, rb + bh3 / 2, color=SLATE)
arrow(s, xb - 0.07, rb + bh3 * 0.36, xa + bw3 + 0.07, rb + bh3 * 0.36,
      color=SLATE)
arrow(s, xa + bw3 + 0.07, rb + bh3 * 0.68, xb - 0.07, rb + bh3 * 0.68,
      color=PINE)
text(s, xa + bw3 + 0.10, rb + bh3 * 0.72, g3 - 0.20, 0.24, "fills", size=8.5,
     color=PINE, align=PP_ALIGN.CENTER, italic=True)
arrow(s, xb + bw3 / 2, rb + bh3, xb + bw3 / 2, rc - 0.05, color=SLATE)
flow_box(s, xb, rc, bw3, bh3, "AlphaPnLTracker", "Greeks + daily CSV",
         fill=TINT_NAVY, accent=NAVY)
caption(s, rc + bh3 + 0.22,
        "Positions update only from execution fills — events, "
        "provenance, risk controls, and replayable PnL.")
footer(s, "System Architecture", "12")

# --- 17. Execution layer v2 -----------------------------------------------
s = d.slide()
title(s, "Execution Layer v2: Why It Matters")
y = table(s, M, 1.44, CW, [
    ["Area", "Before", "Now"],
    ["Hedge fills", "Hedgers could self-fill and update positions.",
     "Hedgers submit orders; execution publishes fills."],
    ["Provenance", "Alpha and hedge fills could blur.",
     "Orders and fills carry producer and order id."],
    ["Lifecycle", "One-shot fill assumption.",
     "Accepted, rejected, partial, filled, canceled reports."],
    ["Risk gates", "Mostly logging.",
     "Block, cancel, and reduce-only behaviour in the router."],
    ["Accounting", "Position updates could precede execution.",
     "Positions update only from execution fills."],
], col_w=[0.18, 0.41, 0.41], align=["l", "l", "l"], row_h=0.58)
note_bottom(s, "Research implication",
            "Once execution is explicit, turnover and transaction costs "
            "become model-selection criteria — not footnotes.",
            h=0.90, accent=CLAY)
footer(s, "System Architecture", "13")

# --- 18. Walk-forward loop ------------------------------------------------
s = d.slide()
title(s, "Walk-Forward Retraining Is First-Class")
steps = ["Daily expanding schedule", "Calibrate + train", "Export ONNX",
         "Neural + BS replay", "Balanced gates", "Promote or reject"]
bw6 = (CW - 5 * 0.34) / 6
for i, st in enumerate(steps):
    bx = M + i * (bw6 + 0.34)
    acc = PINE if i >= 4 else NAVY
    flow_box(s, bx, 1.70, bw6, 1.00, st, None,
             fill=TINT_PINE if i >= 4 else TINT_NAVY, accent=acc, size=10.5)
    if i < 5:
        arrow(s, bx + bw6 + 0.05, 2.20, bx + bw6 + 0.34 - 0.05, 2.20,
              color=SLATE)
# feedback loop back to the schedule
line(s, M + 5 * (bw6 + 0.34) + bw6 / 2, 2.70,
     M + 5 * (bw6 + 0.34) + bw6 / 2, 3.20, color=SLATE)
line(s, M + 5 * (bw6 + 0.34) + bw6 / 2, 3.20, M + bw6 / 2, 3.20, color=SLATE)
arrow(s, M + bw6 / 2, 3.20, M + bw6 / 2, 2.74, color=SLATE)
text(s, M + bw6 + 0.40, 3.24, 6.0, 0.26,
     "next deploy date — the schedule advances one day and the loop "
     "repeats", size=9.5, color=SLATE, italic=True)

split_walk_forward(s, M + 0.10, 3.86, CW - 0.20, folds=4, row_h=0.28,
                   row_gap=0.10)
legend(s, M + 1.55, 5.44, [(TRAIN, "train (expanding)"),
                           (TEST, "deploy day")])
card(s, M, 5.92, CW, 0.94, None,
     "demo/runs/walk_forward/<run_id>/windows/<deploy_date>/  stores "
     "artifacts, replay CSVs, logs, window manifests, and summary reports.",
     tint=TINT_GREY, accent=SLATE, size=11.5, bullet_body=False)
footer(s, "System Architecture", "14")

# --- 19. Promotion gates --------------------------------------------------
s = d.slide()
title(s, "Promotion Gates")
y = table(s, M, 1.44, CW, [
    ["Gate", "Checks"],
    ["Causality", "Train only on rows before the deploy date; record train "
                  "and deploy row counts."],
    ["Artifact", "Checkpoint, ONNX, normalization, Y₀ file, schema, "
                 "hashes."],
    ["ONNX parity", "PyTorch versus ONNX output agreement."],
    ["Delta sanity", "ATM model delta finite and close to BS delta."],
    ["Replay health", "Neural and BS replays complete and write a non-empty "
                      "daily CSV."],
    ["Baseline guard", "Neural hedge must not catastrophically underperform "
                       "BS Delta."],
], col_w=[0.22, 0.78], align=["l", "l"], row_h=0.62)
note_bottom(s, "Default behaviour",
            "No live artifact is overwritten unless every window passes and "
            "--promote-if-pass is explicit.", h=0.90, accent=PINE)
footer(s, "System Architecture", "15")

# --- 20. Section: Results -------------------------------------------------
section_slide(d, 4, "Results",
              "What the delta-only BSDE kept, and what full replication threw "
              "away.", total=SEC)

# --- 21. OOS four-strategy ------------------------------------------------
s = d.slide()
title(s, "OOS Four-Strategy Result",
      "Out-of-sample, 2026-01-02 → 2026-02-06")
y = table(s, M, 1.60, CW, [
    ["OOS metric", "BSDelta", "RoughVol", "BSDE-Full", "BSDE-Delta"],
    ["Hedge PnL", "1,059,605", "698,612", "290,817", "892,723"],
    ["Txn cost", "1,801", "643", "8,201", "9,005"],
    ["Total PnL", "1,060,036", "700,201", "284,847", "885,950"],
    ["vs BSDelta", "100.0%", "66.0%", "26.9%", "83.6%"],
], col_w=[0.24, 0.19, 0.19, 0.19, 0.19], row_h=0.66, highlight_col=4)
note_bottom(s, "Interpretation",
            "Full replication is theoretically clean but economically wrong "
            "for this alpha. The delta-only BSDE keeps most of the "
            "variance-premium exposure — at a transaction cost still small "
            "next to the PnL.", h=1.06, accent=CLAY)
footer(s, "Results", "16")

# --- 22. Cumulative PnL ---------------------------------------------------
s = d.slide()
title(s, "Cumulative OOS PnL")
chart_with_readout(
    s, 1.42, "Cumulative OOS PnL — 4 hedging strategies", [
        "BSDelta and the delta-only BSDE track each other closely.",
        "The full-replication BSDE spends much of the window near zero or "
        "negative.",
        "RoughDelta is smooth but systematically lower — it hedges away "
        "carry.",
    ], h=4.36, image=fig("fig1_cumulative_pnl.png"),
    head="What it says", accent=PINE)
caption(s, 6.02, "Delta-only BSDE closes at 84% of BS Delta; full "
                 "replication closes at 27%.")
footer(s, "Results", "17")

# --- 23. Greek attribution ------------------------------------------------
s = d.slide()
title(s, "Greek Attribution Explains The Difference")
chart_with_readout(
    s, 1.42, "OOS Greek attribution by hedger", [
        "Gamma and vega are the intended alpha sources.",
        "A hedge objective that erases them also erases the strategy.",
        "Transaction cost is visible but second-order next to the residual "
        "that was hedged away.",
    ], h=4.36, image=fig("fig3_greek_attribution.png"),
    head="Why the gap opens", accent=CLAY)
caption(s, 6.02, "The difference between the four bars is not hedge quality "
                 "— it is how much residual each objective left alive.")
footer(s, "Results", "18")

# --- 24. Walk-forward demo ------------------------------------------------
s = d.slide()
title(s, "Five-Day Walk-Forward Demo Result")
kpi_row(s, [("$3.415M", "Total PnL", NAVY),
            ("3.18", "Daily Sharpe", PINE),
            ("138.8", "Turnover\n(fills / day)", NAVY),
            ("0.537", "Model delta\n(BS: 0.504)", PINE)], y=1.52, h=1.44)
y = table(s, M, 3.30, CW, [
    ["Distribution", "Mean daily", "P5", "P50", "P95"],
    ["Daily PnL", "$682,958 ± $214,586", "$311K", "$690K", "$929K"],
], col_w=[0.22, 0.30, 0.16, 0.16, 0.16], row_h=0.56)
note_bottom(s, "Why walk-forward mattered",
            "The unretrained synthetic model was out-of-distribution on SPY. "
            "SPY calibration fixed the scale of V(t), strike, maturity and "
            "skew — the same network went from unusable to a 3.18 daily "
            "Sharpe with no architecture change.", h=1.10, accent=PINE)
footer(s, "Results", "19")

# --- 25. Daily distribution -----------------------------------------------
s = d.slide()
title(s, "Daily PnL Stability")
plot_holder(s, M + 1.30, 1.42, CW - 2.60, 4.62, "Daily PnL distribution",
            "dispersion, transaction costs, residual risk",
            image=fig("fig4_daily_distribution.png"))
caption(s, 6.18, "The model comparison is not only final PnL: daily "
                 "dispersion, transaction costs, and residual risk matter.")
footer(s, "Results", "20")

# --- 26. Section: Outlook -------------------------------------------------
section_slide(d, 5, "Status And Outlook",
              "What is solid, what is still research-grade, and what comes "
              "next.", total=SEC)

# --- 27. What is mature ---------------------------------------------------
s = d.slide()
title(s, "What Is Mature Now?")
compare(s, M, 1.50, CW, 3.16, [
    "Event-driven C++ replay path.",
    "Multi-strategy hedge comparison.",
    "Greek attribution and daily CSV output.",
    "ONNX inference inside replay.",
    "Walk-forward run manifests and gates.",
], [
    "Small public / local panel in the repo.",
    "No live broker connectivity.",
    "Simplified execution simulator.",
    "BSDE objective needs broader regime validation.",
    "Risk sizing and portfolio construction remain MVP-level.",
], left_head="Solid prototype pieces", right_head="Still research-grade",
    left_accent=PINE, right_accent=CLAY)
note_bottom(s, "How to read this project",
            "It is a working prototype of a research process, not a live "
            "trading system.", h=0.88, accent=NAVY)
footer(s, "Status And Outlook", "21")

# --- 28. Limitations ------------------------------------------------------
s = d.slide()
title(s, "Main Limitations")
items = [
    ("Data scale", "The research claim needs more dates, more regimes, and "
                   "stricter OOS splits."),
    ("Cost model", "Fills are replay-realistic but not yet venue- or "
                   "microstructure-realistic."),
    ("Model objective", "Delta-only wins here, but it should be stress-tested "
                        "against alternative hedge losses and residual-risk "
                        "metrics."),
    ("Rough state inference", "U-factors are retained but zeroed at inference "
                              "to avoid learned confounds."),
    ("Promotion policy", "Gates exist; production would need monitoring, "
                         "rollback, and capital / risk budgets."),
]
numbered_steps(s, M, 1.60, CW, items, size=14, gap=0.46, accent=CLAY)
footer(s, "Status And Outlook", "22")

# --- 29. Roadmap ----------------------------------------------------------
s = d.slide()
title(s, "Roadmap")
y = table(s, M, 1.44, CW, [
    ["Next step", "Goal"],
    ["More data regimes", "Test positive VRP, negative VRP, stress, quiet, "
                          "high-rate and low-rate windows."],
    ["Better replay costs", "Add latency, queue position, venue fee model, "
                            "and spread impact."],
    ["Residual-risk gate", "Evaluate hedge residual variance, not only total "
                           "PnL."],
    ["Objective sweep", "Compare full payoff, delta-only, residual variance, "
                        "and cost-aware losses."],
    ["Portfolio layer", "Move from one straddle engine to risk-budgeted "
                        "multi-expiry / multi-strike books."],
], col_w=[0.24, 0.76], align=["l", "l"], row_h=0.64)
note_bottom(s, "The next result worth having",
            "An objective sweep on more regimes is the single experiment that "
            "would most change confidence in the delta-only finding.",
            h=0.90, accent=PINE)
footer(s, "Status And Outlook", "23")

# --- 30. Takeaway ---------------------------------------------------------
s = d.slide()
title(s, "Takeaway")
note(s, M, 1.54, CW, 1.14, "Main message",
     "A neural BSDE should be used as a hedging operator inside a full "
     "trading system — not as an isolated pricing model.", accent=NAVY)
formula(s, M + 1.10, 3.06, CW - 2.20,
        "Which risks should the model hedge,\n"
        "and which residuals are the alpha?", size=20, h=1.34, tint=TINT_PINE,
        color=PINE)
bullets(s, M, 4.86, CW, 1.60, [
    "Full replication can destroy the variance premium it was meant to "
    "trade.",
    "A delta-only BSDE preserves most of the exposure — 83.6% of BS "
    "Delta's OOS PnL.",
    "Both claims survive only because replay, execution, and gates are part "
    "of the experiment.",
], size=14, gap=11)
footer(s, "Status And Outlook", "24")

# --- 31. References -------------------------------------------------------
s = d.slide()
title(s, "References")
refs = [
    "Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate "
    "Liabilities. Journal of Political Economy 81(3), 637–654.",
    "Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic "
    "Volatility. Review of Financial Studies 6(2), 327–343.",
    "Dupire, B. (1994). Pricing with a Smile. Risk 7(1), 18–20.",
    "Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. "
    "Wiley.",
    "Bergomi, L. (2015). Stochastic Volatility Modeling. CRC Press.",
    "Gatheral, J., Jaisson, T. and Rosenbaum, M. (2018). Volatility is Rough. "
    "Quantitative Finance 18(6), 933–949.",
    "Han, J., Jentzen, A. and E, W. (2018). Solving High-Dimensional PDEs "
    "Using Deep Learning. PNAS 115(34), 8505–8510.",
    "Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (2019). Deep Hedging. "
    "Quantitative Finance 19(8), 1271–1291.",
]
bullets(s, M, 1.60, CW, 5.0, refs, size=12.5, gap=22, dot=NAVY)
footer(s, "Effective Engine MVP", "25")

# --- 32. Thank you --------------------------------------------------------
closing_slide(d, "Thank You", "Questions & Discussion", "Effective Engine MVP")


# ===========================================================================
# PART 2 -- COMPONENT LIBRARY
# ===========================================================================

section_slide(d, 6, "Component Library",
              "Blank, reusable versions of every layout above — for "
              "data-analysis talks and quant interviews.", total=SEC)

# --- C1. How to use -------------------------------------------------------
s = d.slide()
title(s, "How To Use This Kit",
      "Every slide in this deck is four to eight lines of Python")
rect(s, M, 1.56, CW * 0.55, 3.16, fill=INK, radius=0.03)
code = [
    "from slidekit import *",
    "",
    "d = Deck()",
    "s = d.slide()",
    "",
    "title(s, 'Does the signal survive costs?')",
    "chart_with_readout(s, 1.5, 'Net Sharpe vs cost',",
    "                   ['Breaks even at 8bp.',",
    "                    'Live cost is 3bp.'])",
    "d.save('out.pptx')",
]
text(s, M + 0.26, 1.78, CW * 0.55 - 0.52, 2.80, code, size=11.5, font=MONO,
     color=RGBColor(0xD8, 0xE2, 0xF5), space_after=1, line_spacing=1.06)
cx = M + CW * 0.55 + 0.36
card(s, cx, 1.56, CW - CW * 0.55 - 0.36, 3.16, "What you get", [
    "One palette and type scale, applied everywhere.",
    "Placeholders that become real figures by passing image=path.",
    "Time-split geometry as data, not as hand-drawn boxes.",
    "Tables, KPI tiles, flows and verdict blocks that already line up.",
], accent=PINE, size=12)
y = table(s, M, 5.02, CW, [
    ["File", "Role"],
    ["slidekit.py", "The components. Palette at the top — change three "
                    "colours and the whole deck re-skins."],
    ["build_deck.py", "This deck. Content only; no layout maths."],
    ["fig/", "Figures. plot_holder(..., image=fig('x.png'))."],
], col_w=[0.20, 0.80], align=["l", "l"], row_h=0.44)
footer(s, "Component Library · how to use", "C1")

# --- C2. Why time splits are different ------------------------------------
s = d.slide()
title(s, "Time Splits — Why The Geometry Matters",
      "The first thing an interviewer checks is whether your split could leak")
split_holdout(s, M + 1.35, 1.72, CW - 1.35, row_h=0.50)
time_axis(s, M + 1.35, 2.44, CW - 1.35, "earliest date", "latest date",
          ticks=8)
text(s, M, 1.80, 1.25, 0.34, "Holdout", size=10.5, color=SLATE)
cw = (CW - 0.40) / 2
card(s, M, 3.20, cw, 2.10, "What a random split does wrong", [
    "A shuffled split puts tomorrow in train and today in test.",
    "Overlapping labels leak across the boundary.",
    "Cross-sectional data leaks through whole dates, not single rows.",
], tint=TINT_GREY, accent=CLAY, size=12)
card(s, M + cw + 0.40, 3.20, cw, 2.10, "What the geometry must show", [
    "Time runs one way, and train is always to the left of test.",
    "Any gap between them is deliberate, and you can name its length.",
    "The number of folds is visible, not asserted.",
], tint=TINT_PINE, accent=PINE, size=12)
note_bottom(s, "The question behind the picture",
            "“Could a row in train have been unknowable at the time the "
            "test row was scored?” If the diagram cannot answer that, "
            "redraw it.", h=0.90, accent=NAVY)
footer(s, "Component Library · time splits", "C2")

# --- C3. Walk-forward -----------------------------------------------------
s = d.slide()
title(s, "Time Splits — Walk-Forward (Expanding Window)",
      "split_walk_forward(s, x, y, w, folds=5)")
split_walk_forward(s, M, 1.68, CW, folds=5, row_h=0.34, row_gap=0.13)
time_axis(s, M + 1.35, 4.24, CW - 1.35, "earliest date", "latest date")
legend(s, M + 1.35, 4.76, [(TRAIN, "train (grows each step)"),
                           (TEST, "deploy / test window")])
cw = (CW - 0.40) / 2
card(s, M, BOTTOM - 1.56, cw, 1.56, "Use when", [
    "You retrain on a schedule and want the deployed model each day.",
    "History is short and you cannot afford to throw the early data away.",
], tint=TINT_PINE, accent=PINE, size=12)
card(s, M + cw + 0.40, BOTTOM - 1.56, cw, 1.56, "Say this out loud", [
    "“Each window trains only on rows strictly before its deploy "
    "date.”",
    "“The reported metric is the concatenation of the deploy windows.”",
], tint=TINT_GREY, accent=SLATE, size=12)
footer(s, "Component Library · time splits", "C3")

# --- C4. Rolling ----------------------------------------------------------
s = d.slide()
title(s, "Time Splits — Rolling (Sliding) Window",
      "split_rolling(s, x, y, w, folds=5, width=0.34)")
split_rolling(s, M, 1.68, CW, folds=5, row_h=0.34, row_gap=0.13)
time_axis(s, M + 1.35, 4.24, CW - 1.35, "earliest date", "latest date")
legend(s, M + 1.35, 4.76, [(TRAIN, "train (fixed length)"),
                           (TEST, "deploy / test window")])
cw = (CW - 0.40) / 2
card(s, M, BOTTOM - 1.56, cw, 1.56, "Use when", [
    "The data-generating process drifts and old regimes actively mislead.",
    "You want every fold trained on the same amount of data.",
], tint=TINT_PINE, accent=PINE, size=12)
card(s, M + cw + 0.40, BOTTOM - 1.56, cw, 1.56, "The trade-off to name", [
    "Expanding uses more data; rolling adapts faster.",
    "Choosing between them is an empirical question — show both curves.",
], tint=TINT_CLAY, accent=CLAY, size=12)
footer(s, "Component Library · time splits", "C4")

# --- C5. Purged K-fold ----------------------------------------------------
s = d.slide()
title(s, "Time Splits — Purged K-Fold With Embargo",
      "split_purged_kfold(s, x, y, w, folds=5, purge=.045, embargo=.045)")
split_purged_kfold(s, M, 1.66, CW, folds=5, row_h=0.32, row_gap=0.12)
legend(s, M + 1.35, 3.96, [(TRAIN, "train"), (TEST, "test"),
                           (PURGE, "purge"), (EMBARGO, "embargo")])
cw = (CW - 0.40) / 2
card(s, M, BOTTOM - 2.14, cw, 2.14, "What the two gaps do", [
    [{"t": "Purge: ", "b": True, "c": NAVY},
     {"t": "drop training rows whose label window overlaps the test block."}],
    [{"t": "Embargo: ", "b": True, "c": NAVY},
     {"t": "drop training rows just after the test block, where serial "
           "correlation still carries test information."}],
], tint=TINT_NAVY, accent=NAVY, size=11.5)
card(s, M + cw + 0.40, BOTTOM - 2.14, cw, 2.14,
     "Set the widths from the data", [
    "Purge width = the label horizon (a 5-day forward return needs 5 days).",
    "Embargo width ≈ the autocorrelation decay of the residual.",
    "Quote the numbers in bars, not as “some gap”.",
], tint=TINT_PINE, accent=PINE, size=11.5)
footer(s, "Component Library · time splits", "C5")

# --- C6. Grouped / date splits --------------------------------------------
s = d.slide()
title(s, "Time Splits — Grouped Date Blocks (Cross-Section)",
      "split_grouped(s, x, y, w, groups=6) — whole dates move together")
split_grouped(s, M, 1.72, CW, groups=6, row_h=0.42, row_gap=0.16)
legend(s, M + 1.35, 3.10, [(TRAIN, "train dates"), (TEST, "held-out dates")])
cw = (CW - 0.40) / 2
card(s, M, 3.70, cw, 1.86, "The unit of analysis is the date", [
    "One date contributes many rows — one per entity.",
    "If a date straddles the split, the cross-section is scored against "
    "itself.",
    "Group by date first, then apply purge and embargo between blocks.",
], tint=TINT_NAVY, accent=NAVY, size=11.5)
card(s, M + cw + 0.40, 3.70, cw, 1.86, "Metrics that fit this shape", [
    "Daily rank IC, then the IC information ratio across dates.",
    "Top-k minus bottom-k spread Sharpe.",
    "Never a pooled R² — it hides which dates carried the result.",
], tint=TINT_PINE, accent=PINE, size=11.5)
note_bottom(s, "In this repo",
            "purged_date_splits and GroupTimeSeriesSplit in rvlab.pipelines "
            "implement exactly this geometry.", h=0.86, accent=SLATE)
footer(s, "Component Library · time splits", "C6")

# --- C7. Hero chart + readout ---------------------------------------------
s = d.slide()
title(s, "Plot Holder — Hero Chart With Readout",
      "chart_with_readout(s, y, label, findings)")
chart_with_readout(s, 1.56, "primary chart", [
    "The one-sentence claim this chart supports.",
    "The second-most important thing a viewer should notice.",
    "The caveat you would rather raise yourself than be asked about.",
    "If you cannot write these three lines, the chart is not ready.",
], h=4.34, hint="pass image=fig('name.png') to drop the real figure in")
caption(s, 6.10, "Default layout for analysis slides: evidence on the left, "
                 "the claim it supports on the right.")
footer(s, "Component Library · plot holders", "C7")

# --- C8. Two-up comparison ------------------------------------------------
s = d.slide()
title(s, "Plot Holder — Two-Up Comparison",
      "plot_grid(s, x, y, w, h, ['A', 'B'], cols=2)")
plot_grid(s, M, 1.56, CW, 3.16, ["baseline", "candidate"], cols=2,
          hints=["what the current model does", "what the change does"])
cw = (CW - 0.40) / 2
card(s, M, BOTTOM - 1.56, cw, 1.56, "Rules for an honest pair", [
    "Same axes, same scale, same window — or say why not.",
    "Difference plotted directly beats two charts read side by side.",
], tint=TINT_GREY, accent=SLATE, size=11.5)
card(s, M + cw + 0.40, BOTTOM - 1.56, cw, 1.56, "Good pairings", [
    "In-sample versus out-of-sample. Before versus after costs.",
    "Model versus the naive baseline it has to beat.",
], tint=TINT_PINE, accent=PINE, size=11.5)
footer(s, "Component Library · plot holders", "C8")

# --- C9. 2x2 diagnostic grid ----------------------------------------------
s = d.slide()
title(s, "Plot Holder — Diagnostic Grid (Small Multiples)",
      "plot_grid(s, x, y, w, h, [...], cols=2)")
plot_grid(s, M, 1.56, CW * 0.66, 4.44,
          ["residuals vs fitted", "QQ / distribution",
           "residual autocorrelation", "prediction vs actual"], cols=2)
card(s, M + CW * 0.66 + 0.36, 1.56, CW * 0.34 - 0.36, 4.44,
     "The four questions", [
         "Is the error structured where it should be noise?",
         "Are the tails what the model assumed?",
         "Is there information left in the residual's own past?",
         "Does the fit hold across the range, or only in the middle?",
         "One panel per question. If a panel answers nothing, cut it.",
     ], accent=NAVY, size=12)
footer(s, "Component Library · plot holders", "C9")

# --- C10. Tearsheet -------------------------------------------------------
s = d.slide()
title(s, "Plot Holder — Strategy Tearsheet",
      "The standard backtest read-out: curve, drawdown, and the numbers")
kpi_row(s, [("00.0%", "Ann. return"), ("0.00", "Sharpe"),
            ("−00%", "Max drawdown", CLAY), ("000", "Turnover"),
            ("0 bp", "Break-even cost", PINE)], y=1.52, h=1.16,
        value_size=25)
plot_holder(s, M, 2.94, CW, 2.30, "cumulative return", "net of costs")
plot_holder(s, M, 5.36, CW, 1.20, "drawdown", "same x-axis as the curve above",
            tint=TINT_CLAY)
footer(s, "Component Library · plot holders", "C10")

# --- C11. KPI tiles -------------------------------------------------------
s = d.slide()
title(s, "Stat Tiles", "kpi_row(s, [(value, label), ...])")
kpi_row(s, [("0.084", "Rank IC"), ("1.30", "Spread Sharpe", PINE),
            ("30", "Sweep cells"), ("0", "Cells that beat carry", CLAY)],
        y=1.60, h=1.50)
kpi_row(s, [("+0.21", "Fitted skew exponent", CLAY),
            ("−0.40", "Implied by H = 0.10")], y=3.34, h=1.30,
        w=CW * 0.5 - 0.11)
card(s, M + CW * 0.5 + 0.11, 3.34, CW * 0.5 - 0.11, 1.30,
     "Pair the number with its comparison", [
         "A statistic alone invites “compared to what?”",
         "Put the benchmark in the label, or in the tile beside it.",
     ], tint=TINT_GREY, accent=SLATE, size=11.5)
note_bottom(s, "Use tiles for the numbers you want quoted back",
            "Three to five per slide. Past five they stop being headlines "
            "and become a table — at which point use a table, where the "
            "reader can compare columns.", h=1.06, accent=NAVY)
footer(s, "Component Library · blocks", "C11")

# --- C12. Claim / test / verdict ------------------------------------------
s = d.slide()
title(s, "Claim → Test → Verdict",
      "verdict_block(s, x, y, w, claim, test, verdict, passed=False)")
verdict_block(s, M, 1.60, CW,
              "Volatility is rough, so the skew exponent should be near "
              "−0.40 with H = 0.10.",
              "Fit the term structure of ATM skew on the panel; compare the "
              "exponent against the prior.",
              "Rejected. The fitted exponent is +0.21 — the wrong sign, "
              "not merely the wrong size.",
              passed=False, h=1.86)
verdict_block(s, M, 3.78, CW,
              "A planted short-term reversal should be recoverable by a "
              "linear model.",
              "Ridge on purged date splits, scored by daily rank IC against a "
              "naive reversal baseline.",
              "Confirmed. Rank IC 0.084, spread Sharpe 1.30, beating the "
              "baseline.",
              passed=True, h=1.86)
note_bottom(s, None,
            "A negative verdict presented cleanly is worth more than a "
            "positive one presented vaguely.", h=0.78, accent=SLATE)
footer(s, "Component Library · blocks", "C12")

# --- C13. Before / after --------------------------------------------------
s = d.slide()
title(s, "Before / After", "compare(s, x, y, w, h, left, right)")
compare(s, M, 1.56, CW, 2.94, [
    "The behaviour, stated plainly enough to be recognisable.",
    "The specific failure it caused.",
    "What it cost — in time, in PnL, or in trust.",
], [
    "The change, in the same terms as the left column.",
    "The failure it removes.",
    "The new cost it introduces — there is always one.",
], left_head="Before", right_head="After", left_accent=SLATE,
    right_accent=PINE)
y = table(s, M, 4.86, CW, [
    ["Keep the columns parallel", "Why"],
    ["Same number of rows, same order",
     "The reader compares by position; ragged columns break that."],
    ["Name the new cost", "A change with no downside reads as a sales "
                          "pitch, not an analysis."],
], col_w=[0.34, 0.66], align=["l", "l"], row_h=0.62)
footer(s, "Component Library · blocks", "C13")

# --- C14. Pipeline flow ---------------------------------------------------
s = d.slide()
title(s, "Pipeline Flow", "flow_row(s, x, y, w, h, steps)")
flow_row(s, M, 1.62, CW, 1.12, [
    ("Raw data", "provenance, schema"),
    ("Features", "leak-free lags"),
    ("Model", "fit on train only"),
    ("Evaluate", "purged splits"),
    ("Decide", "promote or reject"),
], gap=0.42, accent=NAVY, fill=TINT_NAVY)
flow_row(s, M, 3.16, CW, 1.12, [
    ("Ingest", "one loader, one contract"),
    ("Audit", "nulls, dupes, drift"),
    ("Split", "before anything is fitted"),
    ("Fit", "inside the fold"),
    ("Report", "with an interval"),
], gap=0.42, accent=PINE, fill=TINT_PINE)
note_bottom(s, "The ordering claim you are making",
            "A pipeline diagram is an argument that nothing downstream "
            "touched anything upstream. The split box belongs before the fit "
            "box — if it is drawn after, an interviewer will ask why, and "
            "they will be right to.", h=1.30, accent=CLAY)
footer(s, "Component Library · blocks", "C14")

# --- C15. Numbered method -------------------------------------------------
s = d.slide()
title(s, "Numbered Method Steps",
      "numbered_steps(s, x, y, w, [(head, detail), ...])")
cw = (CW - 0.50) / 2
numbered_steps(s, M, 1.58, cw, [
    ("State the estimand", "What number would settle the question, and in "
                           "what units?"),
    ("Fix the split", "Decide the geometry before looking at any score."),
    ("Build features", "Every lag and rolling window closed at t or earlier."),
    ("Fit and score", "Inside the fold. One metric, chosen in advance."),
], size=13.5, gap=0.62, accent=NAVY)
numbered_steps(s, M + cw + 0.50, 1.58, cw, [
    ("Attribute", "Which rows, dates, or features carried the result?"),
    ("Stress it", "Costs, regimes, seeds, and one deliberately hostile "
                  "subsample."),
    ("Report the interval", "A point estimate with no interval is an "
                            "anecdote."),
    ("Say what would change your mind", "The sentence that separates research "
                                        "from advocacy."),
], size=13.5, gap=0.62, accent=PINE)
footer(s, "Component Library · blocks", "C15")

# --- C16. Comparison table ------------------------------------------------
s = d.slide()
title(s, "Model Comparison Table",
      "table(s, x, y, w, rows, highlight=2)")
y = table(s, M, 1.58, CW, [
    ["Model", "OOS metric", "vs baseline", "Turnover", "Verdict"],
    ["Baseline (carry)", "—", "100%", "low", "reference"],
    ["Candidate A", "—", "—", "—", "promoted"],
    ["Candidate B", "—", "—", "—", "rejected"],
    ["Candidate C", "—", "—", "—", "needs more data"],
], col_w=[0.26, 0.18, 0.18, 0.16, 0.22],
    align=["l", "r", "r", "r", "l"], row_h=0.58, highlight=2,
    highlight_tint=TINT_PINE)
cw = (CW - 0.40) / 2
card(s, M, BOTTOM - 1.94, cw, 1.94, "Column order is an argument", [
    "Baseline first, so every later row is read as a difference.",
    "Cost or turnover before the verdict — it is usually why the verdict "
    "went the way it did.",
    "Tint the row you want remembered; leave the others plain.",
], tint=TINT_NAVY, accent=NAVY, size=11.5)
card(s, M + cw + 0.40, BOTTOM - 1.94, cw, 1.94, "What gets asked next", [
    "“What is the baseline, exactly?” Have the definition ready.",
    "“Is that difference inside the noise?” Have the interval "
    "ready.",
    "“How many models did you try?” Have the honest count ready.",
], tint=TINT_CLAY, accent=CLAY, size=11.5)
footer(s, "Component Library · blocks", "C16")

# --- C17. Palette & type --------------------------------------------------
s = d.slide()
title(s, "Palette And Type Scale",
      "Change these six values in slidekit.py and the deck re-skins")
sw = (CW - 5 * 0.24) / 6
for i, (col, name, hexs) in enumerate([
        (NAVY, "NAVY", "191970"), (PINE, "PINE", "01796F"),
        (CLAY, "CLAY", "B85042"), (INK, "INK", "0F1633"),
        (SLATE, "SLATE", "5A6472"), (RULE, "RULE", "D3DAE6")]):
    sx = M + i * (sw + 0.24)
    rect(s, sx, 1.58, sw, 0.90, fill=col, radius=0.06)
    text(s, sx, 2.56, sw, 0.24, name, size=11, bold=True, color=BLACK,
         align=PP_ALIGN.CENTER, font=HEAD)
    text(s, sx, 2.80, sw, 0.24, hexs, size=9.5, color=SLATE,
         align=PP_ALIGN.CENTER, font=MONO)
y = table(s, M, 3.34, CW * 0.56, [
    ["Element", "Size", "Font"],
    ["Slide title", "29 pt bold", "Cambria"],
    ["Standfirst", "13.5 pt italic", "Calibri"],
    ["Card header", "13 pt bold", "Cambria"],
    ["Body / bullets", "12.5 pt", "Calibri"],
    ["Stat tile value", "30 pt bold", "Cambria"],
    ["Caption / footer", "10.5 / 9.5 pt", "Calibri"],
], col_w=[0.44, 0.32, 0.24], align=["l", "l", "l"], row_h=0.38)
card(s, M + CW * 0.56 + 0.40, 3.34, CW * 0.44 - 0.40, y - 3.34,
     "Why these two fonts", [
         "Both ship with Office, on Windows and macOS.",
         "Both have metric-compatible substitutes on Linux, so a PDF "
         "rendered anywhere keeps its line breaks.",
         "Serif headings, sans body — contrast without a third "
         "typeface.",
     ], tint=TINT_GREY, accent=SLATE, size=11.5)
footer(s, "Component Library · reference", "C17")

# --- Closing --------------------------------------------------------------
closing_slide(d, "Component Library",
              "Copy a slide. Change the words. Keep the geometry.",
              "slidekit.py  ·  build_deck.py")


out = os.path.join(HERE, "effective_engine.pptx")
d.save(out)
print(f"wrote {out}  ({len(d.prs.slides.__iter__.__self__._sldIdLst)} slides)")
