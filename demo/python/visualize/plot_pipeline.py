"""
plot_pipeline.py — Variance Alpha Pipeline Visualization Suite
==============================================================

Generates 4 publication-quality figures from the alpha_pnl_test_runner output.

Usage:
    # From demo/build/ (after running alpha_pnl_test_runner):
    python ../python/visualize/plot_pipeline.py

    # From repo root:
    python demo/python/visualize/plot_pipeline.py \
        --results demo/build/results \
        --data    demo/data/spy_chain_panel.csv \
        --outdir  demo/build/results/figures

Output figures:
    fig1_cumulative_pnl.png     — Cumulative PnL curves (4 strategies, OOS)
    fig2_market_context.png     — SPY spot, IV vs RV, VRP, smile structure (127 days)
    fig3_greek_attribution.png  — Stacked bar of Greek contributions per strategy
    fig4_daily_distribution.png — Daily PnL box-plot + scatter (risk profile)
"""

import argparse
import os
import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import datetime

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "BSDelta":      "#2563EB",  # blue
    "RoughDelta":   "#F59E0B",  # amber
    "BSDE-IS-Full": "#EF4444",  # red
    "BSDE-IS-Δonly":"#10B981",  # emerald
    "BSDE-Synth":   "#8B5CF6",  # purple
}
LINESTYLES = {
    "BSDelta":       "-",
    "RoughDelta":    "--",
    "BSDE-IS-Full":  ":",
    "BSDE-IS-Δonly": "-",
    "BSDE-Synth":    "-.",
}
STRATEGY_ORDER = ["BSDelta", "RoughDelta", "BSDE-IS-Full", "BSDE-IS-Δonly"]

IS_SPLIT = "2026-01-02"  # first OOS date


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_oos_csv(results_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "BSDelta":       "oos_bs_delta.csv",
        "RoughDelta":    "oos_rough_delta.csv",
        "BSDE-IS-Full":  "oos_bsde_is_full.csv",
        "BSDE-IS-Δonly": "oos_bsde_is_delta.csv",
        "BSDE-Synth":    "oos_bsde_synth.csv",
    }
    data = {}
    for label, fname in files.items():
        path = results_dir / fname
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df["cum_pnl"] = df["total_pnl"].cumsum()
            data[label] = df
        else:
            print(f"  [warn] {fname} not found, skipping {label}", file=sys.stderr)
    return data


def load_market_csv(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    # Column names may vary; normalise
    df.columns = [c.strip().lower() for c in df.columns]
    # Timestamp is the first column
    ts_col = df.columns[0]
    df["date"] = pd.to_datetime(df[ts_col]).dt.normalize()
    # Take one row per day (last bar of day)
    daily = df.groupby("date").last().reset_index()

    # Map expected columns with fallback
    def get_col(candidates, default=None):
        for c in candidates:
            if c in daily.columns:
                return daily[c]
        return pd.Series([default] * len(daily)) if default is not None else None

    daily["spot"]        = get_col(["underlying_price", "underlying", "spot", "close"])
    daily["atm_iv"]      = get_col(["atm_iv", "iv"])
    daily["rv5"]         = get_col(["rv5_ann", "rv5", "realized_vol"])
    daily["rr25_iv"]     = get_col(["rr25_iv", "rr25"])
    daily["bf25_iv"]     = get_col(["bf25_iv", "bf25"])
    daily["vix_varswap"] = get_col(["vix_varswap"])
    daily["ssvi_rho"]    = get_col(["ssvi_rho"])
    daily["ssvi_phi"]    = get_col(["ssvi_phi"])
    return daily


# ── Figure 1: Cumulative PnL ──────────────────────────────────────────────────

def plot_cumulative_pnl(data: dict, outdir: Path):
    fig, ax = plt.subplots(figsize=(11, 6))

    for label in STRATEGY_ORDER:
        if label not in data:
            continue
        df = data[label]
        ax.plot(df["date"], df["cum_pnl"] / 1e3,
                color=COLORS[label],
                linestyle=LINESTYLES[label],
                linewidth=2.0,
                label=label)
        # Annotate final value
        final = df["cum_pnl"].iloc[-1]
        ax.annotate(f"${final/1e3:,.0f}K",
                    xy=(df["date"].iloc[-1], final / 1e3),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=COLORS[label], va="center")

    # Reference line
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_title("Cumulative OOS PnL — 4 Hedging Strategies (2026-01-02 → 2026-02-06)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($K)")
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Pct-of-BSDelta annotation
    if "BSDelta" in data and "BSDE-IS-Δonly" in data:
        bs_total = data["BSDelta"]["cum_pnl"].iloc[-1]
        for label in ["RoughDelta", "BSDE-IS-Full", "BSDE-IS-Δonly"]:
            if label in data:
                pct = data[label]["cum_pnl"].iloc[-1] / bs_total * 100
                ax.annotate(f"({pct:.0f}% of BS)",
                            xy=(data[label]["date"].iloc[-2],
                                data[label]["cum_pnl"].iloc[-2] / 1e3),
                            xytext=(0, -15), textcoords="offset points",
                            fontsize=8, color=COLORS[label], ha="center")

    fig.tight_layout()
    out = outdir / "fig1_cumulative_pnl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Market Context (2×2) ───────────────────────────────────────────

def plot_market_context(daily: pd.DataFrame, outdir: Path):
    n_days = len(daily)
    date_range = (f"{daily['date'].iloc[0].strftime('%Y-%m-%d')} → "
                  f"{daily['date'].iloc[-1].strftime('%Y-%m-%d')}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"SPY Market Context — {n_days} Trading Days ({date_range})",
                 fontsize=13, fontweight="bold")

    split_dt = pd.Timestamp(IS_SPLIT)

    # Normalise dates to tz-naive for comparison
    first_day = daily["date"].iloc[0].tz_localize(None) if daily["date"].iloc[0].tzinfo else daily["date"].iloc[0]
    last_day  = daily["date"].iloc[-1].tz_localize(None) if daily["date"].iloc[-1].tzinfo else daily["date"].iloc[-1]

    def add_split(ax):
        ylim = ax.get_ylim()
        if first_day <= split_dt <= last_day:
            ax.axvline(split_dt, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.text(split_dt, ylim[1] * 0.97, " OOS →",
                    fontsize=8, color="gray", va="top")

    # [0,0] SPY spot
    ax = axes[0, 0]
    ax.plot(daily["date"], daily["spot"], color="#1E40AF", linewidth=1.5)
    ax.set_title("SPY Spot Price")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    add_split(ax)

    # [0,1] ATM IV vs RV5
    ax = axes[0, 1]
    if daily["atm_iv"].notna().any():
        ax.plot(daily["date"], daily["atm_iv"] * 100, color="#2563EB",
                linewidth=1.5, label="ATM IV (σ)")
    if daily["rv5"].notna().any():
        ax.plot(daily["date"], daily["rv5"] * 100, color="#F59E0B",
                linewidth=1.5, linestyle="--", label="5-bar RV (ann.)")
    ax.set_title("Implied vs Realized Vol")
    ax.set_ylabel("Vol (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    add_split(ax)

    # [0,2] VIX σ²_varswap vs ATM IV² — model-free vs single-strike comparison
    ax = axes[0, 2]
    has_vix = daily["vix_varswap"].notna().any()
    has_atm = daily["atm_iv"].notna().any()
    if has_vix:
        vix_vol = (daily["vix_varswap"].clip(lower=0) ** 0.5) * 100
        ax.plot(daily["date"], vix_vol, color="#EF4444",
                linewidth=1.5, label="VIX σ (model-free)")
    if has_atm:
        ax.plot(daily["date"], daily["atm_iv"] * 100, color="#2563EB",
                linewidth=1.5, linestyle="--", label="ATM IV (BS)")
    if has_vix and has_atm:
        spread = vix_vol - daily["atm_iv"] * 100
        ax2 = ax.twinx()
        ax2.fill_between(daily["date"], spread, 0,
                         where=(spread >= 0), alpha=0.15, color="#EF4444",
                         label="VIX > ATM (smile premium)")
        ax2.fill_between(daily["date"], spread, 0,
                         where=(spread < 0), alpha=0.15, color="#2563EB")
        ax2.set_ylabel("VIX − ATM (%)", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=7, colors="gray")
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title("VIX (model-free) vs ATM IV")
    ax.set_ylabel("Vol (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    add_split(ax)

    # [1,0] VRP = IV - RV
    ax = axes[1, 0]
    if daily["atm_iv"].notna().any() and daily["rv5"].notna().any():
        vrp = (daily["atm_iv"] - daily["rv5"]) * 100
        bar_colors = ["#10B981" if v >= 0 else "#EF4444" for v in vrp]
        ax.bar(daily["date"], vrp, color=bar_colors, width=1.0, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title("Vol Risk Premium (IV − RV5)")
        ax.set_ylabel("VRP (%)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
        add_split(ax)
    else:
        ax.text(0.5, 0.5, "VRP data unavailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Vol Risk Premium (IV − RV5)")

    # [1,1] 25Δ smile structure (skew + curvature)
    ax = axes[1, 1]
    plotted = False
    if daily["rr25_iv"].notna().any():
        ax.plot(daily["date"], daily["rr25_iv"] * 100, color="#7C3AED",
                linewidth=1.5, label="RR25 (skew)")
        plotted = True
    if daily["bf25_iv"].notna().any():
        ax.plot(daily["date"], daily["bf25_iv"] * 100, color="#DB2777",
                linewidth=1.5, linestyle="--", label="BF25 (curvature)")
        plotted = True
    if plotted:
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title("25Δ Smile Structure")
        ax.set_ylabel("Vol (%)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Smile data (rr25, bf25)\nnot in CSV",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("25Δ Smile Structure")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    add_split(ax)

    # [1,2] SSVI parameters — ρ (skew shape) and φ (wing steepness) over time
    ax = axes[1, 2]
    has_rho = daily["ssvi_rho"].notna().any()
    has_phi = daily["ssvi_phi"].notna().any()
    if has_rho:
        ax.plot(daily["date"], daily["ssvi_rho"], color="#7C3AED",
                linewidth=1.5, label="ρ (skew, left axis)")
        ax.set_ylabel("ρ (skew)", color="#7C3AED")
        ax.tick_params(axis="y", colors="#7C3AED")
    if has_phi:
        ax3 = ax.twinx()
        ax3.plot(daily["date"], daily["ssvi_phi"], color="#059669",
                 linewidth=1.5, linestyle="--", label="φ (wings, right axis)")
        ax3.set_ylabel("φ (wing steepness)", color="#059669", fontsize=9)
        ax3.tick_params(axis="y", colors="#059669")
    if has_rho or has_phi:
        # Combined legend
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = (ax3.get_legend_handles_labels() if has_phi else ([], []))
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower right")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    else:
        ax.text(0.5, 0.5, "SSVI data not in CSV",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_title("SSVI Smile Parameters (ρ, φ)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    add_split(ax)

    fig.tight_layout()
    out = outdir / "fig2_market_context.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Greek Attribution (stacked bars) ────────────────────────────────

def plot_greek_attribution(data: dict, outdir: Path):
    labels_to_plot = [l for l in STRATEGY_ORDER if l in data]
    if not labels_to_plot:
        print("  [warn] No data for Greek attribution figure", file=sys.stderr)
        return

    greek_cols = ["gamma_pnl", "vega_pnl", "theta_pnl", "vanna_pnl", "volga_pnl", "delta_hedge_pnl", "txn_cost"]
    greek_labels = ["Γ (gamma)", "ν (vega)", "θ (theta)", "Vanna", "Volga", "Hedge PnL", "Txn Cost"]
    greek_colors = ["#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#10B981", "#6B7280"]

    # Aggregate totals per strategy
    totals = {}
    for label in labels_to_plot:
        df = data[label]
        totals[label] = {col: df[col].sum() for col in greek_cols if col in df.columns}
        # txn_cost is a cost — negate for display
        if "txn_cost" in totals[label]:
            totals[label]["txn_cost"] = -abs(totals[label]["txn_cost"])

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels_to_plot))
    width = 0.5

    bottoms_pos = np.zeros(len(labels_to_plot))
    bottoms_neg = np.zeros(len(labels_to_plot))

    for col, glabel, gcolor in zip(greek_cols, greek_labels, greek_colors):
        vals = np.array([totals[l].get(col, 0.0) for l in labels_to_plot]) / 1e3
        pos_vals = np.where(vals >= 0, vals, 0.0)
        neg_vals = np.where(vals < 0,  vals, 0.0)
        ax.bar(x, pos_vals, width, bottom=bottoms_pos, color=gcolor, alpha=0.85, label=glabel)
        ax.bar(x, neg_vals, width, bottom=bottoms_neg, color=gcolor, alpha=0.85)
        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    # Total PnL line
    total_vals = np.array([data[l]["total_pnl"].sum() / 1e3 for l in labels_to_plot])
    ax.scatter(x, total_vals, s=60, zorder=5, color="black", label="Total PnL", marker="D")
    for xi, tv, label in zip(x, total_vals, labels_to_plot):
        ax.annotate(f"${tv:,.0f}K", xy=(xi, tv),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=9, ha="center", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_to_plot, fontsize=11)
    ax.set_ylabel("PnL ($K)")
    ax.set_title("OOS Greek Attribution by Hedger Strategy", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = outdir / "fig3_greek_attribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Daily PnL Distribution (box + scatter) ─────────────────────────

def plot_daily_distribution(data: dict, outdir: Path):
    labels_to_plot = [l for l in STRATEGY_ORDER if l in data]
    if not labels_to_plot:
        print("  [warn] No data for distribution figure", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    bp_data = [data[l]["total_pnl"].values / 1e3 for l in labels_to_plot]

    bp = ax.boxplot(bp_data,
                    patch_artist=True,
                    widths=0.4,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for patch, label in zip(bp["boxes"], labels_to_plot):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.7)

    # Overlay individual day dots
    for i, (label, vals) in enumerate(zip(labels_to_plot, bp_data), start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=COLORS[label], alpha=0.55, s=18, zorder=3)

    # Annotate stats below each box
    y_bottom = min(v.min() for v in bp_data)
    y_range  = max(v.max() for v in bp_data) - y_bottom
    y_label  = y_bottom - y_range * 0.18
    for i, (label, vals) in enumerate(zip(labels_to_plot, bp_data), start=1):
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        sharpe = mean_v / std_v if std_v > 0 else 0
        ax.text(i, y_label,
                f"μ={mean_v:.0f}\nσ={std_v:.0f}\nSh={sharpe:.2f}",
                ha="center", va="top", fontsize=8, color=COLORS[label])

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(range(1, len(labels_to_plot) + 1))
    ax.set_xticklabels(labels_to_plot, fontsize=11)
    ax.set_ylabel("Daily PnL ($K)")
    ax.set_title("OOS Daily PnL Distribution — Risk Profile Comparison",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = outdir / "fig4_daily_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Absolute paths derived from this script's location, so the script works
    # regardless of the current working directory.
    _script_dir = Path(__file__).resolve().parent          # demo/python/visualize/
    _demo_dir   = _script_dir.parent.parent                # demo/
    _build_dir  = _demo_dir / "build"

    parser = argparse.ArgumentParser(description="Plot variance alpha pipeline results")
    parser.add_argument("--results", default=str(_build_dir / "results"),
                        help="Directory with oos_*.csv files")
    parser.add_argument("--data",    default=str(_demo_dir / "data" / "spy_chain_panel.csv"),
                        help="Path to spy_chain_panel.csv")
    parser.add_argument("--outdir",  default=str(_build_dir / "results" / "figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    data_path   = Path(args.data)
    outdir      = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nVariance Alpha Pipeline — Visualization Suite")
    print(f"  Results dir : {results_dir.resolve()}")
    print(f"  Market data : {data_path.resolve()}")
    print(f"  Output dir  : {outdir.resolve()}\n")

    # Load OOS simulation results (optional — figs 1/3/4 are skipped if absent)
    print("Loading OOS CSV files...")
    oos_data = load_oos_csv(results_dir)
    if not oos_data:
        print("  [info] No OOS CSV files found — skipping figs 1, 3, 4 (run alpha_pnl_test_runner to generate them)")
    else:
        print(f"  Loaded {len(oos_data)} strategies: {list(oos_data.keys())}\n")

    # Load market data
    if data_path.exists():
        print("Loading market data...")
        market = load_market_csv(data_path)
        print(f"  {len(market)} trading days, {len(market.columns)} columns\n")
    else:
        print(f"  [warn] Market data not found at {data_path}, skipping Fig 2\n")
        market = None

    # Generate figures
    print("Generating figures...")
    if oos_data:
        plot_cumulative_pnl(oos_data, outdir)
    else:
        print("  Skipping fig1 (no OOS data)")
    if market is not None:
        plot_market_context(market, outdir)
    else:
        print("  Skipping fig2 (no market data)")
    if oos_data:
        plot_greek_attribution(oos_data, outdir)
        plot_daily_distribution(oos_data, outdir)
    else:
        print("  Skipping figs 3, 4 (no OOS data)")

    print(f"\nAll figures saved to: {outdir.resolve()}/")
    print("Done.")


if __name__ == "__main__":
    main()
