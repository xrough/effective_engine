#!/usr/bin/env python3
"""Validate lab dataset contracts against local CSV inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parents[3] / "lab"
REPO_ROOT = LAB_DIR.parent

POSITIVE_COLUMNS = {
    "underlying_price",
    "atm_strike",
    "time_to_expiry",
    "call_mid",
    "put_mid",
    "call_bid",
    "call_ask",
    "put_bid",
    "put_ask",
    "atm_iv",
    "call25d_strike",
    "call25d_mid",
    "call25d_bid",
    "call25d_ask",
    "put25d_strike",
    "put25d_mid",
    "put25d_bid",
    "put25d_ask",
    "vix_varswap",
    "ssvi_phi",
}
NONNEGATIVE_COLUMNS = {"rv5_ann"}
FINITE_COLUMNS = {
    "rr25_iv",
    "bf25_iv",
    "ssvi_rho",
}
BID_MID_ASK_GROUPS = [
    ("call_bid", "call_mid", "call_ask"),
    ("put_bid", "put_mid", "put_ask"),
    ("call25d_bid", "call25d_mid", "call25d_ask"),
    ("put25d_bid", "put25d_mid", "put25d_ask"),
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp is timezone-naive")
    return dt.astimezone(timezone.utc)


def to_float(row: dict[str, str], col: str) -> float | None:
    value = row.get(col, "")
    if value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, **extra: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, **extra})


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def validate_columns(dataset: dict[str, Any], fieldnames: list[str], checks: list[dict[str, Any]]) -> None:
    required = list(dataset.get("required_columns", []))
    missing = [col for col in required if col not in fieldnames]
    add_check(
        checks,
        "required_columns",
        not missing,
        "ok" if not missing else "missing: " + ", ".join(missing),
        required_count=len(required),
        actual_count=len(fieldnames),
    )


def validate_timestamps(rows: list[dict[str, str]], checks: list[dict[str, Any]]) -> tuple[list[datetime], Counter[str]]:
    timestamps: list[datetime] = []
    bad = 0
    for row in rows:
        try:
            timestamps.append(parse_timestamp(row.get("timestamp_utc", "")))
        except Exception:
            bad += 1

    add_check(checks, "timestamp_parse_utc", bad == 0 and bool(rows), f"bad={bad}, rows={len(rows)}")
    if not timestamps:
        return timestamps, Counter()

    monotonic = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
    add_check(checks, "timestamp_monotonic", monotonic, "ok" if monotonic else "timestamps are not nondecreasing")

    duplicate_count = len(timestamps) - len(set(timestamps))
    add_check(checks, "timestamp_unique", duplicate_count == 0, f"duplicates={duplicate_count}")

    dates = Counter(ts.strftime("%Y-%m-%d") for ts in timestamps)
    add_check(
        checks,
        "date_summary_available",
        bool(dates),
        f"{min(dates)} to {max(dates)}, {len(dates)} trading dates",
    )
    return timestamps, dates


def validate_expected_shape(
    dataset: dict[str, Any],
    rows: list[dict[str, str]],
    dates: Counter[str],
    checks: list[dict[str, Any]],
) -> None:
    expected_rows = dataset.get("row_count")
    if expected_rows is not None:
        add_check(checks, "row_count_matches_registry", len(rows) == int(expected_rows), f"actual={len(rows)}, expected={expected_rows}")

    expected_range = dataset.get("date_range") or {}
    expected_start = expected_range.get("start")
    expected_end = expected_range.get("end")
    expected_n_dates = expected_range.get("trading_dates")
    if dates and expected_start is not None:
        add_check(checks, "start_date_matches_registry", min(dates) == expected_start, f"actual={min(dates)}, expected={expected_start}")
    if dates and expected_end is not None:
        add_check(checks, "end_date_matches_registry", max(dates) == expected_end, f"actual={max(dates)}, expected={expected_end}")
    if dates and expected_n_dates is not None:
        add_check(checks, "trading_date_count_matches_registry", len(dates) == int(expected_n_dates), f"actual={len(dates)}, expected={expected_n_dates}")


def validate_numeric_columns(fieldnames: list[str], rows: list[dict[str, str]], checks: list[dict[str, Any]]) -> None:
    numeric_cols = sorted((POSITIVE_COLUMNS | NONNEGATIVE_COLUMNS | FINITE_COLUMNS).intersection(fieldnames))
    for col in numeric_cols:
        bad = 0
        for row in rows:
            value = to_float(row, col)
            if value is None:
                bad += 1
                continue
            if col in POSITIVE_COLUMNS and value <= 0.0:
                bad += 1
            if col in NONNEGATIVE_COLUMNS and value < 0.0:
                bad += 1
        add_check(checks, f"numeric:{col}", bad == 0, f"bad={bad}, rows={len(rows)}")


def validate_quote_bounds(fieldnames: list[str], rows: list[dict[str, str]], checks: list[dict[str, Any]]) -> None:
    for bid_col, mid_col, ask_col in BID_MID_ASK_GROUPS:
        if not {bid_col, mid_col, ask_col}.issubset(fieldnames):
            continue
        bad = 0
        for row in rows:
            bid = to_float(row, bid_col)
            mid = to_float(row, mid_col)
            ask = to_float(row, ask_col)
            if bid is None or mid is None or ask is None or not (bid <= mid <= ask):
                bad += 1
        add_check(checks, f"quote_bounds:{mid_col}", bad == 0, f"bad={bad}, rows={len(rows)}")


def build_walk_forward_windows(dates: Counter[str], min_train_days: int) -> list[dict[str, Any]]:
    ordered = sorted(dates)
    windows: list[dict[str, Any]] = []
    for idx, deploy_date in enumerate(ordered):
        if idx < min_train_days:
            continue
        train_dates = ordered[:idx]
        train_rows = sum(dates[d] for d in train_dates)
        deploy_rows = dates[deploy_date]
        windows.append(
            {
                "deploy_date": deploy_date,
                "train_start": train_dates[0] if train_dates else None,
                "train_end_exclusive": deploy_date,
                "train_dates": train_dates,
                "train_rows": int(train_rows),
                "deploy_rows": int(deploy_rows),
                "causal": all(d < deploy_date for d in train_dates),
            }
        )
    return windows


def validate_walk_forward(dates: Counter[str], min_train_days: int, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = build_walk_forward_windows(dates, min_train_days)
    add_check(checks, "walk_forward_windows_available", bool(windows), f"windows={len(windows)}, min_train_days={min_train_days}")
    causal = all(window["causal"] for window in windows)
    add_check(checks, "walk_forward_causality", causal, "ok" if causal else "training date leakage detected")
    return windows


def validate_dataset(dataset: dict[str, Any], min_train_days: int) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    path = (REPO_ROOT / dataset["path"]).resolve()
    exists = path.exists()
    add_check(checks, "file_exists", exists, str(path))
    if not exists:
        return {
            "dataset_id": dataset["dataset_id"],
            "asset": dataset.get("asset"),
            "path": str(path),
            "passed": False,
            "checks": checks,
            "summary": {},
            "walk_forward_windows": [],
        }

    fieldnames, rows = read_rows(path)
    add_check(checks, "non_empty_csv", bool(rows), f"rows={len(rows)}")
    validate_columns(dataset, fieldnames, checks)
    timestamps, dates = validate_timestamps(rows, checks)
    validate_expected_shape(dataset, rows, dates, checks)
    validate_numeric_columns(fieldnames, rows, checks)
    validate_quote_bounds(fieldnames, rows, checks)
    windows = validate_walk_forward(dates, min_train_days, checks)

    by_date = [{"date": date, "rows": int(count)} for date, count in sorted(dates.items())]
    summary = {
        "rows": len(rows),
        "columns": len(fieldnames),
        "start_date": min(dates) if dates else None,
        "end_date": max(dates) if dates else None,
        "trading_dates": len(dates),
        "rows_by_date": by_date,
        "first_timestamp_utc": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp_utc": timestamps[-1].isoformat() if timestamps else None,
    }

    return {
        "dataset_id": dataset["dataset_id"],
        "asset": dataset.get("asset"),
        "path": str(path),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "summary": summary,
        "walk_forward_windows": windows,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# Data Contract Validation: {result['created_at']}",
        "",
        f"- Registry: `{result['registry']}`",
        f"- Datasets: {result['n_passed']} / {result['n_datasets']} passed",
        f"- Overall: `{'PASS' if result['passed'] else 'FAIL'}`",
        "",
        "| Dataset | Asset | Rows | Dates | Windows | Status |",
        "|---|---|---:|---|---:|---|",
    ]
    for dataset in result["datasets"]:
        summary = dataset.get("summary", {})
        dates = f"{summary.get('start_date')} -> {summary.get('end_date')}"
        lines.append(
            "| {dataset_id} | {asset} | {rows} | {dates} | {windows} | {status} |".format(
                dataset_id=dataset["dataset_id"],
                asset=dataset.get("asset") or "",
                rows=summary.get("rows", 0),
                dates=dates,
                windows=len(dataset.get("walk_forward_windows", [])),
                status="PASS" if dataset["passed"] else "FAIL",
            )
        )

    lines.extend(["", "## Checks", ""])
    for dataset in result["datasets"]:
        lines.append(f"### {dataset['dataset_id']}")
        lines.append("")
        lines.append("| Check | Status | Detail |")
        lines.append("|---|---|---|")
        for check in dataset["checks"]:
            lines.append(f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate lab dataset contracts.")
    parser.add_argument("--registry", default=str(LAB_DIR / "registry" / "datasets.json"))
    parser.add_argument("--dataset-id", action="append")
    parser.add_argument("--output-json", default=str(LAB_DIR / "reports" / "data_contract_validation.json"))
    parser.add_argument("--output-md")
    parser.add_argument("--min-train-days", type=int, default=3)
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser().resolve()
    registry = load_json(registry_path)
    selected = set(args.dataset_id or [])
    datasets = [
        dataset for dataset in registry.get("datasets", [])
        if not selected or dataset.get("dataset_id") in selected
    ]
    if selected and len(datasets) != len(selected):
        known = {dataset.get("dataset_id") for dataset in registry.get("datasets", [])}
        missing = sorted(selected - known)
        raise SystemExit(f"Unknown dataset id(s): {', '.join(missing)}")

    results = [validate_dataset(dataset, args.min_train_days) for dataset in datasets]
    payload = {
        "created_at": utc_stamp(),
        "registry": str(registry_path),
        "min_train_days": args.min_train_days,
        "n_datasets": len(results),
        "n_passed": sum(1 for dataset in results if dataset["passed"]),
        "passed": all(dataset["passed"] for dataset in results),
        "datasets": results,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, payload)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), payload)

    print(
        f"[data-contracts] datasets={payload['n_passed']}/{payload['n_datasets']} "
        f"status={'PASS' if payload['passed'] else 'FAIL'}"
    )
    print(f"[data-contracts] wrote {output_json}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
