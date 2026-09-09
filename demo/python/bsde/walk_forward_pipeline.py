"""
walk_forward_pipeline.py - first-class walk-forward retraining pipeline.

This orchestrates the existing SPY calibration/retraining flow into a
versioned experiment run:

  schedule -> train/export -> C++ replay -> gates -> optional promotion

The heavy model work stays in calibrate_and_retrain.py. This file owns the
experiment shape, manifests, reports, replay calls, and promotion policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


BSDE_DIR = Path(__file__).resolve().parent
DEMO_ROOT = BSDE_DIR.parents[1]
REPO_ROOT = BSDE_DIR.parents[2]

PROFILE_DEFAULTS = {
    "smoke": {"epochs": 1, "n_syn": 256, "n_steps": 12, "onnx_validation_atol": 1e-4},
    "full": {"epochs": 100, "n_syn": 9500, "n_steps": 50, "onnx_validation_atol": 1e-5},
}

REQUIRED_ARTIFACTS = [
    "neural_bsde.onnx",
    "normalization.json",
    "Y0_init.json",
    "checkpoints/best.pt",
]

PROMOTED_ARTIFACTS = [
    "neural_bsde.onnx",
    "normalization.json",
    "Y0_init.json",
]


@dataclass(frozen=True)
class WindowSpec:
    deploy_date: str
    train_end: str
    train_dates: list[str]
    train_rows: int
    deploy_rows: int


@dataclass(frozen=True)
class PipelineConfig:
    csv_path: Path
    run_id: str
    run_root: Path
    start_date: str | None
    end_date: str | None
    min_train_days: int
    profile: str
    epochs: int
    n_syn: int
    n_steps: int
    seed: int
    promote: bool
    build_dir: Path
    live_artifacts: Path
    delta_tolerance: float = 0.15
    baseline_guard_abs: float = 250_000.0
    baseline_guard_rel: float = 1.0
    onnx_validation_atol: float = 1e-5


RetrainFn = Callable[..., dict[str, Any]]
ReplayFn = Callable[..., dict[str, Any]]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(artifacts_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in PROMOTED_ARTIFACTS:
        path = artifacts_dir / rel
        if path.exists():
            hashes[rel] = sha256_file(path)
    return hashes


def load_panel_dates(csv_path: Path) -> tuple[list[str], dict[str, int]]:
    df = pd.read_csv(csv_path, usecols=["timestamp_utc"])
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    dates = ts.dt.strftime("%Y-%m-%d")
    counts = dates.value_counts().sort_index().to_dict()
    return sorted(counts), {str(k): int(v) for k, v in counts.items()}


def build_walk_forward_schedule(
    csv_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    min_train_days: int = 3,
) -> list[WindowSpec]:
    if min_train_days < 1:
        raise ValueError("min_train_days must be >= 1")

    dates, counts = load_panel_dates(csv_path)
    windows: list[WindowSpec] = []
    for i, deploy_date in enumerate(dates):
        if i < min_train_days:
            continue
        if start_date and deploy_date < start_date:
            continue
        if end_date and deploy_date > end_date:
            continue

        train_dates = dates[:i]
        train_rows = sum(counts[d] for d in train_dates)
        deploy_rows = counts[deploy_date]
        windows.append(
            WindowSpec(
                deploy_date=deploy_date,
                train_end=deploy_date,
                train_dates=train_dates,
                train_rows=train_rows,
                deploy_rows=deploy_rows,
            )
        )
    return windows


def validate_normalization(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "normalization.json missing"

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        return False, f"normalization.json unreadable: {exc}"

    required = {"feature_order", "mean", "std", "state_dim", "m", "K", "T", "r", "model_params"}
    missing = sorted(required - set(data))
    if missing:
        return False, f"missing keys: {', '.join(missing)}"

    state_dim = data.get("state_dim")
    mean = data.get("mean", [])
    std = data.get("std", [])
    if state_dim != 7 or len(mean) != 7 or len(std) != 7:
        return False, "expected 7D state normalization"

    numeric = list(mean) + list(std) + [data.get("K"), data.get("T"), data.get("r")]
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in numeric):
        return False, "normalization contains non-finite values"

    if any(float(x) <= 0.0 for x in std):
        return False, "normalization std must be positive"

    return True, "ok"


def evaluate_artifact_gates(
    artifacts_dir: Path,
    retrain_metrics: dict[str, Any],
    delta_tolerance: float = 0.15,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, **extra: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail, **extra})

    for rel in REQUIRED_ARTIFACTS:
        path = artifacts_dir / rel
        add(f"artifact:{rel}", path.exists() and path.stat().st_size > 0, str(path))

    norm_ok, norm_detail = validate_normalization(artifacts_dir / "normalization.json")
    add("normalization_schema", norm_ok, norm_detail)

    export_metrics = retrain_metrics.get("export", {})
    validation_passed = export_metrics.get("validation_passed") is True
    if export_metrics.get("validation_skipped"):
        detail = "ONNX validation skipped"
    elif export_metrics.get("validated"):
        detail = (
            f"max_err_Y={export_metrics.get('max_err_Y')}, "
            f"max_err_Z={export_metrics.get('max_err_Z')}"
        )
    else:
        detail = "ONNX validation did not run"
    add("onnx_validation", validation_passed, detail)

    delta = retrain_metrics.get("train", {}).get("delta_sanity", {})
    err = delta.get("error")
    finite_delta = all(
        isinstance(delta.get(key), (int, float)) and math.isfinite(float(delta[key]))
        for key in ("model_delta", "bs_delta", "error")
    )
    delta_ok = finite_delta and float(err) <= delta_tolerance
    add(
        "atm_delta_sanity",
        delta_ok,
        f"error={err}, tolerance={delta_tolerance}",
        model_delta=delta.get("model_delta"),
        bs_delta=delta.get("bs_delta"),
    )

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def parse_replay_csv(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "detail": "CSV missing or empty", "rows": 0}

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"ok": False, "detail": "CSV has no data rows", "rows": 0}

    totals: list[float] = []
    fills = 0
    for row in rows:
        try:
            total = float(row["total_pnl"])
            n_fills = int(float(row["n_fills"]))
        except Exception as exc:
            return {"ok": False, "detail": f"bad numeric field: {exc}", "rows": len(rows)}
        if not math.isfinite(total):
            return {"ok": False, "detail": "non-finite total_pnl", "rows": len(rows)}
        totals.append(total)
        fills += n_fills

    return {
        "ok": True,
        "detail": "ok",
        "rows": len(rows),
        "total_pnl": float(sum(totals)),
        "mean_daily_pnl": float(sum(totals) / len(totals)),
        "n_fills": fills,
    }


def evaluate_replay_gates(
    neural_run: dict[str, Any],
    bs_run: dict[str, Any],
    baseline_guard_abs: float,
    baseline_guard_rel: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, **extra: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail, **extra})

    add("neural_replay_exit", neural_run.get("returncode") == 0, f"returncode={neural_run.get('returncode')}")
    add("bs_replay_exit", bs_run.get("returncode") == 0, f"returncode={bs_run.get('returncode')}")

    neural_metrics = neural_run.get("metrics", {})
    bs_metrics = bs_run.get("metrics", {})
    add("neural_replay_csv", neural_metrics.get("ok") is True, neural_metrics.get("detail", "missing"))
    add("bs_replay_csv", bs_metrics.get("ok") is True, bs_metrics.get("detail", "missing"))

    baseline_ok = False
    detail = "baseline metrics unavailable"
    if neural_metrics.get("ok") and bs_metrics.get("ok"):
        neural_pnl = float(neural_metrics["total_pnl"])
        bs_pnl = float(bs_metrics["total_pnl"])
        allowed_shortfall = max(baseline_guard_abs, abs(bs_pnl) * baseline_guard_rel)
        baseline_ok = neural_pnl >= bs_pnl - allowed_shortfall
        detail = (
            f"neural_total_pnl={neural_pnl:.4f}, bs_total_pnl={bs_pnl:.4f}, "
            f"allowed_shortfall={allowed_shortfall:.4f}"
        )
    add("bs_baseline_guard", baseline_ok, detail)

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def default_retrain(**kwargs: Any) -> dict[str, Any]:
    sys.path.insert(0, str(BSDE_DIR))
    from calibrate_and_retrain import run_calibration_retrain

    return run_calibration_retrain(**kwargs)


def default_replay(
    *,
    build_dir: Path,
    csv_path: Path,
    artifacts_dir: Path,
    deploy_date: str,
    hedger: str,
    results_csv: Path,
    log_path: Path,
) -> dict[str, Any]:
    exe = build_dir / "alpha_runner"
    if not exe.exists():
        return {
            "returncode": 127,
            "command": [str(exe)],
            "metrics": {"ok": False, "detail": "alpha_runner not built", "rows": 0},
            "log_path": str(log_path),
        }

    cmd = [
        str(exe),
        "--csv",
        str(csv_path),
        "--start-date",
        deploy_date,
        "--end-date",
        deploy_date,
        "--artifacts",
        str(artifacts_dir),
        "--hedger",
        hedger,
        "--results-csv",
        str(results_csv),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=DEMO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with open(log_path, "w") as f:
        f.write(proc.stdout)

    return {
        "returncode": proc.returncode,
        "command": cmd,
        "metrics": parse_replay_csv(results_csv),
        "log_path": str(log_path),
    }


def promote_artifacts(source_dir: Path, live_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    live_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = live_dir / "archive" / utc_stamp()
    archived: list[str] = []

    if any((live_dir / rel).exists() for rel in PROMOTED_ARTIFACTS + ["manifest.json"]):
        archive_dir.mkdir(parents=True, exist_ok=True)
        for rel in PROMOTED_ARTIFACTS + ["manifest.json"]:
            src = live_dir / rel
            if src.exists():
                dst = archive_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                archived.append(rel)

    copied: list[str] = []
    for rel in PROMOTED_ARTIFACTS:
        src = source_dir / rel
        dst = live_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    live_manifest = {
        **manifest,
        "promoted_at": utc_stamp(),
        "artifact_hashes": artifact_hashes(live_dir),
    }
    write_json(live_dir / "manifest.json", live_manifest)

    return {
        "promoted": True,
        "live_dir": str(live_dir),
        "copied": copied,
        "archived": archived,
        "archive_dir": str(archive_dir) if archived else None,
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Walk-Forward Run {summary['run_id']}",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Windows: {summary['n_windows']}",
        f"- Passed: {summary['n_passed']} / {summary['n_windows']}",
        f"- Promoted: {summary['promotion'].get('promoted', False)}",
        "",
        "| Deploy Date | Train Rows | Deploy Rows | Artifact Gate | Replay Gate | Neural PnL | BS PnL | Passed |",
        "|---|---:|---:|---|---|---:|---:|---|",
    ]
    for w in summary["windows"]:
        neural = w.get("neural_replay", {}).get("metrics", {})
        bs = w.get("bs_replay", {}).get("metrics", {})
        lines.append(
            "| {date} | {train_rows} | {deploy_rows} | {artifact} | {replay} | {neural:.2f} | {bs:.2f} | {passed} |".format(
                date=w["deploy_date"],
                train_rows=w["train_rows"],
                deploy_rows=w["deploy_rows"],
                artifact="PASS" if w["artifact_gate"]["passed"] else "FAIL",
                replay="PASS" if w.get("replay_gate", {}).get("passed") else "FAIL",
                neural=float(neural.get("total_pnl", 0.0) or 0.0),
                bs=float(bs.get("total_pnl", 0.0) or 0.0),
                passed="PASS" if w["passed"] else "FAIL",
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def run_pipeline(
    config: PipelineConfig,
    retrain_fn: RetrainFn = default_retrain,
    replay_fn: ReplayFn = default_replay,
) -> dict[str, Any]:
    run_dir = config.run_root / config.run_id
    windows_dir = run_dir / "windows"
    run_dir.mkdir(parents=True, exist_ok=False)

    windows = build_walk_forward_schedule(
        config.csv_path,
        start_date=config.start_date,
        end_date=config.end_date,
        min_train_days=config.min_train_days,
    )
    if not windows:
        raise ValueError("No walk-forward windows matched the requested dates/min_train_days")

    manifest: dict[str, Any] = {
        "run_id": config.run_id,
        "created_at": utc_stamp(),
        "config": asdict(config),
        "windows": [],
        "promotion": {"promoted": False},
    }
    write_json(run_dir / "manifest.json", manifest)

    live_checkpoint = config.live_artifacts / "checkpoints" / "best.pt"
    previous_checkpoint: Path | None = live_checkpoint if live_checkpoint.exists() else None
    for idx, window in enumerate(windows, start=1):
        print(f"[walk-forward] Window {idx}/{len(windows)} deploy={window.deploy_date}")
        window_dir = windows_dir / window.deploy_date
        artifacts_dir = window_dir / "artifacts"
        replay_dir = window_dir / "replay"
        logs_dir = window_dir / "logs"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        retrain_metrics = retrain_fn(
            csv_path=config.csv_path,
            train_end=window.train_end,
            artifacts_dir=artifacts_dir,
            epochs=config.epochs,
            n_syn=config.n_syn,
            n_steps=config.n_steps,
            seed=config.seed,
            warmstart_path=previous_checkpoint,
            no_warmstart=previous_checkpoint is None,
            validate_export=True,
            export_validation_atol=config.onnx_validation_atol,
        )
        artifact_gate = evaluate_artifact_gates(
            artifacts_dir,
            retrain_metrics,
            delta_tolerance=config.delta_tolerance,
        )

        neural_run: dict[str, Any] = {}
        bs_run: dict[str, Any] = {}
        replay_gate = {"passed": False, "checks": []}
        if artifact_gate["passed"]:
            neural_run = replay_fn(
                build_dir=config.build_dir,
                csv_path=config.csv_path,
                artifacts_dir=artifacts_dir,
                deploy_date=window.deploy_date,
                hedger="neural",
                results_csv=replay_dir / "neural_daily.csv",
                log_path=logs_dir / "alpha_runner_neural.log",
            )
            bs_run = replay_fn(
                build_dir=config.build_dir,
                csv_path=config.csv_path,
                artifacts_dir=artifacts_dir,
                deploy_date=window.deploy_date,
                hedger="bs",
                results_csv=replay_dir / "bs_daily.csv",
                log_path=logs_dir / "alpha_runner_bs.log",
            )
            replay_gate = evaluate_replay_gates(
                neural_run,
                bs_run,
                baseline_guard_abs=config.baseline_guard_abs,
                baseline_guard_rel=config.baseline_guard_rel,
            )

        window_record = {
            **asdict(window),
            "window_dir": str(window_dir),
            "artifacts_dir": str(artifacts_dir),
            "artifact_hashes": artifact_hashes(artifacts_dir),
            "retrain": retrain_metrics,
            "artifact_gate": artifact_gate,
            "neural_replay": neural_run,
            "bs_replay": bs_run,
            "replay_gate": replay_gate,
            "passed": artifact_gate["passed"] and replay_gate["passed"],
        }
        write_json(window_dir / "window_manifest.json", window_record)

        manifest["windows"].append(window_record)
        write_json(run_dir / "manifest.json", manifest)

        checkpoint = artifacts_dir / "checkpoints" / "best.pt"
        if checkpoint.exists():
            previous_checkpoint = checkpoint

    all_passed = all(w["passed"] for w in manifest["windows"])
    if config.promote and all_passed:
        last_window = manifest["windows"][-1]
        source = Path(last_window["artifacts_dir"])
        manifest["promotion"] = promote_artifacts(
            source,
            config.live_artifacts,
            {
                "source_run_id": config.run_id,
                "source_window": last_window["deploy_date"],
                "source_artifacts_dir": str(source),
                "profile": config.profile,
                "gates": {
                    "artifact_gate": last_window["artifact_gate"],
                    "replay_gate": last_window["replay_gate"],
                },
            },
        )
    elif config.promote:
        manifest["promotion"] = {
            "promoted": False,
            "reason": "one or more windows failed gates",
        }

    summary = {
        "run_id": config.run_id,
        "profile": config.profile,
        "n_windows": len(manifest["windows"]),
        "n_passed": sum(1 for w in manifest["windows"] if w["passed"]),
        "all_passed": all_passed,
        "promotion": manifest["promotion"],
        "windows": manifest["windows"],
    }
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir / "summary.md", summary)
    write_json(run_dir / "manifest.json", manifest)

    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "summary": summary,
    }


def resolve_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default.resolve()
    return Path(path).expanduser().resolve()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    defaults = PROFILE_DEFAULTS[args.profile]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    n_syn = args.n_syn if args.n_syn is not None else defaults["n_syn"]
    n_steps = args.n_steps if args.n_steps is not None else defaults["n_steps"]
    run_id = args.run_id or f"wf_{utc_stamp()}"

    return PipelineConfig(
        csv_path=resolve_path(args.csv, DEMO_ROOT / "data" / "spy_chain_panel.csv"),
        run_id=run_id,
        run_root=resolve_path(args.run_root, DEMO_ROOT / "runs" / "walk_forward"),
        start_date=args.start_date,
        end_date=args.end_date,
        min_train_days=args.min_train_days,
        profile=args.profile,
        epochs=epochs,
        n_syn=n_syn,
        n_steps=n_steps,
        seed=args.seed,
        promote=args.promote,
        build_dir=resolve_path(args.build_dir, DEMO_ROOT / "build"),
        live_artifacts=resolve_path(args.live_artifacts, DEMO_ROOT / "artifacts"),
        onnx_validation_atol=defaults["onnx_validation_atol"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily expanding walk-forward BSDE retraining pipeline"
    )
    parser.add_argument("--csv", default=None, help="Path to spy_chain_panel.csv")
    parser.add_argument("--run-id", default=None, help="Run id. Default: timestamped id")
    parser.add_argument("--run-root", default=None, help="Run root. Default: demo/runs/walk_forward")
    parser.add_argument("--start-date", default=None, help="First deploy date, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Last deploy date, YYYY-MM-DD")
    parser.add_argument("--min-train-days", type=int, default=3)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-syn", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    promote = parser.add_mutually_exclusive_group()
    promote.add_argument("--promote-if-pass", dest="promote", action="store_true")
    promote.add_argument("--no-promote", dest="promote", action="store_false")
    parser.set_defaults(promote=False)
    parser.add_argument("--build-dir", default=None, help="Directory containing alpha_runner")
    parser.add_argument("--live-artifacts", default=None, help="Live artifact dir to promote into")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    result = run_pipeline(config)
    summary = result["summary"]
    print(
        f"[walk-forward] Run complete: {result['run_dir']} "
        f"({summary['n_passed']}/{summary['n_windows']} windows passed)"
    )
    if config.promote:
        print(f"[walk-forward] Promotion: {summary['promotion']}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
