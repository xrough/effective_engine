#!/usr/bin/env python3
"""Run a compact, versioned lab experiment.

This is the v0.4 experiment orchestrator: it ties together registry checks,
C++ replay, optional model replay, optional walk-forward smoke, and a single
manifest/summary report under one experiment id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parents[3] / "lab"
REPO_ROOT = LAB_DIR.parent
DEMO_ROOT = REPO_ROOT / "demo"


def utc_stamp(fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    return datetime.now(timezone.utc).strftime(fmt)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def run_text(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return proc.stdout.strip()


def repo_state() -> dict[str, Any]:
    return {
        "branch": run_text(["git", "branch", "--show-current"], REPO_ROOT),
        "head": run_text(["git", "rev-parse", "--short", "HEAD"], REPO_ROOT),
        "dirty": bool(run_text(["git", "status", "--short"], REPO_ROOT)),
    }


def log_tail(path: Path, max_lines: int = 25) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-max_lines:]


def parse_daily_csv(path: Path) -> dict[str, Any]:
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
            pnl = float(row["total_pnl"])
            n_fills = int(float(row.get("n_fills", 0)))
        except Exception as exc:
            return {"ok": False, "detail": f"bad row: {exc}", "rows": len(rows)}
        if not math.isfinite(pnl):
            return {"ok": False, "detail": "non-finite total_pnl", "rows": len(rows)}
        totals.append(pnl)
        fills += n_fills

    return {
        "ok": True,
        "detail": "ok",
        "rows": len(rows),
        "total_pnl": float(sum(totals)),
        "mean_daily_pnl": float(sum(totals) / len(totals)),
        "n_fills": fills,
    }


def run_command(
    *,
    step_id: str,
    name: str,
    command: list[str],
    cwd: Path,
    log_dir: Path,
    required: bool = True,
    artifacts: dict[str, str] | None = None,
    post_check: Any | None = None,
) -> dict[str, Any]:
    started = time.time()
    log_path = log_dir / f"{step_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = proc.stdout
        returncode = proc.returncode
    except FileNotFoundError as exc:
        output = str(exc)
        returncode = 127
    log_path.write_text(output)

    status = "passed" if returncode == 0 else "failed"
    detail = f"returncode={returncode}"
    extra: dict[str, Any] = {}
    if status == "passed" and post_check is not None:
        extra = post_check()
        if not extra.get("ok", False):
            status = "failed"
            detail = extra.get("detail", "post-check failed")

    return {
        "step_id": step_id,
        "name": name,
        "required": required,
        "status": status,
        "detail": detail,
        "command": command,
        "cwd": str(cwd),
        "returncode": returncode,
        "duration_sec": round(time.time() - started, 3),
        "log_path": str(log_path),
        "log_tail": log_tail(log_path),
        "artifacts": artifacts or {},
        "post_check": extra,
    }


def skipped_step(step_id: str, name: str, detail: str) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "name": name,
        "required": False,
        "status": "skipped",
        "detail": detail,
        "command": [],
        "cwd": str(REPO_ROOT),
        "returncode": None,
        "duration_sec": 0.0,
        "log_path": None,
        "artifacts": {},
        "post_check": {},
    }


def onnx_available() -> tuple[bool, str, Path]:
    root = Path(os.environ.get("ONNXRUNTIME_ROOT", str(Path.home() / "onnxruntime"))).expanduser()
    required = [
        root / "include" / "onnxruntime_cxx_api.h",
        root / "lib",
        DEMO_ROOT / "artifacts" / "neural_bsde.onnx",
        DEMO_ROOT / "artifacts" / "normalization.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return False, "missing: " + ", ".join(missing), root
    return True, "available", root


def build_targets(log_dir: Path, include_onnx: bool, onnx_root: Path | None) -> list[dict[str, Any]]:
    cmake_cmd = ["cmake", "-S", str(DEMO_ROOT), "-B", str(DEMO_ROOT / "build")]
    if include_onnx:
        cmake_cmd.extend(["-DBUILD_ONNX_DEMO=ON", f"-DONNXRUNTIME_ROOT={onnx_root}"])
    else:
        cmake_cmd.append("-DBUILD_ONNX_DEMO=OFF")

    return [
        run_command(
            step_id="configure_demo",
            name="Configure demo build",
            command=cmake_cmd,
            cwd=REPO_ROOT,
            log_dir=log_dir,
        ),
        run_command(
            step_id="build_alpha_runner",
            name="Build alpha_runner",
            command=["cmake", "--build", str(DEMO_ROOT / "build"), "--target", "alpha_runner", "-j2"],
            cwd=REPO_ROOT,
            log_dir=log_dir,
        ),
    ]


def run_replay_steps(
    *,
    csv_path: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
    log_dir: Path,
    profile: str,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    bs_csv = out_dir / "bs_daily.csv"
    steps.append(
        run_command(
            step_id="replay_bs",
            name="Replay BS delta strategy",
            command=[
                str(DEMO_ROOT / "build" / "alpha_runner"),
                "--csv",
                csv_path,
                "--hedger",
                "bs",
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--results-csv",
                str(bs_csv),
            ],
            cwd=DEMO_ROOT,
            log_dir=log_dir,
            artifacts={"results_csv": str(bs_csv)},
            post_check=lambda: parse_daily_csv(bs_csv),
        )
    )

    if profile == "model":
        neural_csv = out_dir / "neural_daily.csv"
        steps.append(
            run_command(
                step_id="replay_neural",
                name="Replay neural BSDE strategy",
                command=[
                    str(DEMO_ROOT / "build" / "alpha_runner"),
                    "--csv",
                    csv_path,
                    "--hedger",
                    "neural",
                    "--artifacts",
                    "artifacts",
                    "--start-date",
                    start_date,
                    "--end-date",
                    end_date,
                    "--results-csv",
                    str(neural_csv),
                ],
                cwd=DEMO_ROOT,
                log_dir=log_dir,
                required=False,
                artifacts={"results_csv": str(neural_csv)},
                post_check=lambda: parse_daily_csv(neural_csv),
            )
        )
    return steps


def run_walk_forward_step(
    *,
    profile: str,
    start_date: str,
    end_date: str,
    log_dir: Path,
) -> dict[str, Any]:
    if profile != "model":
        return skipped_step("walk_forward_smoke", "Walk-forward model smoke", "profile is not model")
    run_id = utc_stamp("wf_experiment_%Y%m%dT%H%M%SZ")
    return run_command(
        step_id="walk_forward_smoke",
        name="Walk-forward smoke retraining",
        command=[
            "python3",
            str(DEMO_ROOT / "python" / "bsde" / "walk_forward_pipeline.py"),
            "--profile",
            "smoke",
            "--run-id",
            run_id,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--no-promote",
        ],
        cwd=REPO_ROOT,
        log_dir=log_dir,
        required=False,
        artifacts={"run_id": run_id, "run_root": str(DEMO_ROOT / "runs" / "walk_forward")},
    )


def summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    required = [step for step in steps if step["required"]]
    optional = [step for step in steps if not step["required"]]
    required_passed = all(step["status"] == "passed" for step in required)
    optional_failed = any(step["status"] == "failed" for step in optional)
    if not required_passed:
        recommendation = "stop"
    elif optional_failed:
        recommendation = "hold"
    else:
        recommendation = "merge"
    replay_metrics = {
        step["step_id"]: step.get("post_check", {})
        for step in steps
        if step["step_id"].startswith("replay_")
    }
    return {
        "required_total": len(required),
        "required_passed": sum(1 for step in required if step["status"] == "passed"),
        "optional_total": len(optional),
        "optional_passed": sum(1 for step in optional if step["status"] == "passed"),
        "optional_skipped": sum(1 for step in optional if step["status"] == "skipped"),
        "optional_failed": sum(1 for step in optional if step["status"] == "failed"),
        "default_recommendation": recommendation,
        "replay_metrics": replay_metrics,
    }


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Lab Experiment {payload['experiment_id']}",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Created: `{payload['created_at']}`",
        f"- Date window: `{payload['config']['start_date']}` -> `{payload['config']['end_date']}`",
        f"- Recommendation: `{payload['summary']['default_recommendation']}`",
        "",
        "| Step | Required | Status | Detail |",
        "|---|---|---|---|",
    ]
    for step in payload["steps"]:
        lines.append(f"| `{step['step_id']}` | {step['required']} | {step['status'].upper()} | {step['detail']} |")

    lines.extend(["", "## Replay Metrics", ""])
    metrics = payload["summary"].get("replay_metrics", {})
    if metrics:
        lines.extend(["| Replay | Rows | Total PnL | Fills |", "|---|---:|---:|---:|"])
        for name, row in metrics.items():
            lines.append(
                f"| `{name}` | {row.get('rows', 0)} | {float(row.get('total_pnl', 0.0) or 0.0):.2f} | {int(row.get('n_fills', 0) or 0)} |"
            )
    else:
        lines.append("No replay metrics recorded.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a versioned lab experiment.")
    parser.add_argument("--experiment-id")
    parser.add_argument("--profile", choices=["smoke", "model"], default="smoke")
    parser.add_argument("--csv", default="data/spy_chain_panel.csv")
    parser.add_argument("--start-date", default="2025-08-12")
    parser.add_argument("--end-date", default="2025-08-13")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    experiment_id = args.experiment_id or utc_stamp("experiment_%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else LAB_DIR / "reports" / "artifacts" / experiment_id
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_ok, onnx_detail, onnx_root = onnx_available()
    include_onnx = args.profile == "model" and onnx_ok
    steps: list[dict[str, Any]] = [
        run_command(
            step_id="data_contract_validation",
            name="Validate dataset contracts",
            command=[
                "python3",
                str(DEMO_ROOT / "python" / "lab" / "validate_data_contracts.py"),
                "--output-json",
                str(out_dir / "data_contracts.json"),
                "--output-md",
                str(out_dir / "data_contracts.md"),
            ],
            cwd=REPO_ROOT,
            log_dir=logs_dir,
            artifacts={"result_json": str(out_dir / "data_contracts.json"), "result_md": str(out_dir / "data_contracts.md")},
        ),
        run_command(
            step_id="strategy_registry_validation",
            name="Validate strategy registry",
            command=[
                "python3",
                str(DEMO_ROOT / "python" / "lab" / "validate_strategy_registry.py"),
                "--output-json",
                str(out_dir / "strategy_registry.json"),
                "--output-md",
                str(out_dir / "strategy_registry.md"),
            ],
            cwd=REPO_ROOT,
            log_dir=logs_dir,
            artifacts={"result_json": str(out_dir / "strategy_registry.json"), "result_md": str(out_dir / "strategy_registry.md")},
        ),
    ]
    steps.extend(build_targets(logs_dir, include_onnx, onnx_root if include_onnx else None))
    if args.profile == "model" and not onnx_ok:
        steps.append(skipped_step("onnx_dependency_probe", "ONNX dependency probe", onnx_detail))
    else:
        steps.append(
            {
                "step_id": "onnx_dependency_probe",
                "name": "ONNX dependency probe",
                "required": False,
                "status": "passed" if onnx_ok else "skipped",
                "detail": onnx_detail,
                "command": [],
                "cwd": str(REPO_ROOT),
                "returncode": 0 if onnx_ok else None,
                "duration_sec": 0.0,
                "log_path": None,
                "artifacts": {"onnxruntime_root": str(onnx_root)},
                "post_check": {},
            }
        )
    steps.extend(
        run_replay_steps(
            csv_path=args.csv,
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=out_dir,
            log_dir=logs_dir,
            profile=args.profile if include_onnx else "smoke",
        )
    )
    steps.append(run_walk_forward_step(profile=args.profile if include_onnx else "smoke", start_date=args.start_date, end_date=args.end_date, log_dir=logs_dir))

    payload = {
        "experiment_id": experiment_id,
        "created_at": utc_stamp(),
        "profile": args.profile,
        "repo": repo_state(),
        "config": {
            "csv": args.csv,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "output_dir": str(out_dir),
        },
        "steps": steps,
        "summary": summarize_steps(steps),
    }
    write_json(out_dir / "manifest.json", payload)
    write_json(out_dir / "summary.json", {"experiment_id": experiment_id, **payload["summary"]})
    write_summary_md(out_dir / "summary.md", payload)

    print(
        f"[lab-experiment] {experiment_id} profile={args.profile} "
        f"required={payload['summary']['required_passed']}/{payload['summary']['required_total']} "
        f"optional_failed={payload['summary']['optional_failed']} "
        f"recommendation={payload['summary']['default_recommendation']}"
    )
    print(f"[lab-experiment] wrote {out_dir / 'manifest.json'}")
    return 0 if payload["summary"]["required_passed"] == payload["summary"]["required_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
