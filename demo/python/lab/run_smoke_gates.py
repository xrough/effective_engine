#!/usr/bin/env python3
"""Run versioned lab smoke gates.

The lab control plane is intentionally lightweight: this script executes the
required local confidence checks, captures logs, writes a machine-readable JSON
result, and computes a default merge/hold/stop recommendation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parents[3] / "lab"
REPO_ROOT = LAB_DIR.parent
DEMO_ROOT = REPO_ROOT / "demo"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_text(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return proc.stdout.strip()


def repo_state() -> dict[str, Any]:
    return {
        "branch": run_text(["git", "branch", "--show-current"], REPO_ROOT),
        "head": run_text(["git", "rev-parse", "--short", "HEAD"], REPO_ROOT),
        "dirty": bool(run_text(["git", "status", "--short"], REPO_ROOT)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def log_tail(path: Path, max_lines: int = 30) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-max_lines:]


def parse_daily_csv(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "detail": "CSV missing or empty", "rows": 0}

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"ok": False, "detail": "CSV has no data rows", "rows": 0}

    total = 0.0
    for row in rows:
        if "total_pnl" not in row:
            return {"ok": False, "detail": "missing total_pnl column", "rows": len(rows)}
        try:
            value = float(row["total_pnl"])
        except ValueError:
            return {"ok": False, "detail": "non-numeric total_pnl", "rows": len(rows)}
        if not math.isfinite(value):
            return {"ok": False, "detail": "non-finite total_pnl", "rows": len(rows)}
        total += value

    return {"ok": True, "detail": "ok", "rows": len(rows), "total_pnl": total}


def command_gate(
    *,
    gate_id: str,
    name: str,
    command: list[str],
    cwd: Path,
    required: bool,
    log_dir: Path,
    artifacts: dict[str, str] | None = None,
    post_check: Any | None = None,
) -> dict[str, Any]:
    started = time.time()
    log_path = log_dir / f"{gate_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not command:
        return {
            "gate_id": gate_id,
            "name": name,
            "required": required,
            "status": "failed" if required else "skipped",
            "detail": "empty command",
            "command": [],
            "cwd": str(cwd),
            "returncode": None,
            "duration_sec": 0.0,
            "log_path": str(log_path),
            "artifacts": artifacts or {},
        }

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
    duration = time.time() - started
    status = "passed" if returncode == 0 else "failed"
    detail = f"returncode={returncode}"

    extra: dict[str, Any] = {}
    if returncode == 0 and post_check is not None:
        extra = post_check()
        if not extra.get("ok", False):
            status = "failed"
            detail = extra.get("detail", "post-check failed")

    return {
        "gate_id": gate_id,
        "name": name,
        "required": required,
        "status": status,
        "detail": detail,
        "command": command,
        "cwd": str(cwd),
        "returncode": returncode,
        "duration_sec": round(duration, 3),
        "log_path": str(log_path),
        "log_tail": log_tail(log_path),
        "artifacts": artifacts or {},
        "post_check": extra,
    }


def skipped_gate(gate_id: str, name: str, required: bool, detail: str) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "name": name,
        "required": required,
        "status": "skipped",
        "detail": detail,
        "command": [],
        "cwd": str(REPO_ROOT),
        "returncode": None,
        "duration_sec": 0.0,
        "log_path": None,
        "artifacts": {},
    }


def onnx_probe() -> tuple[bool, str, Path]:
    root = Path(os.environ.get("ONNXRUNTIME_ROOT", str(Path.home() / "onnxruntime"))).expanduser()
    header = root / "include" / "onnxruntime_cxx_api.h"
    lib_dir = root / "lib"
    artifacts = [DEMO_ROOT / "artifacts" / "neural_bsde.onnx", DEMO_ROOT / "artifacts" / "normalization.json"]
    missing = [str(p) for p in [header, lib_dir, *artifacts] if not p.exists()]
    if missing:
        return False, "missing: " + ", ".join(missing), root
    return True, "available", root


def default_output_path(version_id: str) -> Path:
    return LAB_DIR / "reports" / f"{version_id}_gates.json"


def build_required_gates(version_id: str, artifacts_root: Path) -> list[dict[str, Any]]:
    log_dir = artifacts_root / "logs"
    replay_csv = artifacts_root / "bs_daily.csv"
    experiment_dir = artifacts_root / "experiment_orchestrator"
    data_contracts_json = artifacts_root / "data_contracts.json"
    data_contracts_md = artifacts_root / "data_contracts.md"
    strategy_registry_json = artifacts_root / "strategy_registry.json"
    strategy_registry_md = artifacts_root / "strategy_registry.md"
    return [
        command_gate(
            gate_id="data_contract_validation",
            name="Dataset contract validation",
            command=[
                "python3",
                str(DEMO_ROOT / "python" / "lab" / "validate_data_contracts.py"),
                "--output-json",
                str(data_contracts_json),
                "--output-md",
                str(data_contracts_md),
            ],
            cwd=REPO_ROOT,
            required=True,
            log_dir=log_dir,
            artifacts={
                "result_json": str(data_contracts_json),
                "result_md": str(data_contracts_md),
            },
        ),
        command_gate(
            gate_id="strategy_registry_validation",
            name="Strategy registry validation",
            command=[
                "python3",
                str(DEMO_ROOT / "python" / "lab" / "validate_strategy_registry.py"),
                "--output-json",
                str(strategy_registry_json),
                "--output-md",
                str(strategy_registry_md),
            ],
            cwd=REPO_ROOT,
            required=True,
            log_dir=log_dir,
            artifacts={
                "result_json": str(strategy_registry_json),
                "result_md": str(strategy_registry_md),
            },
        ),
        command_gate(
            gate_id="configure_demo",
            name="Configure demo CMake",
            command=["cmake", "-S", str(DEMO_ROOT), "-B", str(DEMO_ROOT / "build"), "-DBUILD_ONNX_DEMO=OFF"],
            cwd=REPO_ROOT,
            required=True,
            log_dir=log_dir,
        ),
        command_gate(
            gate_id="build_core_targets",
            name="Build core smoke targets",
            command=[
                "cmake",
                "--build",
                str(DEMO_ROOT / "build"),
                "--target",
                "alpha_runner",
                "execution_layer_smoke_test",
                "-j2",
            ],
            cwd=REPO_ROOT,
            required=True,
            log_dir=log_dir,
        ),
        command_gate(
            gate_id="execution_layer_smoke",
            name="Execution lifecycle smoke",
            command=[str(DEMO_ROOT / "build" / "execution_layer_smoke_test")],
            cwd=DEMO_ROOT,
            required=True,
            log_dir=log_dir,
        ),
        command_gate(
            gate_id="python_walk_forward_unit",
            name="Walk-forward unit tests",
            command=["python3", str(DEMO_ROOT / "python" / "bsde" / "test_walk_forward_pipeline.py")],
            cwd=REPO_ROOT,
            required=True,
            log_dir=log_dir,
        ),
        command_gate(
            gate_id="bs_replay_smoke",
            name="BS delta replay smoke",
            command=[
                str(DEMO_ROOT / "build" / "alpha_runner"),
                "--csv",
                "data/spy_chain_panel.csv",
                "--hedger",
                "bs",
                "--start-date",
                "2025-08-12",
                "--end-date",
                "2025-08-13",
                "--results-csv",
                str(replay_csv),
            ],
            cwd=DEMO_ROOT,
            required=True,
            log_dir=log_dir,
            artifacts={"results_csv": str(replay_csv)},
            post_check=lambda: parse_daily_csv(replay_csv),
        ),
        command_gate(
            gate_id="experiment_orchestrator_smoke",
            name="Lab experiment orchestrator smoke",
            command=[
                "python3",
                str(DEMO_ROOT / "python" / "lab" / "run_lab_experiment.py"),
                "--experiment-id",
                f"{version_id}_orchestrator_smoke",
                "--profile",
                "smoke",
                "--start-date",
                "2025-08-12",
                "--end-date",
                "2025-08-13",
                "--output-dir",
                str(experiment_dir),
            ],
            cwd=REPO_ROOT,
            required=True,
            log_dir=log_dir,
            artifacts={
                "manifest": str(experiment_dir / "manifest.json"),
                "summary_json": str(experiment_dir / "summary.json"),
                "summary_md": str(experiment_dir / "summary.md"),
            },
        ),
    ]


def build_model_gates(artifacts_root: Path) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    ok, detail, onnx_root = onnx_probe()
    gates.append(
        {
            "gate_id": "onnx_dependency_probe",
            "name": "ONNX dependency probe",
            "required": False,
            "status": "passed" if ok else "skipped",
            "detail": detail,
            "command": [],
            "cwd": str(REPO_ROOT),
            "returncode": 0 if ok else None,
            "duration_sec": 0.0,
            "log_path": None,
            "artifacts": {"onnxruntime_root": str(onnx_root)},
        }
    )
    if not ok:
        gates.append(skipped_gate("neural_replay_smoke", "Neural BSDE replay smoke", False, detail))
        gates.append(skipped_gate("walk_forward_model_smoke", "Walk-forward model smoke", False, detail))
        return gates

    log_dir = artifacts_root / "logs"
    neural_csv = artifacts_root / "neural_daily.csv"
    wf_run_id = datetime.now(timezone.utc).strftime("wf_smoke_lab_%Y%m%dT%H%M%SZ")
    gates.append(
        command_gate(
            gate_id="configure_onnx_demo",
            name="Configure ONNX-enabled demo",
            command=[
                "cmake",
                "-S",
                str(DEMO_ROOT),
                "-B",
                str(DEMO_ROOT / "build"),
                "-DBUILD_ONNX_DEMO=ON",
                f"-DONNXRUNTIME_ROOT={onnx_root}",
            ],
            cwd=REPO_ROOT,
            required=False,
            log_dir=log_dir,
        )
    )
    gates.append(
        command_gate(
            gate_id="build_onnx_alpha_runner",
            name="Build ONNX alpha_runner",
            command=["cmake", "--build", str(DEMO_ROOT / "build"), "--target", "alpha_runner", "-j2"],
            cwd=REPO_ROOT,
            required=False,
            log_dir=log_dir,
        )
    )
    gates.append(
        command_gate(
            gate_id="neural_replay_smoke",
            name="Neural BSDE replay smoke",
            command=[
                str(DEMO_ROOT / "build" / "alpha_runner"),
                "--csv",
                "data/spy_chain_panel.csv",
                "--hedger",
                "neural",
                "--artifacts",
                "artifacts",
                "--start-date",
                "2025-08-12",
                "--end-date",
                "2025-08-13",
                "--results-csv",
                str(neural_csv),
            ],
            cwd=DEMO_ROOT,
            required=False,
            log_dir=log_dir,
            artifacts={"results_csv": str(neural_csv)},
            post_check=lambda: parse_daily_csv(neural_csv),
        )
    )
    gates.append(
        command_gate(
            gate_id="walk_forward_model_smoke",
            name="Walk-forward model smoke",
            command=[
                "python3",
                str(DEMO_ROOT / "python" / "bsde" / "walk_forward_pipeline.py"),
                "--profile",
                "smoke",
                "--run-id",
                wf_run_id,
                "--start-date",
                "2025-08-12",
                "--end-date",
                "2025-08-13",
                "--no-promote",
            ],
            cwd=REPO_ROOT,
            required=False,
            log_dir=log_dir,
            artifacts={"run_id": wf_run_id, "run_root": str(DEMO_ROOT / "runs" / "walk_forward")},
        )
    )
    return gates


def summarize(gates: list[dict[str, Any]]) -> dict[str, Any]:
    required = [g for g in gates if g["required"]]
    optional = [g for g in gates if not g["required"]]
    required_passed = all(g["status"] == "passed" for g in required)
    optional_failed = any(g["status"] == "failed" for g in optional)
    if not required_passed:
        recommendation = "stop"
    elif optional_failed:
        recommendation = "hold"
    else:
        recommendation = "merge"
    return {
        "required_total": len(required),
        "required_passed": sum(1 for g in required if g["status"] == "passed"),
        "optional_total": len(optional),
        "optional_passed": sum(1 for g in optional if g["status"] == "passed"),
        "optional_skipped": sum(1 for g in optional if g["status"] == "skipped"),
        "optional_failed": sum(1 for g in optional if g["status"] == "failed"),
        "default_recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lab smoke gates.")
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--profile", choices=["core", "model"], default="core")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else default_output_path(args.version_id)
    if args.keep_artifacts:
        artifacts_root = LAB_DIR / "reports" / "artifacts" / args.version_id
    else:
        artifacts_root = Path(tempfile.mkdtemp(prefix=f"{args.version_id}_smoke_"))
    artifacts_root.mkdir(parents=True, exist_ok=True)

    gates = build_required_gates(args.version_id, artifacts_root)
    if args.profile == "model":
        gates.extend(build_model_gates(artifacts_root))

    result = {
        "version_id": args.version_id,
        "profile": args.profile,
        "created_at": utc_stamp(),
        "repo": repo_state(),
        "artifacts_root": str(artifacts_root),
        "keep_artifacts": args.keep_artifacts,
        "gates": gates,
        "summary": summarize(gates),
    }
    write_json(output_json, result)

    summary = result["summary"]
    print(
        f"[lab-smoke] {args.version_id} profile={args.profile} "
        f"required={summary['required_passed']}/{summary['required_total']} "
        f"optional_failed={summary['optional_failed']} "
        f"recommendation={summary['default_recommendation']}"
    )
    print(f"[lab-smoke] wrote {output_json}")
    return 0 if summary["required_passed"] == summary["required_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
