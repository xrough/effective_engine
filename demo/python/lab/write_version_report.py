#!/usr/bin/env python3
"""Write a human-readable lab version report from smoke-gate JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parents[3] / "lab"
REPO_ROOT = LAB_DIR.parent
VERSIONS_PATH = LAB_DIR / "versions.json"


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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_version(versions_doc: dict[str, Any], version_id: str) -> dict[str, Any]:
    for version in versions_doc.get("versions", []):
        if version.get("version_id") == version_id:
            return version
    return {
        "version_id": version_id,
        "branch": None,
        "status": "ad-hoc",
        "goal": "Ad-hoc lab version.",
        "main_outcome": "No version metadata was registered before report generation.",
        "comparison": [],
        "risks": ["Version metadata was missing at report time."],
    }


def recommendation_from_gates(gate_doc: dict[str, Any]) -> str:
    return gate_doc.get("summary", {}).get("default_recommendation", "hold")


def status_icon(status: str) -> str:
    return {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(status, status.upper())


def gate_table(gate_doc: dict[str, Any]) -> list[str]:
    lines = [
        "| Gate | Required | Status | Detail |",
        "|---|---|---|---|",
    ]
    for gate in gate_doc.get("gates", []):
        detail = str(gate.get("detail", "")).replace("\n", " ")
        lines.append(
            f"| `{gate.get('gate_id')}` | {gate.get('required')} | {status_icon(gate.get('status', ''))} | {detail} |"
        )
    return lines


def command_table(gate_doc: dict[str, Any]) -> list[str]:
    lines = [
        "| Gate | Command | CWD |",
        "|---|---|---|",
    ]
    for gate in gate_doc.get("gates", []):
        command = gate.get("command") or []
        if not command:
            continue
        cmd = " ".join(command).replace("|", "\\|")
        cwd = gate.get("cwd", "")
        lines.append(f"| `{gate.get('gate_id')}` | `{cmd}` | `{cwd}` |")
    return lines


def artifact_lines(gate_doc: dict[str, Any]) -> list[str]:
    lines = []
    artifacts_root = gate_doc.get("artifacts_root")
    if artifacts_root:
        lines.append(f"- Artifacts root: `{artifacts_root}`")
    for gate in gate_doc.get("gates", []):
        log_path = gate.get("log_path")
        if log_path:
            lines.append(f"- `{gate.get('gate_id')}` log: `{log_path}`")
        for name, value in (gate.get("artifacts") or {}).items():
            lines.append(f"- `{gate.get('gate_id')}` {name}: `{value}`")
    return lines or ["- No artifacts recorded."]


def comparison_table(version: dict[str, Any]) -> list[str]:
    rows = version.get("comparison") or []
    if not rows:
        return ["No comparison rows registered for this version."]
    lines = [
        "| Area | Before | After |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row.get('area', '')} | {row.get('before', '')} | {row.get('after', '')} |")
    return lines


def build_report(version: dict[str, Any], gate_doc: dict[str, Any], recommendation: str) -> str:
    summary = gate_doc.get("summary", {})
    repo = gate_doc.get("repo", {})
    risks = version.get("risks") or ["No risks registered."]
    lines = [
        f"# Lab Version Report: {version['version_id']}",
        "",
        f"- Recommendation: `{recommendation}`",
        f"- Branch: `{version.get('branch') or repo.get('branch') or '<unknown>'}`",
        f"- Status: `{version.get('status', '<unknown>')}`",
        f"- Generated: `{utc_stamp()}`",
        f"- Gate profile: `{gate_doc.get('profile', '<unknown>')}`",
        f"- Repo HEAD: `{repo.get('head', '<unknown>')}`",
        f"- Dirty tree at gate time: `{repo.get('dirty', '<unknown>')}`",
        "",
        "## Goal",
        "",
        version.get("goal", ""),
        "",
        "## Change Summary",
        "",
        version.get("main_outcome", ""),
        "",
        "## Update Comparison",
        "",
        *comparison_table(version),
        "",
        "## Gate Results",
        "",
        f"Required gates: {summary.get('required_passed', 0)} / {summary.get('required_total', 0)} passed.",
        f"Optional gates: {summary.get('optional_passed', 0)} passed, {summary.get('optional_skipped', 0)} skipped, {summary.get('optional_failed', 0)} failed.",
        "",
        *gate_table(gate_doc),
        "",
        "## Commands Run",
        "",
        *command_table(gate_doc),
        "",
        "## Artifacts",
        "",
        *artifact_lines(gate_doc),
        "",
        "## Risks And Review Notes",
        "",
        *[f"- {risk}" for risk in risks],
        "",
        "## Decision",
        "",
        f"Recommended action: `{recommendation}`.",
    ]
    return "\n".join(lines) + "\n"


def update_versions_doc(version_id: str, report_path: Path, gate_path: Path, recommendation: str) -> None:
    versions_doc = load_json(VERSIONS_PATH)
    versions = versions_doc.setdefault("versions", [])
    version = None
    for item in versions:
        if item.get("version_id") == version_id:
            version = item
            break
    if version is None:
        version = {"version_id": version_id, "status": "ad-hoc"}
        versions.append(version)

    version["last_report"] = display_path(report_path)
    version["last_gate_json"] = display_path(gate_path)
    version["last_recommendation"] = recommendation
    version["updated_at"] = utc_stamp()
    if version.get("status") == "in-progress":
        version["status"] = "reported"
    write_json(VERSIONS_PATH, versions_doc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write lab version report.")
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--gate-json")
    parser.add_argument("--out")
    parser.add_argument("--recommendation", choices=["merge", "hold", "stop"])
    args = parser.parse_args()

    gate_path = Path(args.gate_json).expanduser().resolve() if args.gate_json else (LAB_DIR / "reports" / f"{args.version_id}_gates.json").resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else (LAB_DIR / "reports" / f"{args.version_id}.md").resolve()
    versions_doc = load_json(VERSIONS_PATH)
    gate_doc = load_json(gate_path)
    version = find_version(versions_doc, args.version_id)
    recommendation = args.recommendation or recommendation_from_gates(gate_doc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(version, gate_doc, recommendation))
    update_versions_doc(args.version_id, out_path, gate_path, recommendation)
    print(f"[lab-report] wrote {out_path}")
    print(f"[lab-report] recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
