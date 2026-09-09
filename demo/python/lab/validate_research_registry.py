#!/usr/bin/env python3
"""Validate and render the research gate registry."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parents[3] / "lab"
REPO_ROOT = LAB_DIR.parent
REQUIRED_FIELDS = [
    "gate_id",
    "title",
    "family",
    "hypothesis",
    "status",
    "verdict",
    "implementation_paths",
    "latest_outputs",
    "decision_rule",
    "next_action",
]
VALID_STATUSES = {"pass", "rejected", "weak", "narrow-support", "planned", "candidate"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, **extra: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, **extra})


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(nonempty_string(item) for item in value)


def path_exists(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def validate_gate(gate: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [field for field in REQUIRED_FIELDS if field not in gate]
    add_check(checks, "required_fields", not missing, "ok" if not missing else "missing: " + ", ".join(missing))

    gate_id = gate.get("gate_id", "")
    add_check(checks, "gate_id_format", nonempty_string(gate_id) and bool(ID_RE.match(gate_id)), str(gate_id))
    add_check(checks, "status_allowed", gate.get("status") in VALID_STATUSES, str(gate.get("status")))

    for field in ["title", "family", "hypothesis", "verdict", "decision_rule", "next_action"]:
        add_check(checks, f"{field}_nonempty", nonempty_string(gate.get(field)), str(gate.get(field)))

    impl_paths = gate.get("implementation_paths")
    add_check(checks, "implementation_paths_nonempty", nonempty_string_list(impl_paths), str(impl_paths))
    if isinstance(impl_paths, list):
        missing_impl = [path for path in impl_paths if nonempty_string(path) and not path_exists(path)]
        add_check(checks, "implementation_paths_exist", not missing_impl, "ok" if not missing_impl else "missing: " + ", ".join(missing_impl))

    outputs = gate.get("latest_outputs", [])
    add_check(checks, "latest_outputs_list", isinstance(outputs, list), str(outputs))
    if isinstance(outputs, list) and outputs:
        missing_outputs = [path for path in outputs if nonempty_string(path) and not path_exists(path)]
        add_check(checks, "latest_outputs_exist", not missing_outputs, "ok" if not missing_outputs else "missing: " + ", ".join(missing_outputs))
    elif gate.get("status") not in {"candidate", "planned"}:
        add_check(checks, "latest_outputs_present_for_evidence", False, "non-candidate gates need evidence outputs")
    else:
        add_check(checks, "latest_outputs_present_for_evidence", True, "candidate/planned gate")

    return {
        "gate_id": gate.get("gate_id"),
        "title": gate.get("title"),
        "family": gate.get("family"),
        "status": gate.get("status"),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    gates = registry.get("research_gates", [])
    add_check(checks, "research_gates_nonempty", isinstance(gates, list) and bool(gates), f"count={len(gates) if isinstance(gates, list) else 0}")

    ids = [gate.get("gate_id") for gate in gates if isinstance(gate, dict)]
    dupes = sorted([gate_id for gate_id, count in Counter(ids).items() if count > 1])
    add_check(checks, "gate_ids_unique", not dupes, "ok" if not dupes else "duplicates: " + ", ".join(map(str, dupes)))

    statuses = Counter(gate.get("status", "<missing>") for gate in gates if isinstance(gate, dict))
    families = Counter(gate.get("family", "<missing>") for gate in gates if isinstance(gate, dict))
    add_check(checks, "has_passed_research", statuses.get("pass", 0) >= 1, f"statuses={dict(statuses)}")
    add_check(checks, "has_negative_controls", statuses.get("rejected", 0) >= 1, f"statuses={dict(statuses)}")
    add_check(checks, "has_candidate_backlog", statuses.get("candidate", 0) >= 1, f"statuses={dict(statuses)}")
    add_check(checks, "family_coverage", len(families) >= 4, f"families={dict(families)}")

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "status_counts": dict(statuses),
        "family_counts": dict(families),
        "n_gates": len(gates),
    }


def write_markdown(path: Path, payload: dict[str, Any], registry: dict[str, Any]) -> None:
    gates_by_id = {gate["gate_id"]: gate for gate in registry.get("research_gates", [])}
    lines = [
        f"# Research Gate Registry Validation: {payload['created_at']}",
        "",
        f"- Registry: `{payload['registry']}`",
        f"- Gates: {payload['n_passed']} / {payload['n_gates']} passed",
        f"- Overall: `{'PASS' if payload['passed'] else 'FAIL'}`",
        f"- Status counts: `{payload['registry_summary']['status_counts']}`",
        "",
        "| Gate | Family | Status | Validation | Next Action |",
        "|---|---|---|---|---|",
    ]
    for result in payload["research_gates"]:
        gate = gates_by_id.get(result["gate_id"], {})
        lines.append(
            "| {gate_id} | {family} | {status} | {passed} | {next_action} |".format(
                gate_id=result["gate_id"],
                family=result.get("family") or "",
                status=result.get("status") or "",
                passed="PASS" if result["passed"] else "FAIL",
                next_action=gate.get("next_action", ""),
            )
        )

    lines.extend(["", "## Gate Catalog", ""])
    for result in payload["research_gates"]:
        gate = gates_by_id.get(result["gate_id"], {})
        lines.extend(
            [
                f"### {result['gate_id']}",
                "",
                f"- Title: {gate.get('title')}",
                f"- Status: `{gate.get('status')}`",
                f"- Hypothesis: {gate.get('hypothesis')}",
                f"- Verdict: {gate.get('verdict')}",
                f"- Decision rule: {gate.get('decision_rule')}",
                f"- Implementation paths: {'; '.join(gate.get('implementation_paths', []))}",
                f"- Latest outputs: {'; '.join(gate.get('latest_outputs', [])) if gate.get('latest_outputs') else '<none yet>'}",
                "",
            ]
        )

    lines.extend(["## Registry Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in payload["registry_summary"]["checks"]:
        lines.append(f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate research gate registry.")
    parser.add_argument("--registry", default=str(LAB_DIR / "registry" / "research_gates.json"))
    parser.add_argument("--output-json", default=str(LAB_DIR / "reports" / "research_gate_validation.json"))
    parser.add_argument("--output-md")
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser().resolve()
    registry = load_json(registry_path)
    gate_results = [validate_gate(gate) for gate in registry.get("research_gates", [])]
    registry_summary = validate_registry(registry)
    payload = {
        "created_at": utc_stamp(),
        "registry": str(registry_path),
        "n_gates": len(gate_results),
        "n_passed": sum(1 for gate in gate_results if gate["passed"]),
        "passed": registry_summary["passed"] and all(gate["passed"] for gate in gate_results),
        "registry_summary": registry_summary,
        "research_gates": gate_results,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, payload)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), payload, registry)

    print(
        f"[research-registry] gates={payload['n_passed']}/{payload['n_gates']} "
        f"status={'PASS' if payload['passed'] else 'FAIL'}"
    )
    print(f"[research-registry] wrote {output_json}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
