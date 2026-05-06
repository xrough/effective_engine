#!/usr/bin/env python3
"""Validate and render the lab volatility strategy registry."""

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
    "strategy_id",
    "family",
    "asset_scope",
    "signal_inputs",
    "hedger",
    "expected_edge",
    "current_status",
    "replay_command",
    "known_risks",
]
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


def is_nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)


def validate_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [field for field in REQUIRED_FIELDS if field not in strategy]
    add_check(checks, "required_fields", not missing, "ok" if not missing else "missing: " + ", ".join(missing))

    strategy_id = strategy.get("strategy_id", "")
    add_check(checks, "strategy_id_format", isinstance(strategy_id, str) and bool(ID_RE.match(strategy_id)), str(strategy_id))
    add_check(checks, "asset_scope_nonempty", is_nonempty_list(strategy.get("asset_scope")), str(strategy.get("asset_scope")))
    add_check(checks, "signal_inputs_nonempty", is_nonempty_list(strategy.get("signal_inputs")), f"count={len(strategy.get('signal_inputs', [])) if isinstance(strategy.get('signal_inputs'), list) else 0}")
    add_check(checks, "known_risks_nonempty", is_nonempty_list(strategy.get("known_risks")), f"count={len(strategy.get('known_risks', [])) if isinstance(strategy.get('known_risks'), list) else 0}")

    for field in ["family", "hedger", "expected_edge", "current_status", "replay_command"]:
        value = strategy.get(field)
        add_check(checks, f"{field}_nonempty", isinstance(value, str) and bool(value.strip()), str(value))

    replay_command = strategy.get("replay_command", "")
    command_ok = isinstance(replay_command, str) and (
        replay_command.startswith("cd demo &&")
        or replay_command.startswith("python3 ")
        or replay_command.startswith("rg ")
    )
    add_check(checks, "replay_command_shape", command_ok, replay_command)

    return {
        "strategy_id": strategy.get("strategy_id"),
        "family": strategy.get("family"),
        "asset_scope": strategy.get("asset_scope", []),
        "hedger": strategy.get("hedger"),
        "current_status": strategy.get("current_status"),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    strategies = registry.get("strategies", [])
    add_check(checks, "strategies_nonempty", isinstance(strategies, list) and bool(strategies), f"count={len(strategies) if isinstance(strategies, list) else 0}")

    ids = [strategy.get("strategy_id") for strategy in strategies if isinstance(strategy, dict)]
    dupes = sorted([strategy_id for strategy_id, count in Counter(ids).items() if count > 1])
    add_check(checks, "strategy_ids_unique", not dupes, "ok" if not dupes else "duplicates: " + ", ".join(map(str, dupes)))

    families = Counter(strategy.get("family", "<missing>") for strategy in strategies if isinstance(strategy, dict))
    add_check(checks, "family_coverage", len(families) >= 3, f"families={dict(families)}")

    tradable = [s for s in strategies if s.get("hedger") not in {"not-wired", "research-only"}]
    add_check(checks, "tradable_strategy_available", bool(tradable), f"tradable={len(tradable)}")

    components = [s for s in strategies if s.get("family") == "signal-component"]
    add_check(checks, "component_backlog_available", bool(components), f"components={len(components)}")

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "family_counts": dict(families),
        "n_strategies": len(strategies),
        "n_tradable": len(tradable),
        "n_components": len(components),
    }


def write_markdown(path: Path, payload: dict[str, Any], registry: dict[str, Any]) -> None:
    strategies_by_id = {strategy["strategy_id"]: strategy for strategy in registry.get("strategies", [])}
    lines = [
        f"# Strategy Registry Validation: {payload['created_at']}",
        "",
        f"- Registry: `{payload['registry']}`",
        f"- Strategies: {payload['n_passed']} / {payload['n_strategies']} passed",
        f"- Overall: `{'PASS' if payload['passed'] else 'FAIL'}`",
        f"- Families: `{payload['registry_summary']['family_counts']}`",
        "",
        "| Strategy | Family | Hedger | Status | Assets | Validation |",
        "|---|---|---|---|---|---|",
    ]
    for result in payload["strategies"]:
        strategy = strategies_by_id.get(result["strategy_id"], {})
        assets = ", ".join(strategy.get("asset_scope", []))
        lines.append(
            "| {sid} | {family} | {hedger} | {status} | {assets} | {passed} |".format(
                sid=result["strategy_id"],
                family=result.get("family") or "",
                hedger=result.get("hedger") or "",
                status=result.get("current_status") or "",
                assets=assets,
                passed="PASS" if result["passed"] else "FAIL",
            )
        )

    lines.extend(["", "## Strategy Catalog", ""])
    for result in payload["strategies"]:
        strategy = strategies_by_id.get(result["strategy_id"], {})
        lines.extend(
            [
                f"### {result['strategy_id']}",
                "",
                f"- Family: `{strategy.get('family')}`",
                f"- Hedger: `{strategy.get('hedger')}`",
                f"- Status: `{strategy.get('current_status')}`",
                f"- Expected edge: {strategy.get('expected_edge')}",
                f"- Replay command: `{strategy.get('replay_command')}`",
                f"- Signal inputs: {'; '.join(strategy.get('signal_inputs', []))}",
                f"- Known risks: {'; '.join(strategy.get('known_risks', []))}",
                "",
            ]
        )

    lines.extend(["## Registry Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in payload["registry_summary"]["checks"]:
        lines.append(f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate strategy registry.")
    parser.add_argument("--registry", default=str(LAB_DIR / "registry" / "strategies.json"))
    parser.add_argument("--output-json", default=str(LAB_DIR / "reports" / "strategy_registry_validation.json"))
    parser.add_argument("--output-md")
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser().resolve()
    registry = load_json(registry_path)
    strategies = registry.get("strategies", [])
    strategy_results = [validate_strategy(strategy) for strategy in strategies]
    registry_summary = validate_registry(registry)
    payload = {
        "created_at": utc_stamp(),
        "registry": str(registry_path),
        "n_strategies": len(strategy_results),
        "n_passed": sum(1 for result in strategy_results if result["passed"]),
        "passed": registry_summary["passed"] and all(result["passed"] for result in strategy_results),
        "registry_summary": registry_summary,
        "strategies": strategy_results,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, payload)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), payload, registry)

    print(
        f"[strategy-registry] strategies={payload['n_passed']}/{payload['n_strategies']} "
        f"status={'PASS' if payload['passed'] else 'FAIL'}"
    )
    print(f"[strategy-registry] wrote {output_json}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
