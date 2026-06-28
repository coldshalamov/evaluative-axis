#!/usr/bin/env python3
"""Validate, execute, and summarize the serious research system manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sanitize_key(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "default"


@dataclass
class LaneResult:
    lane_id: str
    status: str
    parser: str
    summary_path: str
    exists: bool
    metrics: dict[str, float]
    brief: str
    command: list[str] | None
    execution: dict[str, Any] | None


def validate_manifest(manifest: dict[str, Any]) -> None:
    lane_ids = [lane["id"] for lane in manifest.get("lanes", [])]
    if len(lane_ids) != len(set(lane_ids)):
        raise ValueError("Manifest has duplicate lane ids.")

    gate_ids = [gate["id"] for gate in manifest.get("gates", [])]
    if len(gate_ids) != len(set(gate_ids)):
        raise ValueError("Manifest has duplicate gate ids.")

    lane_id_set = set(lane_ids)
    gate_id_set = set(gate_ids)

    for lane in manifest.get("lanes", []):
        parser_name = lane["result_parser"]
        if parser_name not in PARSERS:
            raise ValueError(f"Unknown parser `{parser_name}` for lane `{lane['id']}`.")
        if not lane.get("summary_path"):
            raise ValueError(f"Lane `{lane['id']}` is missing `summary_path`.")

    for gate in manifest.get("gates", []):
        gate_type = gate["type"]
        if gate_type == "threshold":
            lane_id, _, _ = gate["metric_ref"].partition(".")
            if lane_id not in lane_id_set:
                raise ValueError(f"Gate `{gate['id']}` references unknown lane `{lane_id}`.")
        elif gate_type == "gap":
            for ref_name in ["positive_ref", "negative_ref"]:
                lane_id, _, _ = gate[ref_name].partition(".")
                if lane_id not in lane_id_set:
                    raise ValueError(f"Gate `{gate['id']}` references unknown lane `{lane_id}`.")
        elif gate_type == "count_gap":
            for comparison in gate["comparisons"]:
                for ref_name in ["positive_ref", "negative_ref"]:
                    lane_id, _, _ = comparison[ref_name].partition(".")
                    if lane_id not in lane_id_set:
                        raise ValueError(f"Gate `{gate['id']}` references unknown lane `{lane_id}`.")
        elif gate_type == "all":
            for gate_id in gate["depends_on"]:
                if gate_id not in gate_id_set:
                    raise ValueError(f"Gate `{gate['id']}` depends on unknown gate `{gate_id}`.")
        else:
            raise ValueError(f"Unknown gate type `{gate_type}` in manifest.")


def parse_objective_methods(path: Path) -> tuple[dict[str, float], str]:
    data = read_json(path)
    if all(isinstance(v, dict) and "solve_rate" in v for v in data.values()):
        models = {"default": data}
        aliases = {"default": "default"}
    else:
        models = {}
        aliases = {}
        for raw_name, methods in data.items():
            alias = sanitize_key(raw_name)
            aliases[alias] = raw_name
            models[alias] = methods

    flat: dict[str, float] = {}
    briefs = []
    for alias, methods in models.items():
        baseline_best = max(methods[name]["solve_rate"] for name in methods if name in {"random", "length"})
        direct_names = [name for name in methods if name.startswith("direct_")]
        best_direct_name = max(direct_names, key=lambda name: methods[name]["solve_rate"])
        best_direct = methods[best_direct_name]["solve_rate"]
        flat[f"{alias}.baseline_best_solve_rate"] = baseline_best
        flat[f"{alias}.best_direct_solve_rate"] = best_direct
        flat[f"{alias}.best_direct_minus_baseline"] = best_direct - baseline_best
        flat[f"{alias}.best_direct_method_name"] = 0.0
        for method_name, row in methods.items():
            for metric_name, value in row.items():
                if isinstance(value, (int, float)):
                    flat[f"{alias}.method.{method_name}.{metric_name}"] = float(value)
        raw_name = aliases.get(alias, alias)
        briefs.append(
            f"{raw_name}: best direct `{best_direct_name}` = {best_direct:.1%}, "
            f"best baseline = {baseline_best:.1%}"
        )
    return flat, "; ".join(briefs)


def parse_battery_summary(path: Path) -> tuple[dict[str, float], str]:
    data = read_json(path)
    flat: dict[str, float] = {}
    for metric_name, row in data.get("metrics", {}).items():
        flat[f"metric.{metric_name}.accuracy"] = float(row["accuracy"])
        flat[f"metric.{metric_name}.n"] = float(row["n"])
    for category, metrics in data.get("categories", {}).items():
        for metric_name, row in metrics.items():
            flat[f"category.{category}.metric.{metric_name}.accuracy"] = float(row["accuracy"])
    brief_parts = []
    for name in ["direct_combined", "direct_category_axis", "decomposition_category_axis"]:
        key = f"metric.{name}.accuracy"
        if key in flat:
            brief_parts.append(f"{name} = {flat[key]:.1%}")
    return flat, ", ".join(brief_parts) if brief_parts else "battery summary loaded"


def parse_generic_json(path: Path) -> tuple[dict[str, float], str]:
    data = read_json(path)
    flat = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            flat[key] = float(value)
    return flat, "generic summary loaded"


PARSERS = {
    "objective_methods": parse_objective_methods,
    "battery_summary": parse_battery_summary,
    "generic_json": parse_generic_json,
}


def load_lane_result(lane: dict[str, Any]) -> LaneResult:
    summary_path = ROOT / lane["summary_path"]
    parser_name = lane["result_parser"]
    metrics: dict[str, float] = {}
    brief = "no results yet"
    if summary_path.exists():
        metrics, brief = PARSERS[parser_name](summary_path)
    return LaneResult(
        lane_id=lane["id"],
        status=lane["status"],
        parser=parser_name,
        summary_path=lane["summary_path"],
        exists=summary_path.exists(),
        metrics=metrics,
        brief=brief,
        command=lane.get("command"),
        execution=None,
    )


def resolve_metric(ref: str, results: dict[str, LaneResult]) -> float | None:
    lane_id, metric_key = ref.split(".", 1)
    lane = results.get(lane_id)
    if not lane:
        return None
    value = lane.metrics.get(metric_key)
    return value


def evaluate_gate(gate: dict[str, Any], results: dict[str, LaneResult], gate_states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gate_type = gate["type"]
    if gate_type == "threshold":
        value = resolve_metric(gate["metric_ref"], results)
        if value is None:
            return {"status": "pending", "detail": f"Missing metric `{gate['metric_ref']}`"}
        passed = value >= gate["min_value"]
        return {"status": "pass" if passed else "fail", "detail": f"{value:.3f} >= {gate['min_value']:.3f}"}

    if gate_type == "gap":
        positive = resolve_metric(gate["positive_ref"], results)
        negative = resolve_metric(gate["negative_ref"], results)
        if positive is None or negative is None:
            return {"status": "pending", "detail": "Missing metric for gap comparison"}
        gap = positive - negative
        passed = gap >= gate["min_gap"]
        return {"status": "pass" if passed else "fail", "detail": f"gap {gap:.3f} >= {gate['min_gap']:.3f}"}

    if gate_type == "count_gap":
        pass_count = 0
        pending_count = 0
        total = 0
        details = []
        for comparison in gate["comparisons"]:
            positive = resolve_metric(comparison["positive_ref"], results)
            negative = resolve_metric(comparison["negative_ref"], results)
            total += 1
            if positive is None or negative is None:
                pending_count += 1
                details.append("pending")
                continue
            gap = positive - negative
            if gap >= comparison["min_gap"]:
                pass_count += 1
                details.append(f"pass({gap:.3f})")
            else:
                details.append(f"fail({gap:.3f})")
        if pass_count >= gate["min_pass_count"]:
            return {
                "status": "pass",
                "detail": f"{pass_count}/{total} comparisons passed; {', '.join(details)}",
            }
        if pass_count + pending_count >= gate["min_pass_count"]:
            return {"status": "pending", "detail": ", ".join(details)}
        return {
            "status": "fail",
            "detail": f"{pass_count}/{total} comparisons passed; {', '.join(details)}",
        }

    if gate_type == "all":
        statuses = []
        for gate_id in gate["depends_on"]:
            statuses.append(gate_states.get(gate_id, {}).get("status", "pending"))
        if any(status == "pending" for status in statuses):
            return {"status": "pending", "detail": ", ".join(f"{gid}={gate_states.get(gid, {}).get('status', 'pending')}" for gid in gate["depends_on"])}
        passed = all(status == "pass" for status in statuses)
        return {"status": "pass" if passed else "fail", "detail": ", ".join(f"{gid}={gate_states[gid]['status']}" for gid in gate["depends_on"])}

    raise ValueError(f"Unknown gate type: {gate_type}")


def format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def execute_lane(lane: dict[str, Any]) -> dict[str, Any]:
    command = lane.get("command")
    if not command:
        return {"status": "skipped", "returncode": None, "reason": "No command defined"}
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def build_report(manifest: dict[str, Any], lane_results: dict[str, LaneResult], gate_states: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    completed = sum(1 for lane in lane_results.values() if lane.exists)
    ready = sum(1 for lane in lane_results.values() if lane.status == "ready" and not lane.exists)
    claim_rows = []
    for claim in manifest.get("claims", []):
        supporting = [lane for lane in manifest["lanes"] if claim["id"] in lane.get("claims_supported", [])]
        results_count = sum(1 for lane in supporting if lane_results[lane["id"]].exists)
        claim_rows.append(
            {
                "id": claim["id"],
                "title": claim["title"],
                "supporting_lanes": [lane["id"] for lane in supporting],
                "lanes_with_results": results_count,
                "lane_count": len(supporting),
            }
        )
    report = {
        "program_name": manifest["program_name"],
        "objective": manifest["objective"],
        "completed_lanes_with_results": completed,
        "ready_lanes": ready,
        "claims": claim_rows,
        "gates": gate_states,
        "lanes": {
            lane_id: {
                "status": lane.status,
                "has_results": lane.exists,
                "summary_path": lane.summary_path,
                "brief": lane.brief,
                "metrics": lane.metrics,
                "execution": lane.execution,
            }
            for lane_id, lane in lane_results.items()
        },
    }

    lines = [
        f"# {manifest['program_name']}",
        "",
        manifest["objective"],
        "",
        f"- lanes with results: {completed}",
        f"- lanes ready to run: {ready}",
        "",
        "## Claims",
        "",
        "| Claim | Coverage | Standard |",
        "| --- | --- | --- |",
    ]
    for claim in claim_rows:
        lines.append(
            f"| `{claim['id']}` | {claim['lanes_with_results']} / {claim['lane_count']} supporting lanes with results | "
            f"{next(item['standard'] for item in manifest['claims'] if item['id'] == claim['id'])} |"
        )

    lines.extend(
        [
            "",
        "## Claim Gates",
        "",
        "| Gate | Status | Detail |",
        "| --- | --- | --- |",
        ]
    )
    for gate in manifest["gates"]:
        state = gate_states[gate["id"]]
        lines.append(f"| `{gate['id']}` | `{state['status']}` | {state['detail']} |")

    lines.extend(
        [
            "",
            "## Lane Status",
            "",
            "| Lane | Family | Status | Results | Brief |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for lane in manifest["lanes"]:
        result = lane_results[lane["id"]]
        lines.append(
            f"| `{lane['id']}` | `{lane['family']}` | `{result.status}` | "
            f"`{'yes' if result.exists else 'no'}` | {result.brief} |"
        )

    lines.extend(["", "## Immediate Next Runs", ""])
    lane_priority_map = {lane["id"]: lane.get("priority", 999) for lane in manifest["lanes"]}
    for lane in sorted(manifest["lanes"], key=lambda item: (lane_priority_map[item["id"]], item["id"])):
        result = lane_results[lane["id"]]
        if result.status != "ready":
            continue
        if result.exists and (not result.execution or result.execution.get("status") == "ok"):
            continue
        lines.append(f"- `{lane['id']}`")
        if result.command:
            lines.append(f"  Command: `{format_command(result.command)}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A lane result is evidence, not proof by itself. The system is only strong when multiple objective lanes agree, cheap baselines stay visible, and the capacity gap persists on the same frozen benchmark.",
        ]
    )
    return report, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-ready", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    validate_manifest(manifest)
    selected = set(args.only or [])
    lanes = [
        lane
        for lane in manifest["lanes"]
        if not selected or lane["id"] in selected
    ]
    manifest = {**manifest, "lanes": lanes}

    lane_results = {lane["id"]: load_lane_result(lane) for lane in lanes}

    if args.execute_ready:
        for lane in lanes:
            if lane["status"] != "ready":
                continue
            execution = execute_lane(lane)
            refreshed = load_lane_result(lane)
            refreshed.execution = execution
            lane_results[lane["id"]] = refreshed

    gate_states: dict[str, dict[str, Any]] = {}
    for gate in manifest["gates"]:
        gate_states[gate["id"]] = evaluate_gate(gate, lane_results, gate_states)

    report_json, report_md = build_report(manifest, lane_results, gate_states)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", report_json)
    (args.output / "report.md").write_text(report_md, encoding="utf-8")
    print(report_md)


if __name__ == "__main__":
    main()
