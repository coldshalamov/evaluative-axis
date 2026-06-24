#!/usr/bin/env python3
"""Run controlled evaluative-axis minimal-pair battery."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES, CATEGORY_AXIS_MAP, score_items  # noqa: E402


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def as_items(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for case in cases:
        items.append(
            {
                "id": case["id"],
                "category": case["category"],
                "phenomenon": case["phenomenon"],
                "prompt": case["prompt"],
                "expected_best_candidate_id": "better",
                "candidates": [
                    {
                        "id": "better",
                        "text": case["better"],
                        "decomposition": case.get("better_decomposition", ""),
                    },
                    {
                        "id": "worse",
                        "text": case["worse"],
                        "decomposition": case.get("worse_decomposition", ""),
                    },
                ],
            }
        )
    return items


def group_scores(scores: list[Any]) -> dict[tuple[str, str], Any]:
    return {(row.item_id, row.candidate_id): row for row in scores}


def diff_correct(diff: float) -> float:
    if diff > 0:
        return 1.0
    if diff == 0:
        return 0.5
    return 0.0


def add_metric(summary: dict[str, list[float]], name: str, diff: float) -> None:
    summary.setdefault(name, []).append(diff_correct(diff))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "battery_fastembed_bge_small",
    )
    parser.add_argument("--backend", choices=["lexical", "fastembed", "gemini"], default="fastembed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--interfaces", default="direct,decomposition")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    cases = read_jsonl(args.input)
    items = as_items(cases)
    interfaces = [part.strip() for part in args.interfaces.split(",") if part.strip()]
    scores, metadata = score_items(
        items=items,
        backend=args.backend,
        interfaces=interfaces,
        output_dir=args.output,
        model=args.model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    score_map = group_scores(scores)

    rows = []
    metric_diffs: dict[str, list[float]] = {}
    category_diffs: dict[str, dict[str, list[float]]] = {}
    length_abs_gaps = []

    for case in cases:
        better = score_map[(case["id"], "better")]
        worse = score_map[(case["id"], "worse")]
        better_len = len(words(case["better"]))
        worse_len = len(words(case["worse"]))
        length_gap = better_len - worse_len
        length_abs_gaps.append(abs(length_gap))

        diffs: dict[str, float] = {
            "length": better.baselines["length"] - worse.baselines["length"],
            "sentiment": better.baselines["sentiment"] - worse.baselines["sentiment"],
            "refusal": better.baselines["refusal"] - worse.baselines["refusal"],
        }
        for interface in interfaces:
            diffs[f"{interface}_combined"] = better.combined_scores[interface] - worse.combined_scores[interface]
            category_axis = CATEGORY_AXIS_MAP.get(case["category"], "general_evaluative")
            diffs[f"{interface}_category_axis"] = (
                better.axis_scores[interface][category_axis]
                - worse.axis_scores[interface][category_axis]
            )
            for axis_name in AXES:
                diffs[f"{interface}_{axis_name}"] = (
                    better.axis_scores[interface][axis_name]
                    - worse.axis_scores[interface][axis_name]
                )

        for name, diff in diffs.items():
            add_metric(metric_diffs, name, diff)
            category_diffs.setdefault(case["category"], {}).setdefault(name, []).append(diff_correct(diff))

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "phenomenon": case["phenomenon"],
                "better_len": better_len,
                "worse_len": worse_len,
                "length_gap_better_minus_worse": length_gap,
                **{f"diff_{name}": value for name, value in sorted(diffs.items())},
                **{f"correct_{name}": diff_correct(value) for name, value in sorted(diffs.items())},
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", {"backend": args.backend, "model": args.model, **metadata})
    write_json(args.output / "results.json", rows)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_summary = {
        name: {
            "n": len(vals),
            "accuracy": sum(vals) / len(vals),
        }
        for name, vals in sorted(metric_diffs.items())
    }
    category_summary = {
        category: {
            name: {
                "n": len(vals),
                "accuracy": sum(vals) / len(vals),
            }
            for name, vals in sorted(metrics.items())
        }
        for category, metrics in sorted(category_diffs.items())
    }
    write_json(args.output / "summary.json", {
        "metrics": metric_summary,
        "categories": category_summary,
        "length_abs_gap_mean": sum(length_abs_gaps) / len(length_abs_gaps),
        "length_abs_gap_max": max(length_abs_gaps),
    })

    preferred_order = [
        "length",
        "sentiment",
        "refusal",
        "direct_combined",
        "direct_category_axis",
        "decomposition_combined",
        "decomposition_category_axis",
        "direct_general_evaluative",
        "direct_truthfulness",
        "direct_harm_reduction",
        "direct_persona_honesty",
        "direct_anti_sycophancy",
    ]
    lines = [
        "# Controlled Evaluative-Axis Battery",
        "",
        f"Backend: `{args.backend}`",
        f"Model: `{args.model or metadata.get('model') or 'default'}`",
        f"Cases: {len(cases)}",
        f"Mean absolute length gap: {sum(length_abs_gaps) / len(length_abs_gaps):.2f} words",
        f"Max absolute length gap: {max(length_abs_gaps)} words",
        "",
        "## Pairwise Accuracy",
        "",
        "| Method | Accuracy | n |",
        "| --- | ---: | ---: |",
    ]
    seen = set()
    for name in preferred_order + sorted(metric_summary):
        if name in seen or name not in metric_summary:
            continue
        seen.add(name)
        row = metric_summary[name]
        lines.append(f"| `{name}` | {row['accuracy']:.1%} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.",
            "",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
