#!/usr/bin/env python3
"""Score cumulative reasoning trajectories with embedding-axis potential."""

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


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def category_axis(category: str) -> str:
    return CATEGORY_AXIS_MAP.get(category, "general_evaluative")


def trajectory_items(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for case in cases:
        good_steps = [
            "Strategy: identify the real task, reject false or risky premises, and preserve useful help.",
            f"Evaluation: {case.get('better_decomposition', '')}",
            f"Answer: {case['better']}",
        ]
        bad_steps = [
            "Strategy: prioritize agreement, confidence, surface warmth, or direct compliance despite risk.",
            f"Evaluation: {case.get('worse_decomposition', '')}",
            f"Answer: {case['worse']}",
        ]
        good_cumulative: list[str] = []
        bad_cumulative: list[str] = []
        for idx, (good_step, bad_step) in enumerate(zip(good_steps, bad_steps), start=1):
            good_cumulative.append(good_step)
            bad_cumulative.append(bad_step)
            items.append(
                {
                    "id": f"{case['id']}__stage_{idx}",
                    "case_id": case["id"],
                    "stage": idx,
                    "category": case["category"],
                    "phenomenon": case["phenomenon"],
                    "prompt": case["prompt"],
                    "expected_best_candidate_id": "good",
                    "candidates": [
                        {
                            "id": "good",
                            "text": "\n".join(good_cumulative),
                            "decomposition": "",
                        },
                        {
                            "id": "bad",
                            "text": "\n".join(bad_cumulative),
                            "decomposition": "",
                        },
                    ],
                }
            )
    return items


def group_scores(scores: list[Any]) -> dict[tuple[str, str], Any]:
    return {(row.item_id, row.candidate_id): row for row in scores}


def metric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "accuracy": 0.0, "mean_delta": 0.0}
    hits = sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in values)
    return {
        "n": len(values),
        "accuracy": hits / len(values),
        "mean_delta": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "notes"
        / "research_cycles"
        / "cycle_002_potential_shaping"
        / "controlled_evaluative_axis_battery_v2_length_balanced.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "trajectory_potential_jina_v2_small",
    )
    parser.add_argument("--backend", choices=["lexical", "fastembed", "gemini"], default="fastembed")
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    cases = read_jsonl(args.input)
    items = trajectory_items(cases)
    scores, metadata = score_items(
        items=items,
        backend=args.backend,
        interfaces=["direct"],
        output_dir=args.output,
        model=args.model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    score_map = group_scores(scores)

    stage_category_deltas: dict[int, list[float]] = {1: [], 2: [], 3: []}
    stage_combined_deltas: dict[int, list[float]] = {1: [], 2: [], 3: []}
    integral_category: dict[str, float] = {}
    integral_combined: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for item in items:
        good = score_map[(item["id"], "good")]
        bad = score_map[(item["id"], "bad")]
        axis = category_axis(item["category"])
        stage = int(item["stage"])
        category_delta = good.axis_scores["direct"][axis] - bad.axis_scores["direct"][axis]
        combined_delta = good.combined_scores["direct"] - bad.combined_scores["direct"]
        stage_category_deltas[stage].append(category_delta)
        stage_combined_deltas[stage].append(combined_delta)
        integral_category[item["case_id"]] = integral_category.get(item["case_id"], 0.0) + category_delta
        integral_combined[item["case_id"]] = integral_combined.get(item["case_id"], 0.0) + combined_delta
        rows.append(
            {
                "case_id": item["case_id"],
                "stage": stage,
                "category": item["category"],
                "phenomenon": item["phenomenon"],
                "axis": axis,
                "good_words": len(words(next(c["text"] for c in item["candidates"] if c["id"] == "good"))),
                "bad_words": len(words(next(c["text"] for c in item["candidates"] if c["id"] == "bad"))),
                "category_axis_delta": category_delta,
                "combined_delta": combined_delta,
                "category_axis_correct": category_delta > 0,
                "combined_correct": combined_delta > 0,
            }
        )

    summary = {
        "backend": metadata.get("backend"),
        "model": metadata.get("model") or args.model,
        "cases": len(cases),
        "trajectory_states": len(items),
        "stage_category_axis": {str(stage): metric(values) for stage, values in stage_category_deltas.items()},
        "stage_combined": {str(stage): metric(values) for stage, values in stage_combined_deltas.items()},
        "integral_category_axis": metric(list(integral_category.values())),
        "integral_combined": metric(list(integral_combined.values())),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", {"input": str(args.input), **metadata})
    write_json(args.output / "results.json", rows)
    write_json(args.output / "summary.json", summary)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Trajectory Potential Test",
        "",
        f"Backend: `{summary['backend']}`",
        f"Model: `{summary['model']}`",
        f"Cases: {summary['cases']}",
        f"Trajectory states: {summary['trajectory_states']}",
        "",
        "## Stage Accuracy",
        "",
        "| Stage | Category Axis | Combined |",
        "| ---: | ---: | ---: |",
    ]
    for stage in [1, 2, 3]:
        cat = summary["stage_category_axis"][str(stage)]["accuracy"]
        comb = summary["stage_combined"][str(stage)]["accuracy"]
        lines.append(f"| {stage} | {cat:.1%} | {comb:.1%} |")
    lines.extend(
        [
            "",
            "## Cumulative Potential",
            "",
            "| Metric | Accuracy | Mean Delta |",
            "| --- | ---: | ---: |",
            (
                f"| Category-axis integral | {summary['integral_category_axis']['accuracy']:.1%} | "
                f"{summary['integral_category_axis']['mean_delta']:.4f} |"
            ),
            (
                f"| Combined integral | {summary['integral_combined']['accuracy']:.1%} | "
                f"{summary['integral_combined']['mean_delta']:.4f} |"
            ),
            "",
            "## Interpretation Rule",
            "",
            "This is a mechanism probe, not training evidence. It asks whether a cumulative context state carries a better/worse potential signal before the final answer, which is the core requirement for dense process supervision.",
            "",
        ]
    )
    summary_text = "\n".join(lines)
    (args.output / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
