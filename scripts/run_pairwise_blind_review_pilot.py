#!/usr/bin/env python3
"""Run a small blinded pairwise-review pilot for selected intervention methods."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_combined_summary(results: list[dict[str, Any]], model: str, sample_size: int) -> str:
    lines = [
        "# Pairwise Blind Review Pilot",
        "",
        f"Judge model: `{model}`",
        f"Max sampled rows per comparison: {sample_size}",
        "",
        "This pilot uses blinded LLM adjudication as a sensor, not as human gold truth.",
        "",
        "## Aggregate Results",
        "",
        "| Focus method | Baseline | Focus wins | Baseline wins | Ties | Focus win rate (decided) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| `{row['focus_method']}` | `{row['baseline_method']}` | "
            f"{row['focus_wins']} | {row['baseline_wins']} | {row['ties']} | "
            f"{row['focus_win_rate_decided']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Treat wins over sentiment as weak evidence. The more important comparisons are versus random, length, and refusal heuristics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selections",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_001_next" / "pilot_50_fastembed_bge_small" / "selections.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_007_pairwise_blind_review_pilot" / "runs",
    )
    parser.add_argument(
        "--focus-methods",
        default="direct_category_axis,direct_anti_sycophancy",
        help="Comma-separated focus methods to pilot.",
    )
    parser.add_argument(
        "--baseline-methods",
        default="length,random,sentiment,refusal_heuristic",
        help="Comma-separated baseline methods to compare against.",
    )
    parser.add_argument("--max-per-comparison", type=int, default=10)
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    args = parser.parse_args()

    focus_methods = [part.strip() for part in args.focus_methods.split(",") if part.strip()]
    baseline_methods = [part.strip() for part in args.baseline_methods.split(",") if part.strip()]

    combined_rows: list[dict[str, Any]] = []

    for focus_method in focus_methods:
        run_dir = args.output_root / focus_method
        run_cmd(
            [
                sys.executable,
                "scripts/build_pairwise_review.py",
                "--selections",
                str(args.selections),
                "--output",
                str(run_dir),
                "--focus-methods",
                focus_method,
                "--baseline-methods",
                ",".join(baseline_methods),
                "--max-per-comparison",
                str(args.max_per_comparison),
            ]
        )
        run_cmd(
            [
                sys.executable,
                "scripts/judge_pairwise_with_gemini.py",
                "--review",
                str(run_dir / "review_packet.jsonl"),
                "--output",
                str(run_dir / "gemini_review.jsonl"),
                "--raw-output",
                str(run_dir / "gemini_review_raw.json"),
                "--summary-output",
                str(run_dir / "gemini_review_summary.md"),
                "--model",
                args.model,
            ]
        )
        analysis_dir = run_dir / "analysis"
        run_cmd(
            [
                sys.executable,
                "scripts/analyze_pairwise_review.py",
                "--review",
                str(run_dir / "gemini_review.jsonl"),
                "--key",
                str(run_dir / "review_key.json"),
                "--output",
                str(analysis_dir),
            ]
        )
        aggregate_rows = read_json(analysis_dir / "aggregate.json")
        for row in aggregate_rows:
            comparison = row["comparison"]
            focus_name, baseline_name = comparison.split(" vs ", 1)
            combined_rows.append(
                {
                    "focus_method": focus_name,
                    "baseline_method": baseline_name,
                    **row,
                }
            )

    combined_rows.sort(key=lambda row: (row["focus_method"], row["baseline_method"]))
    write_json(args.output_root / "combined_results.json", combined_rows)
    write_text(
        args.output_root / "combined_summary.md",
        build_combined_summary(combined_rows, model=args.model, sample_size=args.max_per_comparison),
    )
    print(f"Wrote pairwise blind review pilot to {args.output_root}")


if __name__ == "__main__":
    main()
