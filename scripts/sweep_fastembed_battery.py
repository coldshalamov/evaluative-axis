#!/usr/bin/env python3
"""Sweep FastEmbed models on the controlled evaluative-axis battery."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATTERY = (
    ROOT
    / "notes"
    / "research_cycles"
    / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v2_length_balanced.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "notes"
    / "research_cycles"
    / "cycle_002_potential_shaping"
    / "fastembed_model_sweep_v2"
)


DEFAULT_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-base",
    "snowflake/snowflake-arctic-embed-m",
    "jinaai/jina-embeddings-v2-small-en",
    "jinaai/jina-embeddings-v2-base-en",
    "nomic-ai/nomic-embed-text-v1.5-Q",
    "mixedbread-ai/mxbai-embed-large-v1",
]


KEY_METRICS = [
    "length",
    "sentiment",
    "refusal",
    "direct_combined",
    "direct_category_axis",
    "direct_general_evaluative",
    "direct_truthfulness",
    "direct_harm_reduction",
    "direct_persona_honesty",
    "direct_anti_sycophancy",
    "decomposition_combined",
    "decomposition_category_axis",
    "decomposition_general_evaluative",
    "decomposition_truthfulness",
    "decomposition_harm_reduction",
    "decomposition_persona_honesty",
    "decomposition_anti_sycophancy",
]


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_model(model: str, battery: Path, output_root: Path) -> dict[str, Any]:
    out_dir = output_root / slug(model)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_evaluative_axis_battery.py"),
        "--backend",
        "fastembed",
        "--model",
        model,
        "--input",
        str(battery),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    result = {
        "model": model,
        "output_dir": str(out_dir),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode == 0 and (out_dir / "summary.json").exists():
        summary = read_json(out_dir / "summary.json")
        result["metrics"] = summary["metrics"]
        result["length_abs_gap_mean"] = summary["length_abs_gap_mean"]
        result["length_abs_gap_max"] = summary["length_abs_gap_max"]
    return result


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# FastEmbed Model Sweep: Length-Balanced Battery V2",
        "",
        "Date: June 23, 2026",
        "",
        "Battery: `controlled_evaluative_axis_battery_v2_length_balanced.jsonl`",
        "",
        "All candidate pairs in this battery have exact word-count ties, so the",
        "length baseline should be 50%. The sample is small (12 pairs), so this",
        "is a diagnostic sweep, not a publishable model ranking.",
        "",
        "## Key Metrics",
        "",
        "| Model | Status | Best Axis | Best Acc | Direct Combined | Direct Broad | Direct Category | Decomp Category | Anti-Syc |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if result.get("returncode") != 0 or "metrics" not in result:
            lines.append(f"| `{result['model']}` | failed | - | - | - | - | - | - | - |")
            continue
        metrics = result["metrics"]
        embedding_metrics = {
            name: row["accuracy"]
            for name, row in metrics.items()
            if name not in {"length", "sentiment", "refusal"}
        }
        best_name, best_acc = max(embedding_metrics.items(), key=lambda item: item[1])
        lines.append(
            "| `{model}` | ok | `{best}` | {best_acc:.1%} | {direct_combined:.1%} | "
            "{direct_broad:.1%} | {direct_category:.1%} | {decomp_category:.1%} | {anti:.1%} |".format(
                model=result["model"],
                best=best_name,
                best_acc=best_acc,
                direct_combined=metrics.get("direct_combined", {}).get("accuracy", 0.0),
                direct_broad=metrics.get("direct_general_evaluative", {}).get("accuracy", 0.0),
                direct_category=metrics.get("direct_category_axis", {}).get("accuracy", 0.0),
                decomp_category=metrics.get("decomposition_category_axis", {}).get("accuracy", 0.0),
                anti=metrics.get("direct_anti_sycophancy", {}).get("accuracy", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "This sweep answers a narrow question: whether a stronger local ONNX",
            "embedding model changes the result on the same exact-length battery.",
            "If no model beats cheap baselines except on narrow axes, the next move",
            "is trajectory-potential testing and/or Gemini, not more final-answer",
            "BGE-style scoring.",
            "",
        ]
    )

    failures = [result for result in results if result.get("returncode") != 0]
    if failures:
        lines.extend(["## Failures", ""])
        for result in failures:
            stderr = (result.get("stderr") or "").strip().splitlines()
            tail = " | ".join(stderr[-3:]) if stderr else "no stderr"
            lines.append(f"- `{result['model']}`: {tail}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    models = [part.strip() for part in args.models.split(",") if part.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in models:
        print(f"=== {model} ===", flush=True)
        result = run_model(model, args.battery, args.output)
        print(f"returncode={result['returncode']}", flush=True)
        if result.get("stdout"):
            print(result["stdout"], flush=True)
        if result.get("stderr"):
            print(result["stderr"], file=sys.stderr, flush=True)
        results.append(result)
        write_json(args.output / "sweep_results.json", results)
        write_markdown(args.output / "summary.md", results)
    print(f"Wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
