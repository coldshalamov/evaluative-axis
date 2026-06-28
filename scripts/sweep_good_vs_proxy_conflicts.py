#!/usr/bin/env python3
"""Sweep local embedding models on the good-vs-proxy conflict battery."""

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
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "notes"
    / "research_cycles"
    / "cycle_010_oss_good_vs_proxy_sweep"
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


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


def run_model(model: str, battery: Path, output_root: Path) -> dict[str, Any]:
    out_dir = output_root / slug(model)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_good_vs_proxy_conflicts.py"),
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
        result["categories"] = summary["categories"]
        result["comparison"] = summary["comparison"]
    return result


def write_markdown(path: Path, results: list[dict[str, Any]], battery: Path) -> None:
    lines = [
        "# Good vs Proxy Conflict Sweep",
        "",
        "Date: June 27, 2026",
        "",
        f"Battery: `{battery.name}`",
        "",
        "This sweep maps whether raw `good/bad` and nearby proxy-word axes recover",
        "any broad evaluative behavior on the frozen 50-case conflict battery for",
        "the local models we can actually run on this laptop.",
        "",
        "## Key Metrics",
        "",
        "| Model | Status | raw good/bad | sentence good/bad | proxy mean | best proxy | best proxy acc | raw minus proxy mean |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for result in results:
        if result.get("returncode") != 0 or "comparison" not in result:
            lines.append(f"| `{result['model']}` | failed | - | - | - | - | - | - |")
            continue
        comparison = result["comparison"]
        lines.append(
            "| `{model}` | ok | {raw} | {sentence} | {proxy_mean} | `{best_proxy}` | {best_proxy_acc} | {delta} |".format(
                model=result["model"],
                raw=pct(comparison.get("raw_good_bad_accuracy")),
                sentence=pct(comparison.get("sentence_good_bad_accuracy")),
                proxy_mean=pct(comparison.get("proxy_mean_accuracy")),
                best_proxy=comparison.get("best_proxy_axis", "-"),
                best_proxy_acc=pct(comparison.get("best_proxy_accuracy")),
                delta=pct(comparison.get("raw_good_bad_minus_proxy_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Treat this as geometry evidence, not training proof.",
            "If raw `good/bad` stays weak across the local model family as well, that",
            "supports the narrower claim that current usable signal comes from richer",
            "targeted axes rather than from the minimalist broad-word axis.",
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
        write_markdown(args.output / "summary.md", results, args.battery)
    print(f"Wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
