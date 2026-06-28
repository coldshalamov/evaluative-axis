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
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "notes"
    / "research_cycles"
    / "cycle_009_oss_direct_battery_v3_sweep"
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


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


def run_model(model: str, battery: Path, output_root: Path, interfaces: str) -> dict[str, Any]:
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
        "--interfaces",
        interfaces,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    result = {
        "model": model,
        "output_dir": str(out_dir),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "interfaces": interfaces,
    }
    if proc.returncode == 0 and (out_dir / "summary.json").exists():
        summary = read_json(out_dir / "summary.json")
        result["metrics"] = summary["metrics"]
        result["length_abs_gap_mean"] = summary["length_abs_gap_mean"]
        result["length_abs_gap_max"] = summary["length_abs_gap_max"]
    return result


def write_markdown(path: Path, results: list[dict[str, Any]], battery: Path, interfaces: str) -> None:
    interface_set = {part.strip() for part in interfaces.split(",") if part.strip()}
    has_direct = "direct" in interface_set
    has_decomposition = "decomposition" in interface_set
    lines = [
        "# FastEmbed Model Sweep",
        "",
        "Date: June 27, 2026",
        "",
        f"Battery: `{battery.name}`",
        f"Interfaces: `{interfaces}`",
        "",
        "This sweep is diagnostic rather than decisive. Its purpose is to map how",
        "free/local embedding models behave on the same frozen battery so the repo",
        "can separate model-capacity limits from benchmark limits.",
        "",
        "## Key Metrics",
        "",
    ]
    if has_direct and has_decomposition:
        lines.extend(
            [
                "| Model | Status | Best Overall | Best Acc | Length | Refusal | Direct Category | Direct Harm | Direct Persona | Direct Anti-Syc | Decomp Category |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    elif has_direct:
        lines.extend(
            [
                "| Model | Status | Best Direct | Best Acc | Length | Refusal | Direct Combined | Direct Category | Direct Harm | Direct Persona | Direct Anti-Syc |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Model | Status | Best Decomp | Best Acc | Length | Refusal | Decomp Combined | Decomp Category |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for result in results:
        if result.get("returncode") != 0 or "metrics" not in result:
            if has_direct and has_decomposition:
                lines.append(f"| `{result['model']}` | failed | - | - | - | - | - | - | - | - | - |")
            elif has_direct:
                lines.append(f"| `{result['model']}` | failed | - | - | - | - | - | - | - | - | - |")
            else:
                lines.append(f"| `{result['model']}` | failed | - | - | - | - | - | - |")
            continue
        metrics = result["metrics"]
        if has_direct:
            candidate_metrics = {
                name: row["accuracy"]
                for name, row in metrics.items()
                if name.startswith("direct_")
            }
        else:
            candidate_metrics = {
                name: row["accuracy"]
                for name, row in metrics.items()
                if name.startswith("decomposition_")
            }
        if has_decomposition and has_direct:
            candidate_metrics = {
                name: row["accuracy"]
                for name, row in metrics.items()
                if name not in {"length", "sentiment", "refusal"}
            }
        best_name, best_acc = max(candidate_metrics.items(), key=lambda item: item[1])
        length_acc = metrics.get("length", {}).get("accuracy")
        refusal_acc = metrics.get("refusal", {}).get("accuracy")
        if has_direct and has_decomposition:
            lines.append(
                "| `{model}` | ok | `{best}` | {best_acc} | {length} | {refusal} | {direct_category} | {direct_harm} | {direct_persona} | {direct_anti} | {decomp_category} |".format(
                    model=result["model"],
                    best=best_name,
                    best_acc=pct(best_acc),
                    length=pct(length_acc),
                    refusal=pct(refusal_acc),
                    direct_category=pct(metrics.get("direct_category_axis", {}).get("accuracy")),
                    direct_harm=pct(metrics.get("direct_harm_reduction", {}).get("accuracy")),
                    direct_persona=pct(metrics.get("direct_persona_honesty", {}).get("accuracy")),
                    direct_anti=pct(metrics.get("direct_anti_sycophancy", {}).get("accuracy")),
                    decomp_category=pct(metrics.get("decomposition_category_axis", {}).get("accuracy")),
                )
            )
        elif has_direct:
            lines.append(
                "| `{model}` | ok | `{best}` | {best_acc} | {length} | {refusal} | {direct_combined} | {direct_category} | {direct_harm} | {direct_persona} | {direct_anti} |".format(
                    model=result["model"],
                    best=best_name,
                    best_acc=pct(best_acc),
                    length=pct(length_acc),
                    refusal=pct(refusal_acc),
                    direct_combined=pct(metrics.get("direct_combined", {}).get("accuracy")),
                    direct_category=pct(metrics.get("direct_category_axis", {}).get("accuracy")),
                    direct_harm=pct(metrics.get("direct_harm_reduction", {}).get("accuracy")),
                    direct_persona=pct(metrics.get("direct_persona_honesty", {}).get("accuracy")),
                    direct_anti=pct(metrics.get("direct_anti_sycophancy", {}).get("accuracy")),
                )
            )
        else:
            lines.append(
                "| `{model}` | ok | `{best}` | {best_acc} | {length} | {refusal} | {decomp_combined} | {decomp_category} |".format(
                    model=result["model"],
                    best=best_name,
                    best_acc=pct(best_acc),
                    length=pct(length_acc),
                    refusal=pct(refusal_acc),
                    decomp_combined=pct(metrics.get("decomposition_combined", {}).get("accuracy")),
                    decomp_category=pct(metrics.get("decomposition_category_axis", {}).get("accuracy")),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Treat this as a model-landscape map, not as training proof.",
            "If direct-only local models stay near or below cheap baselines while",
            "Gemini remains strong on the same battery, that supports a real",
            "capability gap story rather than a benchmark-only story.",
            "",
        ]
    )
    if has_direct and not has_decomposition:
        lines.extend(
            [
                "These direct-only runs avoid decomposition leakage from hand-authored",
                "notes, so they are the cleaner evidence lane for this battery.",
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
    parser.add_argument("--interfaces", default="direct,decomposition")
    args = parser.parse_args()

    models = [part.strip() for part in args.models.split(",") if part.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in models:
        print(f"=== {model} ===", flush=True)
        result = run_model(model, args.battery, args.output, args.interfaces)
        print(f"returncode={result['returncode']}", flush=True)
        if result.get("stdout"):
            print(result["stdout"], flush=True)
        if result.get("stderr"):
            print(result["stderr"], file=sys.stderr, flush=True)
        results.append(result)
        write_json(args.output / "sweep_results.json", results)
        write_markdown(args.output / "summary.md", results, args.battery, args.interfaces)
    print(f"Wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
