#!/usr/bin/env python3
"""Run the process-potential error/repair localization benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES, CATEGORY_AXIS_MAP, combined, sentiment_proxy, word_count  # noqa: E402
from run_gemini_rerun import GeminiEmbedder, get_api_keys, load_local_env, normalize  # noqa: E402


class SentenceTransformersBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], *, label: str, cache_path: Path | None = None) -> np.ndarray:
        del label, cache_path
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def probe(self) -> dict[str, Any]:
        return {"backend": "sentence_transformers", "model": self.model_name}


class GeminiBackend:
    def __init__(self, model_name: str | None, batch_size: int, max_workers: int):
        load_local_env()
        api_keys, key_source = get_api_keys()
        if not api_keys:
            raise SystemExit("No Gemini API key found in environment or .env.local.")
        self.key_source = key_source
        self.embedder = GeminiEmbedder(
            api_key=api_keys[0],
            api_keys=api_keys,
            model=model_name,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    def encode(self, texts: list[str], *, label: str, cache_path: Path | None = None) -> np.ndarray:
        return self.embedder.encode(texts, label=label, cache_path=cache_path)

    def probe(self) -> dict[str, Any]:
        probe = self.embedder.probe_model()
        probe["backend"] = "gemini"
        probe["key_source"] = self.key_source
        return probe


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def category_axis(category: str) -> str:
    return CATEGORY_AXIS_MAP.get(category, "general_evaluative")


def trace_prefix_text(prompt: str, steps: list[str]) -> str:
    numbered = "\n".join(f"Step {idx + 1}: {step}" for idx, step in enumerate(steps))
    return f"User: {prompt}\nAssistant reasoning trace so far:\n{numbered}"


def answer_only_text(prompt: str, final_step: str) -> str:
    return f"User: {prompt}\nAssistant: {final_step}"


def build_axis_vectors(encoder: Any, output_dir: Path, backend_name: str) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for axis_name, anchors in AXES.items():
        anchor_texts = anchors["positive"] + anchors["negative"]
        cache_path = None
        if backend_name == "gemini":
            cache_path = output_dir / "embedding_cache" / f"{axis_name}_anchors.npy"
        embs = encoder.encode(anchor_texts, label=f"{axis_name} anchors", cache_path=cache_path)
        pos = normalize(embs[: len(anchors["positive"])].mean(axis=0))
        neg = normalize(embs[len(anchors["positive"]) :].mean(axis=0))
        vectors[axis_name] = normalize(pos - neg)
    return vectors


def choose_top1_negative(delta_map: dict[int, float]) -> int | None:
    negatives = {step: value for step, value in delta_map.items() if value < 0}
    if not negatives:
        return None
    return min(negatives, key=lambda step: (negatives[step], step))


def choose_top1_positive(delta_map: dict[int, float]) -> int | None:
    positives = {step: value for step, value in delta_map.items() if value > 0}
    if not positives:
        return None
    return max(positives, key=lambda step: (positives[step], -step))


def accuracy_from_bools(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def compute_prefix_scores(
    traces: list[dict[str, Any]],
    encoder: Any,
    output_dir: Path,
    backend_name: str,
) -> tuple[dict[tuple[str, str, int], dict[str, float]], dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    axis_vectors = build_axis_vectors(encoder, output_dir, backend_name)

    prefix_rows: list[tuple[str, str, int, str]] = []
    prefix_texts: list[str] = []
    final_rows: list[tuple[str, str, str]] = []
    final_texts: list[str] = []

    for trace in traces:
        for variant_name, steps in trace["variants"].items():
            for step_idx in range(1, len(steps) + 1):
                prefix_rows.append((trace["id"], variant_name, step_idx, trace["category"]))
                prefix_texts.append(trace_prefix_text(trace["prompt"], steps[:step_idx]))
            final_rows.append((trace["id"], variant_name, trace["category"]))
            final_texts.append(answer_only_text(trace["prompt"], steps[-1]))

    prefix_cache = None
    final_cache = None
    if backend_name == "gemini":
        prefix_cache = output_dir / "embedding_cache" / "prefix_texts.npy"
        final_cache = output_dir / "embedding_cache" / "final_answer_texts.npy"

    prefix_embs = encoder.encode(prefix_texts, label="process potential prefix texts", cache_path=prefix_cache)
    final_embs = encoder.encode(final_texts, label="process potential final answer texts", cache_path=final_cache)

    prefix_scores: dict[tuple[str, str, int], dict[str, float]] = {}
    for idx, (trace_id, variant_name, step_idx, category) in enumerate(prefix_rows):
        axis_name = category_axis(category)
        axis_scores = {
            name: float(prefix_embs[idx] @ vector)
            for name, vector in axis_vectors.items()
        }
        prefix_scores[(trace_id, variant_name, step_idx)] = {
            "category_axis": axis_scores[axis_name],
            "combined": combined(axis_scores),
            "length": float(word_count(prefix_texts[idx])),
            "sentiment": sentiment_proxy(prefix_texts[idx]),
        }

    final_scores: dict[tuple[str, str], dict[str, float]] = {}
    for idx, (trace_id, variant_name, category) in enumerate(final_rows):
        axis_name = category_axis(category)
        axis_scores = {
            name: float(final_embs[idx] @ vector)
            for name, vector in axis_vectors.items()
        }
        final_scores[(trace_id, variant_name)] = {
            "category_axis": axis_scores[axis_name],
            "combined": combined(axis_scores),
        }

    return prefix_scores, final_scores, encoder.probe()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "experiments" / "research_system_v1" / "benchmarks" / "process_potential_error_repair_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes" / "research_system_v1" / "process_potential_error_repair_v1",
    )
    parser.add_argument("--backend", choices=["gemini", "sentence_transformers"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    if args.backend == "gemini":
        encoder = GeminiBackend(args.model, args.batch_size, args.max_workers)
    else:
        if not args.model:
            raise SystemExit("--model is required for sentence_transformers backend.")
        encoder = SentenceTransformersBackend(args.model)

    payload = read_json(args.input)
    traces = payload["traces"]
    args.output.mkdir(parents=True, exist_ok=True)

    prefix_scores, final_scores, probe = compute_prefix_scores(
        traces=traces,
        encoder=encoder,
        output_dir=args.output,
        backend_name=args.backend,
    )

    metrics_by_name = {
        "category_axis": {
            "error_drop": [],
            "repair_rise": [],
            "error_loc": [],
            "repair_loc": [],
            "clean_beats_persisted": [],
            "repaired_beats_persisted": [],
        },
        "combined": {
            "error_drop": [],
            "repair_rise": [],
            "error_loc": [],
            "repair_loc": [],
            "clean_beats_persisted": [],
            "repaired_beats_persisted": [],
        },
        "length": {
            "error_drop": [],
            "repair_rise": [],
            "error_loc": [],
            "repair_loc": [],
        },
        "sentiment": {
            "error_drop": [],
            "repair_rise": [],
            "error_loc": [],
            "repair_loc": [],
        },
        "final_answer_only_category_axis": {
            "error_drop": [],
            "repair_rise": [],
            "error_loc": [],
            "repair_loc": [],
        },
        "final_answer_only_combined": {
            "error_drop": [],
            "repair_rise": [],
            "error_loc": [],
            "repair_loc": [],
        }
    }
    rows: list[dict[str, Any]] = []

    for trace in traces:
        trace_id = trace["id"]
        error_step = int(trace["error_step"])
        repair_step = int(trace["repair_step"])

        def delta_map_for(metric_name: str, variant_name: str) -> dict[int, float]:
            steps = trace["variants"][variant_name]
            if metric_name.startswith("final_answer_only_"):
                base_metric = metric_name.replace("final_answer_only_", "")
                final_score = final_scores[(trace_id, variant_name)][base_metric]
                prefix_values = [0.0] * len(steps)
                prefix_values[-1] = final_score
            else:
                prefix_values = [
                    prefix_scores[(trace_id, variant_name, step_idx)][metric_name]
                    for step_idx in range(1, len(steps) + 1)
                ]
            return {
                step_idx: prefix_values[step_idx - 1] - (prefix_values[step_idx - 2] if step_idx > 1 else 0.0)
                for step_idx in range(1, len(prefix_values) + 1)
            }

        for metric_name in metrics_by_name:
            error_deltas = delta_map_for(metric_name, "error_persisted")
            repair_deltas = delta_map_for(metric_name, "repaired")
            error_drop = error_deltas.get(error_step, 0.0) < 0
            repair_rise = repair_deltas.get(repair_step, 0.0) > 0
            error_loc = choose_top1_negative(error_deltas) == error_step
            repair_loc = choose_top1_positive(repair_deltas) == repair_step

            metrics_by_name[metric_name]["error_drop"].append(error_drop)
            metrics_by_name[metric_name]["repair_rise"].append(repair_rise)
            metrics_by_name[metric_name]["error_loc"].append(error_loc)
            metrics_by_name[metric_name]["repair_loc"].append(repair_loc)

            if metric_name in {"category_axis", "combined"}:
                metrics_by_name[metric_name]["clean_beats_persisted"].append(
                    final_scores[(trace_id, "clean")][metric_name] > final_scores[(trace_id, "error_persisted")][metric_name]
                )
                metrics_by_name[metric_name]["repaired_beats_persisted"].append(
                    final_scores[(trace_id, "repaired")][metric_name] > final_scores[(trace_id, "error_persisted")][metric_name]
                )

            rows.append(
                {
                    "trace_id": trace_id,
                    "domain": trace["domain"],
                    "category": trace["category"],
                    "metric": metric_name,
                    "error_step": error_step,
                    "repair_step": repair_step,
                    "error_delta_at_step": error_deltas.get(error_step, 0.0),
                    "repair_delta_at_step": repair_deltas.get(repair_step, 0.0),
                    "error_drop": error_drop,
                    "repair_rise": repair_rise,
                    "error_localization_top1": error_loc,
                    "repair_localization_top1": repair_loc,
                    "error_top1_step": choose_top1_negative(error_deltas),
                    "repair_top1_step": choose_top1_positive(repair_deltas),
                }
            )

    metric_summary: dict[str, dict[str, float | int]] = {}
    for metric_name, bucket in metrics_by_name.items():
        summary_row: dict[str, float | int] = {
            "n_traces": len(traces),
            "error_drop_accuracy": accuracy_from_bools(bucket["error_drop"]),
            "repair_rise_accuracy": accuracy_from_bools(bucket["repair_rise"]),
            "error_localization_top1_accuracy": accuracy_from_bools(bucket["error_loc"]),
            "repair_localization_top1_accuracy": accuracy_from_bools(bucket["repair_loc"]),
        }
        summary_row["dense_reward_localization_score"] = (
            float(summary_row["error_localization_top1_accuracy"]) + float(summary_row["repair_localization_top1_accuracy"])
        ) / 2.0
        if "clean_beats_persisted" in bucket:
            summary_row["clean_beats_persisted_accuracy"] = accuracy_from_bools(bucket["clean_beats_persisted"])
            summary_row["repaired_beats_persisted_accuracy"] = accuracy_from_bools(bucket["repaired_beats_persisted"])
        metric_summary[metric_name] = summary_row

    top = metric_summary["category_axis"]
    summary = {
        "backend": probe.get("backend", args.backend),
        "model": probe.get("model", args.model or "default"),
        "n_traces": len(traces),
        "error_drop_accuracy": top["error_drop_accuracy"],
        "repair_rise_accuracy": top["repair_rise_accuracy"],
        "error_localization_top1_accuracy": top["error_localization_top1_accuracy"],
        "repair_localization_top1_accuracy": top["repair_localization_top1_accuracy"],
        "dense_reward_localization_score": top["dense_reward_localization_score"],
        "combined_dense_reward_localization_score": metric_summary["combined"]["dense_reward_localization_score"],
        "length_dense_reward_localization_score": metric_summary["length"]["dense_reward_localization_score"],
        "sentiment_dense_reward_localization_score": metric_summary["sentiment"]["dense_reward_localization_score"],
        "final_answer_only_dense_reward_localization_score": metric_summary["final_answer_only_category_axis"]["dense_reward_localization_score"],
        "metrics": metric_summary,
    }

    write_json(
        args.output / "config.json",
        {
            "input": str(args.input),
            "backend": args.backend,
            "probe": probe,
        },
    )
    write_json(args.output / "results.json", rows)
    write_json(args.output / "summary.json", summary)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Process Potential Error-Repair V1",
        "",
        f"Backend: `{summary['backend']}`",
        f"Model: `{summary['model']}`",
        f"Traces: {summary['n_traces']}",
        "",
        "## Main Metric",
        "",
        "| Scorer | Error drop | Repair rise | Error loc top1 | Repair loc top1 | Dense score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric_name in [
        "category_axis",
        "combined",
        "length",
        "sentiment",
        "final_answer_only_category_axis",
        "final_answer_only_combined",
    ]:
        row = metric_summary[metric_name]
        lines.append(
            f"| `{metric_name}` | {float(row['error_drop_accuracy']):.1%} | {float(row['repair_rise_accuracy']):.1%} | "
            f"{float(row['error_localization_top1_accuracy']):.1%} | {float(row['repair_localization_top1_accuracy']):.1%} | "
            f"{float(row['dense_reward_localization_score']):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "This suite is the bridge from selection to training. A strong result means the evaluative score changes at the injected error and the later repair step, not just at the final answer. A weak result means the signal may still be useful for reranking while remaining too blunt for dense process supervision.",
            "",
        ]
    )
    summary_text = "\n".join(lines)
    (args.output / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
