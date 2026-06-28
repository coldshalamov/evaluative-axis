#!/usr/bin/env python3
"""Compare raw good/bad against nearby proxy-word axes on controlled conflicts."""

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

from run_gemini_rerun import GeminiEmbedder, get_api_keys, load_local_env, normalize  # noqa: E402


AXES: dict[str, dict[str, Any]] = {
    "raw_good_bad": {
        "positive": ["good"],
        "negative": ["bad"],
        "family": "good_bad",
        "kind": "raw_word",
    },
    "sentence_good_bad": {
        "positive": ["This response is good."],
        "negative": ["This response is bad."],
        "family": "good_bad",
        "kind": "sentence_minimal",
    },
    "raw_true_false": {
        "positive": ["true"],
        "negative": ["false"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_truthful_untruthful": {
        "positive": ["truthful"],
        "negative": ["untruthful"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_honest_dishonest": {
        "positive": ["honest"],
        "negative": ["dishonest"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_useful_useless": {
        "positive": ["useful"],
        "negative": ["useless"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_helpful_unhelpful": {
        "positive": ["helpful"],
        "negative": ["unhelpful"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_accurate_inaccurate": {
        "positive": ["accurate"],
        "negative": ["inaccurate"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_correct_incorrect": {
        "positive": ["correct"],
        "negative": ["incorrect"],
        "family": "proxy",
        "kind": "raw_word",
    },
    "raw_safe_unsafe": {
        "positive": ["safe"],
        "negative": ["unsafe"],
        "family": "proxy",
        "kind": "raw_word",
    },
}


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


class FastEmbedBackend:
    def __init__(self, model_name: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise SystemExit(
                "FastEmbed backend requires fastembed. Install it outside the repo, "
                "for example in C:\\Users\\93rob\\.cache\\codex-embedding-venv."
            ) from exc

        self.model_name = model_name
        self.cache_dir = Path.home() / ".cache" / "codex-fastembed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(self.cache_dir))

    def encode(self, texts: list[str], *, label: str, cache_path: Path | None = None) -> np.ndarray:
        del label, cache_path
        vectors = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def probe(self) -> dict[str, Any]:
        return {
            "backend": "fastembed",
            "model": self.model_name,
            "cache_dir": str(self.cache_dir),
        }


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


def direct_text(case: dict[str, Any], answer: str) -> str:
    return f"User: {case['prompt']}\nAssistant: {answer}"


def diff_correct(diff: float) -> float:
    if diff > 0:
        return 1.0
    if diff == 0:
        return 0.5
    return 0.0


def summarize_metric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "accuracy": 0.0, "mean_delta": 0.0}
    return {
        "n": len(values),
        "accuracy": sum(diff_correct(value) for value in values) / len(values),
        "mean_delta": sum(values) / len(values),
    }


def build_axis_vectors(
    encoder: Any,
    output_dir: Path,
    backend_name: str,
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for axis_name, spec in AXES.items():
        anchor_texts = spec["positive"] + spec["negative"]
        cache_path = None
        if backend_name == "gemini":
            cache_path = output_dir / "embedding_cache" / f"{axis_name}_anchors.npy"
        embs = encoder.encode(anchor_texts, label=f"{axis_name} anchors", cache_path=cache_path)
        pos = normalize(embs[: len(spec["positive"])].mean(axis=0))
        neg = normalize(embs[len(spec["positive"]) :].mean(axis=0))
        vectors[axis_name] = normalize(pos - neg)
    return vectors


def score_cases(
    cases: list[dict[str, Any]],
    encoder: Any,
    output_dir: Path,
    backend_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]], dict[str, dict[str, dict[str, float | int]]], dict[str, Any]]:
    axis_vectors = build_axis_vectors(encoder, output_dir, backend_name)

    text_rows: list[tuple[str, str]] = []
    texts: list[str] = []
    for case in cases:
        text_rows.append((case["id"], "better"))
        texts.append(direct_text(case, case["better"]))
        text_rows.append((case["id"], "worse"))
        texts.append(direct_text(case, case["worse"]))

    cache_path = None
    if backend_name == "gemini":
        cache_path = output_dir / "embedding_cache" / "candidate_texts.npy"
    text_embs = encoder.encode(texts, label="good-vs-proxy candidate texts", cache_path=cache_path)

    per_case_scores: dict[tuple[str, str], dict[str, float]] = {}
    for idx, key in enumerate(text_rows):
        per_case_scores[key] = {
            axis_name: float(text_embs[idx] @ axis_vector)
            for axis_name, axis_vector in axis_vectors.items()
        }

    metric_deltas: dict[str, list[float]] = {axis_name: [] for axis_name in AXES}
    category_deltas: dict[str, dict[str, list[float]]] = {}
    rows: list[dict[str, Any]] = []
    length_abs_gaps = []

    for case in cases:
        better_scores = per_case_scores[(case["id"], "better")]
        worse_scores = per_case_scores[(case["id"], "worse")]
        better_len = len(words(case["better"]))
        worse_len = len(words(case["worse"]))
        length_abs_gaps.append(abs(better_len - worse_len))

        row = {
            "id": case["id"],
            "category": case["category"],
            "phenomenon": case["phenomenon"],
            "better_len": better_len,
            "worse_len": worse_len,
            "length_gap_better_minus_worse": better_len - worse_len,
        }
        for axis_name in AXES:
            diff = better_scores[axis_name] - worse_scores[axis_name]
            metric_deltas[axis_name].append(diff)
            category_deltas.setdefault(case["category"], {}).setdefault(axis_name, []).append(diff)
            row[f"diff_{axis_name}"] = diff
            row[f"correct_{axis_name}"] = diff_correct(diff)
        rows.append(row)

    metric_summary = {
        axis_name: summarize_metric(values)
        for axis_name, values in sorted(metric_deltas.items())
    }
    category_summary = {
        category: {
            axis_name: summarize_metric(values)
            for axis_name, values in sorted(metrics.items())
        }
        for category, metrics in sorted(category_deltas.items())
    }

    proxy_axes = [name for name, spec in AXES.items() if spec["family"] == "proxy"]
    proxy_accuracies = [float(metric_summary[name]["accuracy"]) for name in proxy_axes]
    best_axis_name = max(metric_summary, key=lambda name: float(metric_summary[name]["accuracy"]))
    best_proxy_name = max(proxy_axes, key=lambda name: float(metric_summary[name]["accuracy"]))
    comparison = {
        "raw_good_bad_accuracy": float(metric_summary["raw_good_bad"]["accuracy"]),
        "sentence_good_bad_accuracy": float(metric_summary["sentence_good_bad"]["accuracy"]),
        "proxy_mean_accuracy": sum(proxy_accuracies) / len(proxy_accuracies),
        "best_proxy_axis": best_proxy_name,
        "best_proxy_accuracy": float(metric_summary[best_proxy_name]["accuracy"]),
        "best_axis_name": best_axis_name,
        "best_axis_accuracy": float(metric_summary[best_axis_name]["accuracy"]),
        "raw_good_bad_minus_proxy_mean": float(metric_summary["raw_good_bad"]["accuracy"]) - (sum(proxy_accuracies) / len(proxy_accuracies)),
        "sentence_good_bad_minus_proxy_mean": float(metric_summary["sentence_good_bad"]["accuracy"]) - (sum(proxy_accuracies) / len(proxy_accuracies)),
        "raw_good_bad_minus_best_proxy": float(metric_summary["raw_good_bad"]["accuracy"]) - float(metric_summary[best_proxy_name]["accuracy"]),
        "sentence_good_bad_minus_best_proxy": float(metric_summary["sentence_good_bad"]["accuracy"]) - float(metric_summary[best_proxy_name]["accuracy"]),
        "length_abs_gap_mean": sum(length_abs_gaps) / len(length_abs_gaps) if length_abs_gaps else 0.0,
        "length_abs_gap_max": max(length_abs_gaps) if length_abs_gaps else 0,
    }

    return rows, metric_summary, category_summary, comparison


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_md(
    metric_summary: dict[str, dict[str, float | int]],
    category_summary: dict[str, dict[str, dict[str, float | int]]],
    comparison: dict[str, Any],
    backend_name: str,
    model_name: str,
    case_count: int,
) -> str:
    preferred = [
        "raw_good_bad",
        "sentence_good_bad",
        "raw_true_false",
        "raw_truthful_untruthful",
        "raw_honest_dishonest",
        "raw_useful_useless",
        "raw_helpful_unhelpful",
        "raw_accurate_inaccurate",
        "raw_correct_incorrect",
        "raw_safe_unsafe",
    ]
    lines = [
        "# Good vs Proxy Conflicts",
        "",
        f"Backend: `{backend_name}`",
        f"Model: `{model_name}`",
        f"Cases: {case_count}",
        f"Mean absolute length gap: {comparison['length_abs_gap_mean']:.2f} words",
        f"Max absolute length gap: {comparison['length_abs_gap_max']} words",
        "",
        "## Overall Accuracy",
        "",
        "| Axis | Accuracy | Mean delta | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    for axis_name in preferred:
        row = metric_summary[axis_name]
        lines.append(
            f"| `{axis_name}` | {float(row['accuracy']):.1%} | {float(row['mean_delta']):.4f} | {int(row['n'])} |"
        )

    lines.extend(
        [
            "",
            "## Good/Bad vs Proxy Summary",
            "",
            f"- `raw_good_bad`: {comparison['raw_good_bad_accuracy']:.1%}",
            f"- `sentence_good_bad`: {comparison['sentence_good_bad_accuracy']:.1%}",
            f"- proxy mean: {comparison['proxy_mean_accuracy']:.1%}",
            f"- best proxy: `{comparison['best_proxy_axis']}` at {comparison['best_proxy_accuracy']:.1%}",
            f"- raw good/bad minus proxy mean: {comparison['raw_good_bad_minus_proxy_mean']:+.1%}",
            f"- sentence good/bad minus proxy mean: {comparison['sentence_good_bad_minus_proxy_mean']:+.1%}",
            "",
            "## Category Snapshot",
            "",
            "| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    focus_axes = [
        "raw_good_bad",
        "sentence_good_bad",
        "raw_true_false",
        "raw_helpful_unhelpful",
        "raw_safe_unsafe",
    ]
    for category, metrics in category_summary.items():
        values = [
            f"{float(metrics[axis_name]['accuracy']):.1%}"
            for axis_name in focus_axes
        ]
        lines.append(f"| `{category}` | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--backend", choices=["gemini", "sentence_transformers", "fastembed"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    if args.backend == "gemini":
        encoder = GeminiBackend(args.model, args.batch_size, args.max_workers)
    elif args.backend == "fastembed":
        if not args.model:
            raise SystemExit("--model is required for fastembed backend.")
        encoder = FastEmbedBackend(args.model)
    else:
        if not args.model:
            raise SystemExit("--model is required for sentence_transformers backend.")
        encoder = SentenceTransformersBackend(args.model)

    cases = read_jsonl(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, metric_summary, category_summary, comparison = score_cases(
        cases=cases,
        encoder=encoder,
        output_dir=args.output,
        backend_name=args.backend,
    )
    probe = encoder.probe()

    write_json(
        args.output / "config.json",
        {
            "input": str(args.input),
            "backend": args.backend,
            "probe": probe,
            "axis_names": list(AXES.keys()),
        },
    )
    write_json(args.output / "results.json", rows)
    write_results_csv(args.output / "results.csv", rows)
    write_json(
        args.output / "summary.json",
        {
            "metrics": metric_summary,
            "categories": category_summary,
            "comparison": comparison,
        },
    )
    summary_text = build_summary_md(
        metric_summary=metric_summary,
        category_summary=category_summary,
        comparison=comparison,
        backend_name=args.backend,
        model_name=probe.get("model", args.model or "default"),
        case_count=len(cases),
    )
    (args.output / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
