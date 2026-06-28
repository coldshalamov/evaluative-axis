#!/usr/bin/env python3
"""Objective reranking for exact-answer text tasks."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES, COMBINED_WEIGHTS  # noqa: E402
from run_gemini_rerun import GeminiEmbedder, get_api_keys, load_local_env, normalize  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_0-9]+", text))


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_bool(text: str) -> bool | None:
    value = normalize_text(text)
    yes = {"yes", "y", "true", "pass", "passed", "correct"}
    no = {"no", "n", "false", "fail", "failed", "incorrect"}
    if value in yes:
        return True
    if value in no:
        return False
    return None


def parse_number(text: str) -> float | None:
    value = normalize_text(text)
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def grade_candidate(task: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    grading = task["grading"]
    mode = grading["mode"]
    final_answer = candidate["final_answer"]
    if mode == "string":
        observed = normalize_text(final_answer)
        answers = [normalize_text(grading["target"])]
        answers.extend(normalize_text(alias) for alias in grading.get("aliases", []))
        correct = observed in answers
    elif mode == "bool":
        observed = parse_bool(final_answer)
        correct = observed is not None and observed == bool(grading["target"])
    elif mode == "number":
        observed = parse_number(final_answer)
        target = float(grading["target"])
        tolerance = float(grading.get("tolerance", 1e-9))
        correct = observed is not None and abs(observed - target) <= tolerance
    else:
        raise ValueError(f"Unknown grading mode: {mode}")

    return {
        "status": "ok",
        "passed": 1 if correct else 0,
        "total": 1,
        "pass_rate": 1.0 if correct else 0.0,
        "error": "",
        "normalized_answer": normalize_text(final_answer),
    }


class SentenceTransformersBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
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

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, label="objective text reranking")

    def probe(self) -> dict[str, Any]:
        probe = self.embedder.probe_model()
        probe["backend"] = "gemini"
        probe["key_source"] = self.key_source
        return probe


def build_direct_text(task: dict[str, Any], response_text: str) -> str:
    return f"Task:\n{task['prompt']}\n\nResponse:\n{response_text}"


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    solved = sum(1 for row in rows if row["selected_passed"] == row["selected_total"])
    avg_pass_rate = statistics.mean(row["selected_pass_rate"] for row in rows) if rows else 0.0
    avg_regret = statistics.mean(row["best_pass_rate"] - row["selected_pass_rate"] for row in rows) if rows else 0.0
    top_hit_rate = statistics.mean(1.0 if row["selected_is_best_available"] else 0.0 for row in rows) if rows else 0.0
    return {
        "tasks": len(rows),
        "tasks_solved": solved,
        "solve_rate": solved / len(rows) if rows else 0.0,
        "avg_selected_pass_rate": avg_pass_rate,
        "avg_regret_vs_best_available": avg_regret,
        "best_candidate_hit_rate": top_hit_rate,
    }


def combined_score(score_map: dict[str, float]) -> float:
    total = 0.0
    weight_total = 0.0
    for axis_name, weight in COMBINED_WEIGHTS.items():
        total += weight * score_map[axis_name]
        weight_total += weight
    return total / weight_total if weight_total else 0.0


def best_direct_rate(method_summary: dict[str, dict[str, Any]]) -> tuple[str, float]:
    best_name = ""
    best_rate = -1.0
    for name, row in method_summary.items():
        if not name.startswith("direct_"):
            continue
        if row["solve_rate"] > best_rate:
            best_name = name
            best_rate = row["solve_rate"]
    return best_name, best_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["gemini", "sentence_transformers"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    if args.backend == "gemini":
        encoder = GeminiBackend(args.model, args.batch_size, args.max_workers)
    else:
        if not args.model:
            raise SystemExit("--model is required for sentence_transformers backend.")
        encoder = SentenceTransformersBackend(args.model)

    payload = read_json(args.tasks)
    tasks = payload["tasks"]
    rng = random.Random(args.seed)

    axis_vectors = {}
    for axis_name, anchors in AXES.items():
        texts = anchors["positive"] + anchors["negative"]
        embs = encoder.encode(texts)
        pos = normalize(embs[: len(anchors["positive"])].mean(axis=0))
        neg = normalize(embs[len(anchors["positive"]) :].mean(axis=0))
        axis_vectors[axis_name] = normalize(pos - neg)

    candidate_rows: list[dict[str, Any]] = []
    direct_texts: list[str] = []
    for task in tasks:
        for candidate in task["candidates"]:
            evaluation = grade_candidate(task, candidate)
            candidate_row = {
                "task_id": task["id"],
                "candidate_id": candidate["id"],
                "label": candidate.get("label", ""),
                "text": candidate["text"],
                "final_answer": candidate["final_answer"],
                "evaluation": evaluation,
                "length_words": word_count(candidate["text"]),
                "target_axis": task["target_axis"],
            }
            candidate_rows.append(candidate_row)
            direct_texts.append(build_direct_text(task, candidate["text"]))

    direct_embs = encoder.encode(direct_texts)
    for idx, row in enumerate(candidate_rows):
        row["axis_scores"] = {
            axis_name: float(direct_embs[idx] @ axis_vector)
            for axis_name, axis_vector in axis_vectors.items()
        }
        row["scores"] = {
            "direct_combined": combined_score(row["axis_scores"]),
            "direct_general_evaluative": row["axis_scores"]["general_evaluative"],
            "direct_target_axis": row["axis_scores"][row["target_axis"]],
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(row["task_id"], []).append(row)

    methods = ["random", "length", "direct_combined", "direct_general_evaluative", "direct_target_axis"]
    selection_rows: list[dict[str, Any]] = []
    summary_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    for task in tasks:
        task_candidates = grouped[task["id"]]
        best_pass_rate = max(row["evaluation"]["pass_rate"] for row in task_candidates)
        best_candidate_ids = sorted(
            row["candidate_id"] for row in task_candidates if row["evaluation"]["pass_rate"] == best_pass_rate
        )
        for method in methods:
            if method == "random":
                selected = rng.choice(task_candidates)
            elif method == "length":
                selected = max(task_candidates, key=lambda row: (row["length_words"], row["candidate_id"]))
            else:
                selected = max(task_candidates, key=lambda row: (row["scores"][method], row["candidate_id"]))

            result_row = {
                "task_id": task["id"],
                "method": method,
                "selected_candidate_id": selected["candidate_id"],
                "selected_passed": selected["evaluation"]["passed"],
                "selected_total": selected["evaluation"]["total"],
                "selected_pass_rate": selected["evaluation"]["pass_rate"],
                "selected_is_best_available": selected["candidate_id"] in best_candidate_ids,
                "best_pass_rate": best_pass_rate,
                "best_candidate_ids": best_candidate_ids,
            }
            selection_rows.append(result_row)
            summary_rows[method].append(result_row)

    method_summary = {method: summarize_method(rows) for method, rows in summary_rows.items()}
    best_method, best_rate = best_direct_rate(method_summary)

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "config.json",
        {
            "dataset_name": payload.get("dataset_name"),
            "purpose": payload.get("purpose"),
            "backend": args.backend,
            "probe": encoder.probe(),
            "seed": args.seed,
        },
    )
    write_json(args.output / "candidates.json", candidate_rows)
    write_json(args.output / "selections.json", selection_rows)
    write_json(args.output / "summary.json", method_summary)

    lines = [
        "# Objective Text Reranking",
        "",
        f"Backend: `{args.backend}`",
        f"Model: `{encoder.probe().get('model', args.model or 'default')}`",
        f"Tasks: {len(tasks)}",
        "",
        "## Method Results",
        "",
        "| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = method_summary[method]
        lines.append(
            f"| `{method}` | {row['solve_rate']:.1%} | {row['tasks_solved']} / {row['tasks']} | "
            f"{row['avg_selected_pass_rate']:.1%} | {row['best_candidate_hit_rate']:.1%} | "
            f"{row['avg_regret_vs_best_available']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Best direct method: `{best_method}` at {best_rate:.1%} solve rate.",
            "",
            "## Interpretation Rule",
            "",
            "This suite is objective at the final metric layer because each task has an exact grading rule. Treat it as selection evidence rather than as proof of full training value.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
