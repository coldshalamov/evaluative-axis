#!/usr/bin/env python3
"""Download compact RL/preference dataset slices for centroid validation.

This intentionally does not mirror full Hugging Face datasets. It extracts
standardized prompt/better/worse JSONL pairs through Dataset Viewer pagination
or raw streaming so the repo gets proof-ready data under a small disk budget.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes" / "research_cycles" / "rl_dataset_slices"
API = "https://datasets-server.huggingface.co"


DEFAULT_TARGETS = {
    "helpsteer2": 2000,
    "pku_better": 2000,
    "pku_safer": 2000,
    "prm800k": 3000,
    "openr1": 1000,
    "summarize": 2000,
    "webgpt": 1000,
    "rewardbench2": 2000,
}


def stable_id(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def request_json(path: str, params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    url = f"{API}/{path}?{urlencode(params)}"
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=90)
            if response.status_code == 429:
                last_error = RuntimeError(f"rate limited: {url}")
                time.sleep(2 + attempt * 3)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network resilience
            last_error = exc
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"failed GET {url}: {last_error}")


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    max_rows: int | None = None,
    page_size: int = 100,
    start_offset: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = start_offset
    total = None
    while max_rows is None or len(rows) < max_rows:
        length = page_size
        if max_rows is not None:
            length = min(page_size, max_rows - len(rows))
        if length <= 0:
            break

        data = request_json(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
        )
        batch = data.get("rows", [])
        total = data.get("num_rows_total", total)
        if not batch:
            break

        for item in batch:
            row = item.get("row", item)
            row["_row_idx"] = item.get("row_idx", offset + len(rows))
            rows.append(row)

        offset += len(batch)
        if total is not None and offset >= total:
            break
    return rows


def dataset_size(dataset: str) -> dict[str, Any]:
    try:
        return request_json("size", {"dataset": dataset}).get("size", {})
    except Exception as exc:
        return {"error": str(exc)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path.stat().st_size


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        return to_text(value.tolist())
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def make_pair(
    *,
    source_dataset: str,
    source_config: str,
    source_split: str,
    task_type: str,
    prompt: str,
    better: str,
    worse: str,
    label_source: str,
    source_row_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = (prompt or "").strip()
    better = (better or "").strip()
    worse = (worse or "").strip()
    if not prompt or not better or not worse or better == worse:
        return None
    return {
        "id": stable_id(source_dataset, source_config, source_split, task_type, source_row_id, prompt, better, worse),
        "source_dataset": source_dataset,
        "source_config": source_config,
        "source_split": source_split,
        "source_row_id": source_row_id,
        "task_type": task_type,
        "label_source": label_source,
        "prompt": prompt,
        "better": better,
        "worse": worse,
        "metadata": metadata or {},
    }


def build_helpsteer2(target: int) -> list[dict[str, Any]]:
    dataset = "nvidia/HelpSteer2"
    pairs: list[dict[str, Any]] = []
    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/preference/preference.jsonl.gz"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
        for row_idx, line in enumerate(gz):
            if len(pairs) >= target:
                break
            if not line.strip():
                continue
            row = json.loads(line.decode("utf-8"))
            strength = int(row.get("preference_strength") or 0)
            if strength == 0:
                continue
            better_key = "response_2" if strength > 0 else "response_1"
            worse_key = "response_1" if strength > 0 else "response_2"
            split = row.get("split", "train")
            prompt = row.get("prompt", "")
            magnitude = abs(strength)
            pair = make_pair(
                source_dataset=dataset,
                source_config="preference",
                source_split=split,
                task_type="pairwise_helpsteer_preference",
                prompt=prompt,
                better=row.get(better_key, ""),
                worse=row.get(worse_key, ""),
                label_source="preference_strength",
                source_row_id=str(row_idx),
                metadata={
                    "preference_strength": strength,
                    "preference_magnitude": magnitude,
                    "preference_statement": row.get("preference_statement"),
                },
            )
            if pair:
                pairs.append(pair)
    return pairs


def build_pku(target_better: int, target_safer: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = "PKU-Alignment/PKU-SafeRLHF"
    better_pairs: list[dict[str, Any]] = []
    safer_pairs: list[dict[str, Any]] = []
    offset = 0
    total = None
    while len(better_pairs) < target_better or len(safer_pairs) < target_safer:
        data = request_json(
            "rows",
            {
                "dataset": dataset,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            },
        )
        rows = data.get("rows", [])
        total = data.get("num_rows_total", total)
        if not rows:
            break
        for item in rows:
            row = item["row"]
            row_idx = str(item.get("row_idx", offset))
            prompt = row.get("prompt", "")

            def response(response_id: int) -> str:
                return row.get(f"response_{response_id}", "")

            if len(better_pairs) < target_better and row.get("better_response_id") in (0, 1):
                better_id = int(row["better_response_id"])
                worse_id = 1 - better_id
                pair = make_pair(
                    source_dataset=dataset,
                    source_config="default",
                    source_split="train",
                    task_type="pairwise_helpfulness_preference",
                    prompt=prompt,
                    better=response(better_id),
                    worse=response(worse_id),
                    label_source="better_response_id",
                    source_row_id=row_idx,
                    metadata={
                        "better_response_id": better_id,
                        "safer_response_id": row.get("safer_response_id"),
                        "prompt_source": row.get("prompt_source"),
                        "response_0_source": row.get("response_0_source"),
                        "response_1_source": row.get("response_1_source"),
                    },
                )
                if pair:
                    better_pairs.append(pair)

            if len(safer_pairs) < target_safer and row.get("safer_response_id") in (0, 1):
                safer_id = int(row["safer_response_id"])
                riskier_id = 1 - safer_id
                safety_contrast = (
                    row.get("is_response_0_safe") != row.get("is_response_1_safe")
                    or row.get("response_0_severity_level") != row.get("response_1_severity_level")
                )
                pair = make_pair(
                    source_dataset=dataset,
                    source_config="default",
                    source_split="train",
                    task_type="pairwise_safety_preference",
                    prompt=prompt,
                    better=response(safer_id),
                    worse=response(riskier_id),
                    label_source="safer_response_id",
                    source_row_id=row_idx,
                    metadata={
                        "safer_response_id": safer_id,
                        "better_response_id": row.get("better_response_id"),
                        "safety_contrast": bool(safety_contrast),
                        "is_response_0_safe": row.get("is_response_0_safe"),
                        "is_response_1_safe": row.get("is_response_1_safe"),
                        "response_0_severity_level": row.get("response_0_severity_level"),
                        "response_1_severity_level": row.get("response_1_severity_level"),
                    },
                )
                if pair:
                    safer_pairs.append(pair)

        offset += len(rows)
        if total is not None and offset >= total:
            break
    return better_pairs, safer_pairs


def build_rewardbench2(target: int) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    dataset = "allenai/reward-bench-2"
    pairs: list[dict[str, Any]] = []
    import pandas as pd

    url = "https://huggingface.co/datasets/allenai/reward-bench-2/resolve/main/data/test-00000-of-00001.parquet"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    frame = pd.read_parquet(BytesIO(response.content))
    for row_idx, row in frame.iterrows():
        if len(pairs) >= target:
            break
        metadata = row.get("additional_metadata")
        if hasattr(metadata, "as_py"):
            metadata = metadata.as_py()
        if not isinstance(metadata, dict):
            metadata = {}
        pair = make_pair(
            source_dataset=dataset,
            source_config="default",
            source_split="test",
            task_type="heldout_reward_benchmark",
            prompt=row.get("prompt", ""),
            better=to_text(row.get("chosen")),
            worse=to_text(row.get("rejected")),
            label_source="chosen/rejected",
            source_row_id=str(row.get("id", row_idx)),
            metadata={
                "subset": row.get("subset"),
                "num_correct": int(row.get("num_correct")) if row.get("num_correct") is not None else None,
                "num_incorrect": int(row.get("num_incorrect")) if row.get("num_incorrect") is not None else None,
                "category": metadata.get("category"),
                "subcategory": metadata.get("subcategory"),
            },
        )
        if pair:
            pairs.append(pair)
    return pairs[:target]


def build_summarize(target: int) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    dataset = "openai/summarize_from_feedback"
    pairs: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        offset = 0
        while len(pairs) < target:
            rows = fetch_rows(dataset, "comparisons", split, max_rows=min(100, target - len(pairs)), start_offset=offset)
            if not rows:
                break
            for row in rows:
                choice = row.get("choice")
                summaries = row.get("summaries") or []
                if choice not in (0, 1) or len(summaries) < 2:
                    continue
                info = row.get("info") or {}
                prompt = "\n\n".join(
                    part
                    for part in [
                        f"Title: {info.get('title')}" if info.get("title") else "",
                        info.get("post") or info.get("article") or "",
                    ]
                    if part
                )
                rejected_id = 1 - int(choice)
                pair = make_pair(
                    source_dataset=dataset,
                    source_config="comparisons",
                    source_split=split,
                    task_type="summarization_preference",
                    prompt=prompt,
                    better=summaries[int(choice)].get("text", ""),
                    worse=summaries[rejected_id].get("text", ""),
                    label_source="human_choice",
                    source_row_id=str(row.get("_row_idx")),
                    metadata={
                        "choice": choice,
                        "batch": row.get("batch"),
                        "subreddit": info.get("subreddit"),
                    },
                )
                if pair:
                    pairs.append(pair)
            offset += len(rows)
            if len(rows) < 100:
                break
    return pairs[:target]


def build_webgpt(target: int) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    dataset = "openai/webgpt_comparisons"
    pairs: list[dict[str, Any]] = []
    offset = 0
    while len(pairs) < target:
        rows = fetch_rows(dataset, "default", "train", max_rows=min(100, target - len(pairs)), start_offset=offset)
        if not rows:
            break
        for row in rows:
            score_0 = row.get("score_0")
            score_1 = row.get("score_1")
            if score_0 == score_1 or score_0 is None or score_1 is None:
                continue
            better_id = 0 if score_0 > score_1 else 1
            worse_id = 1 - better_id
            question = row.get("question") or {}
            pair = make_pair(
                source_dataset=dataset,
                source_config="default",
                source_split="train",
                task_type="factual_qa_preference",
                prompt=question.get("full_text", ""),
                better=row.get(f"answer_{better_id}", ""),
                worse=row.get(f"answer_{worse_id}", ""),
                label_source="human_score",
                source_row_id=str(row.get("_row_idx")),
                metadata={
                    "score_better": row.get(f"score_{better_id}"),
                    "score_worse": row.get(f"score_{worse_id}"),
                    "question_dataset": question.get("dataset"),
                    "question_id": question.get("id"),
                },
            )
            if pair:
                pairs.append(pair)
        offset += len(rows)
        if len(rows) < 100:
            break
    return pairs[:target]


def build_openr1(target: int, max_scan_rows: int) -> list[dict[str, Any]]:
    dataset = "open-r1/OpenR1-Math-220k"
    pairs: list[dict[str, Any]] = []
    offset = 0
    while len(pairs) < target and offset < max_scan_rows:
        try:
            rows = fetch_rows(dataset, "default", "train", max_rows=min(100, max_scan_rows - offset), start_offset=offset)
        except RuntimeError as exc:
            print(f"  OpenR1 pagination stopped early at offset {offset}: {exc}", flush=True)
            break
        if not rows:
            break
        for row in rows:
            generations = row.get("generations") or []
            if not generations:
                continue
            source = None
            labels = row.get("correctness_math_verify") or []
            if True not in labels or False not in labels:
                labels = row.get("correctness_llama") or []
                source = "correctness_llama"
            else:
                source = "correctness_math_verify"
            if True not in labels or False not in labels:
                continue
            correct_idx = next(i for i, value in enumerate(labels) if value is True)
            incorrect_idx = next(i for i, value in enumerate(labels) if value is False)
            if correct_idx >= len(generations) or incorrect_idx >= len(generations):
                continue
            pair = make_pair(
                source_dataset=dataset,
                source_config="default",
                source_split="train",
                task_type="verifiable_math_generation",
                prompt=row.get("problem", ""),
                better=generations[correct_idx],
                worse=generations[incorrect_idx],
                label_source=source or "correctness",
                source_row_id=str(row.get("uuid") or row.get("_row_idx")),
                metadata={
                    "problem_type": row.get("problem_type"),
                    "question_type": row.get("question_type"),
                    "source": row.get("source"),
                    "correct_idx": correct_idx,
                    "incorrect_idx": incorrect_idx,
                    "correctness_count": row.get("correctness_count"),
                    "answer": row.get("answer"),
                },
            )
            if pair:
                pairs.append(pair)
                if len(pairs) >= target:
                    break
        offset += len(rows)
        time.sleep(1.0)
        if len(rows) < 100:
            break
    return pairs


def iter_prm800k_rows(files: list[str]):
    base = "https://huggingface.co/datasets/tasksource/PRM800K/resolve/main"
    for filename in files:
        url = f"{base}/{filename}"
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    yield filename, json.loads(line)


def build_prm800k(target: int, max_source_rows: int) -> list[dict[str, Any]]:
    dataset = "tasksource/PRM800K"
    files = ["phase1_train.jsonl", "phase1_test.jsonl", "phase2_train.jsonl", "phase2_test.jsonl"]
    pairs: list[dict[str, Any]] = []
    source_rows = 0
    for filename, row in iter_prm800k_rows(files):
        source_rows += 1
        if source_rows > max_source_rows:
            break
        problem = (row.get("question") or {}).get("problem", "")
        previous_steps: list[str] = []
        steps = (row.get("label") or {}).get("steps") or []
        for step_idx, step in enumerate(steps):
            completions = step.get("completions") or []
            rated = [
                (idx, comp)
                for idx, comp in enumerate(completions)
                if comp.get("rating") is not None and comp.get("text")
            ]
            if rated:
                max_rating = max(comp.get("rating") for _, comp in rated)
                min_rating = min(comp.get("rating") for _, comp in rated)
                if max_rating > min_rating:
                    best_idx, best = next((idx, comp) for idx, comp in rated if comp.get("rating") == max_rating)
                    worst_idx, worst = next((idx, comp) for idx, comp in rated if comp.get("rating") == min_rating)
                    prompt_parts = [f"Problem:\n{problem}"]
                    if previous_steps:
                        prompt_parts.append("Previous solution steps:\n" + "\n".join(previous_steps))
                    prompt_parts.append("Choose the next solution step.")
                    pair = make_pair(
                        source_dataset=dataset,
                        source_config="default",
                        source_split=filename.replace(".jsonl", ""),
                        task_type="process_reward_step",
                        prompt="\n\n".join(prompt_parts),
                        better=best.get("text", ""),
                        worse=worst.get("text", ""),
                        label_source="human_step_rating",
                        source_row_id=f"{source_rows}:{step_idx}:{best_idx}:{worst_idx}",
                        metadata={
                            "step_idx": step_idx,
                            "better_rating": max_rating,
                            "worse_rating": min_rating,
                            "ground_truth_answer": (row.get("question") or {}).get("ground_truth_answer"),
                            "generation": row.get("generation"),
                        },
                    )
                    if pair:
                        pairs.append(pair)
                        if len(pairs) >= target:
                            return pairs

            chosen_idx = step.get("chosen_completion")
            chosen_text = None
            if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
                chosen_text = completions[chosen_idx].get("text")
            if not chosen_text and completions:
                chosen_text = max(
                    (comp for comp in completions if comp.get("text")),
                    key=lambda comp: comp.get("rating") if comp.get("rating") is not None else -999,
                    default={},
                ).get("text")
            if chosen_text:
                previous_steps.append(chosen_text.strip())
    return pairs


def write_readme(path: Path, manifest: dict[str, Any]) -> int:
    lines = [
        "# RL Dataset Slices for Centroid Validation",
        "",
        "These files are compact prompt/better/worse extracts, not full mirrors.",
        "They are intended for centroid experiments that split by prompt/problem ID.",
        "",
        "Embedding format remains the repo standard:",
        "",
        "```text",
        "User: {prompt}",
        "Assistant: {response}",
        "```",
        "",
        "## Files",
        "",
    ]
    for file_info in manifest["files"]:
        if file_info["n_pairs"] in (0, None):
            continue
        lines.append(
            f"- `{file_info['file']}`: {file_info['n_pairs']} pairs, "
            f"{file_info['size_bytes'] / (1024 * 1024):.2f} MiB, "
            f"{file_info['task_type']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- RewardBench2 is held out for evaluation only; do not train on it.",
            "- PRM800K prompts include previous solution steps and score candidate next steps.",
            "- OpenR1 pairs are correct vs incorrect generations for the same math problem.",
            "- PKU files preserve separate helpfulness-style and safety-style labels.",
            "- HelpSteer2 pairs come from the repo's pairwise preference file and preserve preference strength.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-local-mb", type=float, default=1500.0)
    parser.add_argument("--openr1-max-scan-rows", type=int, default=8000)
    parser.add_argument("--prm-max-source-rows", type=int, default=20000)
    for name, default in DEFAULT_TARGETS.items():
        parser.add_argument(f"--{name}-target", type=int, default=default)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_by": "scripts/download_centroid_rl_datasets.py",
        "strategy": "compact standardized pair slices; no full dataset mirrors",
        "max_local_mb": args.max_local_mb,
        "dataset_sizes": {},
        "files": [],
    }

    builders = [
        ("helpsteer2_preference_pairs.jsonl", lambda: build_helpsteer2(args.helpsteer2_target)),
        ("pku_better_pairs.jsonl", None),
        ("pku_safer_pairs.jsonl", None),
        ("prm800k_step_pairs.jsonl", lambda: build_prm800k(args.prm800k_target, args.prm_max_source_rows)),
        ("openr1_math_pairs.jsonl", lambda: build_openr1(args.openr1_target, args.openr1_max_scan_rows)),
        ("summarize_feedback_pairs.jsonl", lambda: build_summarize(args.summarize_target)),
        ("webgpt_comparison_pairs.jsonl", lambda: build_webgpt(args.webgpt_target)),
        ("rewardbench2_eval_pairs.jsonl", lambda: build_rewardbench2(args.rewardbench2_target)),
    ]

    for dataset in [
        "nvidia/HelpSteer2",
        "PKU-Alignment/PKU-SafeRLHF",
        "tasksource/PRM800K",
        "open-r1/OpenR1-Math-220k",
        "openai/summarize_from_feedback",
        "openai/webgpt_comparisons",
        "allenai/reward-bench-2",
    ]:
        manifest["dataset_sizes"][dataset] = dataset_size(dataset)

    pku_better, pku_safer = build_pku(args.pku_better_target, args.pku_safer_target)
    pku_outputs = {
        "pku_better_pairs.jsonl": pku_better,
        "pku_safer_pairs.jsonl": pku_safer,
    }

    total_bytes = 0
    for filename, builder in builders:
        if builder is None:
            rows = pku_outputs[filename]
        else:
            print(f"Building {filename}...", flush=True)
            rows = builder()
        path = out_dir / filename
        if not rows:
            if path.exists():
                path.unlink()
            print("  wrote 0 pairs, skipped empty file", flush=True)
            continue
        size_bytes = write_jsonl(path, rows)
        total_bytes += size_bytes
        task_type = rows[0]["task_type"] if rows else "empty"
        manifest["files"].append(
            {
                "file": filename,
                "n_pairs": len(rows),
                "size_bytes": size_bytes,
                "task_type": task_type,
                "source_dataset": rows[0]["source_dataset"] if rows else None,
            }
        )
        print(f"  wrote {len(rows)} pairs, {size_bytes / (1024 * 1024):.2f} MiB", flush=True)
        if total_bytes > args.max_local_mb * 1024 * 1024:
            raise RuntimeError(f"local slice budget exceeded: {total_bytes / (1024 * 1024):.1f} MiB")

    manifest["total_size_bytes"] = total_bytes
    readme_size = write_readme(out_dir / "README.md", manifest)
    manifest["files"].append(
        {
            "file": "README.md",
            "n_pairs": None,
            "size_bytes": readme_size,
            "task_type": "documentation",
            "source_dataset": None,
        }
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"Saved manifest: {manifest_path}", flush=True)
    print(f"Total local slice size: {total_bytes / (1024 * 1024):.2f} MiB", flush=True)


if __name__ == "__main__":
    main()
