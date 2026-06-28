#!/usr/bin/env python3
"""Build blinded pairwise review packets from intervention selections."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_items(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = read_json(path)
    items = payload["items"]
    item_map = {item["id"]: item for item in items}
    return payload, item_map


def load_selections(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    rows = read_json(path)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["item_id"], {})[row["method"]] = row
    return grouped


def find_candidate(item: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in item["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    raise KeyError(f"Candidate not found for item {item['id']}: {candidate_id}")


def build_summary(
    packet_rows: list[dict[str, Any]],
    focus_methods: list[str],
    baseline_methods: list[str],
    counts_by_pair: Counter[tuple[str, str]],
    sampled_by_pair: Counter[tuple[str, str]],
    skipped_same: Counter[tuple[str, str]],
    skipped_missing: Counter[tuple[str, str]],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# Pairwise Blind Review Packet",
        "",
        f"Dataset: `{metadata['dataset']}`",
        f"Selections: `{metadata['selections']}`",
        f"Review rows: {len(packet_rows)}",
        f"Focus methods: {', '.join(f'`{name}`' for name in focus_methods)}",
        f"Baseline methods: {', '.join(f'`{name}`' for name in baseline_methods)}",
        "",
        "## Comparison Counts",
        "",
        "| Focus method | Baseline method | Available review pairs | Sampled review pairs | Same-candidate skips | Missing-method skips |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for focus_method in focus_methods:
        for baseline_method in baseline_methods:
            key = (focus_method, baseline_method)
            lines.append(
                f"| `{focus_method}` | `{baseline_method}` | "
                f"{counts_by_pair.get(key, 0)} | {sampled_by_pair.get(key, 0)} | "
                f"{skipped_same.get(key, 0)} | {skipped_missing.get(key, 0)} |"
            )

    lines.extend(
        [
            "",
            "## Review Rule",
            "",
            "Judge only the prompt and the two blinded responses. Do not open the key until review is finished.",
            "",
            "## Files",
            "",
            f"- Review packet: `{metadata['review_packet']}`",
            f"- Hidden key: `{metadata['review_key']}`",
            f"- Manifest: `{metadata['manifest']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_001_next" / "pilot_50_candidates.json",
    )
    parser.add_argument(
        "--selections",
        type=Path,
        required=True,
        help="Path to selections.json from run_cycle001_intervention.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the pairwise review packet.",
    )
    parser.add_argument(
        "--focus-methods",
        default="direct_category_axis,decomposition_category_axis,direct_combined,decomposition_combined",
    )
    parser.add_argument(
        "--baseline-methods",
        default="length,random,sentiment,refusal_heuristic",
    )
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument(
        "--max-per-comparison",
        type=int,
        default=None,
        help="Optional cap on the number of sampled rows kept for each focus/baseline comparison.",
    )
    args = parser.parse_args()

    payload, item_map = load_items(args.dataset)
    grouped_selections = load_selections(args.selections)
    focus_methods = parse_list(args.focus_methods)
    baseline_methods = parse_list(args.baseline_methods)
    rng = random.Random(args.seed)

    packet_rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    key_rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    counts_by_pair: Counter[tuple[str, str]] = Counter()
    sampled_by_pair: Counter[tuple[str, str]] = Counter()
    skipped_same: Counter[tuple[str, str]] = Counter()
    skipped_missing: Counter[tuple[str, str]] = Counter()
    review_index = 1

    for item_id, item in item_map.items():
        selections_for_item = grouped_selections.get(item_id, {})
        for focus_method in focus_methods:
            for baseline_method in baseline_methods:
                pair_key = (focus_method, baseline_method)
                focus_row = selections_for_item.get(focus_method)
                baseline_row = selections_for_item.get(baseline_method)
                if focus_row is None or baseline_row is None:
                    skipped_missing[pair_key] += 1
                    continue

                focus_candidate_id = focus_row["selected_candidate_id"]
                baseline_candidate_id = baseline_row["selected_candidate_id"]
                if focus_candidate_id == baseline_candidate_id:
                    skipped_same[pair_key] += 1
                    continue

                focus_candidate = find_candidate(item, focus_candidate_id)
                baseline_candidate = find_candidate(item, baseline_candidate_id)
                review_id = f"pair_{review_index:04d}"
                review_index += 1

                responses = [
                    {
                        "text": focus_candidate["text"],
                        "candidate_id": focus_candidate_id,
                        "method": focus_method,
                        "score": focus_row["score"],
                    },
                    {
                        "text": baseline_candidate["text"],
                        "candidate_id": baseline_candidate_id,
                        "method": baseline_method,
                        "score": baseline_row["score"],
                    },
                ]
                rng.shuffle(responses)
                labeled_responses = []
                for index, row in enumerate(responses):
                    labeled = dict(row)
                    labeled["blind_id"] = "A" if index == 0 else "B"
                    labeled_responses.append(labeled)
                blind_rows = [{"blind_id": row["blind_id"], "text": row["text"]} for row in labeled_responses]

                packet_row = (
                    {
                        "review_id": review_id,
                        "item_id": item_id,
                        "category": item.get("category", ""),
                        "prompt": item["prompt"],
                        "responses": blind_rows,
                        "review_fields": {
                            "winner_blind_id": "",
                            "tie": False,
                            "reject_both": False,
                            "notes": "",
                        },
                    }
                )

                key_row = (
                    {
                        "review_id": review_id,
                        "item_id": item_id,
                        "category": item.get("category", ""),
                        "focus_method": focus_method,
                        "baseline_method": baseline_method,
                        "focus_selection": {
                            "candidate_id": focus_candidate_id,
                            "score": focus_row["score"],
                            "matches_expected_fixture": focus_row.get("matches_expected_fixture"),
                        },
                        "baseline_selection": {
                            "candidate_id": baseline_candidate_id,
                            "score": baseline_row["score"],
                            "matches_expected_fixture": baseline_row.get("matches_expected_fixture"),
                        },
                        "blind_map": [
                            {
                                "blind_id": row["blind_id"],
                                "candidate_id": row["candidate_id"],
                                "method": row["method"],
                                "score": row["score"],
                            }
                            for row in labeled_responses
                        ],
                    }
                )
                packet_rows_by_pair.setdefault(pair_key, []).append(packet_row)
                key_rows_by_pair.setdefault(pair_key, []).append(key_row)
                counts_by_pair[pair_key] += 1

    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for focus_method in focus_methods:
        for baseline_method in baseline_methods:
            pair_key = (focus_method, baseline_method)
            pair_packet_rows = packet_rows_by_pair.get(pair_key, [])
            pair_key_rows = key_rows_by_pair.get(pair_key, [])
            if len(pair_packet_rows) != len(pair_key_rows):
                raise ValueError(f"Mismatched packet/key row counts for comparison {pair_key}")

            indices = list(range(len(pair_packet_rows)))
            if args.max_per_comparison is not None and len(indices) > args.max_per_comparison:
                indices = sorted(rng.sample(indices, args.max_per_comparison))

            sampled_by_pair[pair_key] = len(indices)
            for index in indices:
                packet_rows.append(pair_packet_rows[index])
                key_rows.append(pair_key_rows[index])

    args.output.mkdir(parents=True, exist_ok=True)
    review_packet = args.output / "review_packet.jsonl"
    review_key = args.output / "review_key.json"
    manifest = args.output / "manifest.json"
    summary_path = args.output / "summary.md"

    write_jsonl(review_packet, packet_rows)
    write_json(review_key, key_rows)
    manifest_payload = {
        "dataset_name": payload.get("dataset_name"),
        "purpose": payload.get("purpose"),
        "seed": args.seed,
        "focus_methods": focus_methods,
        "baseline_methods": baseline_methods,
        "max_per_comparison": args.max_per_comparison,
        "review_rows": len(packet_rows),
        "counts_by_pair": {
            f"{focus}__vs__{baseline}": counts_by_pair[(focus, baseline)]
            for focus in focus_methods
            for baseline in baseline_methods
        },
        "sampled_by_pair": {
            f"{focus}__vs__{baseline}": sampled_by_pair[(focus, baseline)]
            for focus in focus_methods
            for baseline in baseline_methods
        },
        "skipped_same_candidate": {
            f"{focus}__vs__{baseline}": skipped_same[(focus, baseline)]
            for focus in focus_methods
            for baseline in baseline_methods
        },
        "skipped_missing_method": {
            f"{focus}__vs__{baseline}": skipped_missing[(focus, baseline)]
            for focus in focus_methods
            for baseline in baseline_methods
        },
    }
    write_json(manifest, manifest_payload)

    summary = build_summary(
        packet_rows=packet_rows,
        focus_methods=focus_methods,
        baseline_methods=baseline_methods,
        counts_by_pair=counts_by_pair,
        sampled_by_pair=sampled_by_pair,
        skipped_same=skipped_same,
        skipped_missing=skipped_missing,
        metadata={
            "dataset": args.dataset.as_posix(),
            "selections": args.selections.as_posix(),
            "review_packet": review_packet.as_posix(),
            "review_key": review_key.as_posix(),
            "manifest": manifest.as_posix(),
        },
    )
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
