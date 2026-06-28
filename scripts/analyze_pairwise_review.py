#!/usr/bin/env python3
"""Analyze blinded pairwise review packets after adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def outcome_from_review(review_row: dict[str, Any], key_row: dict[str, Any]) -> dict[str, Any]:
    fields = review_row.get("review_fields", {})
    winner_blind_id = str(fields.get("winner_blind_id", "") or "").strip()
    tie = bool(fields.get("tie"))
    reject_both = bool(fields.get("reject_both"))
    notes = fields.get("notes", "")

    blind_to_method = {row["blind_id"]: row["method"] for row in key_row["blind_map"]}
    comparison_label = f"{key_row['focus_method']} vs {key_row['baseline_method']}"

    if reject_both and (tie or winner_blind_id):
        status = "invalid"
        winner_method = ""
    elif tie and winner_blind_id:
        status = "invalid"
        winner_method = ""
    elif reject_both:
        status = "reject_both"
        winner_method = ""
    elif tie:
        status = "tie"
        winner_method = ""
    elif not winner_blind_id:
        status = "incomplete"
        winner_method = ""
    elif winner_blind_id not in blind_to_method:
        status = "invalid"
        winner_method = ""
    else:
        winner_method = blind_to_method[winner_blind_id]
        if winner_method == key_row["focus_method"]:
            status = "focus_win"
        elif winner_method == key_row["baseline_method"]:
            status = "baseline_win"
        else:
            status = "invalid"

    return {
        "review_id": review_row["review_id"],
        "item_id": key_row["item_id"],
        "category": key_row.get("category", ""),
        "comparison": comparison_label,
        "focus_method": key_row["focus_method"],
        "baseline_method": key_row["baseline_method"],
        "status": status,
        "winner_method": winner_method,
        "winner_blind_id": winner_blind_id,
        "tie": tie,
        "reject_both": reject_both,
        "notes": notes,
    }


def summarized_row(counter: Counter[str]) -> dict[str, float | int]:
    decided = counter["focus_win"] + counter["baseline_win"]
    reviewed = decided + counter["tie"] + counter["reject_both"]
    focus_win_rate = counter["focus_win"] / decided if decided else 0.0
    focus_non_loss = (counter["focus_win"] + 0.5 * counter["tie"]) / reviewed if reviewed else 0.0
    return {
        "focus_wins": counter["focus_win"],
        "baseline_wins": counter["baseline_win"],
        "ties": counter["tie"],
        "reject_both": counter["reject_both"],
        "incomplete": counter["incomplete"],
        "invalid": counter["invalid"],
        "decided": decided,
        "reviewed": reviewed,
        "focus_win_rate_decided": focus_win_rate,
        "focus_non_loss_rate_reviewed": focus_non_loss,
    }


def build_summary(
    aggregate_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    metadata: dict[str, str],
) -> str:
    lines = [
        "# Pairwise Review Analysis",
        "",
        f"Review packet: `{metadata['review']}`",
        f"Key: `{metadata['key']}`",
        "",
        "## Comparison Results",
        "",
        "| Comparison | Focus wins | Baseline wins | Ties | Reject both | Incomplete | Focus win rate (decided) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| `{row['comparison']}` | {row['focus_wins']} | {row['baseline_wins']} | "
            f"{row['ties']} | {row['reject_both']} | {row['incomplete']} | "
            f"{row['focus_win_rate_decided']:.1%} |"
        )

    if category_rows:
        lines.extend(
            [
                "",
                "## Category Breakdown",
                "",
                "| Comparison | Category | Focus wins | Baseline wins | Ties | Reject both | Decided | Focus win rate (decided) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in category_rows:
            lines.append(
                f"| `{row['comparison']}` | `{row['category']}` | {row['focus_wins']} | "
                f"{row['baseline_wins']} | {row['ties']} | {row['reject_both']} | "
                f"{row['decided']} | {row['focus_win_rate_decided']:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Treat incomplete reviews, ties, and reject-both cases as signal about benchmark quality, not noise to hide.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True, help="Completed or partially completed review_packet.jsonl.")
    parser.add_argument("--key", type=Path, required=True, help="Hidden review_key.json generated by build_pairwise_review.py.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for analysis artifacts.")
    args = parser.parse_args()

    review_rows = read_jsonl(args.review)
    key_rows = read_json(args.key)
    key_map = {row["review_id"]: row for row in key_rows}

    result_rows: list[dict[str, Any]] = []
    aggregate_counters: dict[str, Counter[str]] = defaultdict(Counter)
    category_counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for review_row in review_rows:
        review_id = review_row["review_id"]
        key_row = key_map.get(review_id)
        if key_row is None:
            raise KeyError(f"Missing key row for review_id {review_id}")
        result = outcome_from_review(review_row, key_row)
        result_rows.append(result)
        aggregate_counters[result["comparison"]][result["status"]] += 1
        category_counters[(result["comparison"], result["category"])][result["status"]] += 1

    aggregate_rows: list[dict[str, Any]] = []
    for comparison in sorted(aggregate_counters):
        summary = summarized_row(aggregate_counters[comparison])
        aggregate_rows.append({"comparison": comparison, **summary})

    category_rows: list[dict[str, Any]] = []
    for comparison, category in sorted(category_counters):
        summary = summarized_row(category_counters[(comparison, category)])
        if summary["reviewed"] or summary["invalid"]:
            category_rows.append({"comparison": comparison, "category": category, **summary})

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "results.json", result_rows)
    write_csv(args.output / "results.csv", result_rows)
    write_json(args.output / "aggregate.json", aggregate_rows)
    write_csv(args.output / "aggregate.csv", aggregate_rows)
    write_json(args.output / "category_breakdown.json", category_rows)
    write_csv(args.output / "category_breakdown.csv", category_rows)
    summary = build_summary(
        aggregate_rows=aggregate_rows,
        category_rows=category_rows,
        metadata={"review": args.review.as_posix(), "key": args.key.as_posix()},
    )
    (args.output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
