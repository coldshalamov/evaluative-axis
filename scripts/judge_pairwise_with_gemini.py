#!/usr/bin/env python3
"""Fill blinded pairwise review packets with Gemini judgments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gemini_rerun import get_api_keys, load_local_env, quota_retry_delay  # noqa: E402


JUDGE_PROMPT = """You are evaluating two AI assistant responses to the same user message.

User message:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better overall? Consider: helpfulness, accuracy, honesty, safety, and whether it actually addresses the user's question. A good response can be a refusal if the request is harmful, as long as it's appropriate.

Reply with EXACTLY one of:
- "A" if Response A is clearly better
- "B" if Response B is clearly better
- "TIE" if they are roughly equal or you genuinely cannot decide

Then on the next line, give a one-sentence reason. Nothing else."""


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def short(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


class GeminiJudgeClient:
    def __init__(self, *, api_keys: list[str], model_name: str):
        self.api_keys = api_keys
        self.model_name = model_name
        self.session = requests.Session()

    def generate_text(self, prompt: str, *, max_retries: int = 6) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
            },
        }
        for attempt in range(max_retries):
            key = self.api_keys[attempt % len(self.api_keys)]
            response = self.session.post(
                url,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                json=body,
                timeout=90,
            )
            if response.ok:
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"No candidates returned: {data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
                if not text:
                    raise RuntimeError("Empty generateContent response.")
                return text
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                delay = quota_retry_delay(response, attempt, default_cap=90) if response.status_code == 429 else min(30, 2 ** attempt)
                print(f"Retrying judge call after HTTP {response.status_code}; sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini generateContent failed: HTTP {response.status_code}: {response.text[:400]}")
        raise RuntimeError("Gemini generateContent exhausted retries.")


def judge_pair(client: GeminiJudgeClient, prompt: str, response_a: str, response_b: str) -> tuple[str, str]:
    filled = JUDGE_PROMPT.format(
        prompt=short(prompt, 1500),
        response_a=short(response_a, 2500),
        response_b=short(response_b, 2500),
    )
    try:
        text = client.generate_text(filled)
    except Exception as exc:
        return "ERROR", str(exc)[:300]

    first_line = text.splitlines()[0].strip().upper() if text.splitlines() else ""
    if first_line in {"A", "B", "TIE"}:
        reason = text.splitlines()[1].strip() if len(text.splitlines()) > 1 else ""
        return first_line, reason
    for token in ("A", "B", "TIE"):
        if text.upper().startswith(token):
            return token, text
    return "TIE", f"Could not parse: {text[:200]}"


def outcome_to_original_orientation(swapped_outcome: str) -> str:
    if swapped_outcome == "A":
        return "B"
    if swapped_outcome == "B":
        return "A"
    return swapped_outcome


def choose_review_fields(
    direct_outcome: str,
    swapped_outcome: str,
    direct_reason: str,
    swapped_reason: str,
    winner_field: str,
    reject_field: str,
) -> tuple[dict[str, Any], str]:
    if direct_outcome not in {"TIE", "ERROR"} and direct_outcome == swapped_outcome:
        return (
            {
                winner_field: direct_outcome,
                "tie": False,
                reject_field: False,
                "notes": f"Gemini stable win. Direct: {direct_reason} Swapped: {swapped_reason}",
            },
            "stable_win",
        )
    if direct_outcome == "TIE" and swapped_outcome == "TIE":
        return (
            {
                winner_field: "",
                "tie": True,
                reject_field: False,
                "notes": f"Gemini stable tie. Direct: {direct_reason} Swapped: {swapped_reason}",
            },
            "stable_tie",
        )
    if "ERROR" in {direct_outcome, swapped_outcome}:
        return (
            {
                winner_field: "",
                "tie": False,
                reject_field: False,
                "notes": f"Gemini error. Direct={direct_outcome}:{direct_reason} Swapped={swapped_outcome}:{swapped_reason}",
            },
            "error",
        )
    return (
        {
            winner_field: "",
            "tie": True,
            reject_field: False,
            "notes": (
                "Gemini inconsistent across order flip; treated as tie. "
                f"Direct={direct_outcome}:{direct_reason} Swapped={swapped_outcome}:{swapped_reason}"
            ),
        },
        "order_sensitive",
    )


def build_summary(counts: Counter[str], total: int, output_path: Path, raw_path: Path, model_name: str) -> str:
    lines = [
        "# Gemini Pairwise Judge Summary",
        "",
        f"Model: `{model_name}`",
        f"Rows processed: {total}",
        "",
        "## Outcome Counts",
        "",
        f"- Stable wins: {counts['stable_win']}",
        f"- Stable ties: {counts['stable_tie']}",
        f"- Order-sensitive ties: {counts['order_sensitive']}",
        f"- Errors: {counts['error']}",
        "",
        "## Files",
        "",
        f"- Filled review packet: `{output_path.as_posix()}`",
        f"- Raw Gemini judgments: `{raw_path.as_posix()}`",
        "",
        "## Interpretation Rule",
        "",
        "Treat this as blinded LLM adjudication, not human gold truth. Stable order-flip agreement is the minimum bar for trusting a Gemini judgment.",
    ]
    return "\n".join(lines) + "\n"


def status_from_raw(raw_row: dict[str, Any]) -> str:
    return str(raw_row.get("final_status", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True, help="Path to review_packet.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the Gemini-filled review packet.")
    parser.add_argument("--raw-output", type=Path, default=None, help="Optional path for raw Gemini judgments JSON.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional path for a summary markdown file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output/raw-output if present.")
    parser.add_argument(
        "--model",
        default="gemini-flash-lite-latest",
        help="Gemini generation model to use for blind judging.",
    )
    args = parser.parse_args()

    load_local_env()
    api_keys, _ = get_api_keys()
    if not api_keys:
        raise RuntimeError("No Gemini API key for judge")
    client = GeminiJudgeClient(api_keys=api_keys, model_name=args.model)

    review_rows = read_jsonl(args.review)
    if args.limit is not None:
        review_rows = review_rows[: args.limit]

    raw_output = args.raw_output or args.output.with_name(f"{args.output.stem}_raw.json")
    summary_output = args.summary_output or args.output.with_suffix(".summary.md")

    filled_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    if args.resume:
        if args.output.exists():
            filled_rows = read_jsonl(args.output)
            processed_ids = {row["review_id"] for row in filled_rows}
        if raw_output.exists():
            raw_rows = read_json(raw_output)
        print(f"Resuming judge with {len(processed_ids)} existing rows.", flush=True)

    counts: Counter[str] = Counter()
    for raw_row in raw_rows:
        counts[status_from_raw(raw_row)] += 1

    total_rows = len(review_rows)
    try:
        for index, row in enumerate(review_rows, start=1):
            review_id = row.get("review_id") or row.get("item_id") or f"row_{index:04d}"
            if review_id in processed_ids:
                print(f"Skipping existing review {index}/{total_rows}: {review_id}", flush=True)
                continue

            responses = row["responses"]
            if len(responses) != 2:
                raise ValueError(f"Expected exactly 2 responses for review_id {review_id}")
            blind_a = responses[0]["blind_id"]
            blind_b = responses[1]["blind_id"]
            response_a = responses[0]["text"]
            response_b = responses[1]["text"]

            existing_fields = row.get("review_fields", {})
            winner_field = "better_blind_id" if "better_blind_id" in existing_fields else "winner_blind_id"
            reject_field = "both_bad_or_unusable" if "both_bad_or_unusable" in existing_fields else "reject_both"

            direct_outcome, direct_reason = judge_pair(client, row["prompt"], response_a, response_b)
            swapped_outcome_raw, swapped_reason = judge_pair(client, row["prompt"], response_b, response_a)
            swapped_outcome = outcome_to_original_orientation(swapped_outcome_raw)

            if direct_outcome == "A":
                direct_outcome = blind_a
            elif direct_outcome == "B":
                direct_outcome = blind_b
            if swapped_outcome == "A":
                swapped_outcome = blind_a
            elif swapped_outcome == "B":
                swapped_outcome = blind_b

            review_fields, status = choose_review_fields(
                direct_outcome=direct_outcome,
                swapped_outcome=swapped_outcome,
                direct_reason=direct_reason,
                swapped_reason=swapped_reason,
                winner_field=winner_field,
                reject_field=reject_field,
            )
            counts[status] += 1

            filled_row = dict(row)
            filled_row["review_id"] = review_id
            filled_row["review_fields"] = review_fields
            filled_rows.append(filled_row)
            raw_rows.append(
                {
                    "review_id": review_id,
                    "item_id": row.get("item_id"),
                    "category": row.get("category"),
                    "direct_outcome": direct_outcome,
                    "direct_reason": direct_reason,
                    "swapped_outcome_raw": swapped_outcome_raw,
                    "swapped_outcome_mapped": swapped_outcome,
                    "swapped_reason": swapped_reason,
                    "final_status": status,
                    "final_review_fields": review_fields,
                }
            )
            processed_ids.add(review_id)

            write_jsonl(args.output, filled_rows)
            write_json(raw_output, raw_rows)
            summary_output.write_text(
                build_summary(
                    counts=counts,
                    total=len(filled_rows),
                    output_path=args.output,
                    raw_path=raw_output,
                    model_name=args.model,
                ),
                encoding="utf-8",
            )
            print(f"Judged {len(processed_ids)}/{total_rows} rows ({review_id})", flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output, filled_rows)
        write_json(raw_output, raw_rows)
        summary_output.write_text(
            build_summary(
                counts=counts,
                total=len(filled_rows),
                output_path=args.output,
                raw_path=raw_output,
                model_name=args.model,
            ),
            encoding="utf-8",
        )

    print(summary_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
