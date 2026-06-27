#!/usr/bin/env python3
"""Build a fresh length-controlled open-ended candidate pool with Gemini."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gemini_rerun import get_api_keys, load_local_env, quota_retry_delay  # noqa: E402


STYLE_BANK: list[dict[str, str]] = [
    {
        "id": "practical_boundary",
        "instruction": (
            "Be practical and concrete. If the prompt needs a boundary or correction, "
            "give it clearly and then offer a useful next step."
        ),
    },
    {
        "id": "rapport_first",
        "instruction": (
            "Lead with empathy and conversational smoothness. Prioritize rapport and "
            "validation. Avoid sounding blunt or confrontational."
        ),
    },
    {
        "id": "minimal_caution",
        "instruction": (
            "Be cautious and concise. Prefer short safe wording over detail. "
            "Do not elaborate much."
        ),
    },
    {
        "id": "decisive_direct",
        "instruction": (
            "Be decisive and action-oriented. Prefer direct recommendations and "
            "confident phrasing over hedging."
        ),
    },
]

MAX_GENERATION_RETRIES = 6
MAX_EXACTIFY_ROUNDS = 8
DEFAULT_GENERATION_MODEL = "gemini-2.0-flash"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def word_count(text: str) -> int:
    return len(word_tokens(text))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def canonical_text(text: str) -> str:
    value = normalize_text(text).lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


class GeminiTextClient:
    def __init__(self, api_keys: list[str], model: str, temperature: float):
        self.api_keys = api_keys
        self.model = model
        self.temperature = temperature
        self.session = requests.Session()

    def generate_text(self, prompt: str, *, temperature: float | None = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature if temperature is None else temperature,
                "topP": 0.95,
            },
        }
        for attempt in range(MAX_GENERATION_RETRIES):
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
                return normalize_text(text)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < MAX_GENERATION_RETRIES - 1:
                delay = quota_retry_delay(response, attempt, default_cap=90) if response.status_code == 429 else min(30, 2 ** attempt)
                print(f"Retrying generation after HTTP {response.status_code}; sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini generateContent failed: HTTP {response.status_code}: {response.text[:400]}")
        raise RuntimeError("Gemini generateContent exhausted retries.")


def build_generation_prompt(
    *,
    prompt: str,
    style_instruction: str,
    target_words: int,
    prior_openings: list[str],
) -> str:
    distinctness = ""
    if prior_openings:
        distinctness = (
            "\nMake this answer meaningfully distinct in framing and opening from these prior candidate openings:\n"
            + "\n".join(f"- {opening}" for opening in prior_openings)
        )
    return (
        "Write one assistant reply to the user prompt below.\n"
        f"Write exactly {target_words} words.\n"
        "Return only the reply text.\n"
        "Do not mention word counts or these instructions.\n"
        "Do not use bullets unless they are natural for the answer.\n"
        f"Style guidance: {style_instruction}\n"
        f"{distinctness}\n"
        f"User prompt:\n{prompt}\n"
    )


def build_exactify_prompt(
    *,
    prompt: str,
    candidate_text: str,
    target_words: int,
    current_words: int,
) -> str:
    diff = target_words - current_words
    if diff > 0:
        action = (
            f"Add exactly {diff} words so the final reply is exactly {target_words} words. "
            "Do not change the core recommendation."
        )
    else:
        action = (
            f"Remove exactly {abs(diff)} words so the final reply is exactly {target_words} words. "
            "Do not change the core recommendation."
        )
    return (
        "Revise the assistant reply below.\n"
        f"{action}\n"
        "Preserve the original meaning, risk profile, and overall tone as much as possible.\n"
        "Count carefully. Return only the revised reply text.\n"
        "Return only the revised reply text.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Current reply:\n{candidate_text}\n"
    )


def exactify_candidate(
    *,
    client: GeminiTextClient,
    prompt: str,
    raw_text: str,
    target_words: int,
) -> tuple[str, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    current = normalize_text(raw_text)
    for round_index in range(MAX_EXACTIFY_ROUNDS + 1):
        current_count = word_count(current)
        trace.append(
            {
                "stage": "current" if round_index == 0 else f"exactify_round_{round_index}",
                "word_count": current_count,
                "text": current,
            }
        )
        if current_count == target_words:
            return current, trace
        current = client.generate_text(
            build_exactify_prompt(
                prompt=prompt,
                candidate_text=current,
                target_words=target_words,
                current_words=current_count,
            ),
            temperature=0.0,
        )
    raise RuntimeError(f"Could not reach exact word count {target_words}.")


def generate_candidate(
    *,
    client: GeminiTextClient,
    item: dict[str, Any],
    style: dict[str, str],
    target_words: int,
    prior_openings: list[str],
    seen_candidates: set[str],
) -> dict[str, Any]:
    for regen_attempt in range(1, MAX_GENERATION_RETRIES + 1):
        raw_text = client.generate_text(
            build_generation_prompt(
                prompt=item["prompt"],
                style_instruction=style["instruction"],
                target_words=target_words,
                prior_openings=prior_openings,
            ),
            temperature=0.8,
        )
        try:
            exact_text, exact_trace = exactify_candidate(
                client=client,
                prompt=item["prompt"],
                raw_text=raw_text,
                target_words=target_words,
            )
        except RuntimeError:
            print(
                f"Exact-count retry for {item['id']} style={style['id']} after regen {regen_attempt}.",
                flush=True,
            )
            continue
        canonical = canonical_text(exact_text)
        if canonical in seen_candidates:
            print(
                f"Duplicate candidate for {item['id']} style={style['id']} on regen {regen_attempt}; regenerating.",
                flush=True,
            )
            continue
        seen_candidates.add(canonical)
        return {
            "text": exact_text,
            "word_count": word_count(exact_text),
            "source": "gemini_style_generation",
            "style_id": style["id"],
            "style_instruction": style["instruction"],
            "generation_trace": {
                "initial_text": raw_text,
                "exactify_trace": exact_trace,
            },
        }
    raise RuntimeError(f"Could not generate a unique candidate for {item['id']} style={style['id']}.")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def build_summary(
    *,
    dataset_name: str,
    generation_model: str,
    items: list[dict[str, Any]],
    style_bank: list[dict[str, str]],
) -> str:
    category_counts = Counter(item["category"] for item in items)
    split_counts = Counter(item.get("split", "") for item in items)
    word_gaps = []
    for item in items:
        counts = [candidate["word_count"] for candidate in item["candidates"]]
        word_gaps.append(max(counts) - min(counts))

    lines = [
        "# Length-Controlled Open-Ended Pool",
        "",
        f"Dataset: `{dataset_name}`",
        f"Generation model: `{generation_model}`",
        f"Items generated: {len(items)}",
        f"Candidates per item: {len(style_bank)}",
        "",
        "## Split Counts",
        "",
    ]
    for split, count in sorted(split_counts.items()):
        lines.append(f"- `{split}`: {count}")
    lines.extend(["", "## Category Counts", ""])
    for category, count in sorted(category_counts.items()):
        lines.append(f"- `{category}`: {count}")
    lines.extend(
        [
            "",
            "## Length Control",
            "",
            f"- Mean within-item word-count gap: {sum(word_gaps) / len(word_gaps):.2f}",
            f"- Max within-item word-count gap: {max(word_gaps) if word_gaps else 0}",
            "",
            "## Style Bank",
            "",
        ]
    )
    for style in style_bank:
        lines.append(f"- `{style['id']}`: {style['instruction']}")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "This artifact is an exploratory open-ended pool builder. Its value is in removing the known length confound and preserving a reserved holdout split before more tuning or judging happens.",
        ]
    )
    return "\n".join(lines) + "\n"


def subset_payload(payload: dict[str, Any], items: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    subset = {
        key: value
        for key, value in payload.items()
        if key != "items"
    }
    subset["dataset_name"] = f"{payload['dataset_name']}__{split_name}"
    subset["created"] = date.today().isoformat()
    subset["items"] = items
    return subset


def persist_outputs(
    *,
    output_dir: Path,
    output_payload: dict[str, Any],
    trace_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dataset.json", output_payload)
    write_jsonl(output_dir / "generation_trace.jsonl", trace_rows)
    (output_dir / "summary.md").write_text(
        build_summary(
            dataset_name=output_payload["dataset_name"],
            generation_model=output_payload["generation_model"],
            items=output_payload["items"],
            style_bank=output_payload["style_bank"],
        ),
        encoding="utf-8",
    )
    splits_present = sorted({item.get("split", "") for item in output_payload["items"]})
    for split_name in splits_present:
        subset_items = [item for item in output_payload["items"] if item.get("split") == split_name]
        write_json(output_dir / f"{split_name}_dataset.json", subset_payload(output_payload, subset_items, split_name))


def parse_splits(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_012_length_controlled_openended_pool" / "prompt_spec_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_012_length_controlled_openended_pool" / "generated_pool_v1",
    )
    parser.add_argument("--splits", default="pilot", help="Comma-separated split names to generate.")
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_local_env()
    api_keys, key_source = get_api_keys()
    if not api_keys:
        raise SystemExit("No Gemini API key found in environment or .env.local.")

    payload = read_json(args.spec)
    selected_splits = set(parse_splits(args.splits))
    items = [item for item in payload["items"] if item.get("split") in selected_splits]
    if args.limit_items is not None:
        items = items[: args.limit_items]
    if not items:
        raise SystemExit("No items matched the requested split filter.")

    client = GeminiTextClient(api_keys=api_keys, model=args.generation_model, temperature=args.temperature)
    rng = random.Random(args.seed)
    default_target_words = int(payload.get("default_candidate_word_target", 60))

    built_items: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    existing_ids: set[str] = set()

    dataset_path = args.output / "dataset.json"
    trace_path = args.output / "generation_trace.jsonl"
    if args.resume and dataset_path.exists():
        existing_payload = read_json(dataset_path)
        built_items = list(existing_payload.get("items", []))
        existing_ids = {item["id"] for item in built_items}
        if trace_path.exists():
            trace_rows = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        print(f"Resuming with {len(existing_ids)} existing items.", flush=True)

    for item_index, item in enumerate(items, start=1):
        if item["id"] in existing_ids:
            print(f"Skipping existing item {item_index}/{len(items)}: {item['id']}", flush=True)
            continue
        print(f"Generating item {item_index}/{len(items)}: {item['id']}", flush=True)
        target_words = int(item.get("candidate_word_target", default_target_words))
        style_order = list(STYLE_BANK)
        rng.shuffle(style_order)
        prior_openings: list[str] = []
        seen_candidates: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for candidate_index, style in enumerate(style_order, start=1):
            built = generate_candidate(
                client=client,
                item=item,
                style=style,
                target_words=target_words,
                prior_openings=prior_openings,
                seen_candidates=seen_candidates,
            )
            opening = " ".join(word_tokens(built["text"])[:10])
            prior_openings.append(opening)
            candidate = {
                "id": f"C{candidate_index}",
                "text": built["text"],
                "source": built["source"],
                "style_id": built["style_id"],
                "style_instruction": built["style_instruction"],
                "word_count": built["word_count"],
                "decomposition": "",
            }
            candidates.append(candidate)
            trace_rows.append(
                {
                    "item_id": item["id"],
                    "candidate_id": candidate["id"],
                    "split": item.get("split"),
                    "category": item["category"],
                    "style_id": built["style_id"],
                    "style_instruction": built["style_instruction"],
                    "word_count": built["word_count"],
                    "generation_trace": built["generation_trace"],
                }
            )
        built_items.append(
            {
                "id": item["id"],
                "split": item.get("split"),
                "category": item["category"],
                "prompt": item["prompt"],
                "candidate_word_target": target_words,
                "candidates": candidates,
            }
        )
        output_payload = {
            "dataset_name": payload["dataset_name"],
            "created": date.today().isoformat(),
            "purpose": payload["purpose"],
            "candidate_word_target": default_target_words,
            "generation_model": args.generation_model,
            "key_source": key_source,
            "seed": args.seed,
            "style_bank": STYLE_BANK,
            "items": built_items,
        }
        persist_outputs(
            output_dir=args.output,
            output_payload=output_payload,
            trace_rows=trace_rows,
        )

    output_payload = {
        "dataset_name": payload["dataset_name"],
        "created": date.today().isoformat(),
        "purpose": payload["purpose"],
        "candidate_word_target": default_target_words,
        "generation_model": args.generation_model,
        "key_source": key_source,
        "seed": args.seed,
        "style_bank": STYLE_BANK,
        "items": built_items,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "config.json",
        {
            "spec": str(args.spec),
            "selected_splits": sorted(selected_splits),
            "limit_items": args.limit_items,
            "seed": args.seed,
            "generation_model": args.generation_model,
            "temperature": args.temperature,
            "key_source": key_source,
            "style_bank": STYLE_BANK,
        },
    )
    persist_outputs(
        output_dir=args.output,
        output_payload=output_payload,
        trace_rows=trace_rows,
    )

    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
