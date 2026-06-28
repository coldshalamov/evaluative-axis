#!/usr/bin/env python3
"""Word-stripping ablation: test whether evaluative signal survives
removal of explicit evaluative vocabulary from response texts.

If the signal depends on word-overlap between responses and anchor texts,
stripping shared vocabulary should collapse accuracy to chance. If the
signal reflects deeper semantic structure, accuracy should survive.

Usage:
    python scripts/run_word_stripping_ablation.py \
        --backend fastembed --model BAAI/bge-small-en-v1.5
    python scripts/run_word_stripping_ablation.py \
        --backend gemini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES, CATEGORY_AXIS_MAP, score_items  # noqa: E402

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

EVALUATIVE_WORDS = {
    "good", "bad", "better", "worse", "best", "worst",
    "right", "wrong", "correct", "incorrect",
    "safe", "unsafe", "dangerous", "harmful", "harmless",
    "helpful", "unhelpful", "useful", "useless",
    "honest", "dishonest", "truthful", "untruthful",
    "ethical", "unethical", "moral", "immoral",
    "appropriate", "inappropriate", "responsible", "irresponsible",
    "excellent", "terrible", "positive", "negative",
    "beneficial", "detrimental", "constructive", "destructive",
    "proper", "improper", "reliable", "unreliable",
    "accurate", "inaccurate", "valid", "invalid",
    "sycophantic", "flattery", "flatters",
    "refuses", "endorses", "corrects", "condemns", "discloses", "hides",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "and", "but", "or", "nor", "not", "no",
    "so", "if", "than", "that", "this", "these", "those", "it", "its",
}

ANCHOR_WORDS: set[str] = set()
for phrase in [
    "The assistant refuses wrongdoing",
    "Condemns harmful acts",
    "Corrects false claims",
    "Discloses serious risks",
    "Tells uncomfortable truths",
    "De-escalates harm",
    "Provides instructions for wrongdoing",
    "Endorses harmful acts",
    "Agrees with false claims",
    "Hides serious risks",
    "Lies to avoid discomfort",
    "Helps harmful activity",
]:
    for w in re.findall(r"[a-zA-Z']+", phrase.lower()):
        if w not in STOP_WORDS:
            ANCHOR_WORDS.add(w)

ALL_STRIP_WORDS = EVALUATIVE_WORDS | ANCHOR_WORDS


def strip_evaluative(text: str, words_to_strip: set[str]) -> str:
    def replacer(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() in words_to_strip:
            return "___"
        return w
    return re.sub(r"[a-zA-Z']+", replacer, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def as_items(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for case in cases:
        items.append({
            "id": case["id"],
            "category": case["category"],
            "phenomenon": case["phenomenon"],
            "prompt": case["prompt"],
            "expected_best_candidate_id": "better",
            "candidates": [
                {"id": "better", "text": case["better"], "decomposition": ""},
                {"id": "worse", "text": case["worse"], "decomposition": ""},
            ],
        })
    return items


def diff_correct(diff: float) -> float:
    if diff > 0:
        return 1.0
    if diff == 0:
        return 0.5
    return 0.0


def score_battery(items: list[dict[str, Any]], cases: list[dict[str, Any]],
                   backend: str, model: str | None, output_dir: Path,
                   batch_size: int) -> dict[str, float]:
    scores, _ = score_items(
        items=items, backend=backend, interfaces=["direct"],
        output_dir=output_dir, model=model, batch_size=batch_size,
        max_workers=2, sleep_between=0.0,
    )
    score_map = {(r.item_id, r.candidate_id): r for r in scores}

    metric_accs: dict[str, list[float]] = {}
    for case in cases:
        better = score_map[(case["id"], "better")]
        worse = score_map[(case["id"], "worse")]
        for axis_name in AXES:
            diff = better.axis_scores["direct"][axis_name] - worse.axis_scores["direct"][axis_name]
            metric_accs.setdefault(f"direct_{axis_name}", []).append(diff_correct(diff))
        diff_combined = better.combined_scores["direct"] - worse.combined_scores["direct"]
        metric_accs.setdefault("direct_combined", []).append(diff_correct(diff_combined))
        category_axis = CATEGORY_AXIS_MAP.get(case["category"], "general_evaluative")
        diff_cat = better.axis_scores["direct"][category_axis] - worse.axis_scores["direct"][category_axis]
        metric_accs.setdefault("direct_category_axis", []).append(diff_correct(diff_cat))

    return {name: sum(vals) / len(vals) for name, vals in sorted(metric_accs.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes" / "research_cycles" / "word_stripping_ablation")
    parser.add_argument("--backend", choices=["fastembed", "gemini"], default="fastembed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cases = read_jsonl(args.battery)
    print(f"Loaded {len(cases)} cases from {args.battery.name}")

    # --- Condition 1: Original (baseline) ---
    print("\n=== CONDITION 1: Original texts ===")
    items_orig = as_items(cases)
    accs_orig = score_battery(items_orig, cases, args.backend, args.model,
                               args.output / "original", args.batch_size)

    # --- Condition 2: Evaluative words stripped ---
    print("\n=== CONDITION 2: Evaluative words stripped ===")
    cases_eval_stripped = []
    for case in cases:
        c = dict(case)
        c["better"] = strip_evaluative(case["better"], EVALUATIVE_WORDS)
        c["worse"] = strip_evaluative(case["worse"], EVALUATIVE_WORDS)
        cases_eval_stripped.append(c)
    items_eval = as_items(cases_eval_stripped)
    accs_eval = score_battery(items_eval, cases_eval_stripped, args.backend, args.model,
                               args.output / "eval_stripped", args.batch_size)

    # --- Condition 3: Evaluative + anchor words stripped ---
    print("\n=== CONDITION 3: Evaluative + anchor words stripped ===")
    cases_all_stripped = []
    for case in cases:
        c = dict(case)
        c["better"] = strip_evaluative(case["better"], ALL_STRIP_WORDS)
        c["worse"] = strip_evaluative(case["worse"], ALL_STRIP_WORDS)
        cases_all_stripped.append(c)
    items_all = as_items(cases_all_stripped)
    accs_all = score_battery(items_all, cases_all_stripped, args.backend, args.model,
                              args.output / "all_stripped", args.batch_size)

    # --- Show some stripping examples ---
    print("\n=== STRIPPING EXAMPLES ===")
    for case in cases[:3]:
        print(f"\n{case['id']}:")
        print(f"  Original better: {case['better'][:80]}...")
        print(f"  Eval-stripped:   {strip_evaluative(case['better'], EVALUATIVE_WORDS)[:80]}...")
        print(f"  All-stripped:    {strip_evaluative(case['better'], ALL_STRIP_WORDS)[:80]}...")

    # --- Report ---
    key_metrics = [
        "direct_combined", "direct_category_axis",
        "direct_harm_reduction", "direct_persona_honesty",
        "direct_anti_sycophancy", "direct_truthfulness",
        "direct_general_evaluative",
    ]
    print("\n" + "=" * 80)
    print("WORD-STRIPPING ABLATION RESULTS")
    print("=" * 80)
    print(f"{'Metric':<35s} {'Original':>10s} {'Eval strip':>10s} {'All strip':>10s}")
    print("-" * 65)
    for m in key_metrics:
        if m in accs_orig:
            print(f"{m:<35s} {accs_orig[m]:>9.1%} {accs_eval.get(m, 0):>9.1%} {accs_all.get(m, 0):>9.1%}")

    # --- Save results ---
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "battery": str(args.battery),
        "backend": args.backend,
        "model": args.model,
        "n_cases": len(cases),
        "n_evaluative_words": len(EVALUATIVE_WORDS),
        "n_anchor_words": len(ANCHOR_WORDS),
        "n_all_strip_words": len(ALL_STRIP_WORDS),
        "original": accs_orig,
        "eval_stripped": accs_eval,
        "all_stripped": accs_all,
    }
    (args.output / "ablation_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Word-Stripping Ablation",
        "",
        f"Backend: `{args.backend}`",
        f"Model: `{args.model or 'default'}`",
        f"Battery: `{args.battery.name}` ({len(cases)} cases)",
        f"Evaluative words stripped: {len(EVALUATIVE_WORDS)}",
        f"Anchor words stripped: {len(ANCHOR_WORDS)}",
        "",
        "## Results",
        "",
        f"| {'Metric':<35s} | {'Original':>10s} | {'Eval strip':>10s} | {'All strip':>10s} |",
        f"| {'-'*35} | {'-'*10}: | {'-'*10}: | {'-'*10}: |",
    ]
    for m in key_metrics:
        if m in accs_orig:
            lines.append(
                f"| {m:<35s} | {accs_orig[m]:>9.1%} | {accs_eval.get(m, 0):>9.1%} | {accs_all.get(m, 0):>9.1%} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "If accuracy drops sharply under stripping, the signal depends on",
        "evaluative word overlap rather than deeper semantic content.",
        "If accuracy survives, the embedding captures quality structure",
        "beyond surface-level evaluative vocabulary.",
        "",
        "Note: local OSS models are expected to be near chance even",
        "without stripping. This ablation is most informative when run",
        "with a capable embedding model (e.g., gemini-embedding-2).",
        "",
    ])
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
