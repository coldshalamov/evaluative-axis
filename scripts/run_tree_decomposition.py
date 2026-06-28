#!/usr/bin/env python3
"""E-01: score the good/bad tree on the balanced battery.

Hypothesis: child terms of "good" may score better than the broad root.
Depends on: methodology/EXPERIMENT_SPECS.md and the original + warmth batteries.
Output: notes/research_cycles/tree_decomposition/tree_results.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BATTERY_ORIGINAL = (
    ROOT
    / "notes"
    / "research_cycles"
    / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BATTERY_WARMTH = (
    ROOT
    / "notes"
    / "research_cycles"
    / "battery_rebalancing"
    / "warmth_cases.jsonl"
)
OUTPUT_PATH = (
    ROOT / "notes" / "research_cycles" / "tree_decomposition" / "tree_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TREE_LEVELS = {
    "level_0_root": {
        "good": {"positive": ["Good"], "negative": ["Bad"], "parent": None},
    },
    "level_1_primary_children": {
        "careful": {"positive": ["Careful"], "negative": ["Reckless"], "parent": "good"},
        "honest": {"positive": ["Honest"], "negative": ["Dishonest"], "parent": "good"},
        "kind": {"positive": ["Kind"], "negative": ["Cruel"], "parent": "good"},
        "wise": {"positive": ["Wise"], "negative": ["Foolish"], "parent": "good"},
        "helpful": {"positive": ["Helpful"], "negative": ["Unhelpful"], "parent": "good"},
        "thorough": {"positive": ["Thorough"], "negative": ["Superficial"], "parent": "good"},
        "fair": {"positive": ["Fair"], "negative": ["Unfair"], "parent": "good"},
        "responsible": {
            "positive": ["Responsible"],
            "negative": ["Irresponsible"],
            "parent": "good",
        },
        "clear": {"positive": ["Clear"], "negative": ["Confusing"], "parent": "good"},
        "respectful": {
            "positive": ["Respectful"],
            "negative": ["Disrespectful"],
            "parent": "good",
        },
    },
    "level_2_grandchildren": {
        "deliberate": {
            "positive": ["Deliberate"],
            "negative": ["Impulsive"],
            "parent": "careful",
        },
        "attentive": {
            "positive": ["Attentive"],
            "negative": ["Inattentive"],
            "parent": "careful",
        },
        "precise": {"positive": ["Precise"], "negative": ["Sloppy"], "parent": "careful"},
        "cautious": {
            "positive": ["Cautious"],
            "negative": ["Careless"],
            "parent": "careful",
        },
        "methodical": {
            "positive": ["Methodical"],
            "negative": ["Haphazard"],
            "parent": "careful",
        },
        "truthful": {
            "positive": ["Truthful"],
            "negative": ["Deceptive"],
            "parent": "honest",
        },
        "transparent": {
            "positive": ["Transparent"],
            "negative": ["Opaque"],
            "parent": "honest",
        },
        "sincere": {
            "positive": ["Sincere"],
            "negative": ["Insincere"],
            "parent": "honest",
        },
        "forthright": {
            "positive": ["Forthright"],
            "negative": ["Evasive"],
            "parent": "honest",
        },
        "candid": {"positive": ["Candid"], "negative": ["Misleading"], "parent": "honest"},
        "compassionate": {
            "positive": ["Compassionate"],
            "negative": ["Indifferent"],
            "parent": "kind",
        },
        "patient": {"positive": ["Patient"], "negative": ["Impatient"], "parent": "kind"},
        "gentle": {"positive": ["Gentle"], "negative": ["Harsh"], "parent": "kind"},
        "encouraging": {
            "positive": ["Encouraging"],
            "negative": ["Discouraging"],
            "parent": "kind",
        },
        "supportive": {
            "positive": ["Supportive"],
            "negative": ["Dismissive"],
            "parent": "kind",
        },
    },
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def framed(case: dict, key: str) -> str:
    return f"User: {case['prompt']}\nAssistant: {case[key]}"


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def compute_axis(embed_fn, positive: list[str], negative: list[str]) -> np.ndarray:
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    return normalize(pos_embs.mean(axis=0) - neg_embs.mean(axis=0))


def accuracy_from_margins(margins: np.ndarray) -> dict:
    correct = float(np.sum(margins > 0))
    ties = float(np.sum(margins == 0))
    total = int(len(margins))
    return {
        "accuracy": (correct + 0.5 * ties) / total if total else 0.0,
        "correct": int(correct),
        "ties": int(ties),
        "total": total,
    }


def score_axis(better_embs: np.ndarray, worse_embs: np.ndarray, axis: np.ndarray) -> dict:
    better_scores = better_embs @ axis
    worse_scores = worse_embs @ axis
    margins = better_scores - worse_scores
    stats = accuracy_from_margins(margins)
    stats["mean_margin"] = float(np.mean(margins)) if len(margins) else 0.0
    stats["margins"] = [float(x) for x in margins]
    return stats


def aggregate_sum(axis_results: dict[str, dict], split: str) -> dict:
    all_margins = np.array([result[split]["margins"] for result in axis_results.values()])
    summed = all_margins.sum(axis=0)
    stats = accuracy_from_margins(summed)
    stats["mean_margin"] = float(np.mean(summed)) if len(summed) else 0.0
    stats["margins"] = [float(x) for x in summed]
    return stats


def aggregate_vote(axis_results: dict[str, dict], split: str) -> dict:
    all_margins = np.array([result[split]["margins"] for result in axis_results.values()])
    votes = np.sign(all_margins).sum(axis=0)
    stats = accuracy_from_margins(votes)
    stats["vote_margins"] = [float(x) for x in votes]
    return stats


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def main() -> None:
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    splits = {
        "original": original,
        "warmth": warmth,
        "combined": original + warmth,
    }

    print("E-01 Tree Decomposition of Good")
    print(f"Original cases: {len(original)}")
    print(f"Warmth cases:   {len(warmth)}")
    print(f"Combined cases: {len(splits['combined'])}")

    results = {
        "experiment": "E-01 Tree Decomposition of Good",
        "models": MODELS,
        "batteries": {
            "original": str(BATTERY_ORIGINAL),
            "warmth": str(BATTERY_WARMTH),
        },
        "tree_levels": TREE_LEVELS,
        "results": {},
    }

    for model_name in MODELS:
        print("\n" + "=" * 100)
        print(f"MODEL: {model_name}")
        print("=" * 100)

        model = SentenceTransformer(model_name, trust_remote_code=True)

        def embed_fn(texts: list[str]) -> np.ndarray:
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        split_embs = {}
        for split_name, rows in splits.items():
            split_embs[split_name] = {
                "better": embed_fn([framed(case, "better") for case in rows]),
                "worse": embed_fn([framed(case, "worse") for case in rows]),
            }

        model_results = {}
        for level_name, terms in TREE_LEVELS.items():
            level_results = {"terms": {}, "aggregates": {}}
            for term_name, anchors in terms.items():
                axis = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
                term_result = {
                    "positive": anchors["positive"],
                    "negative": anchors["negative"],
                    "parent": anchors["parent"],
                }
                for split_name, embs in split_embs.items():
                    term_result[split_name] = score_axis(embs["better"], embs["worse"], axis)
                level_results["terms"][term_name] = term_result

            for split_name in splits:
                level_results["aggregates"].setdefault("sum_of_projections", {})[
                    split_name
                ] = aggregate_sum(level_results["terms"], split_name)
                level_results["aggregates"].setdefault("majority_vote", {})[
                    split_name
                ] = aggregate_vote(level_results["terms"], split_name)

            model_results[level_name] = level_results

            print(f"\n{level_name}")
            print(f"{'term':<18} {'original':>10} {'warmth':>10} {'combined':>10}")
            print("-" * 54)
            for term_name, term_result in level_results["terms"].items():
                print(
                    f"{term_name:<18} "
                    f"{pct(term_result['original']['accuracy']):>10} "
                    f"{pct(term_result['warmth']['accuracy']):>10} "
                    f"{pct(term_result['combined']['accuracy']):>10}"
                )
            for aggregate_name, aggregate_result in level_results["aggregates"].items():
                print(
                    f"{aggregate_name:<18} "
                    f"{pct(aggregate_result['original']['accuracy']):>10} "
                    f"{pct(aggregate_result['warmth']['accuracy']):>10} "
                    f"{pct(aggregate_result['combined']['accuracy']):>10}"
                )

        results["results"][model_name] = model_results
        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
