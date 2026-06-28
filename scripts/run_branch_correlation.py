#!/usr/bin/env python3
"""E-08: compare same-branch and cross-branch term margin correlations.

Hypothesis: terms from the same branch of the good tree may have more similar
per-case margins than terms from different branches.
Depends on: methodology/EXPERIMENT_SPECS.md and the original + warmth batteries.
Output: notes/research_cycles/branch_correlation/branch_results.json
"""

from __future__ import annotations

import gc
import itertools
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
    ROOT / "notes" / "research_cycles" / "branch_correlation" / "branch_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

BRANCHES = {
    "firmness": ["careful", "thorough", "precise", "responsible", "honest"],
    "warmth": ["kind", "patient", "gentle", "encouraging", "supportive"],
    "competence": ["clear", "useful", "helpful", "wise", "constructive"],
}

ANCHORS = {
    "careful": {"positive": ["Careful"], "negative": ["Reckless"]},
    "thorough": {"positive": ["Thorough"], "negative": ["Superficial"]},
    "precise": {"positive": ["Precise"], "negative": ["Sloppy"]},
    "responsible": {"positive": ["Responsible"], "negative": ["Irresponsible"]},
    "honest": {"positive": ["Honest"], "negative": ["Dishonest"]},
    "kind": {"positive": ["Kind"], "negative": ["Cruel"]},
    "patient": {"positive": ["Patient"], "negative": ["Impatient"]},
    "gentle": {"positive": ["Gentle"], "negative": ["Harsh"]},
    "encouraging": {"positive": ["Encouraging"], "negative": ["Discouraging"]},
    "supportive": {"positive": ["Supportive"], "negative": ["Dismissive"]},
    "clear": {"positive": ["Clear"], "negative": ["Confusing"]},
    "useful": {"positive": ["Useful"], "negative": ["Useless"]},
    "helpful": {"positive": ["Helpful"], "negative": ["Unhelpful"]},
    "wise": {"positive": ["Wise"], "negative": ["Foolish"]},
    "constructive": {"positive": ["Constructive"], "negative": ["Destructive"]},
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


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def pct_corr(value: float | None) -> str:
    if value is None:
        return "  n/a"
    return f"{value:5.3f}"


def main() -> None:
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    splits = {
        "original": original,
        "warmth": warmth,
        "combined": original + warmth,
    }

    term_to_branch = {
        term: branch for branch, terms in BRANCHES.items() for term in terms
    }
    terms = [term for branch_terms in BRANCHES.values() for term in branch_terms]

    print("E-08 Same-Branch vs Cross-Branch Correlation")
    print(f"Original cases: {len(original)}")
    print(f"Warmth cases:   {len(warmth)}")
    print(f"Combined cases: {len(splits['combined'])}")

    results = {
        "experiment": "E-08 Same-Branch vs Cross-Branch Correlation",
        "models": MODELS,
        "batteries": {
            "original": str(BATTERY_ORIGINAL),
            "warmth": str(BATTERY_WARMTH),
        },
        "branches": BRANCHES,
        "anchors": ANCHORS,
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

        axes = {
            term: compute_axis(embed_fn, ANCHORS[term]["positive"], ANCHORS[term]["negative"])
            for term in terms
        }

        model_results = {}
        print(f"\n{'split':<12} {'within_mean':>12} {'cross_mean':>12} {'difference':>12}")
        print("-" * 52)

        for split_name, embs in split_embs.items():
            margins = {
                term: (embs["better"] @ axis) - (embs["worse"] @ axis)
                for term, axis in axes.items()
            }

            pair_results = []
            within_values = []
            cross_values = []
            for left, right in itertools.combinations(terms, 2):
                corr = correlation(margins[left], margins[right])
                same_branch = term_to_branch[left] == term_to_branch[right]
                pair = {
                    "left": left,
                    "right": right,
                    "left_branch": term_to_branch[left],
                    "right_branch": term_to_branch[right],
                    "same_branch": same_branch,
                    "correlation": corr,
                }
                pair_results.append(pair)
                if corr is not None:
                    if same_branch:
                        within_values.append(corr)
                    else:
                        cross_values.append(corr)

            within_summary = summarize(within_values)
            cross_summary = summarize(cross_values)
            difference = None
            if within_summary["mean"] is not None and cross_summary["mean"] is not None:
                difference = within_summary["mean"] - cross_summary["mean"]

            model_results[split_name] = {
                "term_margins": {
                    term: [float(x) for x in term_margins]
                    for term, term_margins in margins.items()
                },
                "pairwise_correlations": pair_results,
                "within_branch": within_summary,
                "cross_branch": cross_summary,
                "within_minus_cross_mean": difference,
            }

            print(
                f"{split_name:<12} "
                f"{pct_corr(within_summary['mean']):>12} "
                f"{pct_corr(cross_summary['mean']):>12} "
                f"{pct_corr(difference):>12}"
            )

        results["results"][model_name] = model_results
        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
