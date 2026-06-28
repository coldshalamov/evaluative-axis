#!/usr/bin/env python3
"""E-04: compare bipolar and single-pole scoring methods.

Hypothesis: different projection/cosine formulas may rank responses differently.
Depends on: methodology/EXPERIMENT_SPECS.md and the original + warmth batteries.
Output: notes/research_cycles/scoring_methods/scoring_results.json
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
    ROOT / "notes" / "research_cycles" / "scoring_methods" / "scoring_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = {
    "careful_reckless": {"positive": "Careful", "negative": "Reckless"},
    "honest_dishonest": {"positive": "Honest", "negative": "Dishonest"},
    "kind_cruel": {"positive": "Kind", "negative": "Cruel"},
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


def cosine_to_anchor(response_embs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    anchor_norm = np.linalg.norm(anchor) + 1e-12
    response_norms = np.linalg.norm(response_embs, axis=1) + 1e-12
    return (response_embs @ anchor) / (response_norms * anchor_norm)


def accuracy_from_scores(better_scores: np.ndarray, worse_scores: np.ndarray) -> dict:
    margins = better_scores - worse_scores
    correct = float(np.sum(margins > 0))
    ties = float(np.sum(margins == 0))
    total = int(len(margins))
    return {
        "accuracy": (correct + 0.5 * ties) / total if total else 0.0,
        "correct": int(correct),
        "ties": int(ties),
        "total": total,
        "mean_margin": float(np.mean(margins)) if total else 0.0,
        "margins": [float(x) for x in margins],
    }


def score_methods(
    better_embs: np.ndarray,
    worse_embs: np.ndarray,
    positive_emb: np.ndarray,
    negative_emb: np.ndarray,
) -> dict:
    bipolar_axis = normalize(positive_emb - negative_emb)
    raw_axis = positive_emb - negative_emb

    methods = {}
    methods["bipolar_axis"] = accuracy_from_scores(
        better_embs @ bipolar_axis,
        worse_embs @ bipolar_axis,
    )
    methods["cosine_to_positive"] = accuracy_from_scores(
        cosine_to_anchor(better_embs, positive_emb),
        cosine_to_anchor(worse_embs, positive_emb),
    )
    methods["cosine_to_negative_inverted"] = accuracy_from_scores(
        -cosine_to_anchor(better_embs, negative_emb),
        -cosine_to_anchor(worse_embs, negative_emb),
    )
    methods["difference_of_cosines"] = accuracy_from_scores(
        cosine_to_anchor(better_embs, positive_emb)
        - cosine_to_anchor(better_embs, negative_emb),
        cosine_to_anchor(worse_embs, positive_emb)
        - cosine_to_anchor(worse_embs, negative_emb),
    )
    methods["unnormalized_projection"] = accuracy_from_scores(
        better_embs @ raw_axis,
        worse_embs @ raw_axis,
    )
    return methods


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

    print("E-04 Bipolar vs Single-Pole Scoring")
    print(f"Original cases: {len(original)}")
    print(f"Warmth cases:   {len(warmth)}")
    print(f"Combined cases: {len(splits['combined'])}")

    results = {
        "experiment": "E-04 Bipolar vs Single-Pole Scoring",
        "models": MODELS,
        "batteries": {
            "original": str(BATTERY_ORIGINAL),
            "warmth": str(BATTERY_WARMTH),
        },
        "axes": AXES,
        "methods": [
            "bipolar_axis",
            "cosine_to_positive",
            "cosine_to_negative_inverted",
            "difference_of_cosines",
            "unnormalized_projection",
        ],
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
        for axis_name, anchors in AXES.items():
            positive_emb = embed_fn([anchors["positive"]])[0]
            negative_emb = embed_fn([anchors["negative"]])[0]
            axis_results = {}

            for split_name, embs in split_embs.items():
                axis_results[split_name] = score_methods(
                    embs["better"], embs["worse"], positive_emb, negative_emb
                )

            model_results[axis_name] = axis_results

            print(f"\n{axis_name}: {anchors['positive']} / {anchors['negative']}")
            print(f"{'method':<32} {'original':>10} {'warmth':>10} {'combined':>10}")
            print("-" * 66)
            for method_name in results["methods"]:
                print(
                    f"{method_name:<32} "
                    f"{pct(axis_results['original'][method_name]['accuracy']):>10} "
                    f"{pct(axis_results['warmth'][method_name]['accuracy']):>10} "
                    f"{pct(axis_results['combined'][method_name]['accuracy']):>10}"
                )

        results["results"][model_name] = model_results
        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
