#!/usr/bin/env python3
"""Test anchor format variations on the balanced battery.

Depends on: methodology/RESEARCH_CONTEXT.md and the original + warmth batteries.
Output: notes/research_cycles/anchor_formats/format_results.json
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
    ROOT / "notes" / "research_cycles" / "anchor_formats" / "format_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

FORMATS = {
    "careful": {
        "1_word": {"pos": ["Careful"], "neg": ["Reckless"]},
        "compound_2word": {"pos": ["Careful deliberate"], "neg": ["Reckless impulsive"]},
        "compound_3word": {
            "pos": ["Careful deliberate attentive"],
            "neg": ["Reckless impulsive negligent"],
        },
        "bare_clause": {"pos": ["is careful"], "neg": ["is reckless"]},
        "this_is": {"pos": ["This is careful"], "neg": ["This is reckless"]},
        "response_is": {
            "pos": ["The response is careful"],
            "neg": ["The response is reckless"],
        },
        "short_judgment": {
            "pos": ["Careful and considered"],
            "neg": ["Reckless and careless"],
        },
        "mixed_tree": {
            "pos": ["Careful, deliberate, and attentive"],
            "neg": ["Reckless, impulsive, and negligent"],
        },
    },
    "honest": {
        "1_word": {"pos": ["Honest"], "neg": ["Dishonest"]},
        "compound_2word": {"pos": ["Honest truthful"], "neg": ["Dishonest deceptive"]},
        "compound_3word": {
            "pos": ["Honest truthful sincere"],
            "neg": ["Dishonest deceptive misleading"],
        },
        "bare_clause": {"pos": ["is honest"], "neg": ["is dishonest"]},
        "this_is": {"pos": ["This is honest"], "neg": ["This is dishonest"]},
        "response_is": {
            "pos": ["The response is honest"],
            "neg": ["The response is dishonest"],
        },
        "short_judgment": {
            "pos": ["Honest and forthright"],
            "neg": ["Dishonest and evasive"],
        },
        "mixed_tree": {
            "pos": ["Honest, truthful, and sincere"],
            "neg": ["Dishonest, deceptive, and misleading"],
        },
    },
    "thorough": {
        "1_word": {"pos": ["Thorough"], "neg": ["Superficial"]},
        "compound_2word": {"pos": ["Thorough complete"], "neg": ["Superficial incomplete"]},
        "compound_3word": {
            "pos": ["Thorough complete detailed"],
            "neg": ["Superficial incomplete vague"],
        },
        "bare_clause": {"pos": ["is thorough"], "neg": ["is superficial"]},
        "this_is": {"pos": ["This is thorough"], "neg": ["This is superficial"]},
        "response_is": {
            "pos": ["The response is thorough"],
            "neg": ["The response is superficial"],
        },
        "short_judgment": {
            "pos": ["Thorough and complete"],
            "neg": ["Superficial and incomplete"],
        },
        "mixed_tree": {
            "pos": ["Thorough, complete, and detailed"],
            "neg": ["Superficial, incomplete, and vague"],
        },
    },
}

METHODS = ["bipolar_axis", "cosine_to_positive"]


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


def score_format(
    better_embs: np.ndarray,
    worse_embs: np.ndarray,
    positive_emb: np.ndarray,
    negative_emb: np.ndarray,
) -> dict:
    axis = normalize(positive_emb - negative_emb)
    return {
        "bipolar_axis": accuracy_from_scores(better_embs @ axis, worse_embs @ axis),
        "cosine_to_positive": accuracy_from_scores(
            cosine_to_anchor(better_embs, positive_emb),
            cosine_to_anchor(worse_embs, positive_emb),
        ),
    }


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

    print("Anchor Format Variations")
    print(f"Original cases: {len(original)}")
    print(f"Warmth cases:   {len(warmth)}")
    print(f"Combined cases: {len(splits['combined'])}")

    results = {
        "experiment": "Anchor Format Variations",
        "models": MODELS,
        "batteries": {
            "original": str(BATTERY_ORIGINAL),
            "warmth": str(BATTERY_WARMTH),
        },
        "formats": FORMATS,
        "methods": METHODS,
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
        for concept, format_specs in FORMATS.items():
            concept_results = {}
            print(f"\nCONCEPT: {concept}")
            for method_name in METHODS:
                print(f"\nMethod: {method_name}")
                print(f"{'format':<20} {'original':>10} {'warmth':>10} {'combined':>10}")
                print("-" * 56)

                for format_name, anchors in format_specs.items():
                    if format_name not in concept_results:
                        pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
                        neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
                        format_result = {"pos": anchors["pos"], "neg": anchors["neg"]}
                        for split_name, embs in split_embs.items():
                            format_result[split_name] = score_format(
                                embs["better"], embs["worse"], pos_emb, neg_emb
                            )
                        concept_results[format_name] = format_result

                    row = concept_results[format_name]
                    print(
                        f"{format_name:<20} "
                        f"{pct(row['original'][method_name]['accuracy']):>10} "
                        f"{pct(row['warmth'][method_name]['accuracy']):>10} "
                        f"{pct(row['combined'][method_name]['accuracy']):>10}"
                    )

            model_results[concept] = concept_results

        results["results"][model_name] = model_results
        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
