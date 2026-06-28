#!/usr/bin/env python3
"""Test the deep-research recommended terms and template format.

New candidates: complete/incomplete, moderate/excessive, effective/ineffective
New template: "This response is [X]" (midpoint between naked token and character projection)
Also test the full recommended shortlist as a composite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

# --- Naked single words ---
SINGLE_AXES = {
    "good_bad":           {"positive": ["Good"],        "negative": ["Bad"]},
    "true_false":         {"positive": ["True"],        "negative": ["False"]},
    "useful_useless":     {"positive": ["Useful"],      "negative": ["Useless"]},
    "honest_dishonest":   {"positive": ["Honest"],      "negative": ["Dishonest"]},
    "strong_weak":        {"positive": ["Strong"],      "negative": ["Weak"]},
    "complete_incomplete": {"positive": ["Complete"],   "negative": ["Incomplete"]},
    "kind_cruel":         {"positive": ["Kind"],        "negative": ["Cruel"]},
    "moderate_excessive": {"positive": ["Moderate"],    "negative": ["Excessive"]},
    "careful_reckless":   {"positive": ["Careful"],     "negative": ["Reckless"]},
    "clear_confused":     {"positive": ["Clear"],       "negative": ["Confused"]},
    "effective_ineffective": {"positive": ["Effective"], "negative": ["Ineffective"]},
}

# --- "This response is [X]" template ---
TEMPLATE_AXES = {
    "good_bad":           {"positive": ["This response is good"],
                           "negative": ["This response is bad"]},
    "true_false":         {"positive": ["This response is true"],
                           "negative": ["This response is false"]},
    "useful_useless":     {"positive": ["This response is useful"],
                           "negative": ["This response is useless"]},
    "honest_dishonest":   {"positive": ["This response is honest"],
                           "negative": ["This response is dishonest"]},
    "strong_weak":        {"positive": ["This response is strong"],
                           "negative": ["This response is weak"]},
    "complete_incomplete": {"positive": ["This response is complete"],
                            "negative": ["This response is incomplete"]},
    "kind_cruel":         {"positive": ["This response is kind"],
                           "negative": ["This response is cruel"]},
    "moderate_excessive": {"positive": ["This response is moderate"],
                           "negative": ["This response is excessive"]},
    "careful_reckless":   {"positive": ["This response is careful"],
                           "negative": ["This response is reckless"]},
    "clear_confused":     {"positive": ["This response is clear"],
                           "negative": ["This response is confused"]},
    "effective_ineffective": {"positive": ["This response is effective"],
                              "negative": ["This response is ineffective"]},
}

# --- Templated centroids (multiple phrases per pole) ---
CENTROID_AXES = {
    "core_6_positive": {
        "positive": [
            "This response is good",
            "This response is true",
            "This response is useful",
            "This response is honest",
            "This response is strong",
            "This response is complete",
        ],
        "negative": [
            "This response is bad",
            "This response is false",
            "This response is useless",
            "This response is dishonest",
            "This response is weak",
            "This response is incomplete",
        ],
    },
    "core_4_stripped": {
        "positive": [
            "This response is good",
            "This response is true",
            "This response is useful",
            "This response is honest",
        ],
        "negative": [
            "This response is bad",
            "This response is false",
            "This response is useless",
            "This response is dishonest",
        ],
    },
    "extended_8": {
        "positive": [
            "This response is good",
            "This response is true",
            "This response is useful",
            "This response is honest",
            "This response is strong",
            "This response is complete",
            "This response is kind",
            "This response is moderate",
        ],
        "negative": [
            "This response is bad",
            "This response is false",
            "This response is useless",
            "This response is dishonest",
            "This response is weak",
            "This response is incomplete",
            "This response is cruel",
            "This response is excessive",
        ],
    },
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def score_battery(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        s_better = float(np.dot(better_embs[i], axis))
        s_worse = float(np.dot(worse_embs[i], axis))
        if s_better > s_worse:
            correct += 1
        elif s_better == s_worse:
            correct += 0.5
    return correct / len(better_embs)


MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")
    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        # ML-jargon baseline (top 3 only for brevity)
        print("\n--- ML-jargon baseline ---")
        for axis_name, anchors in ML_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"ml/{axis_name}"] = round(acc, 4)
            print(f"  ml/{axis_name:30s}: {acc:.0%}")

        # Single words
        print("\n--- Single words ---")
        for axis_name, anchors in SINGLE_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"single/{axis_name}"] = round(acc, 4)
            print(f"  single/{axis_name:28s}: {acc:.0%}")

        # "This response is [X]" template
        print("\n--- Template: 'This response is [X]' ---")
        for axis_name, anchors in TEMPLATE_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"template/{axis_name}"] = round(acc, 4)
            print(f"  template/{axis_name:26s}: {acc:.0%}")

        # Templated centroids
        print("\n--- Templated centroids ---")
        for axis_name, anchors in CENTROID_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"centroid/{axis_name}"] = round(acc, 4)
            print(f"  centroid/{axis_name:25s}: {acc:.0%}")

        all_results[model_name] = model_results

        # Summary
        best_ml = max([(k,v) for k,v in model_results.items() if k.startswith("ml/")], key=lambda x: x[1])
        best_other = max([(k,v) for k,v in model_results.items() if not k.startswith("ml/")], key=lambda x: x[1])
        print(f"\n  Best ML:     {best_ml[0]:40s} = {best_ml[1]:.0%}")
        print(f"  Best other:  {best_other[0]:40s} = {best_other[1]:.0%}")

        del model

    out_dir = ROOT / "notes" / "research_cycles" / "deep_research_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
