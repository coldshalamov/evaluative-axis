#!/usr/bin/env python3
"""Test composite axes from best-performing universal terms.

The vocabulary depth experiment found "careful", "noble", and "kind"
as the most cross-model robust universal terms. This script tests
whether combining them (mean of multiple single-word directions)
produces stronger axes than any individual term.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

# Best single-term pairs identified from vocabulary depth experiment
COMPOSITE_AXES = {
    "careful_only":    {"positive": ["Careful"],
                        "negative": ["Reckless"]},
    "noble_only":      {"positive": ["Noble"],
                        "negative": ["Base"]},
    "kind_only":       {"positive": ["Kind"],
                        "negative": ["Cruel"]},
    "brave_only":      {"positive": ["Brave"],
                        "negative": ["Cowardly"]},

    # Composites: averaging directions from multiple universal terms
    "careful+noble":   {"positive": ["Careful", "Noble"],
                        "negative": ["Reckless", "Base"]},
    "careful+noble+kind": {"positive": ["Careful", "Noble", "Kind"],
                           "negative": ["Reckless", "Base", "Cruel"]},
    "careful+noble+kind+brave": {"positive": ["Careful", "Noble", "Kind", "Brave"],
                                  "negative": ["Reckless", "Base", "Cruel", "Cowardly"]},
    "top5_universal":  {"positive": ["Careful", "Noble", "Kind", "Brave", "Wise"],
                        "negative": ["Reckless", "Base", "Cruel", "Cowardly", "Foolish"]},
    "top7_universal":  {"positive": ["Careful", "Noble", "Kind", "Brave", "Wise", "Just", "Helpful"],
                        "negative": ["Reckless", "Base", "Cruel", "Cowardly", "Foolish", "Unjust", "Harmful"]},

    # Character projection composites
    "char_careful+noble+kind": {
        "positive": ["A careful person said this", "A noble person said this", "A kind person said this"],
        "negative": ["A reckless person said this", "A base person said this", "A cruel person said this"],
    },

    # Hybrid: single words + character projection
    "hybrid_top5": {
        "positive": ["Careful", "Noble", "Kind", "A careful person said this", "A kind person said this"],
        "negative": ["Reckless", "Base", "Cruel", "A reckless person said this", "A cruel person said this"],
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

        # ML-jargon baseline
        print("\n--- ML-jargon baseline ---")
        for axis_name, anchors in ML_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"ml/{axis_name}"] = round(acc, 4)
            print(f"  ml/{axis_name:25s}: {acc:.0%}")

        # Composite universal axes
        print("\n--- Universal composites ---")
        for axis_name, anchors in COMPOSITE_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"universal/{axis_name}"] = round(acc, 4)
            print(f"  universal/{axis_name:30s}: {acc:.0%}")

        all_results[model_name] = model_results

        # Print model summary
        best_ml = max([(k,v) for k,v in model_results.items() if k.startswith("ml/")], key=lambda x: x[1])
        best_uni = max([(k,v) for k,v in model_results.items() if k.startswith("universal/")], key=lambda x: x[1])
        print(f"\n  Best ML:        {best_ml[0]:40s} = {best_ml[1]:.0%}")
        print(f"  Best universal: {best_uni[0]:40s} = {best_uni[1]:.0%}")

        del model

    # Save
    out_dir = ROOT / "notes" / "research_cycles" / "universal_composite_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composite_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
