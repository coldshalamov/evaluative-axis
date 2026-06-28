#!/usr/bin/env python3
"""Compare projection magnitudes and margins between diffuse and specific axes.

Theory: "good" has thousands of senses so projections will be smaller
and margins tighter. "careful" has fewer senses so projections should
be larger and margins wider. This tests whether "good" fails because
the direction is wrong or because the signal-to-noise is too low.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

ORIGINAL_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
WARMTH_BATTERY = (
    ROOT / "notes" / "research_cycles" / "battery_rebalancing"
    / "warmth_cases.jsonl"
)

TEST_AXES = {
    "good":     {"positive": ["Good"],     "negative": ["Bad"]},
    "careful":  {"positive": ["Careful"],   "negative": ["Reckless"]},
    "hard":     {"positive": ["Hard"],      "negative": ["Soft"]},
    "honest":   {"positive": ["Honest"],    "negative": ["Dishonest"]},
    "kind":     {"positive": ["Kind"],      "negative": ["Cruel"]},
    "thorough": {"positive": ["Thorough"],  "negative": ["Superficial"]},
}

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


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
    raw_norm = float(np.linalg.norm(axis))
    return axis / (raw_norm + 1e-12), raw_norm


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(ORIGINAL_BATTERY)
    warmth = read_jsonl(WARMTH_BATTERY)

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Original battery
        orig_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in original])
        orig_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in original])

        # Warmth battery
        warm_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in warmth])
        warm_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in warmth])

        for axis_name, anchors in TEST_AXES.items():
            axis_vec, raw_norm = compute_axis(embed_fn, anchors["positive"], anchors["negative"])

            # Original battery scores
            orig_better_scores = [float(np.dot(orig_better[i], axis_vec)) for i in range(len(original))]
            orig_worse_scores = [float(np.dot(orig_worse[i], axis_vec)) for i in range(len(original))]
            orig_margins = [b - w for b, w in zip(orig_better_scores, orig_worse_scores)]

            # Warmth battery scores
            warm_better_scores = [float(np.dot(warm_better[i], axis_vec)) for i in range(len(warmth))]
            warm_worse_scores = [float(np.dot(warm_worse[i], axis_vec)) for i in range(len(warmth))]
            warm_margins = [b - w for b, w in zip(warm_better_scores, warm_worse_scores)]

            # Pairwise accuracy
            orig_acc = sum(1 for m in orig_margins if m > 0) / len(orig_margins)
            warm_acc = sum(1 for m in warm_margins if m > 0) / len(warm_margins)

            print(f"\n--- {axis_name} ---")
            print(f"  Axis vector norm (before normalization): {raw_norm:.4f}")
            print(f"  ORIGINAL (50 firmness-heavy cases):")
            print(f"    Pairwise accuracy: {orig_acc:.0%}")
            print(f"    Better scores: mean={np.mean(orig_better_scores):.4f}, std={np.std(orig_better_scores):.4f}")
            print(f"    Worse scores:  mean={np.mean(orig_worse_scores):.4f}, std={np.std(orig_worse_scores):.4f}")
            print(f"    Margins (better-worse): mean={np.mean(orig_margins):.5f}, std={np.std(orig_margins):.5f}")
            print(f"    Margin sign: {sum(1 for m in orig_margins if m>0)} positive, {sum(1 for m in orig_margins if m<0)} negative, {sum(1 for m in orig_margins if m==0)} zero")

            print(f"  WARMTH (20 warmth cases):")
            print(f"    Pairwise accuracy: {warm_acc:.0%}")
            print(f"    Better scores: mean={np.mean(warm_better_scores):.4f}, std={np.std(warm_better_scores):.4f}")
            print(f"    Worse scores:  mean={np.mean(warm_worse_scores):.4f}, std={np.std(warm_worse_scores):.4f}")
            print(f"    Margins (better-worse): mean={np.mean(warm_margins):.5f}, std={np.std(warm_margins):.5f}")
            print(f"    Margin sign: {sum(1 for m in warm_margins if m>0)} positive, {sum(1 for m in warm_margins if m<0)} negative, {sum(1 for m in warm_margins if m==0)} zero")

        # Also show ML-jargon for comparison
        for ml_name in ["anti_sycophancy", "general_evaluative"]:
            anchors = ML_AXES[ml_name]
            axis_vec, raw_norm = compute_axis(embed_fn, anchors["positive"], anchors["negative"])

            orig_better_scores = [float(np.dot(orig_better[i], axis_vec)) for i in range(len(original))]
            orig_worse_scores = [float(np.dot(orig_worse[i], axis_vec)) for i in range(len(original))]
            orig_margins = [b - w for b, w in zip(orig_better_scores, orig_worse_scores)]

            warm_better_scores = [float(np.dot(warm_better[i], axis_vec)) for i in range(len(warmth))]
            warm_worse_scores = [float(np.dot(warm_worse[i], axis_vec)) for i in range(len(warmth))]
            warm_margins = [b - w for b, w in zip(warm_better_scores, warm_worse_scores)]

            orig_acc = sum(1 for m in orig_margins if m > 0) / len(orig_margins)
            warm_acc = sum(1 for m in warm_margins if m > 0) / len(warm_margins)

            print(f"\n--- ML: {ml_name} ---")
            print(f"  Axis vector norm (before normalization): {raw_norm:.4f}")
            print(f"  ORIGINAL: acc={orig_acc:.0%}, margin mean={np.mean(orig_margins):.5f}, std={np.std(orig_margins):.5f}")
            print(f"  WARMTH:   acc={warm_acc:.0%}, margin mean={np.mean(warm_margins):.5f}, std={np.std(warm_margins):.5f}")

        del model


if __name__ == "__main__":
    main()
