#!/usr/bin/env python3
"""Test whether axes survive on a rebalanced battery.

The original 50-case battery is ~64% firmness-biased (better = pushback).
We add 20 warmth cases (better = warm/accommodating/patient) and retest.

If "Hard/Soft" collapses on the warmth cases but "Careful/Reckless"
survives, that confirms Hard was a firmness artifact while Careful
captures something real. If both collapse, the battery was driving
all our results.
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
    "careful":      {"positive": ["Careful"],      "negative": ["Reckless"]},
    "hard":         {"positive": ["Hard"],         "negative": ["Soft"]},
    "heavy":        {"positive": ["Heavy"],        "negative": ["Light"]},
    "moderate":     {"positive": ["Moderate"],     "negative": ["Excessive"]},
    "good":         {"positive": ["Good"],         "negative": ["Bad"]},
    "thorough":     {"positive": ["Thorough"],     "negative": ["Superficial"]},
    "noble":        {"positive": ["Noble"],        "negative": ["Base"]},
    "kind":         {"positive": ["Kind"],         "negative": ["Cruel"]},
    "honest":       {"positive": ["Honest"],       "negative": ["Dishonest"]},
    "wise":         {"positive": ["Wise"],         "negative": ["Foolish"]},
    "responsible":  {"positive": ["Responsible"],  "negative": ["Irresponsible"]},
    "constructive": {"positive": ["Constructive"], "negative": ["Destructive"]},
    "fair":         {"positive": ["Fair"],         "negative": ["Unfair"]},
    "gentle":       {"positive": ["Gentle"],       "negative": ["Harsh"]},
    "patient":      {"positive": ["Patient"],      "negative": ["Impatient"]},
    "supportive":   {"positive": ["Supportive"],   "negative": ["Dismissive"]},
    "respectful":   {"positive": ["Respectful"],   "negative": ["Contemptuous"]},
    "warm":         {"positive": ["Warm"],         "negative": ["Cold"]},
    "encouraging":  {"positive": ["Encouraging"],  "negative": ["Discouraging"]},
    "compassionate": {"positive": ["Compassionate"], "negative": ["Indifferent"]},
    "thoughtful":   {"positive": ["Thoughtful"],   "negative": ["Thoughtless"]},
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
    return axis / (np.linalg.norm(axis) + 1e-12)


def score_split(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        s_b = float(np.dot(better_embs[i], axis))
        s_w = float(np.dot(worse_embs[i], axis))
        if s_b > s_w:
            correct += 1
        elif s_b == s_w:
            correct += 0.5
    return correct / len(better_embs) if len(better_embs) > 0 else 0


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(ORIGINAL_BATTERY)
    warmth = read_jsonl(WARMTH_BATTERY)
    combined = original + warmth

    print(f"Original battery: {len(original)} cases (firmness-biased)")
    print(f"Warmth cases: {len(warmth)} cases (warmth-biased)")
    print(f"Combined: {len(combined)} cases")

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed all three splits
        orig_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in original])
        orig_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in original])
        warm_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in warmth])
        warm_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in warmth])
        comb_better = np.vstack([orig_better, warm_better])
        comb_worse = np.vstack([orig_worse, warm_worse])

        model_results = {}

        # ML-jargon axes
        print(f"\n{'Axis':<25s} {'Original':>10s} {'Warmth':>10s} {'Combined':>10s} {'Delta':>8s}")
        print("-" * 70)

        print("--- ML-jargon axes ---")
        for name, anchors in ML_AXES.items():
            ax = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            orig_acc = score_split(orig_better, orig_worse, ax)
            warm_acc = score_split(warm_better, warm_worse, ax)
            comb_acc = score_split(comb_better, comb_worse, ax)
            delta = warm_acc - orig_acc
            marker = " !!!" if abs(delta) > 0.20 else (" **" if abs(delta) > 0.10 else "")
            print(f"  ml/{name:<21s} {orig_acc:>9.0%} {warm_acc:>9.0%} {comb_acc:>9.0%} {delta:>+7.0%}{marker}")
            model_results[f"ml/{name}"] = {
                "original": round(orig_acc, 4),
                "warmth": round(warm_acc, 4),
                "combined": round(comb_acc, 4),
            }

        print("\n--- Single-word axes ---")
        for name, anchors in TEST_AXES.items():
            ax = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            orig_acc = score_split(orig_better, orig_worse, ax)
            warm_acc = score_split(warm_better, warm_worse, ax)
            comb_acc = score_split(comb_better, comb_worse, ax)
            delta = warm_acc - orig_acc
            marker = " !!!" if abs(delta) > 0.20 else (" **" if abs(delta) > 0.10 else "")
            print(f"  {name:<23s} {orig_acc:>9.0%} {warm_acc:>9.0%} {comb_acc:>9.0%} {delta:>+7.0%}{marker}")
            model_results[name] = {
                "original": round(orig_acc, 4),
                "warmth": round(warm_acc, 4),
                "combined": round(comb_acc, 4),
            }

        all_results[model_name] = model_results
        del model

    # Summary: which axes survive rebalancing?
    print(f"\n{'='*80}")
    print("SURVIVAL ANALYSIS: axes that score >50% on BOTH splits")
    print(f"{'='*80}")

    all_axes = list(TEST_AXES.keys())
    for name in all_axes:
        orig_accs = [all_results[m][name]["original"] for m in MODELS]
        warm_accs = [all_results[m][name]["warmth"] for m in MODELS]
        comb_accs = [all_results[m][name]["combined"] for m in MODELS]

        orig_min = min(orig_accs)
        warm_min = min(warm_accs)
        comb_min = min(comb_accs)

        survives = orig_min > 0.50 and warm_min > 0.50
        marker = " SURVIVES" if survives else ""
        orig_str = "/".join(f"{a:.0%}" for a in orig_accs)
        warm_str = "/".join(f"{a:.0%}" for a in warm_accs)
        comb_str = "/".join(f"{a:.0%}" for a in comb_accs)
        print(f"  {name:<20s} orig=[{orig_str}]  warm=[{warm_str}]  comb=[{comb_str}]{marker}")

    out_dir = ROOT / "notes" / "research_cycles" / "battery_rebalancing"
    out_path = out_dir / "rebalance_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
