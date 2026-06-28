#!/usr/bin/env python3
"""Search for the optimal small basis set of evaluative axes.

Tests every combination of 1, 2, 3, and 4 axes from our best-performing
single-word pairs. For each combination, scores via sum-of-projections
and majority vote. Evaluates on:
  - Overall accuracy
  - Worst-case category accuracy (the weakest category)
  - Cross-model consistency (min accuracy across models)
  - Per-category coverage map

The goal: find the smallest set of terms that together represent
"good" completely enough to work as a universal evaluative signal.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

# Candidate axes — our best performers plus a few more for coverage
CANDIDATE_AXES = {
    "careful":   {"positive": ["Careful"],   "negative": ["Reckless"]},
    "hard":      {"positive": ["Hard"],      "negative": ["Soft"]},
    "moderate":  {"positive": ["Moderate"],  "negative": ["Excessive"]},
    "heavy":     {"positive": ["Heavy"],     "negative": ["Light"]},
    "noble":     {"positive": ["Noble"],     "negative": ["Base"]},
    "wise":      {"positive": ["Wise"],      "negative": ["Foolish"]},
    "honest":    {"positive": ["Honest"],    "negative": ["Dishonest"]},
    "precise":   {"positive": ["Precise"],   "negative": ["Sloppy"]},
    "kind":      {"positive": ["Kind"],      "negative": ["Cruel"]},
    "clear":     {"positive": ["Clear"],     "negative": ["Confused"]},
    "useful":    {"positive": ["Useful"],    "negative": ["Useless"]},
    "good":      {"positive": ["Good"],      "negative": ["Bad"]},
    "active":    {"positive": ["Active"],    "negative": ["Passive"]},
    "fair":      {"positive": ["Fair"],      "negative": ["Unfair"]},
    "sound":     {"positive": ["Sound"],     "negative": ["Unsound"]},
    "thorough":  {"positive": ["Thorough"],  "negative": ["Superficial"]},
    "responsible": {"positive": ["Responsible"], "negative": ["Irresponsible"]},
    "constructive": {"positive": ["Constructive"], "negative": ["Destructive"]},
}

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

CATEGORIES = [
    "anti_sycophancy", "persona_honesty", "harm_reduction",
    "truthfulness", "reasoning_rigor", "helpfulness",
    "context_binding", "mixed"
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


def score_cases_sum(better_embs, worse_embs, axes_list):
    """Score using sum of projections. Returns per-case 0/1 array."""
    results = []
    for i in range(len(better_embs)):
        s_better = sum(float(np.dot(better_embs[i], ax)) for ax in axes_list)
        s_worse = sum(float(np.dot(worse_embs[i], ax)) for ax in axes_list)
        results.append(1 if s_better > s_worse else (0.5 if s_better == s_worse else 0))
    return np.array(results)


def score_cases_vote(better_embs, worse_embs, axes_list):
    """Score using majority vote across axes. Returns per-case 0/1 array."""
    results = []
    for i in range(len(better_embs)):
        votes = 0
        for ax in axes_list:
            s_b = float(np.dot(better_embs[i], ax))
            s_w = float(np.dot(worse_embs[i], ax))
            if s_b > s_w:
                votes += 1
            elif s_b == s_w:
                votes += 0.5
        results.append(1 if votes > len(axes_list) / 2 else
                      (0.5 if votes == len(axes_list) / 2 else 0))
    return np.array(results)


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")

    cat_indices = {}
    for i, c in enumerate(cases):
        cat = c.get("category", "unknown")
        cat_indices.setdefault(cat, []).append(i)

    axis_names = list(CANDIDATE_AXES.keys())
    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        # Pre-compute all axis vectors
        axis_vecs = {}
        for name, anchors in CANDIDATE_AXES.items():
            axis_vecs[name] = compute_axis(embed_fn, anchors["positive"], anchors["negative"])

        # Score each individual axis
        print("\n--- Individual axes ---")
        individual_scores = {}
        for name in axis_names:
            per_case = score_cases_sum(better_embs, worse_embs, [axis_vecs[name]])
            acc = per_case.mean()
            individual_scores[name] = acc

            # Per-category
            cat_accs = {}
            for cat, idx in cat_indices.items():
                cat_accs[cat] = per_case[idx].mean()
            worst_cat = min(cat_accs.values())
            best_cat = max(cat_accs.values())
            print(f"  {name:<15s}: {acc:.0%}  worst_cat={worst_cat:.0%}  best_cat={best_cat:.0%}")

        # Test combinations of size 2, 3, 4 using sum-of-projections
        # Only test combinations of the top performers to keep it manageable
        top_axes = sorted(individual_scores, key=individual_scores.get, reverse=True)[:10]
        print(f"\nTop 10 individual: {top_axes}")

        model_combos = {}

        for size in [2, 3, 4]:
            print(f"\n--- Best combos of size {size} (sum-of-projections) ---")
            best_combos = []

            for combo in combinations(top_axes, size):
                axes_list = [axis_vecs[name] for name in combo]
                per_case = score_cases_sum(better_embs, worse_embs, axes_list)
                acc = per_case.mean()

                cat_accs = {}
                for cat, idx in cat_indices.items():
                    cat_accs[cat] = per_case[idx].mean()
                worst_cat = min(cat_accs.values())
                min_cat_name = min(cat_accs, key=cat_accs.get)

                best_combos.append({
                    "combo": combo,
                    "accuracy": round(acc, 4),
                    "worst_category": round(worst_cat, 4),
                    "worst_category_name": min_cat_name,
                    "per_category": {k: round(v, 4) for k, v in cat_accs.items()},
                })

            # Sort by accuracy, then by worst-category
            best_combos.sort(key=lambda x: (x["accuracy"], x["worst_category"]), reverse=True)

            for rank, combo_info in enumerate(best_combos[:5]):
                combo_str = "+".join(combo_info["combo"])
                print(f"  #{rank+1}: {combo_str:<45s} acc={combo_info['accuracy']:.0%}  "
                      f"worst={combo_info['worst_category']:.0%} ({combo_info['worst_category_name']})")
                # Show per-category
                cats_str = "  ".join(f"{k[:6]}={v:.0%}" for k, v in
                                    sorted(combo_info['per_category'].items()))
                print(f"       {cats_str}")

            model_combos[size] = best_combos[:10]

        # Also test majority vote for best combos
        print(f"\n--- Majority vote for top combos ---")
        for size in [3]:
            for combo_info in model_combos[size][:3]:
                combo = combo_info["combo"]
                axes_list = [axis_vecs[name] for name in combo]
                per_case_vote = score_cases_vote(better_embs, worse_embs, axes_list)
                acc_vote = per_case_vote.mean()

                per_case_sum = score_cases_sum(better_embs, worse_embs, axes_list)
                acc_sum = per_case_sum.mean()

                combo_str = "+".join(combo)
                print(f"  {combo_str:<45s} sum={acc_sum:.0%}  vote={acc_vote:.0%}")

        all_results[model_name] = {
            "individual": {name: round(individual_scores[name], 4) for name in axis_names},
            "top_axes": top_axes,
            "combos": {str(k): [
                {**c, "combo": list(c["combo"])} for c in v
            ] for k, v in model_combos.items()},
        }
        del model

    # Cross-model analysis: find combos that work best across all models
    print(f"\n{'='*70}")
    print("CROSS-MODEL ANALYSIS")
    print(f"{'='*70}")

    # Individual cross-model consistency
    print("\n--- Individual axes: cross-model min accuracy ---")
    cross_model_individual = {}
    for name in axis_names:
        accs = []
        for m in MODELS:
            accs.append(all_results[m]["individual"][name])
        min_acc = min(accs)
        mean_acc = sum(accs) / len(accs)
        cross_model_individual[name] = {"min": min_acc, "mean": mean_acc, "all": accs}

    sorted_by_min = sorted(cross_model_individual.items(), key=lambda x: x[1]["min"], reverse=True)
    for name, info in sorted_by_min[:10]:
        accs_str = ", ".join(f"{a:.0%}" for a in info["all"])
        print(f"  {name:<15s}: min={info['min']:.0%}  mean={info['mean']:.0%}  [{accs_str}]")

    out_dir = ROOT / "notes" / "research_cycles" / "optimal_basis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    out_path = out_dir / "optimal_basis_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Save cross-model summary
    summary = {
        "cross_model_individual": {
            name: {"min": info["min"], "mean": info["mean"],
                   "models": dict(zip(MODELS, info["all"]))}
            for name, info in sorted_by_min
        }
    }
    (out_dir / "cross_model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
