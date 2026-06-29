#!/usr/bin/env python3
"""Test: do per-category centroids beat a single global centroid?

The global centroid is a compromise across quality dimensions.
If we train separate centroids per category and combine them,
does the ensemble outperform the single direction?

Also tests: what if you split by quality dimension rather than category?
"""

import json, sys, io
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def make_centroid(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    n = norm(d)
    return d / (n + 1e-12) if n > 0 else d


def pairwise_accuracy(better_embs, worse_embs, direction):
    return float((np.dot(better_embs, direction) > np.dot(worse_embs, direction)).mean())


def main():
    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    # Group by category
    bat_by_cat = defaultdict(list)
    for i, c in enumerate(battery):
        cat = c.get("category", "unknown")
        bat_by_cat[cat].append(i)

    exp_by_cat = defaultdict(list)
    for i, c in enumerate(expansion):
        cat = c.get("category", "unknown")
        exp_by_cat[cat].append(i)

    print(f"Battery: {len(battery)} cases across {len(bat_by_cat)} categories")
    for cat, idxs in sorted(bat_by_cat.items()):
        print(f"  {cat}: {len(idxs)} train, {len(exp_by_cat.get(cat, []))} test")
    print(f"Expansion: {len(expansion)} cases\n")

    from sentence_transformers import SentenceTransformer

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        bat_b = model.encode([c["better"] for c in battery], batch_size=32,
                              show_progress_bar=False, convert_to_numpy=True)
        bat_w = model.encode([c["worse"] for c in battery], batch_size=32,
                              show_progress_bar=False, convert_to_numpy=True)
        exp_b = model.encode([c["better"] for c in expansion], batch_size=32,
                              show_progress_bar=False, convert_to_numpy=True)
        exp_w = model.encode([c["worse"] for c in expansion], batch_size=32,
                              show_progress_bar=False, convert_to_numpy=True)

        # 1. Global centroid
        global_dir = make_centroid(bat_b, bat_w)
        global_acc = pairwise_accuracy(exp_b, exp_w, global_dir)
        print(f"\n  Global centroid OOS: {global_acc:.1%} ({len(expansion)} cases)")

        # Per-category breakdown of global centroid
        print(f"\n  Global centroid per-category:")
        for cat in sorted(exp_by_cat.keys()):
            idxs = exp_by_cat[cat]
            acc = pairwise_accuracy(exp_b[idxs], exp_w[idxs], global_dir)
            print(f"    {cat}: {acc:.1%} ({len(idxs)} cases)")

        # 2. Per-category centroids tested on their own category
        print(f"\n  Per-category centroids (trained & tested within category):")
        cat_centroids = {}
        cat_accs = {}
        weighted_correct = 0
        weighted_total = 0
        for cat in sorted(bat_by_cat.keys()):
            train_idx = bat_by_cat[cat]
            test_idx = exp_by_cat.get(cat, [])
            if len(train_idx) < 3 or len(test_idx) < 2:
                print(f"    {cat}: skipped (too few cases)")
                continue
            cat_dir = make_centroid(bat_b[train_idx], bat_w[train_idx])
            cat_centroids[cat] = cat_dir
            acc = pairwise_accuracy(exp_b[test_idx], exp_w[test_idx], cat_dir)
            cat_accs[cat] = acc
            weighted_correct += acc * len(test_idx)
            weighted_total += len(test_idx)
            # Also compare to global on same cases
            global_on_cat = pairwise_accuracy(exp_b[test_idx], exp_w[test_idx], global_dir)
            delta = acc - global_on_cat
            marker = "+" if delta > 0 else ""
            print(f"    {cat}: {acc:.1%} (global: {global_on_cat:.1%}, "
                  f"delta: {marker}{delta:.1%}, n_train={len(train_idx)}, n_test={len(test_idx)})")

        if weighted_total > 0:
            weighted_avg = weighted_correct / weighted_total
            print(f"  Weighted per-category average: {weighted_avg:.1%} (global: {global_acc:.1%})")

        # 3. Ensemble: for each test case, use the centroid from its category
        print(f"\n  Ensemble (use matching category centroid):")
        ensemble_correct = 0
        ensemble_total = 0
        for cat, test_idx in exp_by_cat.items():
            if cat in cat_centroids:
                d = cat_centroids[cat]
            else:
                d = global_dir  # fallback
            for i in test_idx:
                if np.dot(exp_b[i], d) > np.dot(exp_w[i], d):
                    ensemble_correct += 1
                ensemble_total += 1
        ensemble_acc = ensemble_correct / ensemble_total if ensemble_total > 0 else 0
        print(f"  Ensemble OOS: {ensemble_acc:.1%} (global: {global_acc:.1%}, "
              f"delta: {'+' if ensemble_acc > global_acc else ''}{ensemble_acc - global_acc:.1%})")

        # 4. Multi-centroid voting: score each test case against ALL category centroids
        print(f"\n  Multi-centroid voting (majority of category centroids):")
        vote_correct = 0
        for i in range(len(expansion)):
            votes = 0
            total_votes = 0
            for cat, d in cat_centroids.items():
                if np.dot(exp_b[i], d) > np.dot(exp_w[i], d):
                    votes += 1
                total_votes += 1
            if votes > total_votes / 2:
                vote_correct += 1
        vote_acc = vote_correct / len(expansion)
        print(f"  Majority vote OOS: {vote_acc:.1%} (global: {global_acc:.1%}, "
              f"delta: {'+' if vote_acc > global_acc else ''}{vote_acc - global_acc:.1%})")

        # 5. Score-sum: sum dot products across all category centroids
        print(f"\n  Multi-centroid score-sum:")
        sum_correct = 0
        for i in range(len(expansion)):
            score_b = sum(np.dot(exp_b[i], d) for d in cat_centroids.values())
            score_w = sum(np.dot(exp_w[i], d) for d in cat_centroids.values())
            if score_b > score_w:
                sum_correct += 1
        sum_acc = sum_correct / len(expansion)
        print(f"  Score-sum OOS: {sum_acc:.1%} (global: {global_acc:.1%}, "
              f"delta: {'+' if sum_acc > global_acc else ''}{sum_acc - global_acc:.1%})")

        # 6. Cosine between category centroids
        print(f"\n  Cosine between category centroids:")
        cats = sorted(cat_centroids.keys())
        for i in range(len(cats)):
            for j in range(i+1, len(cats)):
                cos = float(np.dot(cat_centroids[cats[i]], cat_centroids[cats[j]]))
                print(f"    {cats[i]} vs {cats[j]}: {cos:.3f}")

        all_results[model_name] = {
            "global_oos": round(global_acc, 4),
            "weighted_category_avg": round(weighted_avg, 4) if weighted_total > 0 else None,
            "ensemble_oos": round(ensemble_acc, 4),
            "majority_vote_oos": round(vote_acc, 4),
            "score_sum_oos": round(sum_acc, 4),
            "per_category": {cat: round(acc, 4) for cat, acc in cat_accs.items()},
        }

        del model
        import gc; gc.collect()

    # Save
    out_path = OUT_DIR / "category_ensemble_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Per-category centroids vs global")
    print(f"{'='*70}")
    for mname, r in all_results.items():
        short = mname.split("/")[-1]
        print(f"\n  {short}:")
        print(f"    Global:       {r['global_oos']:.1%}")
        print(f"    Ensemble:     {r['ensemble_oos']:.1%}")
        print(f"    Majority:     {r['majority_vote_oos']:.1%}")
        print(f"    Score-sum:    {r['score_sum_oos']:.1%}")


if __name__ == "__main__":
    main()
