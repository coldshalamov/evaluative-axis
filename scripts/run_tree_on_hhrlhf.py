#!/usr/bin/env python3
"""Test the five-term tree (careful, honest, helpful, thorough, restrained)
on 493 HH-RLHF preference pairs using OR logic.

This is the decisive generalization test: the tree gets 89-94% OOS on the
hand-crafted battery. If it also works on real preference data, the method
generalizes. If it fails, the battery result doesn't transfer.

Uses cached embeddings from run_probe_fixed.py.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

CACHE_DIR = Path(__file__).resolve().parents[1] / "notes" / "research_cycles" / "real_data_test"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TREE_POSITIVE = ["Careful", "Honest", "Helpful", "Thorough", "Restrained"]
TREE_NEGATIVE = ["Reckless", "Dishonest", "Unhelpful", "Superficial", "Unrestrained"]


def bootstrap_ci(scores, n_boot=5000, ci=0.95):
    arr = np.array(scores, dtype=float)
    n = len(arr)
    boot_means = np.array([np.mean(np.random.choice(arr, size=n, replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha))


def main():
    from sentence_transformers import SentenceTransformer

    with open(CACHE_DIR / "hh_rlhf_sample.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)

    n = len(pairs)
    print(f"Dataset: {n} preference pairs from Anthropic HH-RLHF\n")
    np.random.seed(42)

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        cache_file = CACHE_DIR / f"embeddings_{short}.npz"
        if cache_file.exists():
            print(f"  Loading cached embeddings...")
            data = np.load(cache_file)
            chosen_embs = data["chosen"]
            rejected_embs = data["rejected"]
        else:
            print(f"  ERROR: No cached embeddings found at {cache_file}")
            continue

        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Embed tree anchor words
        pos_embs = {}
        neg_embs = {}
        for word in TREE_POSITIVE:
            e = model.encode([word], convert_to_numpy=True)[0]
            pos_embs[word] = e / (norm(e) + 1e-12)
        for word in TREE_NEGATIVE:
            e = model.encode([word], convert_to_numpy=True)[0]
            neg_embs[word] = e / (norm(e) + 1e-12)

        del model
        gc.collect()

        # --- Test 1: Individual axis cosine-to-positive ---
        print(f"\n--- Individual axis accuracy (cosine to positive anchor) ---")
        print(f"  {'Axis':20s} {'Acc':>6s} {'CI':>13s}")
        print(f"  {'-'*45}")
        axis_correct = {}
        for pos_word, neg_word in zip(TREE_POSITIVE, TREE_NEGATIVE):
            a = pos_embs[pos_word]
            correct = [float(np.dot(chosen_embs[i], a)) > float(np.dot(rejected_embs[i], a))
                       for i in range(n)]
            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)
            print(f"  {pos_word:20s} {acc:5.1%} [{lo:.0%},{hi:.0%}]")
            axis_correct[pos_word] = correct

        # --- Test 2: Individual axis bipolar ---
        print(f"\n--- Individual axis accuracy (bipolar: cos_pos - cos_neg) ---")
        print(f"  {'Axis':20s} {'Acc':>6s} {'CI':>13s}")
        print(f"  {'-'*45}")
        axis_bipolar_correct = {}
        for pos_word, neg_word in zip(TREE_POSITIVE, TREE_NEGATIVE):
            a_pos = pos_embs[pos_word]
            a_neg = neg_embs[neg_word]
            correct = []
            for i in range(n):
                chosen_score = float(np.dot(chosen_embs[i], a_pos)) - float(np.dot(chosen_embs[i], a_neg))
                rejected_score = float(np.dot(rejected_embs[i], a_pos)) - float(np.dot(rejected_embs[i], a_neg))
                correct.append(chosen_score > rejected_score)
            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)
            print(f"  {pos_word}/{neg_word:15s} {acc:5.1%} [{lo:.0%},{hi:.0%}]")
            axis_bipolar_correct[pos_word] = correct

        # --- Test 3: Tree combinations (OR, majority, 2-of-5) ---
        print(f"\n--- Tree combinations (cosine-to-positive) ---")
        for threshold_name, threshold in [("ANY (1 of 5)", 1), ("2 of 5", 2), ("Majority (3 of 5)", 3)]:
            correct = []
            for i in range(n):
                votes = sum(1 for word in TREE_POSITIVE if axis_correct[word][i])
                correct.append(votes >= threshold)
            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)
            print(f"  {threshold_name:20s} {acc:5.1%} [{lo:.0%},{hi:.0%}]")

        print(f"\n--- Tree combinations (bipolar) ---")
        for threshold_name, threshold in [("ANY (1 of 5)", 1), ("2 of 5", 2), ("Majority (3 of 5)", 3)]:
            correct = []
            for i in range(n):
                votes = sum(1 for word in TREE_POSITIVE if axis_bipolar_correct[word][i])
                correct.append(votes >= threshold)
            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)
            print(f"  {threshold_name:20s} {acc:5.1%} [{lo:.0%},{hi:.0%}]")

        # --- Test 4: Sum of bipolar scores ---
        print(f"\n--- Sum of all 5 bipolar scores ---")
        correct = []
        for i in range(n):
            chosen_sum = sum(float(np.dot(chosen_embs[i], pos_embs[w])) - float(np.dot(chosen_embs[i], neg_embs[TREE_NEGATIVE[j]]))
                            for j, w in enumerate(TREE_POSITIVE))
            rejected_sum = sum(float(np.dot(rejected_embs[i], pos_embs[w])) - float(np.dot(rejected_embs[i], neg_embs[TREE_NEGATIVE[j]]))
                              for j, w in enumerate(TREE_POSITIVE))
            correct.append(chosen_sum > rejected_sum)
        acc = sum(correct) / n
        lo, hi = bootstrap_ci(correct)
        print(f"  Sum of 5 bipolar    {acc:5.1%} [{lo:.0%},{hi:.0%}]")

        # --- Store results ---
        results[short] = {
            "individual_cospos": {w: sum(axis_correct[w])/n for w in TREE_POSITIVE},
            "individual_bipolar": {w: sum(axis_bipolar_correct[w])/n for w in TREE_POSITIVE},
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'':20s}", end="")
    for short in results:
        print(f" {short:>12s}", end="")
    print()

    for word in TREE_POSITIVE:
        print(f"  cos({word}):{'':10s}", end="")
        for short, r in results.items():
            print(f" {r['individual_cospos'][word]:11.1%}", end="")
        print()

    print()
    for word in TREE_POSITIVE:
        neg = TREE_NEGATIVE[TREE_POSITIVE.index(word)]
        print(f"  bip({word}/{neg}):{'':4s}", end="")
        for short, r in results.items():
            print(f" {r['individual_bipolar'][word]:11.1%}", end="")
        print()

    with open(CACHE_DIR / "tree_hhrlhf_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {CACHE_DIR / 'tree_hhrlhf_results.json'}")


if __name__ == "__main__":
    main()
