#!/usr/bin/env python3
"""Learn the quality direction from battery embeddings, test on expansion cases.

This answers a fundamentally different question from cosine-to-anchor:
"Is quality geometrically separable in these embeddings at all?"

If a supervised direction (mean_better - mean_worse) trained on the 70-case
battery transfers to 35 held-out expansion cases, quality IS encoded — the
problem is just that anchor words can't find it unsupervised.

If even the supervised direction fails, quality is not linearly separable
in these embedding spaces, full stop.

Also compares the learned direction to "careful" and "good" axis vectors
to see if anchor words are even pointing in the right neighborhood.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
CACHE_DIR = ROOT / "notes" / "research_cycles" / "supervised_direction"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

ANCHOR_WORDS = {
    "Careful": "Reckless",
    "Good": "Bad",
    "Honest": "Dishonest",
    "Helpful": "Unhelpful",
    "Thorough": "Superficial",
    "Restrained": "Unrestrained",
}


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main():
    from sentence_transformers import SentenceTransformer

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load battery (training set)
    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    print(f"Training set: {len(battery)} battery cases (50 orig + 20 warmth)")

    # Load expansion (test set — never used during development)
    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))
    print(f"Test set: {len(expansion)} expansion cases (OOS)")

    # Show category breakdown
    cats = {}
    for c in expansion:
        cat = c.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  Categories: {cats}")

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Embed battery responses
        battery_better = model.encode([c["better"] for c in battery],
                                      convert_to_numpy=True, batch_size=32)
        battery_worse = model.encode([c["worse"] for c in battery],
                                     convert_to_numpy=True, batch_size=32)

        # Normalize
        for i in range(len(battery)):
            battery_better[i] /= norm(battery_better[i]) + 1e-12
            battery_worse[i] /= norm(battery_worse[i]) + 1e-12

        # Embed expansion responses
        exp_better = model.encode([c["better"] for c in expansion],
                                  convert_to_numpy=True, batch_size=32)
        exp_worse = model.encode([c["worse"] for c in expansion],
                                 convert_to_numpy=True, batch_size=32)

        for i in range(len(expansion)):
            exp_better[i] /= norm(exp_better[i]) + 1e-12
            exp_worse[i] /= norm(exp_worse[i]) + 1e-12

        # === METHOD 1: Supervised quality direction ===
        # quality_dir = mean(better) - mean(worse)
        quality_dir = battery_better.mean(axis=0) - battery_worse.mean(axis=0)
        quality_dir /= norm(quality_dir) + 1e-12

        # Test on battery (in-sample)
        battery_correct = sum(
            1 for i in range(len(battery))
            if np.dot(battery_better[i], quality_dir) > np.dot(battery_worse[i], quality_dir)
        )
        battery_acc = battery_correct / len(battery)

        # Test on expansion (OOS)
        exp_correct = sum(
            1 for i in range(len(expansion))
            if np.dot(exp_better[i], quality_dir) > np.dot(exp_worse[i], quality_dir)
        )
        exp_acc = exp_correct / len(expansion)

        print(f"\n  Supervised direction (mean_better - mean_worse):")
        print(f"    Battery (in-sample, n={len(battery)}): {battery_acc:.1%}")
        print(f"    Expansion (OOS, n={len(expansion)}):   {exp_acc:.1%}")

        # Per-category OOS
        print(f"    Per-category OOS:")
        cat_results = {}
        for i, c in enumerate(expansion):
            cat = c.get("category", "unknown")
            correct = np.dot(exp_better[i], quality_dir) > np.dot(exp_worse[i], quality_dir)
            cat_results.setdefault(cat, []).append(correct)
        for cat, vals in sorted(cat_results.items()):
            print(f"      {cat:30s}: {sum(vals)/len(vals):.0%} ({sum(vals)}/{len(vals)})")

        # === METHOD 2: Leave-one-out on battery ===
        loo_correct = 0
        for i in range(len(battery)):
            mask = np.ones(len(battery), dtype=bool)
            mask[i] = False
            loo_dir = battery_better[mask].mean(axis=0) - battery_worse[mask].mean(axis=0)
            loo_dir /= norm(loo_dir) + 1e-12
            if np.dot(battery_better[i], loo_dir) > np.dot(battery_worse[i], loo_dir):
                loo_correct += 1
        loo_acc = loo_correct / len(battery)
        print(f"\n  Leave-one-out on battery: {loo_acc:.1%}")

        # === METHOD 3: Logistic regression on battery, test on expansion ===
        from sklearn.linear_model import LogisticRegression

        # Train on battery diff vectors
        diff_train = battery_better - battery_worse
        X_train = np.vstack([diff_train, -diff_train])
        y_train = np.concatenate([np.ones(len(battery)), np.zeros(len(battery))])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        # Test on expansion diff vectors
        diff_test = exp_better - exp_worse
        X_test = np.vstack([diff_test, -diff_test])
        y_test = np.concatenate([np.ones(len(expansion)), np.zeros(len(expansion))])
        probe_acc = clf.score(X_test, y_test)

        probe_dir = clf.coef_[0]
        probe_dir /= norm(probe_dir) + 1e-12

        print(f"\n  Logistic probe (train=battery, test=expansion): {probe_acc:.1%}")

        # === Compare directions to anchor words ===
        print(f"\n  Direction comparisons (cosine similarity):")
        anchor_dirs = {}
        for pos, neg in ANCHOR_WORDS.items():
            pos_emb = model.encode([pos], convert_to_numpy=True)[0]
            neg_emb = model.encode([neg], convert_to_numpy=True)[0]
            axis = pos_emb - neg_emb
            axis /= norm(axis) + 1e-12
            anchor_dirs[pos] = axis

        print(f"    {'':20s} {'quality_dir':>12s} {'probe_dir':>12s}")
        for name, axis in anchor_dirs.items():
            cos_q = float(np.dot(quality_dir, axis))
            cos_p = float(np.dot(probe_dir, axis))
            print(f"    {name:20s} {cos_q:+11.3f}  {cos_p:+11.3f}")

        # quality_dir vs probe_dir
        cos_qp = float(np.dot(quality_dir, probe_dir))
        print(f"    quality_dir vs probe_dir: {cos_qp:+.3f}")

        # === Anchor words scored on expansion (for comparison) ===
        print(f"\n  Anchor word accuracy on expansion (OOS, n={len(expansion)}):")
        for pos, neg in ANCHOR_WORDS.items():
            axis = anchor_dirs[pos]
            correct = sum(
                1 for i in range(len(expansion))
                if np.dot(exp_better[i], axis) > np.dot(exp_worse[i], axis)
            )
            print(f"    {pos}/{neg:15s}: {correct/len(expansion):.1%}")

        # === Store ===
        results[short] = {
            "supervised_battery": battery_acc,
            "supervised_expansion": exp_acc,
            "loo_battery": loo_acc,
            "probe_expansion": probe_acc,
        }

        del model
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Method':35s}", end="")
    for short in results:
        print(f" {short:>12s}", end="")
    print()
    for method in ["supervised_battery", "supervised_expansion", "loo_battery", "probe_expansion"]:
        label = method.replace("_", " ")
        print(f"  {label:35s}", end="")
        for short, r in results.items():
            print(f" {r[method]:11.1%}", end="")
        print()

    with open(CACHE_DIR / "supervised_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {CACHE_DIR / 'supervised_results.json'}")


if __name__ == "__main__":
    main()
