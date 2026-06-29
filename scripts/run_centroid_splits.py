#!/usr/bin/env python3
"""Test centroid direction transfer across quality types.

Key questions:
1. Train on warmth-only (20 cases) → test on firmness (50 cases): does it transfer?
2. Train on firmness-only (50 cases) → test on warmth (20 cases): does it transfer?
3. Train on expansion (35) → test on battery (70): reverse validation
4. Per-category leave-one-category-out: train on all but one category, test on held-out
5. What's the cosine between warmth-trained and firmness-trained directions?
6. Logistic probe with proper GroupKFold on battery
"""

import json, gc, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from numpy.linalg import norm

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


def embed_and_norm(model, texts):
    embs = model.encode(texts, convert_to_numpy=True, batch_size=32)
    for i in range(len(embs)):
        embs[i] /= norm(embs[i]) + 1e-12
    return embs


def accuracy(better_embs, worse_embs, direction):
    n = len(better_embs)
    correct = sum(
        1 for i in range(n)
        if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)
    )
    return correct / n


def make_dir(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    d /= norm(d) + 1e-12
    return d


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orig_cases = load_cases(BATTERY_50)
    warmth_cases = load_cases(WARMTH_20)
    battery = orig_cases + warmth_cases

    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    # Categorize battery cases
    battery_cats = {}
    for i, c in enumerate(battery):
        cat = c.get("category", "unknown")
        battery_cats.setdefault(cat, []).append(i)

    print(f"Battery: {len(orig_cases)} orig + {len(warmth_cases)} warmth = {len(battery)}")
    print(f"Expansion: {len(expansion)}")
    print(f"Battery categories: {sorted(battery_cats.keys())}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        bat_better = embed_and_norm(model, [c["better"] for c in battery])
        bat_worse = embed_and_norm(model, [c["worse"] for c in battery])

        orig_better = bat_better[:len(orig_cases)]
        orig_worse = bat_worse[:len(orig_cases)]
        warm_better = bat_better[len(orig_cases):]
        warm_worse = bat_worse[len(orig_cases):]

        exp_better = embed_and_norm(model, [c["better"] for c in expansion])
        exp_worse = embed_and_norm(model, [c["worse"] for c in expansion])

        model_results = {}

        # =================================================================
        # 1. CROSS-TYPE TRANSFER
        # =================================================================
        print(f"\n--- 1. Cross-Type Direction Transfer ---")

        # Train on firmness (orig 50), test on warmth (20) and expansion (35)
        dir_firm = make_dir(orig_better, orig_worse)
        firm_on_warm = accuracy(warm_better, warm_worse, dir_firm)
        firm_on_exp = accuracy(exp_better, exp_worse, dir_firm)
        firm_on_firm = accuracy(orig_better, orig_worse, dir_firm)

        print(f"  Trained on FIRMNESS (50 orig cases):")
        print(f"    → Firmness:  {firm_on_firm:.0%} (in-sample)")
        print(f"    → Warmth:    {firm_on_warm:.0%} (cross-type transfer)")
        print(f"    → Expansion: {firm_on_exp:.0%} (OOS)")

        # Train on warmth (20), test on firmness (50) and expansion (35)
        dir_warm = make_dir(warm_better, warm_worse)
        warm_on_firm = accuracy(orig_better, orig_worse, dir_warm)
        warm_on_exp = accuracy(exp_better, exp_worse, dir_warm)
        warm_on_warm = accuracy(warm_better, warm_worse, dir_warm)

        print(f"\n  Trained on WARMTH (20 cases):")
        print(f"    → Warmth:    {warm_on_warm:.0%} (in-sample)")
        print(f"    → Firmness:  {warm_on_firm:.0%} (cross-type transfer)")
        print(f"    → Expansion: {warm_on_exp:.0%} (OOS)")

        # Full battery for comparison
        dir_full = make_dir(bat_better, bat_worse)
        full_on_exp = accuracy(exp_better, exp_worse, dir_full)
        print(f"\n  Trained on FULL battery (70 cases):")
        print(f"    → Expansion: {full_on_exp:.0%} (OOS)")

        # Cosine between directions
        cos_fw = float(np.dot(dir_firm, dir_warm))
        cos_ff = float(np.dot(dir_firm, dir_full))
        cos_wf = float(np.dot(dir_warm, dir_full))
        print(f"\n  Direction cosines:")
        print(f"    Firmness ↔ Warmth:   {cos_fw:+.4f}")
        print(f"    Firmness ↔ Full:     {cos_ff:+.4f}")
        print(f"    Warmth   ↔ Full:     {cos_wf:+.4f}")

        model_results["cross_type"] = {
            "firm_on_warm": float(firm_on_warm),
            "firm_on_exp": float(firm_on_exp),
            "warm_on_firm": float(warm_on_firm),
            "warm_on_exp": float(warm_on_exp),
            "full_on_exp": float(full_on_exp),
            "cos_firm_warm": cos_fw,
            "cos_firm_full": cos_ff,
            "cos_warm_full": cos_wf,
        }

        # =================================================================
        # 2. REVERSE VALIDATION
        # =================================================================
        print(f"\n--- 2. Reverse Validation (train expansion → test battery) ---")

        dir_exp = make_dir(exp_better, exp_worse)
        rev_on_bat = accuracy(bat_better, bat_worse, dir_exp)
        rev_on_orig = accuracy(orig_better, orig_worse, dir_exp)
        rev_on_warm = accuracy(warm_better, warm_worse, dir_exp)

        print(f"  Trained on EXPANSION (35 cases):")
        print(f"    → Full battery: {rev_on_bat:.0%}")
        print(f"    → Orig 50:      {rev_on_orig:.0%}")
        print(f"    → Warmth 20:    {rev_on_warm:.0%}")

        cos_exp_full = float(np.dot(dir_exp, dir_full))
        print(f"    Direction cosine (expansion ↔ full battery): {cos_exp_full:+.4f}")

        model_results["reverse_validation"] = {
            "exp_on_battery": float(rev_on_bat),
            "exp_on_orig": float(rev_on_orig),
            "exp_on_warm": float(rev_on_warm),
            "cos_exp_full": cos_exp_full,
        }

        # =================================================================
        # 3. LEAVE-ONE-CATEGORY-OUT (battery)
        # =================================================================
        print(f"\n--- 3. Leave-One-Category-Out (battery) ---")

        for cat, indices in sorted(battery_cats.items()):
            train_mask = np.ones(len(battery), dtype=bool)
            train_mask[indices] = False
            test_idx = np.array(indices)

            direction = make_dir(bat_better[train_mask], bat_worse[train_mask])
            cat_acc = accuracy(bat_better[test_idx], bat_worse[test_idx], direction)
            print(f"  Hold out '{cat}' ({len(indices)} cases): {cat_acc:.0%}")

        # =================================================================
        # 4. LOGISTIC PROBE WITH GROUPKFOLD
        # =================================================================
        print(f"\n--- 4. Logistic Probe with GroupKFold ---")

        diff_bat = bat_better - bat_worse
        X = np.vstack([diff_bat, -diff_bat])
        y = np.concatenate([np.ones(len(battery)), np.zeros(len(battery))])
        groups = np.concatenate([np.arange(len(battery)), np.arange(len(battery))])

        gkf = GroupKFold(n_splits=5)
        fold_accs = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))

        mean_gkf = np.mean(fold_accs)
        std_gkf = np.std(fold_accs)
        print(f"  5-fold GroupKFold: {mean_gkf:.1%} +/- {std_gkf:.1%}")
        print(f"  Folds: {[f'{a:.0%}' for a in fold_accs]}")

        # Train probe on full battery, test on expansion
        clf_full = LogisticRegression(max_iter=1000, C=1.0)
        clf_full.fit(X, y)
        diff_exp = exp_better - exp_worse
        X_exp = np.vstack([diff_exp, -diff_exp])
        y_exp = np.concatenate([np.ones(len(expansion)), np.zeros(len(expansion))])
        probe_oos = clf_full.score(X_exp, y_exp)
        print(f"  Train battery → test expansion: {probe_oos:.0%}")

        # Compare probe direction to centroid direction
        probe_dir = clf_full.coef_[0]
        probe_dir /= norm(probe_dir) + 1e-12
        cos_probe_centroid = float(np.dot(probe_dir, dir_full))
        print(f"  Cosine (probe ↔ centroid): {cos_probe_centroid:+.4f}")

        model_results["probe_groupkfold"] = {
            "mean": float(mean_gkf),
            "std": float(std_gkf),
            "folds": [float(a) for a in fold_accs],
            "oos": float(probe_oos),
            "cos_probe_centroid": cos_probe_centroid,
        }

        # =================================================================
        # 5. HOW DIFFERENT IS EACH CASE FROM THE MEAN?
        # =================================================================
        print(f"\n--- 5. Training Pair Importance ---")

        # For each training pair, compute: if we remove it, how much does
        # expansion accuracy change? (Leave-one-out sensitivity)
        base_acc = accuracy(exp_better, exp_worse, dir_full)
        deltas = []
        for i in range(len(battery)):
            mask = np.ones(len(battery), dtype=bool)
            mask[i] = False
            loo_dir = make_dir(bat_better[mask], bat_worse[mask])
            loo_acc = accuracy(exp_better, exp_worse, loo_dir)
            deltas.append((i, battery[i].get("category", "?"),
                          battery[i].get("phenomenon", "?"),
                          loo_acc - base_acc))

        deltas.sort(key=lambda x: x[3])

        print(f"  Most HELPFUL pairs (removing them hurts OOS accuracy):")
        for idx, cat, phen, d in deltas[:5]:
            print(f"    #{idx:2d} [{cat}] {phen}: {d:+.1%}")

        print(f"\n  Most HARMFUL pairs (removing them helps OOS accuracy):")
        for idx, cat, phen, d in deltas[-5:]:
            print(f"    #{idx:2d} [{cat}] {phen}: {d:+.1%}")

        all_results[short] = model_results

        del model
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Cross-type transfer:")
    print(f"  {'':30s}", end="")
    for short in all_results:
        print(f"  {short:>20s}", end="")
    print()
    for metric in ["firm_on_warm", "warm_on_firm", "firm_on_exp", "warm_on_exp", "full_on_exp"]:
        label = metric.replace("_", " ")
        print(f"  {label:30s}", end="")
        for short, r in all_results.items():
            print(f"  {r['cross_type'][metric]:19.0%}", end="")
        print()

    print(f"\n  Direction cosines (firm ↔ warm):")
    for short, r in all_results.items():
        print(f"  {short}: {r['cross_type']['cos_firm_warm']:+.4f}")

    print(f"\n  Reverse validation (train exp → test battery):")
    for short, r in all_results.items():
        rv = r["reverse_validation"]
        print(f"  {short}: battery={rv['exp_on_battery']:.0%} (cos={rv['cos_exp_full']:+.4f})")

    with open(OUT_DIR / "centroid_splits_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'centroid_splits_results.json'}")


if __name__ == "__main__":
    main()
