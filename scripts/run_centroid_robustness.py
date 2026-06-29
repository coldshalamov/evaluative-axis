#!/usr/bin/env python3
"""Centroid robustness: how stable is the direction across different training data?

Key experiments:
1. Reverse validation: train on expansion (35/61), test on battery (70)
2. Cross-validation with stratified splits
3. Direction stability: how much does the direction change when we use
   completely different training cases?
4. Ensemble: average multiple centroid directions from different splits
5. Comparison to random directions at each training size
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
    return sum(1 for i in range(n)
               if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)) / n


def make_dir(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    d /= norm(d) + 1e-12
    return d


def main():
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    # Pool everything
    all_cases = battery + expansion
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}, Total: {len(all_cases)}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        all_better = embed_and_norm(model, [c["better"] for c in all_cases])
        all_worse = embed_and_norm(model, [c["worse"] for c in all_cases])

        bat_better = all_better[:len(battery)]
        bat_worse = all_worse[:len(battery)]
        exp_better = all_better[len(battery):]
        exp_worse = all_worse[len(battery):]

        mr = {}

        # =================================================================
        # 1. REVERSE VALIDATION (expansion -> battery)
        # =================================================================
        print(f"\n--- 1. Reverse Validation ---")
        dir_bat = make_dir(bat_better, bat_worse)
        dir_exp = make_dir(exp_better, exp_worse)

        bat_on_exp = accuracy(exp_better, exp_worse, dir_bat)
        exp_on_bat = accuracy(bat_better, bat_worse, dir_exp)
        exp_on_exp = accuracy(exp_better, exp_worse, dir_exp)
        bat_on_bat = accuracy(bat_better, bat_worse, dir_bat)

        cos_be = float(np.dot(dir_bat, dir_exp))
        print(f"  Battery dir -> Expansion: {bat_on_exp:.0%} (battery self: {bat_on_bat:.0%})")
        print(f"  Expansion dir -> Battery: {exp_on_bat:.0%} (expansion self: {exp_on_exp:.0%})")
        print(f"  Cosine(battery dir, expansion dir): {cos_be:+.4f}")

        mr["reverse"] = {
            "bat_on_exp": float(bat_on_exp), "exp_on_bat": float(exp_on_bat),
            "bat_self": float(bat_on_bat), "exp_self": float(exp_on_exp),
            "cos_dirs": cos_be,
        }

        # =================================================================
        # 2. POOLED DIRECTION (all cases together)
        # =================================================================
        print(f"\n--- 2. Pooled Direction ---")
        dir_all = make_dir(all_better, all_worse)
        all_on_bat = accuracy(bat_better, bat_worse, dir_all)
        all_on_exp = accuracy(exp_better, exp_worse, dir_all)
        cos_ab = float(np.dot(dir_all, dir_bat))
        cos_ae = float(np.dot(dir_all, dir_exp))
        print(f"  Pooled ({len(all_cases)} cases) -> Battery: {all_on_bat:.0%}")
        print(f"  Pooled -> Expansion: {all_on_exp:.0%}")
        print(f"  Cosine(pooled, battery): {cos_ab:+.4f}")
        print(f"  Cosine(pooled, expansion): {cos_ae:+.4f}")

        mr["pooled"] = {
            "on_bat": float(all_on_bat), "on_exp": float(all_on_exp),
            "cos_bat": cos_ab, "cos_exp": cos_ae,
        }

        # =================================================================
        # 3. DISJOINT SPLIT STABILITY
        # =================================================================
        print(f"\n--- 3. Disjoint Split Stability ---")
        # Split pooled data into two non-overlapping halves, compute direction from
        # each, check cosine between them and cross-test accuracy

        n_total = len(all_cases)
        n_reps = 50
        cos_splits = []
        cross_accs = []
        for _ in range(n_reps):
            perm = np.random.permutation(n_total)
            half1 = perm[:n_total//2]
            half2 = perm[n_total//2:]

            d1 = make_dir(all_better[half1], all_worse[half1])
            d2 = make_dir(all_better[half2], all_worse[half2])
            cos_splits.append(float(np.dot(d1, d2)))

            # d1 tested on half2, d2 tested on half1
            a12 = accuracy(all_better[half2], all_worse[half2], d1)
            a21 = accuracy(all_better[half1], all_worse[half1], d2)
            cross_accs.append((a12 + a21) / 2)

        print(f"  Cosine between disjoint halves (50 splits):")
        print(f"    Mean: {np.mean(cos_splits):+.4f}  Std: {np.std(cos_splits):.4f}")
        print(f"    Min: {np.min(cos_splits):+.4f}  Max: {np.max(cos_splits):+.4f}")
        print(f"  Cross-accuracy (train half1, test half2 and vice versa):")
        print(f"    Mean: {np.mean(cross_accs):.1%}  Std: {np.std(cross_accs):.1%}")

        mr["split_stability"] = {
            "cos_mean": float(np.mean(cos_splits)),
            "cos_std": float(np.std(cos_splits)),
            "cos_min": float(np.min(cos_splits)),
            "acc_mean": float(np.mean(cross_accs)),
            "acc_std": float(np.std(cross_accs)),
        }

        # =================================================================
        # 4. ENSEMBLE: AVERAGE DIRECTIONS FROM K DISJOINT SPLITS
        # =================================================================
        print(f"\n--- 4. Ensemble Directions ---")

        for k in [3, 5]:
            perm = np.random.permutation(n_total)
            chunk_size = n_total // k
            dirs = []
            for j in range(k):
                idx = perm[j*chunk_size:(j+1)*chunk_size]
                d = make_dir(all_better[idx], all_worse[idx])
                dirs.append(d)
            ens_dir = np.mean(dirs, axis=0)
            ens_dir /= norm(ens_dir) + 1e-12

            ens_bat = accuracy(bat_better, bat_worse, ens_dir)
            ens_exp = accuracy(exp_better, exp_worse, ens_dir)
            cos_ens_full = float(np.dot(ens_dir, dir_all))
            print(f"  Ensemble of {k} disjoint chunks -> Battery: {ens_bat:.0%}  Expansion: {ens_exp:.0%}  cos(pooled): {cos_ens_full:+.4f}")

        # =================================================================
        # 5. RANDOM DIRECTION COMPARISON AT EACH TRAINING SIZE
        # =================================================================
        print(f"\n--- 5. Random vs. Centroid at Each Training Size ---")
        dim = all_better.shape[1]
        for n_train in [10, 20, 35, 50, 70]:
            cent_accs = []
            rand_accs = []
            for _ in range(100):
                idx = np.random.choice(len(battery), size=min(n_train, len(battery)), replace=False)
                d = make_dir(bat_better[idx], bat_worse[idx])
                cent_accs.append(accuracy(exp_better, exp_worse, d))

                r = np.random.randn(dim)
                r /= norm(r)
                rand_accs.append(accuracy(exp_better, exp_worse, r))

            print(f"  n={n_train:3d}: centroid {np.mean(cent_accs):.1%} [{np.percentile(cent_accs,5):.0%},{np.percentile(cent_accs,95):.0%}]  random {np.mean(rand_accs):.1%} [{np.percentile(rand_accs,5):.0%},{np.percentile(rand_accs,95):.0%}]")

        all_results[short] = mr
        del model
        gc.collect()

    with open(OUT_DIR / "centroid_robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'centroid_robustness_results.json'}")


if __name__ == "__main__":
    main()
