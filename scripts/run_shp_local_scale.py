#!/usr/bin/env python3
"""Test centroid on SHP at scale with LOCAL models.

No API rate limits. Addresses the core question: does more training data
make the centroid work on a real public dataset?

Previous test: 70 SHP train → 430 test → 55-57% (chance).
This test: 200, 500, 1000, 2000 SHP train → rest test.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def load_shp(min_score_ratio=2.0, max_pairs=5000):
    from datasets import load_dataset
    pairs = []
    for split in ["train", "validation", "test"]:
        print(f"  Loading SHP {split}...")
        ds = load_dataset("stanfordnlp/SHP", split=split)
        for row in ds:
            sr = row.get("score_ratio") or 0
            if sr < min_score_ratio:
                continue
            if row["labels"] == 1:
                better, worse = row["human_ref_A"], row["human_ref_B"]
            else:
                better, worse = row["human_ref_B"], row["human_ref_A"]
            if len(better) < 20 or len(worse) < 20:
                continue
            if len(better) > 3000 or len(worse) > 3000:
                continue
            pairs.append({"better": better, "worse": worse,
                           "score_ratio": sr})
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    np.random.RandomState(42).shuffle(pairs)
    return pairs


def make_centroid(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, direction):
    return float((np.dot(better_embs, direction) > np.dot(worse_embs, direction)).mean())


def permutation_test(train_b, train_w, test_b, test_w, observed, n_perms=1000):
    n = len(train_b)
    count = 0
    for _ in range(n_perms):
        mask = np.random.randint(0, 2, size=n).astype(bool)
        pb = np.where(mask[:, None], train_b, train_w)
        pw = np.where(mask[:, None], train_w, train_b)
        d = make_centroid(pb, pw)
        if pairwise_accuracy(test_b, test_w, d) >= observed:
            count += 1
    return count / n_perms


def main():
    # Load SHP — response-only format (no prompt, matching meta-validity finding)
    print("Loading SHP...")
    pairs = load_shp(min_score_ratio=2.0, max_pairs=5000)
    print(f"Loaded {len(pairs)} SHP pairs\n")

    from sentence_transformers import SentenceTransformer

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Response-only embeddings
        print("  Embedding responses (response-only format)...")
        all_better = model.encode([c["better"] for c in pairs],
                                   batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True)
        all_worse = model.encode([c["worse"] for c in pairs],
                                  batch_size=64, show_progress_bar=True,
                                  convert_to_numpy=True)

        n = len(pairs)

        # Full in-sample
        full_dir = make_centroid(all_better, all_worse)
        full_acc = pairwise_accuracy(all_better, all_worse, full_dir)
        print(f"\n  Full in-sample ({n} pairs): {full_acc:.1%}")

        # Learning curve with 50/50 split
        half = n // 2
        results = {"model": model_name, "n_total": n,
                    "full_insample": round(full_acc, 4)}

        print(f"\n  Learning curve (5 random splits each):")
        lc = {}
        for n_train in [70, 200, 500, 1000, 2000, half]:
            if n_train >= n - 50:
                continue
            accs = []
            for rep in range(5):
                perm = np.random.RandomState(rep * 100 + n_train).permutation(n)
                tr, te = perm[:n_train], perm[n_train:]
                d = make_centroid(all_better[tr], all_worse[tr])
                accs.append(pairwise_accuracy(all_better[te], all_worse[te], d))
            mean_a = np.mean(accs)
            std_a = np.std(accs)
            n_test = n - n_train
            print(f"    {n_train:>5d} train / {n_test:>5d} test: "
                  f"{mean_a:.1%} ± {std_a:.1%}")
            lc[str(n_train)] = {"mean": round(float(mean_a), 4),
                                 "std": round(float(std_a), 4),
                                 "n_test": n_test}
        results["learning_curve"] = lc

        # Main result: 50/50 split with permutation
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        tr, te = perm[:half], perm[half:]
        d = make_centroid(all_better[tr], all_worse[tr])
        main_oos = pairwise_accuracy(all_better[te], all_worse[te], d)

        print(f"\n  === MAIN: {half}/{n-half} split ===")
        print(f"  OOS: {main_oos:.1%}")

        print(f"  Permutation test (1000 shuffles)...")
        p_val = permutation_test(all_better[tr], all_worse[tr],
                                  all_better[te], all_worse[te],
                                  main_oos, n_perms=1000)
        print(f"  p = {p_val:.4f}")

        results["main_oos"] = round(float(main_oos), 4)
        results["main_perm_p"] = round(float(p_val), 4)
        results["main_train_n"] = int(half)
        results["main_test_n"] = int(n - half)

        # 10 random 50/50 splits for stability
        split_accs = []
        for rep in range(10):
            p2 = np.random.RandomState(200 + rep).permutation(n)
            tr2, te2 = p2[:half], p2[half:]
            d2 = make_centroid(all_better[tr2], all_worse[tr2])
            split_accs.append(pairwise_accuracy(all_better[te2], all_worse[te2], d2))
        print(f"  10 splits: {np.mean(split_accs):.1%} ± {np.std(split_accs):.1%}")
        results["splits_mean"] = round(float(np.mean(split_accs)), 4)
        results["splits_std"] = round(float(np.std(split_accs)), 4)

        all_results[model_name] = results
        del model, all_better, all_worse
        gc.collect()

    # Save
    out_path = OUT_DIR / "shp_local_scale_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: SHP at scale with local models")
    print(f"{'='*70}")
    for mname, r in all_results.items():
        short = mname.split("/")[-1]
        print(f"\n  {short}:")
        print(f"    Main OOS ({r['main_train_n']}/{r['main_test_n']}): "
              f"{r['main_oos']:.1%} (p={r['main_perm_p']:.4f})")
        print(f"    10 splits: {r['splits_mean']:.1%} ± {r['splits_std']:.1%}")
        for n_tr, v in r.get("learning_curve", {}).items():
            print(f"    {n_tr:>5s} train: {v['mean']:.1%} ± {v['std']:.1%}")


if __name__ == "__main__":
    main()
