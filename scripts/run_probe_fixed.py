#!/usr/bin/env python3
"""Linear probe with proper cross-validation (no data leakage).

The first probe run doubled data with diff/-diff but StratifiedKFold could
split a pair's two halves across train/test — free accuracy. This version
uses GroupKFold to keep each pair's diff and -diff in the same fold.

Also adds: PCA dimensionality check, null distribution via label shuffle.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold

CACHE_DIR = Path(__file__).resolve().parents[1] / "notes" / "research_cycles" / "real_data_test"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TOP_ANCHORS = ["Good", "Helpful", "Honest", "Careful", "Genuine", "Thoughtful"]


def embed_or_load(model_name, pairs, cache_dir):
    """Embed responses or load from cache."""
    short = model_name.split("/")[-1]
    cache_file = cache_dir / f"embeddings_{short}.npz"

    if cache_file.exists():
        print(f"  Loading cached embeddings for {short}...")
        data = np.load(cache_file)
        return data["chosen"], data["rejected"]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embed = lambda texts: model.encode(texts, show_progress_bar=True,
                                       convert_to_numpy=True, batch_size=32)

    print(f"  Embedding chosen responses...")
    chosen = embed([p['chosen'] for p in pairs])
    print(f"  Embedding rejected responses...")
    rejected = embed([p['rejected'] for p in pairs])

    n = len(pairs)
    for i in range(n):
        chosen[i] = chosen[i] / (norm(chosen[i]) + 1e-12)
        rejected[i] = rejected[i] / (norm(rejected[i]) + 1e-12)

    np.savez(cache_file, chosen=chosen, rejected=rejected)
    print(f"  Cached to {cache_file}")

    del model
    gc.collect()

    return chosen, rejected


def main():
    from sentence_transformers import SentenceTransformer

    with open(CACHE_DIR / "hh_rlhf_sample.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)

    n = len(pairs)
    print(f"Dataset: {n} preference pairs from Anthropic HH-RLHF\n")
    np.random.seed(42)

    groups = np.concatenate([np.arange(n), np.arange(n)])
    y_bal = np.concatenate([np.ones(n), np.zeros(n)])
    cv = GroupKFold(n_splits=10)

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        chosen_embs, rejected_embs = embed_or_load(model_name, pairs, CACHE_DIR)

        diff = chosen_embs - rejected_embs
        X = np.vstack([diff, -diff])

        # --- GroupKFold probe (no leakage) ---
        print("\n--- Linear probe (GroupKFold, no leakage) ---")
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(max_iter=1000, C=C)
            scores = cross_val_score(clf, X, y_bal, cv=cv, groups=groups, scoring='accuracy')
            print(f"  C={C:5.2f}: {scores.mean():.1%} +/- {scores.std():.1%}  folds: {[f'{s:.1%}' for s in scores]}")

        # --- Null distribution: shuffle labels 20 times ---
        print("\n--- Null distribution (shuffled labels) ---")
        null_accs = []
        for trial in range(20):
            y_shuf = y_bal.copy()
            perm = np.random.permutation(n)
            y_shuf[:n] = y_bal[:n][perm]
            y_shuf[n:] = 1 - y_shuf[:n]  # keep diff/-diff label symmetry
            clf = LogisticRegression(max_iter=1000, C=1.0)
            sc = cross_val_score(clf, X, y_shuf, cv=cv, groups=groups, scoring='accuracy')
            null_accs.append(sc.mean())
        print(f"  Null mean: {np.mean(null_accs):.1%} +/- {np.std(null_accs):.1%}")
        print(f"  Null range: [{np.min(null_accs):.1%}, {np.max(null_accs):.1%}]")

        # --- Probe direction analysis ---
        clf_final = LogisticRegression(max_iter=1000, C=1.0)
        clf_final.fit(X, y_bal)
        w = clf_final.coef_[0]
        w_norm = w / (norm(w) + 1e-12)

        model_for_anchors = SentenceTransformer(model_name, trust_remote_code=True)
        anchor_embs = {}
        for name in TOP_ANCHORS:
            e = model_for_anchors.encode([name], convert_to_numpy=True)[0]
            anchor_embs[name] = e / (norm(e) + 1e-12)
        del model_for_anchors
        gc.collect()

        print("\n  Probe direction vs anchors:")
        for name in TOP_ANCHORS:
            cos_sim = float(np.dot(w_norm, anchor_embs[name]))
            print(f"    cos(probe, '{name}'): {cos_sim:+.4f}")

        # Length correlation
        len_diffs = np.array([len(p['chosen']) - len(p['rejected']) for p in pairs])
        probe_scores = diff @ w_norm
        corr = np.corrcoef(len_diffs, probe_scores)[0, 1]
        print(f"\n  Correlation(probe_score, length_diff): {corr:.4f}")

        # Cosine overlap between chosen and rejected
        overlaps = [float(np.dot(chosen_embs[i], rejected_embs[i])) for i in range(n)]
        print(f"  Embedding overlap: mean={np.mean(overlaps):.4f} median={np.median(overlaps):.4f}")

        # What fraction of variance is in top PCs?
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, diff.shape[1]))
        pca.fit(diff)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"\n  PCA on diff vectors:")
        for k in [1, 5, 10, 20, 50]:
            if k <= len(cumvar):
                print(f"    Top {k:2d} PCs: {cumvar[k-1]:.1%} variance")

        # Probe on top-k PCs
        print("\n  Probe on PCA-reduced diff vectors (C=1.0):")
        for k in [5, 10, 20, 50]:
            if k <= diff.shape[1]:
                diff_pca = PCA(n_components=k).fit_transform(diff)
                X_pca = np.vstack([diff_pca, -diff_pca])
                clf_pca = LogisticRegression(max_iter=1000, C=1.0)
                sc = cross_val_score(clf_pca, X_pca, y_bal, cv=cv, groups=groups, scoring='accuracy')
                print(f"    k={k:3d}: {sc.mean():.1%} +/- {sc.std():.1%}")

        # Recompute C=1.0 score for results dict
        clf_c1 = LogisticRegression(max_iter=1000, C=1.0)
        sc_c1 = cross_val_score(clf_c1, X, y_bal, cv=cv, groups=groups, scoring='accuracy')
        results[short] = {
            "probe_C1": float(sc_c1.mean()),
            "probe_C1_std": float(sc_c1.std()),
            "null_mean": float(np.mean(null_accs)),
            "null_std": float(np.std(null_accs)),
            "length_corr": float(corr),
            "overlap_mean": float(np.mean(overlaps)),
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\n  {'Model':25s} {'Probe':>6s} {'Null':>6s} {'len_corr':>9s} {'overlap':>8s}")
    for short, r in results.items():
        print(f"  {short:25s} {r['probe_C1']:5.1%}  {r['null_mean']:5.1%}  {r['length_corr']:+8.4f}  {r['overlap_mean']:7.4f}")

    with open(CACHE_DIR / "probe_fixed_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {CACHE_DIR / 'probe_fixed_results.json'}")


if __name__ == "__main__":
    main()
