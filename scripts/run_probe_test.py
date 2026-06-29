#!/usr/bin/env python3
"""Two follow-up experiments on HH-RLHF data:

1. Response-only cosine: Strip the prompt, embed just the assistant response.
   Tests whether the shared prompt was drowning out the signal.

2. Linear probe: Logistic regression on emb(chosen) - emb(rejected).
   Tests whether quality info exists ANYWHERE in the embedding, not just
   along the "Good" axis. This is the decisive test.

If the probe hits chance too, the negative result is airtight.
If the probe hits 65%+, quality info IS encoded — cosine-to-anchor just can't reach it.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

CACHE_DIR = Path(__file__).resolve().parents[1] / "notes" / "research_cycles" / "real_data_test"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TOP_ANCHORS = ["Good", "Helpful", "Honest", "Careful", "Genuine", "Thoughtful"]


def bootstrap_ci(scores, n_boot=5000, ci=0.95):
    arr = np.array(scores, dtype=float)
    n = len(arr)
    boot_means = np.array([np.mean(np.random.choice(arr, size=n, replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha))


def main():
    from sentence_transformers import SentenceTransformer

    # Load cached data
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

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=True,
                                           convert_to_numpy=True, batch_size=32)

        # =============================================
        # PART 1: Response-only cosine
        # =============================================
        print("\n--- PART 1: Response-only cosine ---")
        print("  Embedding chosen responses (response only)...")
        chosen_resp_embs = embed([p['chosen'] for p in pairs])
        print("  Embedding rejected responses (response only)...")
        rejected_resp_embs = embed([p['rejected'] for p in pairs])

        for i in range(n):
            chosen_resp_embs[i] = chosen_resp_embs[i] / (norm(chosen_resp_embs[i]) + 1e-12)
            rejected_resp_embs[i] = rejected_resp_embs[i] / (norm(rejected_resp_embs[i]) + 1e-12)

        anchor_embs = {}
        for name in TOP_ANCHORS:
            e = embed([name])[0]
            anchor_embs[name] = e / (norm(e) + 1e-12)

        print(f"\n  Response-only cosine results:")
        print(f"  {'Anchor':15s} {'Acc':>6s} {'CI':>13s}")
        print(f"  {'-'*40}")
        resp_only_results = {}
        for name in TOP_ANCHORS:
            a = anchor_embs[name]
            correct = [float(np.dot(chosen_resp_embs[i], a)) > float(np.dot(rejected_resp_embs[i], a))
                       for i in range(n)]
            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)
            print(f"  {name:15s} {acc:5.1%} [{lo:.0%},{hi:.0%}]")
            resp_only_results[name] = acc

        # =============================================
        # PART 2: With-prompt cosine (for comparison)
        # =============================================
        print("\n--- PART 2: With-prompt cosine (for comparison) ---")
        print("  Embedding chosen responses (with prompt)...")
        chosen_full_embs = embed([f"User: {p['prompt']}\nAssistant: {p['chosen']}" for p in pairs])
        print("  Embedding rejected responses (with prompt)...")
        rejected_full_embs = embed([f"User: {p['prompt']}\nAssistant: {p['rejected']}" for p in pairs])

        for i in range(n):
            chosen_full_embs[i] = chosen_full_embs[i] / (norm(chosen_full_embs[i]) + 1e-12)
            rejected_full_embs[i] = rejected_full_embs[i] / (norm(rejected_full_embs[i]) + 1e-12)

        print(f"\n  With-prompt cosine results:")
        print(f"  {'Anchor':15s} {'Acc':>6s} {'CI':>13s}")
        print(f"  {'-'*40}")
        full_results = {}
        for name in TOP_ANCHORS:
            a = anchor_embs[name]
            correct = [float(np.dot(chosen_full_embs[i], a)) > float(np.dot(rejected_full_embs[i], a))
                       for i in range(n)]
            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)
            print(f"  {name:15s} {acc:5.1%} [{lo:.0%},{hi:.0%}]")
            full_results[name] = acc

        # How much do the pair embeddings overlap?
        overlaps = [float(np.dot(chosen_full_embs[i], rejected_full_embs[i])) for i in range(n)]
        print(f"\n  Embedding overlap (cos between chosen & rejected):")
        print(f"    With prompt:    mean={np.mean(overlaps):.4f}  median={np.median(overlaps):.4f}")
        overlaps_resp = [float(np.dot(chosen_resp_embs[i], rejected_resp_embs[i])) for i in range(n)]
        print(f"    Response only:  mean={np.mean(overlaps_resp):.4f}  median={np.median(overlaps_resp):.4f}")

        # =============================================
        # PART 3: Linear probe (the decisive test)
        # =============================================
        print("\n--- PART 3: Linear probe ---")

        diff_resp = chosen_resp_embs - rejected_resp_embs
        diff_full = chosen_full_embs - rejected_full_embs

        X_resp = np.vstack([diff_resp, -diff_resp])
        X_full = np.vstack([diff_full, -diff_full])
        y_bal = np.concatenate([np.ones(n), np.zeros(n)])

        cv_bal = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Response-only probe
        clf_resp = LogisticRegression(max_iter=1000, C=1.0)
        scores_resp = cross_val_score(clf_resp, X_resp, y_bal, cv=cv_bal, scoring='accuracy')
        resp_mean = scores_resp.mean()
        resp_std = scores_resp.std()

        # Full (with-prompt) probe
        clf_full = LogisticRegression(max_iter=1000, C=1.0)
        scores_full = cross_val_score(clf_full, X_full, y_bal, cv=cv_bal, scoring='accuracy')
        full_mean = scores_full.mean()
        full_std = scores_full.std()

        print(f"\n  Linear probe accuracy (10-fold CV):")
        print(f"    Response only:  {resp_mean:.1%} +/- {resp_std:.1%}")
        print(f"    With prompt:    {full_mean:.1%} +/- {full_std:.1%}")
        print(f"    Chance:         50.0%")

        # Also try different regularization strengths
        print(f"\n  Regularization sweep (response only):")
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(max_iter=1000, C=C)
            sc = cross_val_score(clf, X_resp, y_bal, cv=cv_bal, scoring='accuracy')
            print(f"    C={C:6.2f}: {sc.mean():.1%} +/- {sc.std():.1%}")

        # What directions does the probe find?
        clf_final = LogisticRegression(max_iter=1000, C=1.0)
        clf_final.fit(X_resp, y_bal)
        w = clf_final.coef_[0]
        w_norm = w / (norm(w) + 1e-12)

        # How does the learned direction compare to our anchor words?
        print(f"\n  Learned probe direction vs anchor words:")
        for name in TOP_ANCHORS:
            cos_sim = float(np.dot(w_norm, anchor_embs[name]))
            print(f"    cos(probe, '{name}'): {cos_sim:+.4f}")

        # Does the probe direction correspond to length?
        len_diffs = np.array([len(p['chosen']) - len(p['rejected']) for p in pairs])
        probe_scores = diff_resp @ w_norm  # project differences onto probe direction
        corr = np.corrcoef(len_diffs, probe_scores)[0, 1]
        print(f"\n  Correlation(probe_score, length_diff): {corr:.4f}")

        results[short] = {
            "resp_only_cosine": resp_only_results,
            "full_cosine": full_results,
            "probe_resp": {"mean": float(resp_mean), "std": float(resp_std),
                           "folds": [float(s) for s in scores_resp]},
            "probe_full": {"mean": float(full_mean), "std": float(full_std),
                           "folds": [float(s) for s in scores_full]},
        }

        del model
        gc.collect()

    # =============================================
    # CROSS-MODEL SUMMARY
    # =============================================
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'':15s}", end="")
    for short in results:
        print(f" {short:>20s}", end="")
    print()

    print(f"  {'Probe (resp)':15s}", end="")
    for short, r in results.items():
        print(f" {r['probe_resp']['mean']:19.1%}", end="")
    print()

    print(f"  {'Probe (full)':15s}", end="")
    for short, r in results.items():
        print(f" {r['probe_full']['mean']:19.1%}", end="")
    print()

    print(f"  {'cos(Good) resp':15s}", end="")
    for short, r in results.items():
        print(f" {r['resp_only_cosine']['Good']:19.1%}", end="")
    print()

    print(f"  {'cos(Good) full':15s}", end="")
    for short, r in results.items():
        print(f" {r['full_cosine']['Good']:19.1%}", end="")
    print()

    # Save results
    with open(CACHE_DIR / "probe_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {CACHE_DIR / 'probe_results.json'}")


if __name__ == "__main__":
    main()
