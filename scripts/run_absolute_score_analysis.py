#!/usr/bin/env python3
"""Absolute-score analysis: discrimination vs training-signal quality.

Our battery tests DISCRIMINATION (pairwise): can the axis pick A over B?
Training tests a different thing: does the axis give a positive score to
good responses and a negative score to bad responses?

"Good" might fail at discrimination (can't tell warm-but-wrong from correct-
but-firm) while still being a useful training signal (gives positive scores
to genuinely good responses). This test checks by computing absolute
projections of each response onto axis vectors, not just deltas.

Key question: Do the score DISTRIBUTIONS of better vs worse responses
overlap, or are they separable? If they're separable, the axis is a useful
training signal even when pairwise discrimination fails.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = [
    ("good", "Good", "Bad"),
    ("careful", "Careful", "Reckless"),
    ("restrained", "Restrained", "Unrestrained"),
    ("thorough", "Thorough", "Superficial"),
    ("kind", "Kind", "Cruel"),
    ("helpful", "Helpful", "Unhelpful"),
    ("honest", "Honest", "Dishonest"),
]


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - half), min(1, center + half)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def main():
    from sentence_transformers import SentenceTransformer

    battery = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    all_cases = battery + warmth

    firmness_cases = [c for c in battery if c["category"] != "anti_sycophancy"]
    warmth_cases = warmth
    syc_cases = [c for c in battery if c["category"] == "anti_sycophancy"]

    print(f"Total cases: {len(all_cases)}")
    print(f"  Firmness-biased: {len(firmness_cases)}")
    print(f"  Warmth-aligned: {len(warmth_cases)}")
    print(f"  Anti-sycophancy: {len(syc_cases)}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed all responses individually (not as conversations)
        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases]

        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        for axis_name, pos, neg in AXES:
            p_emb = embed_fn([pos])[0]
            n_emb = embed_fn([neg])[0]
            axis = (p_emb - n_emb) / (norm(p_emb - n_emb) + 1e-12)

            # Absolute scores for all responses
            better_scores = [float(np.dot(better_embs[i], axis)) for i in range(len(all_cases))]
            worse_scores = [float(np.dot(worse_embs[i], axis)) for i in range(len(all_cases))]

            # Pairwise accuracy (our standard test)
            pairwise_correct = sum(1 for b, w in zip(better_scores, worse_scores) if b > w)
            pairwise_acc = pairwise_correct / len(all_cases)

            # Absolute-score threshold test: what fraction of responses
            # would be scored "positive" (above median of all scores)?
            all_scores = better_scores + worse_scores
            median_score = np.median(all_scores)

            better_above_med = sum(1 for s in better_scores if s > median_score)
            worse_above_med = sum(1 for s in worse_scores if s > median_score)
            better_rate = better_above_med / len(better_scores)
            worse_rate = worse_above_med / len(worse_scores)

            # Cohen's d: effect size of better vs worse score distributions
            d = cohens_d(better_scores, worse_scores)

            # Overlap: what fraction of worse scores exceed the minimum better score?
            min_better = min(better_scores)
            max_worse = max(worse_scores)
            overlap_count = sum(1 for w in worse_scores if w > min_better)
            overlap_rate = overlap_count / len(worse_scores)

            # Score by case type
            splits = {
                "firmness": [i for i, c in enumerate(all_cases) if c in firmness_cases],
                "warmth": [i for i, c in enumerate(all_cases) if c in warmth_cases],
                "sycophancy": [i for i, c in enumerate(all_cases) if c in syc_cases],
            }

            split_results = {}
            for split_name, idxs in splits.items():
                if not idxs:
                    continue
                b_sc = [better_scores[i] for i in idxs]
                w_sc = [worse_scores[i] for i in idxs]
                pair_k = sum(1 for b, w in zip(b_sc, w_sc) if b > w)
                pair_a = pair_k / len(idxs)
                sd = cohens_d(b_sc, w_sc)
                split_results[split_name] = {
                    "n": len(idxs),
                    "pairwise_acc": round(pair_a, 3),
                    "cohens_d": round(sd, 3),
                    "better_mean": round(float(np.mean(b_sc)), 6),
                    "worse_mean": round(float(np.mean(w_sc)), 6),
                }

            model_results[axis_name] = {
                "pairwise_acc": round(pairwise_acc, 3),
                "cohens_d": round(d, 3),
                "better_mean": round(float(np.mean(better_scores)), 6),
                "worse_mean": round(float(np.mean(worse_scores)), 6),
                "better_std": round(float(np.std(better_scores)), 6),
                "worse_std": round(float(np.std(worse_scores)), 6),
                "above_median_better": round(better_rate, 3),
                "above_median_worse": round(worse_rate, 3),
                "overlap_rate": round(overlap_rate, 3),
                "splits": split_results,
            }

        # Print summary table
        print(f"\n{'Axis':12s} {'Pairwise':>8s} {'d':>6s} {'B_mean':>8s} {'W_mean':>8s} {'B>med':>6s} {'W>med':>6s} {'Overlap':>8s}")
        print("-" * 75)
        for axis_name, _, _ in AXES:
            r = model_results[axis_name]
            print(f"{axis_name:12s} {r['pairwise_acc']:8.0%} {r['cohens_d']:6.3f} "
                  f"{r['better_mean']:8.5f} {r['worse_mean']:8.5f} "
                  f"{r['above_median_better']:6.0%} {r['above_median_worse']:6.0%} "
                  f"{r['overlap_rate']:8.0%}")

        # Per-split breakdown for key axes
        for axis_name in ["good", "careful", "restrained"]:
            r = model_results[axis_name]
            print(f"\n  {axis_name} by content split:")
            for split_name, sr in r["splits"].items():
                print(f"    {split_name:12s} n={sr['n']:2d}  pairwise={sr['pairwise_acc']:.0%}  "
                      f"d={sr['cohens_d']:+.3f}  B_mean={sr['better_mean']:+.5f}  W_mean={sr['worse_mean']:+.5f}")

        all_results[short] = model_results

        del model
        gc.collect()

    # Cross-model summary
    print(f"\n{'='*80}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    print(f"\n--- Pairwise accuracy vs Cohen's d ---")
    print(f"If d is near zero but pairwise accuracy is low, the distributions")
    print(f"overlap heavily and the axis can't discriminate OR train.")
    print(f"If d is positive but pairwise accuracy is low, there's a mean shift")
    print(f"but too much overlap for pairwise -- could still work for training.")
    print()
    print(f"{'Axis':12s}", end="")
    for m in all_results:
        print(f" {m:>25s}", end="")
    print()
    print(f"{'':12s}", end="")
    for m in all_results:
        print(f" {'pair':>8s} {'d':>8s} {'split':>8s}", end="")
    print()
    print("-" * 90)

    for axis_name, _, _ in AXES:
        print(f"{axis_name:12s}", end="")
        for m in all_results:
            r = all_results[m][axis_name]
            # Split: firmness d vs warmth d
            fd = r["splits"].get("firmness", {}).get("cohens_d", 0)
            wd = r["splits"].get("warmth", {}).get("cohens_d", 0)
            print(f" {r['pairwise_acc']:8.0%} {r['cohens_d']:+8.3f} {fd:+4.2f}/{wd:+4.2f}", end="")
        print()

    # Save results
    out = ROOT / "notes/research_cycles/absolute_score_analysis"
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / "absolute_score_results.json"
    outfile.write_text(json.dumps({
        "experiment": "Absolute score analysis: discrimination vs training signal",
        "date": "2026-06-28",
        "n_cases": len(all_cases),
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
