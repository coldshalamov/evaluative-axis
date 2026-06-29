#!/usr/bin/env python3
"""Composite training signal: can independent scoring + summing produce
a training signal with positive Cohen's d across ALL content splits?

Individual axes have complementary weaknesses:
- careful: d > 0 on firmness, d < 0 on warmth
- restrained: d > 0 on firmness, d ~ 0 on warmth
- good: d < 0 on firmness, d > 0 on warmth

Can we construct a composite that is positive on both splits? This tests
the scalar-plus-basis idea from §5.5: use multiple independent scores
to build a training signal that covers multiple quality dimensions.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from itertools import combinations

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
    ("precise", "Precise", "Imprecise"),
    ("deliberate", "Deliberate", "Impulsive"),
    ("measured", "Measured", "Extreme"),
]


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
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

    firmness_idx = [i for i, c in enumerate(all_cases) if c in [x for x in battery if x["category"] != "anti_sycophancy"]]
    warmth_idx = [i for i, c in enumerate(all_cases) if c in warmth]
    syc_idx = [i for i, c in enumerate(all_cases) if c.get("category") == "anti_sycophancy"]

    n = len(all_cases)
    print(f"Cases: {n} (firmness={len(firmness_idx)}, warmth={len(warmth_idx)}, syc={len(syc_idx)})")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        # Compute scores on all axes
        axis_scores = {}
        for axis_name, pos, neg in AXES:
            p_emb = embed_fn([pos])[0]
            n_emb = embed_fn([neg])[0]
            axis = (p_emb - n_emb) / (norm(p_emb - n_emb) + 1e-12)
            b_scores = [float(np.dot(better_embs[i], axis)) for i in range(n)]
            w_scores = [float(np.dot(worse_embs[i], axis)) for i in range(n)]
            axis_scores[axis_name] = (b_scores, w_scores)

        # Individual axis d values by split
        print(f"\n--- Individual axes ---")
        print(f"{'Axis':12s} {'All d':>7s} {'Firm d':>7s} {'Warm d':>7s} {'Syc d':>7s} {'Pair':>6s}")
        print("-" * 55)

        for axis_name, _, _ in AXES:
            b, w = axis_scores[axis_name]
            d_all = cohens_d(b, w)
            d_firm = cohens_d([b[i] for i in firmness_idx], [w[i] for i in firmness_idx])
            d_warm = cohens_d([b[i] for i in warmth_idx], [w[i] for i in warmth_idx])
            d_syc = cohens_d([b[i] for i in syc_idx], [w[i] for i in syc_idx])
            pair = sum(1 for bb, ww in zip(b, w) if bb > ww) / n
            print(f"{axis_name:12s} {d_all:+7.3f} {d_firm:+7.3f} {d_warm:+7.3f} {d_syc:+7.3f} {pair:6.0%}")

        # Test composites: sum of z-scored individual axes
        # Z-score each axis to put them on common scale
        axis_z = {}
        for axis_name in axis_scores:
            b, w = axis_scores[axis_name]
            all_sc = np.array(b + w)
            mu, sigma = np.mean(all_sc), np.std(all_sc)
            if sigma == 0:
                sigma = 1
            axis_z[axis_name] = (
                [(s - mu) / sigma for s in b],
                [(s - mu) / sigma for s in w],
            )

        # Try all 2-3 axis combinations from the independent terms
        independent_terms = ["careful", "restrained", "thorough", "deliberate", "measured", "precise"]
        warmth_terms = ["good", "kind", "helpful", "honest"]

        print(f"\n--- Composite scores (sum of z-scored axes) ---")
        print(f"{'Composite':35s} {'All d':>7s} {'Firm d':>7s} {'Warm d':>7s} {'Syc d':>7s} {'Pair':>6s}")
        print("-" * 75)

        composites = []

        # Test combinations of 2-3 independent terms
        for size in [2, 3]:
            for combo in combinations(independent_terms, size):
                b_sum = np.zeros(n)
                w_sum = np.zeros(n)
                for ax in combo:
                    bz, wz = axis_z[ax]
                    b_sum += np.array(bz)
                    w_sum += np.array(wz)

                d_all = cohens_d(b_sum, w_sum)
                d_firm = cohens_d(b_sum[firmness_idx], w_sum[firmness_idx])
                d_warm = cohens_d(b_sum[warmth_idx], w_sum[warmth_idx])
                d_syc = cohens_d(b_sum[syc_idx], w_sum[syc_idx])
                pair = sum(1 for bb, ww in zip(b_sum, w_sum) if bb > ww) / n
                name = "+".join(combo)

                composites.append({
                    "name": name,
                    "d_all": d_all,
                    "d_firm": d_firm,
                    "d_warm": d_warm,
                    "d_syc": d_syc,
                    "pair": pair,
                })

        # Also test mixed: independent + warmth (to see if mixing helps)
        for ind in independent_terms[:3]:
            for wrm in warmth_terms[:2]:
                b_sum = np.array(axis_z[ind][0]) + np.array(axis_z[wrm][0])
                w_sum = np.array(axis_z[ind][1]) + np.array(axis_z[wrm][1])
                d_all = cohens_d(b_sum, w_sum)
                d_firm = cohens_d(b_sum[firmness_idx], w_sum[firmness_idx])
                d_warm = cohens_d(b_sum[warmth_idx], w_sum[warmth_idx])
                d_syc = cohens_d(b_sum[syc_idx], w_sum[syc_idx])
                pair = sum(1 for bb, ww in zip(b_sum, w_sum) if bb > ww) / n
                name = f"{ind}+{wrm}"
                composites.append({
                    "name": name,
                    "d_all": d_all,
                    "d_firm": d_firm,
                    "d_warm": d_warm,
                    "d_syc": d_syc,
                    "pair": pair,
                })

        # Sort by minimum d across splits (want ALL positive)
        composites.sort(key=lambda x: min(x["d_firm"], x["d_warm"]), reverse=True)

        for c in composites[:25]:
            min_d = min(c["d_firm"], c["d_warm"])
            flag = " ***" if min_d > 0 else ""
            print(f"{c['name']:35s} {c['d_all']:+7.3f} {c['d_firm']:+7.3f} "
                  f"{c['d_warm']:+7.3f} {c['d_syc']:+7.3f} {c['pair']:6.0%}{flag}")

        # Count how many composites have ALL splits positive
        all_positive = [c for c in composites if c["d_firm"] > 0 and c["d_warm"] > 0]
        print(f"\nComposites with d > 0 on BOTH firmness and warmth: {len(all_positive)}/{len(composites)}")
        for c in all_positive:
            print(f"  {c['name']:35s} firm={c['d_firm']:+.3f} warm={c['d_warm']:+.3f} "
                  f"syc={c['d_syc']:+.3f} pair={c['pair']:.0%}")

        all_results[short] = {
            "composites_all_positive": [c["name"] for c in all_positive],
            "best_composite": composites[0] if composites else None,
        }

        del model
        gc.collect()

    # Save
    out = ROOT / "notes/research_cycles/absolute_score_analysis/composite_training_signal.json"
    out.write_text(json.dumps({
        "experiment": "Composite training signal analysis",
        "date": "2026-06-28",
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
