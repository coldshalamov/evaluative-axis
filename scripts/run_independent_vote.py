#!/usr/bin/env python3
"""Test majority voting among the 9 warmth-independent terms.

From comprehensive analysis: 9 terms are independent of good on 2+ models.
All share the semantic property of restraint/discipline/self-control.
Question: does combining them beat careful alone?

Previous experiments showed L1 majority vote (all 10 children) FAILS because
8 warmth-biased children outvote 2 independent ones. This tests the opposite:
voting among ONLY the independent terms.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

INDEPENDENT_TERMS = [
    ("careful", "Careful", "Reckless"),
    ("thorough", "Thorough", "Superficial"),
    ("deliberate", "Deliberate", "Impulsive"),
    ("cautious", "Cautious", "Careless"),
    ("methodical", "Methodical", "Haphazard"),
    ("patient", "Patient", "Impatient"),
    ("prudent", "Prudent", "Reckless"),
    ("measured", "Measured", "Impulsive"),
    ("rigorous", "Rigorous", "Lax"),
]

REFERENCE = [
    ("good", "Good", "Bad"),
    ("careful_ref", "Careful", "Reckless"),
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

def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    cases = orig + warmth
    n = len(cases)
    n_orig = len(orig)

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases])

        axis_correct = {}
        axis_deltas = {}

        all_axes = INDEPENDENT_TERMS + [("good", "Good", "Bad")]
        for name, pos, neg in all_axes:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)
            deltas = []
            correct = []
            for i in range(n):
                d = float(np.dot(better_embs[i], axis) - np.dot(worse_embs[i], axis))
                deltas.append(d)
                correct.append(1 if d > 0 else 0)
            axis_deltas[name] = deltas
            axis_correct[name] = correct

        # Individual accuracies
        print(f"\nIndividual term accuracies:")
        for name, _, _ in INDEPENDENT_TERMS:
            acc = sum(axis_correct[name]) / n
            orig_acc = sum(axis_correct[name][i] for i in range(n_orig)) / n_orig
            warm_acc = sum(axis_correct[name][i] for i in range(n_orig, n)) / (n - n_orig)
            print(f"  {name:15s} {acc:5.0%}  (orig={orig_acc:.0%}, warm={warm_acc:.0%})")

        good_acc = sum(axis_correct["good"]) / n
        print(f"  {'good':15s} {good_acc:5.0%}  (reference)")

        # Majority vote among all 9
        vote_correct_9 = []
        for i in range(n):
            votes = sum(axis_correct[name][i] for name, _, _ in INDEPENDENT_TERMS)
            vote_correct_9.append(1 if votes > len(INDEPENDENT_TERMS) / 2 else 0)

        vote_acc_9 = sum(vote_correct_9) / n
        lo9, hi9 = wilson_ci(sum(vote_correct_9), n)
        orig_9 = sum(vote_correct_9[i] for i in range(n_orig)) / n_orig
        warm_9 = sum(vote_correct_9[i] for i in range(n_orig, n)) / (n - n_orig)

        # Majority vote among top-3 (careful, thorough, deliberate — 3/3 independent)
        top3 = ["careful", "thorough", "deliberate"]
        vote_correct_3 = []
        for i in range(n):
            votes = sum(axis_correct[name][i] for name in top3)
            vote_correct_3.append(1 if votes >= 2 else 0)

        vote_acc_3 = sum(vote_correct_3) / n
        lo3, hi3 = wilson_ci(sum(vote_correct_3), n)
        orig_3 = sum(vote_correct_3[i] for i in range(n_orig)) / n_orig
        warm_3 = sum(vote_correct_3[i] for i in range(n_orig, n)) / (n - n_orig)

        # Majority vote among top-5 (careful, thorough, deliberate, cautious, patient)
        top5 = ["careful", "thorough", "deliberate", "cautious", "patient"]
        vote_correct_5 = []
        for i in range(n):
            votes = sum(axis_correct[name][i] for name in top5)
            vote_correct_5.append(1 if votes >= 3 else 0)

        vote_acc_5 = sum(vote_correct_5) / n
        lo5, hi5 = wilson_ci(sum(vote_correct_5), n)
        orig_5 = sum(vote_correct_5[i] for i in range(n_orig)) / n_orig
        warm_5 = sum(vote_correct_5[i] for i in range(n_orig, n)) / (n - n_orig)

        # Sum of deltas (continuous scoring) among all 9
        sum_deltas = [sum(axis_deltas[name][i] for name, _, _ in INDEPENDENT_TERMS) for i in range(n)]
        sum_correct = [1 if d > 0 else 0 for d in sum_deltas]
        sum_acc = sum(sum_correct) / n
        lo_s, hi_s = wilson_ci(sum(sum_correct), n)
        orig_s = sum(sum_correct[i] for i in range(n_orig)) / n_orig
        warm_s = sum(sum_correct[i] for i in range(n_orig, n)) / (n - n_orig)

        careful_acc = sum(axis_correct["careful"]) / n
        lo_c, hi_c = wilson_ci(sum(axis_correct["careful"]), n)
        orig_c = sum(axis_correct["careful"][i] for i in range(n_orig)) / n_orig
        warm_c = sum(axis_correct["careful"][i] for i in range(n_orig, n)) / (n - n_orig)

        print(f"\n--- Combination strategies ---")
        print(f"  careful alone:         {careful_acc:5.0%} [{lo_c:.0%}, {hi_c:.0%}]  (orig={orig_c:.0%}, warm={warm_c:.0%})")
        print(f"  top-3 vote:            {vote_acc_3:5.0%} [{lo3:.0%}, {hi3:.0%}]  (orig={orig_3:.0%}, warm={warm_3:.0%})")
        print(f"  top-5 vote:            {vote_acc_5:5.0%} [{lo5:.0%}, {hi5:.0%}]  (orig={orig_5:.0%}, warm={warm_5:.0%})")
        print(f"  all-9 vote:            {vote_acc_9:5.0%} [{lo9:.0%}, {hi9:.0%}]  (orig={orig_9:.0%}, warm={warm_9:.0%})")
        print(f"  all-9 delta sum:       {sum_acc:5.0%} [{lo_s:.0%}, {hi_s:.0%}]  (orig={orig_s:.0%}, warm={warm_s:.0%})")
        print(f"  good (reference):      {good_acc:5.0%}")

        # Check r_good for each combination
        r_vote9, _ = pearsonr(vote_correct_9, axis_correct["good"])
        r_sum9, _ = pearsonr(sum_deltas, axis_deltas["good"])
        print(f"\n  r_good(9-vote): {r_vote9:+.2f}")
        print(f"  r_good(9-sum):  {r_sum9:+.2f}")

        # Per-case analysis: where does voting rescue careful's failures?
        careful_wrong = [i for i in range(n) if axis_correct["careful"][i] == 0]
        vote9_rescues = sum(1 for i in careful_wrong if vote_correct_9[i] == 1)
        careful_right = [i for i in range(n) if axis_correct["careful"][i] == 1]
        vote9_breaks = sum(1 for i in careful_right if vote_correct_9[i] == 0)
        print(f"\n  Careful wrong ({len(careful_wrong)} cases): 9-vote rescues {vote9_rescues}")
        print(f"  Careful right ({len(careful_right)} cases): 9-vote breaks  {vote9_breaks}")
        print(f"  Net: {vote9_rescues - vote9_breaks:+d}")

        all_results[short] = {
            "careful_alone": round(float(careful_acc), 3),
            "top3_vote": round(float(vote_acc_3), 3),
            "top5_vote": round(float(vote_acc_5), 3),
            "all9_vote": round(float(vote_acc_9), 3),
            "all9_sum": round(float(sum_acc), 3),
            "good": round(float(good_acc), 3),
            "net_rescue_9vote": vote9_rescues - vote9_breaks,
        }

        del model
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for method in ["careful_alone", "top3_vote", "top5_vote", "all9_vote", "all9_sum", "good"]:
        vals = [all_results[m][method] for m in all_results]
        print(f"  {method:20s}: {', '.join(f'{v:.0%}' for v in vals)}  mean={np.mean(vals):.0%}")

    out = ROOT / "notes/research_cycles/tree_decomposition/independent_vote_results.json"
    out.write_text(json.dumps({
        "experiment": "Majority vote among 9 warmth-independent terms",
        "date": "2026-06-28",
        "terms_tested": [name for name, _, _ in INDEPENDENT_TERMS],
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
