#!/usr/bin/env python3
"""The tree as a diagnostic profile, not a voting system.

For each case, compute the projection delta (better - worse) on:
- Level 0: good/bad
- Level 1: careful, honest, helpful, thorough, restrained
- Level 2: decompose the L1 terms further

Report the PROFILE for each case type. The question isn't "does it vote right?"
but "does the profile distinguish case types in a useful way?"

Also: compare to random baselines for OR/voting to establish what's real signal
vs threshold artifact.
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

TREE = {
    "good":       ("Good", "Bad"),
    "careful":    ("Careful", "Reckless"),
    "honest":     ("Honest", "Dishonest"),
    "helpful":    ("Helpful", "Unhelpful"),
    "thorough":   ("Thorough", "Superficial"),
    "restrained": ("Restrained", "Unrestrained"),
}

# Level 2: decompose L1 terms further
LEVEL2 = {
    "accurate":   ("Accurate", "Inaccurate"),
    "precise":    ("Precise", "Vague"),
    "logical":    ("Logical", "Illogical"),
    "truthful":   ("Truthful", "Deceptive"),
    "sincere":    ("Sincere", "Insincere"),
    "clear":      ("Clear", "Confusing"),
    "useful":     ("Useful", "Useless"),
    "kind":       ("Kind", "Cruel"),
    "complete":   ("Complete", "Incomplete"),
    "measured":   ("Measured", "Extreme"),
    "moderate":   ("Moderate", "Excessive"),
    "prudent":    ("Prudent", "Imprudent"),
}


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def main():
    from sentence_transformers import SentenceTransformer

    battery = read_jsonl(BATTERY)
    warmth_cases = read_jsonl(WARMTH)
    all_cases = battery + warmth_cases
    n = len(all_cases)

    labels = []
    for c in battery:
        labels.append("sycophancy" if c["category"] == "anti_sycophancy" else "firmness")
    for c in warmth_cases:
        labels.append("warmth")

    all_axes = {**TREE, **LEVEL2}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        # Compute deltas for each axis
        axis_deltas = {}
        axis_dirs = {}
        for name, (pos, neg) in all_axes.items():
            p = embed([pos])[0]
            nn = embed([neg])[0]
            d = (p - nn) / (norm(p - nn) + 1e-12)
            axis_dirs[name] = d
            deltas = [float(np.dot(better_embs[i], d)) - float(np.dot(worse_embs[i], d))
                      for i in range(n)]
            axis_deltas[name] = deltas

        # DIAGNOSTIC PROFILE: mean delta by case type
        print(f"\n  MEAN DELTA BY CASE TYPE (positive = better response scores higher)")
        print(f"  {'Axis':12s} {'Firm':>8s} {'Warm':>8s} {'Syc':>8s}  {'Firm d':>8s} {'Warm d':>8s}")
        print(f"  {'-'*60}")

        firm_idx = [i for i in range(n) if labels[i] == "firmness"]
        warm_idx = [i for i in range(n) if labels[i] == "warmth"]
        syc_idx = [i for i in range(n) if labels[i] == "sycophancy"]

        for name in ["good", "careful", "honest", "helpful", "thorough", "restrained"]:
            d = axis_deltas[name]
            firm_mean = np.mean([d[i] for i in firm_idx])
            warm_mean = np.mean([d[i] for i in warm_idx])
            syc_mean = np.mean([d[i] for i in syc_idx])

            # Cohen's d: better vs worse projection (positive = correct direction)
            better_projs = [float(np.dot(better_embs[i], axis_dirs[name])) for i in range(n)]
            worse_projs = [float(np.dot(worse_embs[i], axis_dirs[name])) for i in range(n)]
            firm_d = cohens_d([better_projs[i] for i in firm_idx], [worse_projs[i] for i in firm_idx])
            warm_d = cohens_d([better_projs[i] for i in warm_idx], [worse_projs[i] for i in warm_idx])

            print(f"  {name:12s} {firm_mean:+8.4f} {warm_mean:+8.4f} {syc_mean:+8.4f}  {firm_d:+8.3f} {warm_d:+8.3f}")

        print(f"\n  LEVEL 2:")
        for name in LEVEL2:
            d = axis_deltas[name]
            firm_mean = np.mean([d[i] for i in firm_idx])
            warm_mean = np.mean([d[i] for i in warm_idx])
            syc_mean = np.mean([d[i] for i in syc_idx])
            better_projs = [float(np.dot(better_embs[i], axis_dirs[name])) for i in range(n)]
            worse_projs = [float(np.dot(worse_embs[i], axis_dirs[name])) for i in range(n)]
            firm_d = cohens_d([better_projs[i] for i in firm_idx], [worse_projs[i] for i in firm_idx])
            warm_d = cohens_d([better_projs[i] for i in warm_idx], [worse_projs[i] for i in warm_idx])
            print(f"  {name:12s} {firm_mean:+8.4f} {warm_mean:+8.4f} {syc_mean:+8.4f}  {firm_d:+8.3f} {warm_d:+8.3f}")

        # RANDOM BASELINE for OR
        print(f"\n  RANDOM OR BASELINES:")
        rng = np.random.RandomState(42)
        for k_axes in [3, 5, 7]:
            random_accs = []
            for trial in range(1000):
                random_outcomes = rng.random((k_axes, n)) > 0.5
                any_correct = np.any(random_outcomes, axis=0)
                random_accs.append(np.mean(any_correct))
            print(f"    {k_axes} random axes, OR: {np.mean(random_accs):.1%} (expected {1 - 0.5**k_axes:.1%})")

        # Actual OR accuracy for comparison
        actual_or = []
        for i in range(n):
            correct = any(axis_deltas[name][i] > 0
                         for name in ["careful", "honest", "helpful", "thorough", "restrained"])
            actual_or.append(correct)
        print(f"    5 tree axes, actual OR: {np.mean(actual_or):.1%}")

        # What about scoring: sum of deltas across tree (not directions, but scores)
        print(f"\n  SUM-OF-DELTAS SCORING:")
        sum_deltas = [sum(axis_deltas[name][i] for name in ["careful", "honest", "helpful", "thorough", "restrained"])
                      for i in range(n)]
        sum_correct = [d > 0 for d in sum_deltas]
        sum_acc = sum(sum_correct) / n
        sum_firm = sum(sum_correct[i] for i in firm_idx) / len(firm_idx)
        sum_warm = sum(sum_correct[i] for i in warm_idx) / len(warm_idx)
        sum_syc = sum(sum_correct[i] for i in syc_idx) / len(syc_idx)
        print(f"    sum(5 deltas) > 0: {sum_acc:.0%}  f={sum_firm:.0%}  w={sum_warm:.0%}  s={sum_syc:.0%}")

        # Weighted sum: weight by each axis's individual accuracy
        axis_accs = {}
        for name in ["careful", "honest", "helpful", "thorough", "restrained"]:
            axis_accs[name] = sum(1 for i in range(n) if axis_deltas[name][i] > 0) / n

        weighted_deltas = [sum(axis_accs[name] * axis_deltas[name][i]
                              for name in ["careful", "honest", "helpful", "thorough", "restrained"])
                          for i in range(n)]
        w_correct = [d > 0 for d in weighted_deltas]
        w_acc = sum(w_correct) / n
        w_firm = sum(w_correct[i] for i in firm_idx) / len(firm_idx)
        w_warm = sum(w_correct[i] for i in warm_idx) / len(warm_idx)
        w_syc = sum(w_correct[i] for i in syc_idx) / len(syc_idx)
        print(f"    accuracy-weighted sum: {w_acc:.0%}  f={w_firm:.0%}  w={w_warm:.0%}  s={w_syc:.0%}")

        # MIN of deltas (worst axis determines quality)
        min_deltas = [min(axis_deltas[name][i] for name in ["careful", "honest", "helpful", "thorough", "restrained"])
                      for i in range(n)]
        min_correct = [d > 0 for d in min_deltas]
        min_acc = sum(min_correct) / n
        min_firm = sum(min_correct[i] for i in firm_idx) / len(firm_idx)
        min_warm = sum(min_correct[i] for i in warm_idx) / len(warm_idx)
        min_syc = sum(min_correct[i] for i in syc_idx) / len(syc_idx)
        print(f"    min(5 deltas) > 0: {min_acc:.0%}  f={min_firm:.0%}  w={min_warm:.0%}  s={min_syc:.0%}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
