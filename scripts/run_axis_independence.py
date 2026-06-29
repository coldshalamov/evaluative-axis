#!/usr/bin/env python3
"""How many independent quality dimensions exist in these embedding models?

Compute pairwise cosine similarities between axis DIRECTIONS (not words)
to find which axes are genuinely independent vs duplicates.

Then test aggregate scoring with balanced matrices — equal numbers of
warmth-leaning and competence-leaning terms — to see if the aggregate
cancels the biases.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
EXP_DIR = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# Candidate axes — a broad matrix
CANDIDATES = {
    # Competence/rigor cluster
    "careful":    ("Careful", "Reckless"),
    "precise":    ("Precise", "Vague"),
    "rigorous":   ("Rigorous", "Lax"),
    "logical":    ("Logical", "Illogical"),
    "accurate":   ("Accurate", "Inaccurate"),
    # Restraint/discipline cluster
    "restrained": ("Restrained", "Unrestrained"),
    "measured":   ("Measured", "Extreme"),
    "moderate":   ("Moderate", "Excessive"),
    "prudent":    ("Prudent", "Imprudent"),
    # Integrity cluster
    "honest":     ("Honest", "Dishonest"),
    "truthful":   ("Truthful", "Deceptive"),
    "sincere":    ("Sincere", "Insincere"),
    "fair":       ("Fair", "Unfair"),
    # Utility/warmth cluster
    "helpful":    ("Helpful", "Unhelpful"),
    "kind":       ("Kind", "Cruel"),
    "respectful": ("Respectful", "Disrespectful"),
    "clear":      ("Clear", "Confusing"),
    # Completeness
    "thorough":   ("Thorough", "Superficial"),
    "complete":   ("Complete", "Incomplete"),
    # Reference
    "good":       ("Good", "Bad"),
}


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

    firm_idx = [i for i in range(n) if labels[i] == "firmness"]
    warm_idx = [i for i in range(n) if labels[i] == "warmth"]
    syc_idx = [i for i in range(n) if labels[i] == "sycophancy"]

    # OOS
    oos = []
    for f in sorted(EXP_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                oos.append(json.loads(line))
    n_oos = len(oos)

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_in = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_in = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])
        better_oos = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in oos])
        worse_oos = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in oos])

        # Compute axis directions and deltas
        dirs = {}
        deltas_in = {}
        deltas_oos = {}
        firm_d = {}
        warm_d = {}

        for name, (pos, neg) in CANDIDATES.items():
            p = embed([pos])[0]
            nn = embed([neg])[0]
            d = (p - nn) / (norm(p - nn) + 1e-12)
            dirs[name] = d
            deltas_in[name] = [float(np.dot(better_in[i], d)) - float(np.dot(worse_in[i], d))
                               for i in range(n)]
            deltas_oos[name] = [float(np.dot(better_oos[i], d)) - float(np.dot(worse_oos[i], d))
                                for i in range(n_oos)]

            bp = [float(np.dot(better_in[i], d)) for i in range(n)]
            wp = [float(np.dot(worse_in[i], d)) for i in range(n)]
            firm_d[name] = cohens_d([bp[i] for i in firm_idx], [wp[i] for i in firm_idx])
            warm_d[name] = cohens_d([bp[i] for i in warm_idx], [wp[i] for i in warm_idx])

        # Classification: warmth-leaning vs firmness-leaning
        print(f"\n  AXIS BIAS CLASSIFICATION:")
        print(f"  {'Axis':12s} {'Firm d':>8s} {'Warm d':>8s} {'Bias':>12s} {'Pairwise':>8s}")
        print(f"  {'-'*55}")

        warmth_axes = []
        firm_axes = []

        for name in CANDIDATES:
            acc = sum(1 for d in deltas_in[name] if d > 0) / n
            if firm_d[name] > 0 and warm_d[name] <= 0:
                bias = "FIRMNESS"
                firm_axes.append(name)
            elif warm_d[name] > 0 and firm_d[name] <= 0:
                bias = "WARMTH"
                warmth_axes.append(name)
            elif firm_d[name] > 0 and warm_d[name] > 0:
                bias = "both+"
                # Assign to whichever is weaker (it's the one that needs help)
                if firm_d[name] > warm_d[name]:
                    firm_axes.append(name)
                else:
                    warmth_axes.append(name)
            else:
                bias = "both-"
                warmth_axes.append(name)  # default
            print(f"  {name:12s} {firm_d[name]:+8.3f} {warm_d[name]:+8.3f} {bias:>12s} {acc:8.0%}")

        print(f"\n  Firmness-leaning axes ({len(firm_axes)}): {', '.join(firm_axes)}")
        print(f"  Warmth-leaning axes ({len(warmth_axes)}): {', '.join(warmth_axes)}")

        # BALANCED AGGREGATE: equal weight to warmth and firmness clusters
        print(f"\n  BALANCED AGGREGATE SCORING:")

        # Strategy 1: pick top N from each cluster, sum
        for k in [2, 3, 4]:
            f_pick = firm_axes[:k] if len(firm_axes) >= k else firm_axes
            w_pick = warmth_axes[:k] if len(warmth_axes) >= k else warmth_axes
            selected = f_pick + w_pick

            agg_in = [sum(deltas_in[name][i] for name in selected) for i in range(n)]
            agg_correct_in = [d > 0 for d in agg_in]
            acc_in = sum(agg_correct_in) / n
            f_acc = sum(agg_correct_in[i] for i in firm_idx) / len(firm_idx)
            w_acc = sum(agg_correct_in[i] for i in warm_idx) / len(warm_idx)
            s_acc = sum(agg_correct_in[i] for i in syc_idx) / len(syc_idx)

            agg_oos = [sum(deltas_oos[name][i] for name in selected) for i in range(n_oos)]
            acc_oos = sum(1 for d in agg_oos if d > 0) / n_oos

            terms = "+".join(selected)
            print(f"  {k}+{k} ({len(selected)} axes): in={acc_in:.0%} (f={f_acc:.0%} w={w_acc:.0%} s={s_acc:.0%}) oos={acc_oos:.0%}")
            print(f"    terms: {terms}")

        # Strategy 2: normalize each axis's delta by its std, then sum
        # This prevents high-magnitude axes from dominating
        print(f"\n  NORMALIZED AGGREGATE (z-score each axis, then sum):")
        for k in [2, 3]:
            f_pick = firm_axes[:k] if len(firm_axes) >= k else firm_axes
            w_pick = warmth_axes[:k] if len(warmth_axes) >= k else warmth_axes
            selected = f_pick + w_pick

            # Z-score normalize
            norm_deltas_in = {}
            norm_deltas_oos = {}
            for name in selected:
                d_arr = np.array(deltas_in[name])
                mu, sigma = d_arr.mean(), d_arr.std()
                if sigma > 0:
                    norm_deltas_in[name] = [(deltas_in[name][i] - mu) / sigma for i in range(n)]
                    norm_deltas_oos[name] = [(deltas_oos[name][i] - mu) / sigma for i in range(n_oos)]
                else:
                    norm_deltas_in[name] = deltas_in[name]
                    norm_deltas_oos[name] = deltas_oos[name]

            nagg_in = [sum(norm_deltas_in[name][i] for name in selected) for i in range(n)]
            nagg_correct = [d > 0 for d in nagg_in]
            nacc_in = sum(nagg_correct) / n
            nf = sum(nagg_correct[i] for i in firm_idx) / len(firm_idx)
            nw = sum(nagg_correct[i] for i in warm_idx) / len(warm_idx)
            ns = sum(nagg_correct[i] for i in syc_idx) / len(syc_idx)

            nagg_oos = [sum(norm_deltas_oos[name][i] for name in selected) for i in range(n_oos)]
            nacc_oos = sum(1 for d in nagg_oos if d > 0) / n_oos

            print(f"  {k}+{k} normed: in={nacc_in:.0%} (f={nf:.0%} w={nw:.0%} s={ns:.0%}) oos={nacc_oos:.0%}")

        # ALL axes aggregate (balanced by cluster mean)
        print(f"\n  CLUSTER-MEAN BALANCED (avg firmness cluster + avg warmth cluster):")
        if firm_axes and warmth_axes:
            firm_mean_in = [np.mean([deltas_in[name][i] for name in firm_axes]) for i in range(n)]
            warm_mean_in = [np.mean([deltas_in[name][i] for name in warmth_axes]) for i in range(n)]
            balanced_in = [firm_mean_in[i] + warm_mean_in[i] for i in range(n)]
            bal_correct = [d > 0 for d in balanced_in]
            bal_acc = sum(bal_correct) / n
            bf = sum(bal_correct[i] for i in firm_idx) / len(firm_idx)
            bw = sum(bal_correct[i] for i in warm_idx) / len(warm_idx)
            bs = sum(bal_correct[i] for i in syc_idx) / len(syc_idx)

            firm_mean_oos = [np.mean([deltas_oos[name][i] for name in firm_axes]) for i in range(n_oos)]
            warm_mean_oos = [np.mean([deltas_oos[name][i] for name in warmth_axes]) for i in range(n_oos)]
            balanced_oos = [firm_mean_oos[i] + warm_mean_oos[i] for i in range(n_oos)]
            bal_oos = sum(1 for d in balanced_oos if d > 0) / n_oos

            print(f"  mean({len(firm_axes)} firm) + mean({len(warmth_axes)} warm): in={bal_acc:.0%} (f={bf:.0%} w={bw:.0%} s={bs:.0%}) oos={bal_oos:.0%}")

        # For reference: raw good
        good_in_acc = sum(1 for d in deltas_in["good"] if d > 0) / n
        good_oos_acc = sum(1 for d in deltas_oos["good"] if d > 0) / n_oos
        care_in_acc = sum(1 for d in deltas_in["careful"] if d > 0) / n
        care_oos_acc = sum(1 for d in deltas_oos["careful"] if d > 0) / n_oos
        print(f"\n  Reference: good in={good_in_acc:.0%} oos={good_oos_acc:.0%}")
        print(f"  Reference: careful in={care_in_acc:.0%} oos={care_oos_acc:.0%}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
