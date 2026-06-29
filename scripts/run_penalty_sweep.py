#!/usr/bin/env python3
"""Sweep a large vocabulary of COMMON words as penalties against good.

Guiding principle: the best penalty terms are common everyday words
that describe actual failure-mode behaviors, not clinical jargon.

For each candidate penalty term:
  score = cos(response, Good) - alpha * cos(response, penalty)

Report which terms improve firmness/sycophancy without destroying warmth.
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

# Large vocabulary of COMMON words that might describe failure modes
# Organized by what failure they might capture, but tested agnostically
PENALTY_CANDIDATES = [
    # Warmth excess / people-pleasing (common words)
    "flattering", "agreeable", "pleasing", "nice", "sweet",
    "smooth", "charming", "reassuring", "comforting", "soothing",
    "encouraging", "accommodating", "obliging", "eager",
    "enthusiastic", "positive", "cheerful", "friendly",
    "generous", "gentle", "soft", "tender", "loving",
    # Deference / submission
    "compliant", "submissive", "obedient", "passive", "deferential",
    "apologetic", "meek", "timid", "servile",
    # Deception / evasion
    "misleading", "deceptive", "dishonest", "manipulative",
    "evasive", "vague", "ambiguous", "unclear", "confusing",
    "dodgy", "slippery",
    # Superficiality
    "shallow", "superficial", "simplistic", "trivial", "obvious",
    "generic", "bland", "boring", "dull", "ordinary",
    "predictable", "formulaic", "cliche",
    # Recklessness / carelessness
    "careless", "reckless", "hasty", "rushed", "sloppy",
    "lazy", "negligent", "irresponsible",
    # Excess / verbosity
    "verbose", "excessive", "extreme", "exaggerated", "dramatic",
    "overwhelming", "overblown", "pompous", "pretentious",
    "wordy", "rambling", "repetitive", "redundant",
    # Arrogance / presumption
    "arrogant", "condescending", "patronizing", "presumptuous",
    "dismissive", "superior", "smug", "pompous",
    # Bias / unfairness
    "biased", "unfair", "partial", "prejudiced", "one-sided",
    # Clinical/rare (expected to work LESS well — control group)
    "sycophantic", "obsequious", "fawning", "ingratiating",
    "disingenuous", "perfunctory", "platitudinous",
    # Interesting edge cases
    "fake", "false", "wrong", "bad", "harmful", "dangerous",
    "toxic", "corrupt", "broken", "flawed",
    # Emotional manipulation
    "emotional", "needy", "clingy", "desperate", "pushy",
    "insistent", "demanding", "controlling",
]

# Remove duplicates
PENALTY_CANDIDATES = list(dict.fromkeys(PENALTY_CANDIDATES))


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

    results_all_models = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])
        better_oos = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in oos])
        worse_oos = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in oos])

        # Normalize response embeddings for cosine
        for i in range(n):
            better_embs[i] = better_embs[i] / (norm(better_embs[i]) + 1e-12)
            worse_embs[i] = worse_embs[i] / (norm(worse_embs[i]) + 1e-12)
        for i in range(n_oos):
            better_oos[i] = better_oos[i] / (norm(better_oos[i]) + 1e-12)
            worse_oos[i] = worse_oos[i] / (norm(worse_oos[i]) + 1e-12)

        # Embed good
        good_emb = embed(["Good"])[0]
        good_emb = good_emb / (norm(good_emb) + 1e-12)

        # Baseline: raw cos(Good)
        b_good = [float(np.dot(better_embs[i], good_emb)) for i in range(n)]
        w_good = [float(np.dot(worse_embs[i], good_emb)) for i in range(n)]
        base_correct = [b_good[i] > w_good[i] for i in range(n)]
        base_acc = sum(base_correct) / n
        base_f = sum(base_correct[i] for i in firm_idx) / len(firm_idx)
        base_w = sum(base_correct[i] for i in warm_idx) / len(warm_idx)
        base_s = sum(base_correct[i] for i in syc_idx) / len(syc_idx)

        b_good_oos = [float(np.dot(better_oos[i], good_emb)) for i in range(n_oos)]
        w_good_oos = [float(np.dot(worse_oos[i], good_emb)) for i in range(n_oos)]
        base_oos = sum(b_good_oos[i] > w_good_oos[i] for i in range(n_oos)) / n_oos

        print(f"\n  BASELINE: cos(Good) = {base_acc:.0%} (f={base_f:.0%} w={base_w:.0%} s={base_s:.0%})  OOS={base_oos:.0%}")

        # Embed all penalty candidates
        penalty_embs = {}
        cos_to_good = {}
        for term in PENALTY_CANDIDATES:
            t_emb = embed([term.capitalize()])[0]
            t_emb = t_emb / (norm(t_emb) + 1e-12)
            penalty_embs[term] = t_emb
            cos_to_good[term] = float(np.dot(good_emb, t_emb))

        # Test each penalty at alpha=1.0
        results = []
        for term in PENALTY_CANDIDATES:
            t_emb = penalty_embs[term]
            b_pen = [float(np.dot(better_embs[i], t_emb)) for i in range(n)]
            w_pen = [float(np.dot(worse_embs[i], t_emb)) for i in range(n)]

            b_score = [b_good[i] - b_pen[i] for i in range(n)]
            w_score = [w_good[i] - w_pen[i] for i in range(n)]
            correct = [b_score[i] > w_score[i] for i in range(n)]
            acc = sum(correct) / n
            f_acc = sum(correct[i] for i in firm_idx) / len(firm_idx)
            w_acc = sum(correct[i] for i in warm_idx) / len(warm_idx)
            s_acc = sum(correct[i] for i in syc_idx) / len(syc_idx)

            b_pen_oos = [float(np.dot(better_oos[i], t_emb)) for i in range(n_oos)]
            w_pen_oos = [float(np.dot(worse_oos[i], t_emb)) for i in range(n_oos)]
            b_s_oos = [b_good_oos[i] - b_pen_oos[i] for i in range(n_oos)]
            w_s_oos = [w_good_oos[i] - w_pen_oos[i] for i in range(n_oos)]
            oos_acc = sum(b_s_oos[i] > w_s_oos[i] for i in range(n_oos)) / n_oos

            # Balance score: geometric mean of split accuracies (penalizes imbalance)
            splits = [f_acc, w_acc, s_acc]
            balance = (max(f_acc, 0.01) * max(w_acc, 0.01) * max(s_acc, 0.01)) ** (1/3)

            results.append({
                "term": term,
                "acc": acc,
                "firm": f_acc,
                "warm": w_acc,
                "syc": s_acc,
                "oos": oos_acc,
                "balance": balance,
                "cos_good": cos_to_good[term],
                "delta_acc": acc - base_acc,
            })

        # Sort by balance (geometric mean of splits)
        results.sort(key=lambda x: -x["balance"])

        print(f"\n  TOP 25 PENALTIES BY BALANCED ACCURACY (alpha=1.0):")
        print(f"  {'Term':20s} {'Acc':>5s} {'Firm':>5s} {'Warm':>5s} {'Syc':>5s} {'OOS':>5s} {'Bal':>5s} {'cos(G)':>7s} {'dAcc':>6s}")
        print(f"  {'-'*85}")
        for r in results[:25]:
            print(f"  {r['term']:20s} {r['acc']:4.0%}  {r['firm']:4.0%}  {r['warm']:4.0%}  {r['syc']:4.0%}  {r['oos']:4.0%}  {r['balance']:.3f}  {r['cos_good']:+.4f}  {r['delta_acc']:+4.0%}")

        print(f"\n  BOTTOM 10 (worst penalties):")
        for r in results[-10:]:
            print(f"  {r['term']:20s} {r['acc']:4.0%}  {r['firm']:4.0%}  {r['warm']:4.0%}  {r['syc']:4.0%}  {r['oos']:4.0%}  {r['balance']:.3f}  {r['cos_good']:+.4f}  {r['delta_acc']:+4.0%}")

        # What terms improve ALL splits vs baseline?
        improvers = [r for r in results
                     if r["firm"] > base_f and r["warm"] >= base_w * 0.8 and r["syc"] > base_s]
        if improvers:
            print(f"\n  TERMS THAT IMPROVE FIRMNESS+SYCOPHANCY WITHOUT DESTROYING WARMTH:")
            for r in sorted(improvers, key=lambda x: -x["acc"]):
                print(f"  {r['term']:20s} {r['acc']:4.0%}  f={r['firm']:4.0%}(+{r['firm']-base_f:+.0%})  w={r['warm']:4.0%}  s={r['syc']:4.0%}(+{r['syc']-base_s:+.0%})")

        # Test top 3 at multiple alphas
        top3 = [r["term"] for r in results[:3]]
        print(f"\n  ALPHA SWEEP for top 3 ({', '.join(top3)}):")
        for term in top3:
            t_emb = penalty_embs[term]
            b_pen = [float(np.dot(better_embs[i], t_emb)) for i in range(n)]
            w_pen = [float(np.dot(worse_embs[i], t_emb)) for i in range(n)]
            for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                b_score = [b_good[i] - alpha * b_pen[i] for i in range(n)]
                w_score = [w_good[i] - alpha * w_pen[i] for i in range(n)]
                correct = [b_score[i] > w_score[i] for i in range(n)]
                acc = sum(correct) / n
                f_acc = sum(correct[i] for i in firm_idx) / len(firm_idx)
                w_acc = sum(correct[i] for i in warm_idx) / len(warm_idx)
                s_acc = sum(correct[i] for i in syc_idx) / len(syc_idx)
                print(f"    {term:15s} a={alpha:.1f}: {acc:.0%} (f={f_acc:.0%} w={w_acc:.0%} s={s_acc:.0%})")

        # Test combinations of top penalties
        print(f"\n  COMBINED PENALTIES (top terms together):")
        top5 = [r["term"] for r in results[:5]]
        for k in [2, 3, 4, 5]:
            combo = top5[:k]
            combo_embs = [penalty_embs[t] for t in combo]
            for alpha in [0.5, 1.0]:
                b_score = list(b_good)
                w_score = list(w_good)
                for t_emb in combo_embs:
                    for i in range(n):
                        b_score[i] -= alpha * float(np.dot(better_embs[i], t_emb)) / k
                        w_score[i] -= alpha * float(np.dot(worse_embs[i], t_emb)) / k
                correct = [b_score[i] > w_score[i] for i in range(n)]
                acc = sum(correct) / n
                f_acc = sum(correct[i] for i in firm_idx) / len(firm_idx)
                w_acc = sum(correct[i] for i in warm_idx) / len(warm_idx)
                s_acc = sum(correct[i] for i in syc_idx) / len(syc_idx)
                names = "+".join(combo)
                print(f"    mean({k}) a={alpha}: {acc:.0%} (f={f_acc:.0%} w={w_acc:.0%} s={s_acc:.0%})  [{names}]")

        results_all_models[short] = results
        del model
        gc.collect()

    # Cross-model comparison: terms that appear in top 10 for ALL models
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL CONSENSUS")
    print(f"{'='*80}")
    for model_short, results in results_all_models.items():
        top10 = set(r["term"] for r in results[:10])
        print(f"  {model_short} top 10: {', '.join(r['term'] for r in results[:10])}")

    # Find terms in top 20 for all models
    all_top20 = None
    for results in results_all_models.values():
        top20 = set(r["term"] for r in results[:20])
        if all_top20 is None:
            all_top20 = top20
        else:
            all_top20 &= top20
    if all_top20:
        print(f"\n  Terms in top 20 for ALL 3 models: {', '.join(sorted(all_top20))}")


if __name__ == "__main__":
    main()
