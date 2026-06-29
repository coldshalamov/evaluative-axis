#!/usr/bin/env python3
"""Build and test balanced matrices — sets of terms chosen for
COMPLEMENTARY failure modes, not synonyms.

Key insight: raw sum lets high-magnitude axes dominate. Use vote counting
(how many axes favor the better response) to give each axis equal voice.

Test various matrix compositions with diverse term sets.
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

# Diverse term sets — each group targets a DIFFERENT failure mode
AXES = {
    # Root
    "good":         ("Good", "Bad"),
    # Anti-sycophancy (counter: flattery, people-pleasing)
    "skeptical":    ("Skeptical", "Credulous"),
    "critical":     ("Critical", "Uncritical"),
    "frank":        ("Frank", "Diplomatic"),
    "direct":       ("Direct", "Evasive"),
    # Anti-recklessness (counter: acting without checking)
    "careful":      ("Careful", "Reckless"),
    "cautious":     ("Cautious", "Rash"),
    "prudent":      ("Prudent", "Imprudent"),
    # Anti-superficiality (counter: shallow/incomplete work)
    "thorough":     ("Thorough", "Superficial"),
    "analytical":   ("Analytical", "Simplistic"),
    "substantive":  ("Substantive", "Shallow"),
    # Pro-integrity (counter: deception, withholding)
    "honest":       ("Honest", "Dishonest"),
    "forthright":   ("Forthright", "Withholding"),
    # Pro-discipline (counter: verbosity, excess)
    "restrained":   ("Restrained", "Unrestrained"),
    "concise":      ("Concise", "Verbose"),
    "measured":     ("Measured", "Extreme"),
    # Pro-competence (counter: incompetence)
    "competent":    ("Competent", "Incompetent"),
    "rigorous":     ("Rigorous", "Lax"),
    "meticulous":   ("Meticulous", "Sloppy"),
    # Pro-fairness (counter: bias, partiality)
    "impartial":    ("Impartial", "Partial"),
    "objective":    ("Objective", "Biased"),
    # Pro-warmth (counter: coldness, cruelty)
    "helpful":      ("Helpful", "Unhelpful"),
    "respectful":   ("Respectful", "Disrespectful"),
    "kind":         ("Kind", "Cruel"),
    "considerate":  ("Considerate", "Inconsiderate"),
    # Pro-reliability
    "responsible":  ("Responsible", "Irresponsible"),
    "reliable":     ("Reliable", "Unreliable"),
    "diligent":     ("Diligent", "Negligent"),
    # Pro-clarity
    "clear":        ("Clear", "Confusing"),
    "precise":      ("Precise", "Vague"),
    "accurate":     ("Accurate", "Inaccurate"),
    # Pro-insight
    "insightful":   ("Insightful", "Obvious"),
    "wise":         ("Wise", "Foolish"),
    "thoughtful":   ("Thoughtful", "Thoughtless"),
}

# Manually curated matrices — each picks ONE term per failure mode
MATRICES = {
    "core_6": ["good", "skeptical", "careful", "thorough", "restrained", "honest"],
    "diverse_8": ["good", "skeptical", "careful", "thorough", "restrained", "honest", "analytical", "impartial"],
    "max_coverage_10": ["good", "skeptical", "careful", "thorough", "restrained", "honest",
                        "analytical", "impartial", "frank", "meticulous"],
    "kitchen_sink_15": ["good", "skeptical", "careful", "thorough", "restrained", "honest",
                        "analytical", "impartial", "frank", "meticulous", "cautious",
                        "helpful", "direct", "responsible", "measured"],
    "no_good_5": ["skeptical", "careful", "thorough", "restrained", "honest"],
    "no_good_8": ["skeptical", "careful", "thorough", "restrained", "honest", "analytical", "impartial", "frank"],
    "all_30": [k for k in AXES if k != "good"],
    "all_31": list(AXES.keys()),
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

        # Compute all axis deltas
        deltas_in = {}
        deltas_oos = {}
        for name, (pos, neg) in AXES.items():
            p = embed([pos])[0]
            nn = embed([neg])[0]
            d = (p - nn) / (norm(p - nn) + 1e-12)
            deltas_in[name] = [float(np.dot(better_in[i], d)) - float(np.dot(worse_in[i], d))
                               for i in range(n)]
            deltas_oos[name] = [float(np.dot(better_oos[i], d)) - float(np.dot(worse_oos[i], d))
                                for i in range(n_oos)]

        print(f"\n  {'Matrix':25s} | {'Votes':>5s} {'V-f':>5s} {'V-w':>5s} {'V-s':>5s} | {'Sum':>5s} {'S-f':>5s} {'S-w':>5s} {'S-s':>5s} | {'V-OOS':>5s} {'S-OOS':>5s}")
        print(f"  {'-'*105}")

        for mname, terms in MATRICES.items():
            k = len(terms)

            # VOTE COUNT: how many axes favor the better response?
            votes_in = [sum(1 for t in terms if deltas_in[t][i] > 0) for i in range(n)]
            vote_correct_in = [v > k/2 for v in votes_in]
            v_acc = sum(vote_correct_in) / n
            v_f = sum(vote_correct_in[i] for i in firm_idx) / len(firm_idx)
            v_w = sum(vote_correct_in[i] for i in warm_idx) / len(warm_idx)
            v_s = sum(vote_correct_in[i] for i in syc_idx) / len(syc_idx)

            # RAW SUM
            sums_in = [sum(deltas_in[t][i] for t in terms) for i in range(n)]
            sum_correct_in = [s > 0 for s in sums_in]
            s_acc = sum(sum_correct_in) / n
            s_f = sum(sum_correct_in[i] for i in firm_idx) / len(firm_idx)
            s_w = sum(sum_correct_in[i] for i in warm_idx) / len(warm_idx)
            s_s = sum(sum_correct_in[i] for i in syc_idx) / len(syc_idx)

            # OOS
            votes_oos = [sum(1 for t in terms if deltas_oos[t][i] > 0) for i in range(n_oos)]
            v_oos = sum(v > k/2 for v in votes_oos) / n_oos
            sums_oos = [sum(deltas_oos[t][i] for t in terms) for i in range(n_oos)]
            s_oos = sum(s > 0 for s in sums_oos) / n_oos

            print(f"  {mname:25s} | {v_acc:4.0%}  {v_f:4.0%}  {v_w:4.0%}  {v_s:4.0%}  | {s_acc:4.0%}  {s_f:4.0%}  {s_w:4.0%}  {s_s:4.0%}  | {v_oos:4.0%}  {s_oos:4.0%}")

        # DETAILED: for the best matrix, show per-case diagnostic
        print(f"\n  DETAILED DIAGNOSTIC for 'diverse_8':")
        terms = MATRICES["diverse_8"]
        for i in range(n):
            scores = {t: deltas_in[t][i] for t in terms}
            pos_count = sum(1 for v in scores.values() if v > 0)
            total = sum(scores.values())
            correct = "OK" if total > 0 else "FAIL"
            if correct == "FAIL" and labels[i] in ("firmness", "sycophancy"):
                signs = " ".join(f"{t[0]}{'+'if scores[t]>0 else '-'}" for t in terms)
                print(f"    case {i:2d} [{labels[i]:10s}] {pos_count}/{len(terms)} positive  total={total:+.4f}  {signs}")

        # Also: what's the MEAN aggregate score for each case type?
        print(f"\n  MEAN AGGREGATE SCORE by case type (diverse_8):")
        for label, idx in [("firmness", firm_idx), ("warmth", warm_idx), ("sycophancy", syc_idx)]:
            total_scores = [sum(deltas_in[t][i] for t in terms) for i in idx]
            vote_scores = [sum(1 for t in terms if deltas_in[t][i] > 0) for i in idx]
            print(f"    {label:12s}: mean_sum={np.mean(total_scores):+.4f}  mean_votes={np.mean(vote_scores):.1f}/{len(terms)}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
