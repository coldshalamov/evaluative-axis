#!/usr/bin/env python3
"""For each case where "good" fails, which terms succeed?

Build the matrix empirically: find terms that COMPLEMENT good by catching
its specific failure modes, not terms that duplicate it.

Strategy:
1. Find all cases where good/bad gets the wrong answer
2. Test a large vocabulary of candidate axes
3. For each axis, count which good-failure cases it rescues
4. Greedily select axes that maximize coverage of failure cases
5. Test the aggregate
"""

import json, gc, math, itertools
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

# Large vocabulary — each pair chosen for a DISTINCT failure mode it might counter
CANDIDATES = {
    # Counters to sycophancy
    "direct":       ("Direct", "Evasive"),
    "candid":       ("Candid", "Guarded"),
    "frank":        ("Frank", "Diplomatic"),
    "principled":   ("Principled", "Unprincipled"),
    "honest":       ("Honest", "Dishonest"),
    "forthright":   ("Forthright", "Withholding"),
    # Counters to recklessness/lack of rigor
    "careful":      ("Careful", "Reckless"),
    "cautious":     ("Cautious", "Rash"),
    "diligent":     ("Diligent", "Negligent"),
    "rigorous":     ("Rigorous", "Lax"),
    "meticulous":   ("Meticulous", "Sloppy"),
    "methodical":   ("Methodical", "Haphazard"),
    # Counters to excessive agreeableness
    "critical":     ("Critical", "Uncritical"),
    "discerning":   ("Discerning", "Undiscerning"),
    "skeptical":    ("Skeptical", "Credulous"),
    "objective":    ("Objective", "Biased"),
    "impartial":    ("Impartial", "Partial"),
    # Counters to superficiality
    "thorough":     ("Thorough", "Superficial"),
    "substantive":  ("Substantive", "Shallow"),
    "insightful":   ("Insightful", "Obvious"),
    "analytical":   ("Analytical", "Simplistic"),
    # Counters to verbosity/people-pleasing
    "concise":      ("Concise", "Verbose"),
    "restrained":   ("Restrained", "Unrestrained"),
    "measured":     ("Measured", "Extreme"),
    "disciplined":  ("Disciplined", "Undisciplined"),
    # General quality
    "competent":    ("Competent", "Incompetent"),
    "reliable":     ("Reliable", "Unreliable"),
    "precise":      ("Precise", "Vague"),
    "accurate":     ("Accurate", "Inaccurate"),
    "helpful":      ("Helpful", "Unhelpful"),
    "responsible":  ("Responsible", "Irresponsible"),
    # Reference
    "good":         ("Good", "Bad"),
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

        # Compute all axes
        axis_dirs = {}
        axis_correct_in = {}
        axis_correct_oos = {}
        axis_deltas_in = {}
        axis_deltas_oos = {}

        for name, (pos, neg) in CANDIDATES.items():
            p = embed([pos])[0]
            nn = embed([neg])[0]
            d = (p - nn) / (norm(p - nn) + 1e-12)
            axis_dirs[name] = d

            deltas = [float(np.dot(better_in[i], d)) - float(np.dot(worse_in[i], d))
                      for i in range(n)]
            axis_deltas_in[name] = deltas
            axis_correct_in[name] = [d > 0 for d in deltas]

            deltas_oos = [float(np.dot(better_oos[i], d)) - float(np.dot(worse_oos[i], d))
                          for i in range(n_oos)]
            axis_deltas_oos[name] = deltas_oos
            axis_correct_oos[name] = [d > 0 for d in deltas_oos]

        # Where does good fail?
        good_fails = [i for i in range(n) if not axis_correct_in["good"][i]]
        good_fail_firm = [i for i in good_fails if labels[i] == "firmness"]
        good_fail_warm = [i for i in good_fails if labels[i] == "warmth"]
        good_fail_syc = [i for i in good_fails if labels[i] == "sycophancy"]

        good_acc = sum(axis_correct_in["good"]) / n
        print(f"\n  Good accuracy: {good_acc:.0%} ({sum(axis_correct_in['good'])}/{n})")
        print(f"  Good failures: {len(good_fails)} ({len(good_fail_firm)}f/{len(good_fail_warm)}w/{len(good_fail_syc)}s)")

        # For each non-good axis: how many good-failures does it rescue?
        print(f"\n  RESCUE RATES (how many good-failures each term catches):")
        print(f"  {'Term':15s} {'Rescue':>7s} {'Own acc':>7s} {'Firm':>6s} {'Warm':>6s} {'Syc':>6s}  Rescues by type")
        print(f"  {'-'*75}")

        rescue_data = {}
        for name in CANDIDATES:
            if name == "good":
                continue
            rescued = [i for i in good_fails if axis_correct_in[name][i]]
            own_acc = sum(axis_correct_in[name]) / n
            rescue_firm = sum(1 for i in rescued if labels[i] == "firmness")
            rescue_warm = sum(1 for i in rescued if labels[i] == "warmth")
            rescue_syc = sum(1 for i in rescued if labels[i] == "sycophancy")

            # Also: does this term break cases good gets right?
            breaks = [i for i in range(n) if axis_correct_in["good"][i] and not axis_correct_in[name][i]]

            rescue_data[name] = {
                "rescued": set(rescued),
                "n_rescued": len(rescued),
                "own_acc": own_acc,
                "breaks": len(breaks),
            }

            f_acc = sum(axis_correct_in[name][i] for i in firm_idx) / len(firm_idx)
            w_acc = sum(axis_correct_in[name][i] for i in warm_idx) / len(warm_idx)
            s_acc = sum(axis_correct_in[name][i] for i in syc_idx) / len(syc_idx)

            print(f"  {name:15s} {len(rescued):3d}/{len(good_fails):3d}  {own_acc:6.0%}  {f_acc:5.0%}  {w_acc:5.0%}  {s_acc:5.0%}   {rescue_firm}f/{rescue_warm}w/{rescue_syc}s  (breaks {len(breaks)})")

        # GREEDY SET COVER: pick terms that maximally cover good's failures
        print(f"\n  GREEDY SET COVER (maximize coverage of good's failures):")
        remaining = set(good_fails)
        selected = []
        available = set(CANDIDATES.keys()) - {"good"}

        while remaining and available:
            best_name = None
            best_new = 0
            best_rescued = set()
            for name in available:
                new_rescued = rescue_data[name]["rescued"] & remaining
                if len(new_rescued) > best_new:
                    best_new = len(new_rescued)
                    best_name = name
                    best_rescued = new_rescued
            if best_name is None or best_new == 0:
                break
            selected.append(best_name)
            remaining -= best_rescued
            available.discard(best_name)
            covered = len(good_fails) - len(remaining)
            print(f"  +{best_name:15s}  covers {best_new:3d} new  (total {covered}/{len(good_fails)})")

        print(f"  Selected {len(selected)} terms: {', '.join(selected)}")
        print(f"  Uncovered failures: {len(remaining)}")

        # Test the greedy-selected aggregate
        print(f"\n  AGGREGATE SCORING WITH SELECTED TERMS:")
        # Include good + selected terms
        full_set = ["good"] + selected

        # Sum of raw deltas
        sum_deltas = [sum(axis_deltas_in[name][i] for name in full_set) for i in range(n)]
        sum_correct = [d > 0 for d in sum_deltas]
        sum_acc = sum(sum_correct) / n
        sf = sum(sum_correct[i] for i in firm_idx) / len(firm_idx)
        sw = sum(sum_correct[i] for i in warm_idx) / len(warm_idx)
        ss = sum(sum_correct[i] for i in syc_idx) / len(syc_idx)
        lo, hi = wilson_ci(sum(sum_correct), n)
        print(f"  sum(good + {len(selected)} terms): {sum_acc:.0%} [{lo:.0%}-{hi:.0%}]  f={sf:.0%}  w={sw:.0%}  s={ss:.0%}")

        # Without good
        sum_deltas_ng = [sum(axis_deltas_in[name][i] for name in selected) for i in range(n)]
        sum_correct_ng = [d > 0 for d in sum_deltas_ng]
        acc_ng = sum(sum_correct_ng) / n
        sf_ng = sum(sum_correct_ng[i] for i in firm_idx) / len(firm_idx)
        sw_ng = sum(sum_correct_ng[i] for i in warm_idx) / len(warm_idx)
        ss_ng = sum(sum_correct_ng[i] for i in syc_idx) / len(syc_idx)
        print(f"  sum({len(selected)} terms, no good): {acc_ng:.0%}  f={sf_ng:.0%}  w={sw_ng:.0%}  s={ss_ng:.0%}")

        # Z-score normalized sum (equal weight per axis regardless of magnitude)
        norm_deltas = {}
        norm_deltas_oos_dict = {}
        for name in full_set:
            arr = np.array(axis_deltas_in[name])
            mu, sigma = arr.mean(), arr.std()
            if sigma > 0:
                norm_deltas[name] = [(axis_deltas_in[name][i] - mu) / sigma for i in range(n)]
                norm_deltas_oos_dict[name] = [(axis_deltas_oos[name][i] - mu) / sigma for i in range(n_oos)]
            else:
                norm_deltas[name] = axis_deltas_in[name]
                norm_deltas_oos_dict[name] = axis_deltas_oos[name]

        znorm_in = [sum(norm_deltas[name][i] for name in full_set) for i in range(n)]
        zn_correct = [d > 0 for d in znorm_in]
        zn_acc = sum(zn_correct) / n
        zf = sum(zn_correct[i] for i in firm_idx) / len(firm_idx)
        zw = sum(zn_correct[i] for i in warm_idx) / len(warm_idx)
        zs = sum(zn_correct[i] for i in syc_idx) / len(syc_idx)
        print(f"  z-normed sum(good + {len(selected)}): {zn_acc:.0%}  f={zf:.0%}  w={zw:.0%}  s={zs:.0%}")

        # OOS
        sum_oos = [sum(axis_deltas_oos[name][i] for name in full_set) for i in range(n_oos)]
        oos_acc = sum(1 for d in sum_oos if d > 0) / n_oos
        zn_oos = [sum(norm_deltas_oos_dict[name][i] for name in full_set) for i in range(n_oos)]
        zn_oos_acc = sum(1 for d in zn_oos if d > 0) / n_oos
        print(f"  OOS raw sum: {oos_acc:.0%}  z-normed: {zn_oos_acc:.0%}")

        # Compare to baselines
        good_oos = sum(axis_correct_oos["good"]) / n_oos
        care_oos = sum(axis_correct_oos["careful"]) / n_oos
        print(f"  OOS baselines: good={good_oos:.0%}  careful={care_oos:.0%}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
