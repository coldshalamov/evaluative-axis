#!/usr/bin/env python3
"""Test the "training wheels" idea: keep good as the primary signal,
add penalty terms for its known failure modes.

score = good_projection - w * failure_mode_projection

Failure modes of "good" (via warm):
- sycophantic (the bad part of warm/helpful)
- reckless (the bad part of being overly helpful)
- deceptive (the bad part of being pleasant)

Also test: good + honest (training for honest to counter sycophancy)
And: good + careful (training for careful to counter recklessness)
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


def make_dir(embed, pos, neg):
    p = embed([pos])[0]
    n_ = embed([neg])[0]
    d = p - n_
    return d / (norm(d) + 1e-12)


def score_cases(better_embs, worse_embs, direction, n):
    return [float(np.dot(better_embs[i], direction)) > float(np.dot(worse_embs[i], direction))
            for i in range(n)]


def report(name, outcomes, labels, n):
    acc = sum(outcomes) / n
    firm_idx = [i for i in range(n) if labels[i] == "firmness"]
    warm_idx = [i for i in range(n) if labels[i] == "warmth"]
    syc_idx = [i for i in range(n) if labels[i] == "sycophancy"]
    firm = sum(outcomes[i] for i in firm_idx) / len(firm_idx) if firm_idx else 0
    warm = sum(outcomes[i] for i in warm_idx) / len(warm_idx) if warm_idx else 0
    syc = sum(outcomes[i] for i in syc_idx) / len(syc_idx) if syc_idx else 0
    lo, hi = wilson_ci(sum(outcomes), n)
    print(f"  {name:40s} {acc:5.0%} [{lo:.0%}-{hi:.0%}]  f={firm:.0%}  w={warm:.0%}  s={syc:.0%}")


def main():
    from sentence_transformers import SentenceTransformer

    battery = read_jsonl(BATTERY)
    warmth_cases = read_jsonl(WARMTH)
    in_sample = battery + warmth_cases
    n = len(in_sample)

    labels = []
    for c in battery:
        labels.append("sycophancy" if c["category"] == "anti_sycophancy" else "firmness")
    for c in warmth_cases:
        labels.append("warmth")

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

        better_in = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in in_sample])
        worse_in = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in in_sample])
        better_oos = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in oos])
        worse_oos = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in oos])

        # Directions
        good_dir = make_dir(embed, "Good", "Bad")
        careful_dir = make_dir(embed, "Careful", "Reckless")
        honest_dir = make_dir(embed, "Honest", "Dishonest")
        helpful_dir = make_dir(embed, "Helpful", "Unhelpful")
        thorough_dir = make_dir(embed, "Thorough", "Superficial")
        restrained_dir = make_dir(embed, "Restrained", "Unrestrained")

        # Failure mode directions (things to penalize)
        sycophantic_dir = make_dir(embed, "Sycophantic", "Genuine")
        reckless_dir = make_dir(embed, "Reckless", "Careful")
        deceptive_dir = make_dir(embed, "Deceptive", "Truthful")
        flattering_dir = make_dir(embed, "Flattering", "Honest")
        agreeable_dir = make_dir(embed, "Agreeable", "Principled")
        obedient_dir = make_dir(embed, "Obedient", "Independent")

        print(f"\n  --- IN-SAMPLE (n={n}) ---")
        print(f"  {'Strategy':40s} {'All':>5s} {'CI':>10s}  {'Firm':>5s}  {'Warm':>5s}  {'Syc':>5s}")
        print(f"  {'-'*80}")

        # Baselines
        report("raw good", score_cases(better_in, worse_in, good_dir, n), labels, n)
        report("careful alone", score_cases(better_in, worse_in, careful_dir, n), labels, n)

        # Good + single corrective (additive)
        for w in [0.3, 0.5, 0.7, 1.0]:
            combo = good_dir + w * careful_dir
            combo = combo / (norm(combo) + 1e-12)
            report(f"good + {w:.1f}*careful", score_cases(better_in, worse_in, combo, n), labels, n)

        print()
        # Good + honest (counters sycophancy)
        for w in [0.3, 0.5, 0.7, 1.0]:
            combo = good_dir + w * honest_dir
            combo = combo / (norm(combo) + 1e-12)
            report(f"good + {w:.1f}*honest", score_cases(better_in, worse_in, combo, n), labels, n)

        print()
        # Good - failure modes (subtractive)
        for w in [0.3, 0.5, 0.7, 1.0]:
            combo = good_dir - w * sycophantic_dir
            combo = combo / (norm(combo) + 1e-12)
            report(f"good - {w:.1f}*sycophantic", score_cases(better_in, worse_in, combo, n), labels, n)

        print()
        for w in [0.3, 0.5, 0.7, 1.0]:
            combo = good_dir - w * flattering_dir
            combo = combo / (norm(combo) + 1e-12)
            report(f"good - {w:.1f}*flattering", score_cases(better_in, worse_in, combo, n), labels, n)

        print()
        # Good + multiple correctives
        for w in [0.3, 0.5]:
            combo = good_dir + w * (careful_dir + honest_dir + restrained_dir)
            combo = combo / (norm(combo) + 1e-12)
            report(f"good + {w:.1f}*(care+hon+rest)", score_cases(better_in, worse_in, combo, n), labels, n)

        # Good + all 5 tree terms
        for w in [0.3, 0.5, 0.7, 1.0]:
            combo = good_dir + w * (careful_dir + honest_dir + helpful_dir + thorough_dir + restrained_dir)
            combo = combo / (norm(combo) + 1e-12)
            report(f"good + {w:.1f}*(all 5 tree)", score_cases(better_in, worse_in, combo, n), labels, n)

        print()
        # Good - failure modes + tree terms (the full training wheels)
        combo = good_dir + 0.5 * (careful_dir + honest_dir + restrained_dir) - 0.5 * (sycophantic_dir + flattering_dir)
        combo = combo / (norm(combo) + 1e-12)
        report("good+0.5*correctives-0.5*failures", score_cases(better_in, worse_in, combo, n), labels, n)

        # Sum of all 5 tree directions (no good at all)
        combo = careful_dir + honest_dir + helpful_dir + thorough_dir + restrained_dir
        combo = combo / (norm(combo) + 1e-12)
        report("sum(all 5 tree) no good", score_cases(better_in, worse_in, combo, n), labels, n)

        # Best OOS strategies
        print(f"\n  --- OUT-OF-SAMPLE (n={n_oos}) ---")
        print(f"  {'Strategy':40s} {'Acc':>5s}")
        print(f"  {'-'*50}")

        strategies = {
            "raw good": good_dir,
            "careful alone": careful_dir,
        }
        for w in [0.5, 1.0]:
            c = good_dir + w * careful_dir
            strategies[f"good + {w:.1f}*careful"] = c / (norm(c) + 1e-12)
            c = good_dir + w * honest_dir
            strategies[f"good + {w:.1f}*honest"] = c / (norm(c) + 1e-12)
            c = good_dir - w * sycophantic_dir
            strategies[f"good - {w:.1f}*sycophantic"] = c / (norm(c) + 1e-12)

        for w in [0.5, 1.0]:
            c = good_dir + w * (careful_dir + honest_dir + helpful_dir + thorough_dir + restrained_dir)
            strategies[f"good + {w:.1f}*(all 5)"] = c / (norm(c) + 1e-12)

        c = careful_dir + honest_dir + helpful_dir + thorough_dir + restrained_dir
        strategies["sum(all 5) no good"] = c / (norm(c) + 1e-12)

        for name, direction in strategies.items():
            outcomes = score_cases(better_oos, worse_oos, direction, n_oos)
            print(f"  {name:40s} {sum(outcomes)/n_oos:5.0%}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
