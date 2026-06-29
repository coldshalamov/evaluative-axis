#!/usr/bin/env python3
"""Cross-model disagreement analysis + warmth-independent voting.

Questions:
1. What predicts whether models agree or disagree on "careful"?
   - Response length difference? Delta magnitude? Warmth content?
2. Does majority voting among ONLY warmth-independent axes
   (careful, thorough, deliberate) beat single-axis "careful"?
3. Is the disagreement structured (e.g., one model consistently wrong
   on a case type) or random?
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
BATTERY_EXPANSION = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

WARMTH_INDEPENDENT = [
    ("careful", "Careful", "Reckless"),
    ("thorough", "Thorough", "Superficial"),
    ("deliberate", "Deliberate", "Impulsive"),
]

ALL_AXES = WARMTH_INDEPENDENT + [
    ("good", "Good", "Bad"),
    ("kind", "Kind", "Cruel"),
    ("honest", "Honest", "Dishonest"),
    ("rigorous", "Rigorous", "Sloppy"),
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

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    main_cases = original + warmth
    n_main = len(main_cases)
    n_orig = len(original)

    expansion_cases = []
    for f in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(f))
    n_exp = len(expansion_cases)

    all_cases = main_cases + expansion_cases
    n_all = len(all_cases)

    model_axis_correct = {}
    model_axis_deltas = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"Processing {short}...")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        model_axis_correct[short] = {}
        model_axis_deltas[short] = {}

        for name, pos, neg in ALL_AXES:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)

            deltas = []
            correct = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                d = sb - sw
                deltas.append(d)
                correct.append(1 if d > 0 else 0)

            model_axis_correct[short][name] = correct
            model_axis_deltas[short][name] = deltas

        del model
        gc.collect()

    models = list(model_axis_correct.keys())

    # ============================================================
    # 1. CROSS-MODEL AGREEMENT ON CAREFUL
    # ============================================================
    print(f"\n{'='*60}")
    print(f"1. CROSS-MODEL AGREEMENT ON CAREFUL (main, n={n_main})")
    print(f"{'='*60}")

    careful_votes_main = []
    for i in range(n_main):
        votes = sum(model_axis_correct[m]["careful"][i] for m in models)
        careful_votes_main.append(votes)

    for v in [3, 2, 1, 0]:
        n_v = sum(1 for cv in careful_votes_main if cv == v)
        print(f"  {v}/3 agree: {n_v} ({n_v/n_main:.0%})")

    print(f"\n  Pairwise agreement:")
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            agree = sum(1 for j in range(n_main)
                       if model_axis_correct[m1]["careful"][j] == model_axis_correct[m2]["careful"][j])
            print(f"    {m1} vs {m2}: {agree}/{n_main} ({agree/n_main:.0%})")

    # ============================================================
    # 2. WHAT PREDICTS DISAGREEMENT?
    # ============================================================
    print(f"\n{'='*60}")
    print(f"2. PREDICTORS OF DISAGREEMENT")
    print(f"{'='*60}")

    # 2a. Response length difference
    len_diffs = [len(c["better"]) - len(c["worse"]) for c in main_cases]
    abs_len_diffs = [abs(d) for d in len_diffs]

    for category, indices in [
        ("all-agree (3/3)", [i for i in range(n_main) if careful_votes_main[i] == 3]),
        ("all-disagree (0/3)", [i for i in range(n_main) if careful_votes_main[i] == 0]),
        ("mixed (1-2/3)", [i for i in range(n_main) if careful_votes_main[i] in [1, 2]]),
    ]:
        if not indices:
            continue
        mean_abs = np.mean([abs_len_diffs[i] for i in indices])
        mean_signed = np.mean([len_diffs[i] for i in indices])
        pct_better_longer = np.mean([1 if len_diffs[i] > 0 else 0 for i in indices])
        print(f"  {category:25s}: n={len(indices):3d}  "
              f"|len_diff|={mean_abs:.0f}  signed={mean_signed:+.0f}  "
              f"better_longer={pct_better_longer:.0%}")

    # 2b. Delta magnitude on careful
    print(f"\n  Delta magnitude by agreement level:")
    for category, indices in [
        ("all-agree (3/3)", [i for i in range(n_main) if careful_votes_main[i] == 3]),
        ("all-disagree (0/3)", [i for i in range(n_main) if careful_votes_main[i] == 0]),
        ("mixed (1-2/3)", [i for i in range(n_main) if careful_votes_main[i] in [1, 2]]),
    ]:
        if not indices:
            continue
        for m in models:
            mean_d = np.mean([abs(model_axis_deltas[m]["careful"][i]) for i in indices])
            print(f"    {category:25s} {m:30s}: mean|delta|={mean_d:.4f}")

    # 2c. Content type: orig vs warmth
    print(f"\n  Orig vs warmth agreement:")
    for source, idx_range in [("orig", range(n_orig)), ("warmth", range(n_orig, n_main))]:
        n_src = len(list(idx_range))
        all_agree = sum(1 for i in idx_range if careful_votes_main[i] == 3)
        none_agree = sum(1 for i in idx_range if careful_votes_main[i] == 0)
        print(f"    {source:8s}: n={n_src}  all-agree={all_agree} ({all_agree/n_src:.0%})  "
              f"none-agree={none_agree} ({none_agree/n_src:.0%})")

    # 2d. Per-model accuracy by content type
    print(f"\n  Per-model careful accuracy by content type:")
    for m in models:
        orig_acc = np.mean([model_axis_correct[m]["careful"][i] for i in range(n_orig)])
        warm_acc = np.mean([model_axis_correct[m]["careful"][i] for i in range(n_orig, n_main)])
        print(f"    {m:30s}: orig={orig_acc:.0%}  warmth={warm_acc:.0%}  gap={warm_acc-orig_acc:+.0%}")

    # ============================================================
    # 3. WARMTH-INDEPENDENT VOTING
    # ============================================================
    print(f"\n{'='*60}")
    print(f"3. WARMTH-INDEPENDENT VOTING")
    print(f"{'='*60}")

    wi_axes = [name for name, _, _ in WARMTH_INDEPENDENT]

    # 3a. Same-model voting: for each model, vote among warmth-independent axes
    print(f"\n  Same-model multi-axis voting (main battery, n={n_main}):")
    for m in models:
        # Single axis baselines
        careful_acc = np.mean([model_axis_correct[m]["careful"][i] for i in range(n_main)])
        thorough_acc = np.mean([model_axis_correct[m]["thorough"][i] for i in range(n_main)])
        deliberate_acc = np.mean([model_axis_correct[m]["deliberate"][i] for i in range(n_main)])

        # Majority vote: 2/3 or 3/3 agree
        vote_correct = 0
        for i in range(n_main):
            wi_votes = sum(model_axis_correct[m][ax][i] for ax in wi_axes)
            if wi_votes >= 2:
                vote_correct += 1
        vote_acc = vote_correct / n_main
        lo, hi = wilson_ci(vote_correct, n_main)

        # Sum of deltas (continuous)
        sum_correct = 0
        for i in range(n_main):
            total_delta = sum(model_axis_deltas[m][ax][i] for ax in wi_axes)
            if total_delta > 0:
                sum_correct += 1
        sum_acc = sum_correct / n_main

        print(f"    {m:30s}: careful={careful_acc:.0%}  thorough={thorough_acc:.0%}  "
              f"deliberate={deliberate_acc:.0%}  vote={vote_acc:.0%} [{lo:.0%},{hi:.0%}]  "
              f"sum={sum_acc:.0%}")

    # 3b. Same-model voting on EXPANSION
    print(f"\n  Same-model multi-axis voting (expansion, n={n_exp}):")
    for m in models:
        careful_acc = np.mean([model_axis_correct[m]["careful"][i] for i in range(n_main, n_all)])
        vote_correct = 0
        for i in range(n_main, n_all):
            wi_votes = sum(model_axis_correct[m][ax][i] for ax in wi_axes)
            if wi_votes >= 2:
                vote_correct += 1
        vote_acc = vote_correct / n_exp

        sum_correct = 0
        for i in range(n_main, n_all):
            total_delta = sum(model_axis_deltas[m][ax][i] for ax in wi_axes)
            if total_delta > 0:
                sum_correct += 1
        sum_acc = sum_correct / n_exp

        print(f"    {m:30s}: careful={careful_acc:.0%}  vote={vote_acc:.0%}  sum={sum_acc:.0%}")

    # 3c. Cross-model voting on careful
    print(f"\n  Cross-model voting on careful:")
    # Majority vote across models
    cross_correct_main = 0
    for i in range(n_main):
        cv = sum(model_axis_correct[m]["careful"][i] for m in models)
        if cv >= 2:
            cross_correct_main += 1
    cross_acc_main = cross_correct_main / n_main
    lo, hi = wilson_ci(cross_correct_main, n_main)
    print(f"    Main battery: {cross_acc_main:.0%} [{lo:.0%},{hi:.0%}]")

    cross_correct_exp = 0
    for i in range(n_main, n_all):
        cv = sum(model_axis_correct[m]["careful"][i] for m in models)
        if cv >= 2:
            cross_correct_exp += 1
    cross_acc_exp = cross_correct_exp / n_exp
    print(f"    Expansion: {cross_acc_exp:.0%}")

    # Best individual model comparison
    print(f"\n  Best single model comparison:")
    for m in models:
        main_acc = np.mean([model_axis_correct[m]["careful"][i] for i in range(n_main)])
        exp_acc = np.mean([model_axis_correct[m]["careful"][i] for i in range(n_main, n_all)])
        print(f"    {m:30s}: main={main_acc:.0%}  exp={exp_acc:.0%}")

    # ============================================================
    # 4. ORIG/WARMTH SPLIT FOR VOTING
    # ============================================================
    print(f"\n{'='*60}")
    print(f"4. CONTENT SPLIT: VOTING vs SINGLE AXIS")
    print(f"{'='*60}")

    for m in models:
        print(f"\n  {m}:")
        for source, idx_range in [("orig(50)", range(n_orig)), ("warmth(20)", range(n_orig, n_main))]:
            indices = list(idx_range)
            n_src = len(indices)

            careful_acc = np.mean([model_axis_correct[m]["careful"][i] for i in indices])
            good_acc = np.mean([model_axis_correct[m]["good"][i] for i in indices])

            vote_correct = sum(1 for i in indices
                             if sum(model_axis_correct[m][ax][i] for ax in wi_axes) >= 2)
            vote_acc = vote_correct / n_src

            sum_correct = sum(1 for i in indices
                            if sum(model_axis_deltas[m][ax][i] for ax in wi_axes) > 0)
            sum_acc = sum_correct / n_src

            print(f"    {source:12s}: good={good_acc:.0%}  careful={careful_acc:.0%}  "
                  f"vote={vote_acc:.0%}  sum={sum_acc:.0%}")

    # ============================================================
    # 5. CONFIDENCE ANALYSIS: does delta magnitude predict correctness?
    # ============================================================
    print(f"\n{'='*60}")
    print(f"5. CONFIDENCE CALIBRATION")
    print(f"{'='*60}")

    for m in models:
        deltas = [abs(model_axis_deltas[m]["careful"][i]) for i in range(n_main)]
        correct = [model_axis_correct[m]["careful"][i] for i in range(n_main)]

        median_d = np.median(deltas)
        high_conf = [(c, d) for c, d in zip(correct, deltas) if d >= median_d]
        low_conf = [(c, d) for c, d in zip(correct, deltas) if d < median_d]

        high_acc = np.mean([c for c, _ in high_conf]) if high_conf else 0
        low_acc = np.mean([c for c, _ in low_conf]) if low_conf else 0

        print(f"  {m:30s}: high-conf={high_acc:.0%} (n={len(high_conf)})  "
              f"low-conf={low_acc:.0%} (n={len(low_conf)})  "
              f"gap={high_acc-low_acc:+.0%}")

    # Save results
    summary = {
        "experiment": "Cross-model disagreement + warmth-independent voting",
        "date": "2026-06-28",
        "main_battery_n": n_main,
        "expansion_n": n_exp,
    }
    out_path = ROOT / "notes/research_cycles/tree_decomposition/crossmodel_disagreement.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
