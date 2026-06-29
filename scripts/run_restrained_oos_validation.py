#!/usr/bin/env python3
"""OOS validation of 'restrained' as evaluative axis.

In-sample (balanced battery, n=70): restrained scored 54-64% accuracy.
This tests it on the held-out expansion battery (n=20, 4 categories)
to check whether the signal holds out-of-sample or was noise from
searching ~50 terms on a fixed battery.

Also tests 'temperate' (independent on 3/3 models in-sample).
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]

BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

EXPANSION_FILES = [
    ROOT / "notes/research_cycles/battery_expansion/factual_accuracy.jsonl",
    ROOT / "notes/research_cycles/battery_expansion/conciseness_completeness.jsonl",
    ROOT / "notes/research_cycles/battery_expansion/creative_quality.jsonl",
    ROOT / "notes/research_cycles/battery_expansion/nuance_context.jsonl",
]

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = [
    ("good", "Good", "Bad"),
    ("careful", "Careful", "Reckless"),
    ("restrained", "Restrained", "Unrestrained"),
    ("temperate", "Temperate", "Intemperate"),
    ("thorough", "Thorough", "Superficial"),
    ("deliberate", "Deliberate", "Impulsive"),
    ("rigorous", "Rigorous", "Lax"),
    ("measured", "Measured", "Impulsive"),
    ("prudent", "Prudent", "Reckless"),
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

def score_cases(cases, embed_fn, axes):
    better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases])
    worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases])

    results = {}
    all_deltas = {}
    for name, pos, neg in axes:
        p = embed_fn([pos])[0]
        n_ = embed_fn([neg])[0]
        axis = (p - n_) / (norm(p - n_) + 1e-12)

        deltas = []
        correct = []
        for i in range(len(cases)):
            d = float(np.dot(better_embs[i], axis) - np.dot(worse_embs[i], axis))
            deltas.append(d)
            correct.append(1 if d > 0 else 0)

        all_deltas[name] = deltas
        k = sum(correct)
        n = len(cases)
        lo, hi = wilson_ci(k, n)
        results[name] = {
            "accuracy": round(k / n, 3),
            "correct": k,
            "total": n,
            "wilson_95_ci": [round(lo, 3), round(hi, 3)],
            "correct_list": correct,
        }

    good_deltas = all_deltas.get("good", [0] * len(cases))
    for name in all_deltas:
        if name != "good" and len(all_deltas[name]) > 2:
            r, p = pearsonr(all_deltas[name], good_deltas)
            results[name]["r_good"] = round(float(r), 3)

    return results

def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    in_sample = orig + warmth

    oos = []
    oos_by_cat = {}
    for f in EXPANSION_FILES:
        cases = read_jsonl(f)
        cat = f.stem
        oos_by_cat[cat] = cases
        oos.extend(cases)

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        print(f"\n--- In-sample (balanced battery, n={len(in_sample)}) ---")
        is_results = score_cases(in_sample, embed_fn, AXES)
        print(f"{'Term':15s} {'Acc':>5s} {'CI 95%':>15s} {'r_good':>7s}")
        print("-" * 50)
        for name, _, _ in AXES:
            r = is_results[name]
            ci = f"[{r['wilson_95_ci'][0]:.0%}, {r['wilson_95_ci'][1]:.0%}]"
            rg = f"{r.get('r_good', 0):+.2f}" if 'r_good' in r else "  ref"
            print(f"{name:15s} {r['accuracy']:5.0%} {ci:>15s} {rg:>7s}")

        print(f"\n--- OOS expansion (n={len(oos)}) ---")
        oos_results = score_cases(oos, embed_fn, AXES)
        print(f"{'Term':15s} {'Acc':>5s} {'CI 95%':>15s} {'r_good':>7s}")
        print("-" * 50)
        for name, _, _ in AXES:
            r = oos_results[name]
            ci = f"[{r['wilson_95_ci'][0]:.0%}, {r['wilson_95_ci'][1]:.0%}]"
            rg = f"{r.get('r_good', 0):+.2f}" if 'r_good' in r else "  ref"
            print(f"{name:15s} {r['accuracy']:5.0%} {ci:>15s} {rg:>7s}")

        print(f"\n--- OOS by category ---")
        for cat, cases in oos_by_cat.items():
            cat_results = score_cases(cases, embed_fn, AXES)
            accs = {name: cat_results[name]['accuracy'] for name, _, _ in AXES}
            print(f"  {cat:30s} " + "  ".join(f"{name}={accs[name]:.0%}" for name, _, _ in AXES[:4]))

        # Per-case OOS detail for restrained vs careful
        print(f"\n--- Per-case OOS: restrained vs careful ---")
        restrained_correct = oos_results["restrained"]["correct_list"]
        careful_correct = oos_results["careful"]["correct_list"]
        for i, c in enumerate(oos):
            r_ok = "Y" if restrained_correct[i] else "."
            c_ok = "Y" if careful_correct[i] else "."
            print(f"  {c['id']:35s} restrained={r_ok}  careful={c_ok}")

        agree = sum(1 for i in range(len(oos)) if restrained_correct[i] == careful_correct[i])
        print(f"  Agreement: {agree}/{len(oos)}")

        all_results[short] = {
            "in_sample": {name: {k: v for k, v in is_results[name].items() if k != 'correct_list'}
                         for name, _, _ in AXES},
            "oos": {name: {k: v for k, v in oos_results[name].items() if k != 'correct_list'}
                   for name, _, _ in AXES},
        }

        del model
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: In-sample vs OOS accuracy")
    print(f"{'='*80}")
    print(f"{'Term':15s} {'In-sample (3 models)':>25s}  {'OOS (3 models)':>25s}")
    print("-" * 70)
    for name, _, _ in AXES:
        is_accs = [all_results[m]["in_sample"][name]["accuracy"] for m in all_results]
        oos_accs = [all_results[m]["oos"][name]["accuracy"] for m in all_results]
        is_str = ", ".join(f"{a:.0%}" for a in is_accs)
        oos_str = ", ".join(f"{a:.0%}" for a in oos_accs)
        is_mean = np.mean(is_accs)
        oos_mean = np.mean(oos_accs)
        delta = oos_mean - is_mean
        print(f"{name:15s} [{is_str}] mean={is_mean:.0%}  [{oos_str}] mean={oos_mean:.0%}  Δ={delta:+.0%}")

    out = ROOT / "notes/research_cycles/tree_decomposition/restrained_oos_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "OOS validation of restrained axis",
        "date": "2026-06-28",
        "in_sample_n": len(in_sample),
        "oos_n": len(oos),
        "axes_tested": [name for name, _, _ in AXES],
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
