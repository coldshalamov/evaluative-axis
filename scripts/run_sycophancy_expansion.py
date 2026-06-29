#!/usr/bin/env python3
"""Expanded anti-sycophancy test: good vs careful vs restrained on 20 cases.

The original battery has 5 anti-sycophancy cases where "good" picks the
sycophantic response 80-100% of the time and "careful" picks the correct
pushback 80% of the time. This is striking but rests on n=5.

This test expands to 20 anti-sycophancy cases (5 original + 15 new) to see
if the pattern holds with a sample large enough for Wilson CIs to be
informative.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
EXPANSION = ROOT / "notes/research_cycles/battery_expansion/anti_sycophancy_expansion.jsonl"

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

    all_battery = read_jsonl(BATTERY)
    orig_syc = [c for c in all_battery if c["category"] == "anti_sycophancy"]
    new_syc = read_jsonl(EXPANSION)
    all_syc = orig_syc + new_syc

    non_syc = [c for c in all_battery if c["category"] != "anti_sycophancy"]

    print(f"Original anti-sycophancy cases: {len(orig_syc)}")
    print(f"New anti-sycophancy cases: {len(new_syc)}")
    print(f"Total anti-sycophancy: {len(all_syc)}")
    print(f"Non-sycophancy battery cases: {len(non_syc)}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Score all anti-sycophancy cases
        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_syc])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_syc])

        # Also score non-sycophancy for comparison
        nonsyc_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in non_syc])
        nonsyc_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in non_syc])

        model_results = {}

        for axis_name, pos, neg in AXES:
            p = embed_fn([pos])[0]
            n_ = embed_fn([neg])[0]
            axis = (p - n_) / (norm(p - n_) + 1e-12)

            # Anti-sycophancy performance
            syc_correct = []
            for i in range(len(all_syc)):
                d = float(np.dot(better_embs[i], axis) - np.dot(worse_embs[i], axis))
                syc_correct.append(1 if d > 0 else 0)

            # Non-sycophancy performance
            nonsyc_correct = []
            for i in range(len(non_syc)):
                d = float(np.dot(nonsyc_better[i], axis) - np.dot(nonsyc_worse[i], axis))
                nonsyc_correct.append(1 if d > 0 else 0)

            syc_k = sum(syc_correct)
            syc_n = len(all_syc)
            syc_acc = syc_k / syc_n
            syc_lo, syc_hi = wilson_ci(syc_k, syc_n)

            orig_k = sum(syc_correct[:len(orig_syc)])
            orig_acc = orig_k / len(orig_syc) if orig_syc else 0

            new_k = sum(syc_correct[len(orig_syc):])
            new_acc = new_k / len(new_syc) if new_syc else 0

            nonsyc_k = sum(nonsyc_correct)
            nonsyc_acc = nonsyc_k / len(non_syc) if non_syc else 0

            model_results[axis_name] = {
                "syc_accuracy": round(syc_acc, 3),
                "syc_correct": syc_k,
                "syc_total": syc_n,
                "syc_wilson_ci": [round(syc_lo, 3), round(syc_hi, 3)],
                "orig_accuracy": round(orig_acc, 3),
                "new_accuracy": round(new_acc, 3),
                "nonsyc_accuracy": round(nonsyc_acc, 3),
                "per_case": syc_correct,
            }

        # Print results
        print(f"\n{'Axis':12s} {'Syc(20)':>8s} {'CI 95%':>15s} {'Orig(5)':>8s} {'New(15)':>8s} {'NonSyc':>8s}")
        print("-" * 65)
        for axis_name, _, _ in AXES:
            r = model_results[axis_name]
            ci = f"[{r['syc_wilson_ci'][0]:.0%}, {r['syc_wilson_ci'][1]:.0%}]"
            print(f"{axis_name:12s} {r['syc_accuracy']:8.0%} {ci:>15s} {r['orig_accuracy']:8.0%} {r['new_accuracy']:8.0%} {r['nonsyc_accuracy']:8.0%}")

        # Per-case detail
        print(f"\n--- Per-case: good vs careful vs restrained ---")
        for i, c in enumerate(all_syc):
            g = "Y" if model_results["good"]["per_case"][i] else "."
            ca = "Y" if model_results["careful"]["per_case"][i] else "."
            re = "Y" if model_results["restrained"]["per_case"][i] else "."
            src = "orig" if i < len(orig_syc) else "new"
            print(f"  {c['id']:25s} [{src:4s}] good={g} careful={ca} restrained={re}")

        # Key comparison
        g_syc = model_results["good"]["syc_accuracy"]
        c_syc = model_results["careful"]["syc_accuracy"]
        g_non = model_results["good"]["nonsyc_accuracy"]
        c_non = model_results["careful"]["nonsyc_accuracy"]
        print(f"\n  Good:    syc={g_syc:.0%}  non-syc={g_non:.0%}  gap={g_syc-g_non:+.0%}")
        print(f"  Careful: syc={c_syc:.0%}  non-syc={c_non:.0%}  gap={c_syc-c_non:+.0%}")

        # Clean per_case from results for JSON
        for ax in model_results:
            del model_results[ax]["per_case"]

        all_results[short] = model_results

        del model
        gc.collect()

    # Cross-model summary
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL SUMMARY: Anti-sycophancy accuracy (n=20)")
    print(f"{'='*80}")
    for axis_name, _, _ in AXES:
        accs = [all_results[m][axis_name]["syc_accuracy"] for m in all_results]
        mean_acc = np.mean(accs)
        print(f"  {axis_name:12s}: {', '.join(f'{a:.0%}' for a in accs)}  mean={mean_acc:.0%}")

    out = ROOT / "notes/research_cycles/tree_decomposition/sycophancy_expansion_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "Expanded anti-sycophancy test (n=20)",
        "date": "2026-06-28",
        "n_original": len(orig_syc),
        "n_new": len(new_syc),
        "n_total": len(all_syc),
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
