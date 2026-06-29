#!/usr/bin/env python3
"""Content split OOS validation: does the warmth-bias pattern hold on the
20-case expansion battery?

Tests whether:
1. Good shows warmth sensitivity on expansion cases
2. Careful stays stable
3. Per-category patterns are consistent

Also computes the Snowflake-specific question: why is Snowflake's good gap
so small (+12pt) compared to BGE-M3/Nomic (+68-69pt)?
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

    expansion_cases = []
    exp_categories = {}
    for f in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        cases = read_jsonl(f)
        cat = f.stem
        for c in cases:
            c["_category"] = cat
        exp_categories[cat] = cases
        expansion_cases.extend(cases)

    print(f"Main battery: {len(original)} orig + {len(warmth)} warmth = {len(main_cases)}")
    print(f"Expansion: {len(expansion_cases)} cases across {len(exp_categories)} categories")
    for cat, cases in exp_categories.items():
        print(f"  {cat}: {len(cases)} cases")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {short}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed everything
        all_cases = main_cases + expansion_cases
        n_all = len(all_cases)
        n_main = len(main_cases)
        n_orig = len(original)
        n_warmth = len(warmth)
        n_exp = len(expansion_cases)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        # Build axes
        terms = [
            ("good", "Good", "Bad"),
            ("careful", "Careful", "Reckless"),
            ("kind", "Kind", "Cruel"),
            ("thorough", "Thorough", "Superficial"),
            ("honest", "Honest", "Dishonest"),
        ]

        deltas = {}
        for name, pos, neg in terms:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)
            d = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                d.append(sb - sw)
            deltas[name] = d

        correct = {name: [1 if d > 0 else 0 for d in deltas[name]] for name in deltas}

        # --- Content split on MAIN battery ---
        print(f"\n--- Main battery content split ---")
        print(f"{'Term':12s} {'Orig(50)':>10s} {'Warmth(20)':>12s} {'Gap':>8s}")
        for name in ["good", "careful", "kind", "thorough", "honest"]:
            orig_acc = np.mean([correct[name][i] for i in range(n_orig)])
            warm_acc = np.mean([correct[name][i] for i in range(n_orig, n_main)])
            gap = warm_acc - orig_acc
            print(f"{name:12s} {orig_acc:10.0%} {warm_acc:12.0%} {gap:+8.0%}")

        # --- Expansion battery overall ---
        print(f"\n--- Expansion battery overall (n={n_exp}) ---")
        for name in ["good", "careful", "kind", "thorough", "honest"]:
            exp_correct = sum(correct[name][i] for i in range(n_main, n_all))
            acc = exp_correct / n_exp
            lo, hi = wilson_ci(exp_correct, n_exp)
            print(f"  {name:12s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]")

        # --- Expansion per-category ---
        print(f"\n--- Expansion per-category ---")
        offset = n_main
        for cat, cases in exp_categories.items():
            n_cat = len(cases)
            print(f"\n  {cat} (n={n_cat}):")
            for name in ["good", "careful"]:
                cat_correct = sum(correct[name][offset + j] for j in range(n_cat))
                acc = cat_correct / n_cat
                print(f"    {name:12s}: {acc:.0%} ({cat_correct}/{n_cat})")
            offset += n_cat

        # --- Kind_delta split on expansion (for comparison) ---
        print(f"\n--- Expansion kind_delta split ---")
        exp_kind = deltas["kind"][n_main:]
        warmer_worse = [j for j in range(n_exp) if exp_kind[j] < 0]
        warmer_better = [j for j in range(n_exp) if exp_kind[j] >= 0]
        print(f"  Worse-is-warmer: n={len(warmer_worse)}")
        print(f"  Better-is-warmer: n={len(warmer_better)}")

        for name in ["good", "careful"]:
            if warmer_worse:
                ww = np.mean([correct[name][n_main + j] for j in warmer_worse])
            else:
                ww = float("nan")
            if warmer_better:
                wb = np.mean([correct[name][n_main + j] for j in warmer_better])
            else:
                wb = float("nan")
            gap = wb - ww if not (np.isnan(ww) or np.isnan(wb)) else float("nan")
            print(f"  {name:12s}: worse-warmer={ww:.0%}  better-warmer={wb:.0%}  gap={gap:+.0%}")

        # --- Snowflake investigation: why is good's gap only +12pt? ---
        if "snowflake" in short.lower():
            print(f"\n--- Snowflake good gap investigation ---")
            # Check per-case deltas to see if there's a bimodal pattern
            orig_good_deltas = [deltas["good"][i] for i in range(n_orig)]
            warm_good_deltas = [deltas["good"][i] for i in range(n_orig, n_main)]
            print(f"  Good delta stats:")
            print(f"    Orig mean={np.mean(orig_good_deltas):.4f}  std={np.std(orig_good_deltas):.4f}")
            print(f"    Warmth mean={np.mean(warm_good_deltas):.4f}  std={np.std(warm_good_deltas):.4f}")
            # Check careful
            orig_careful_deltas = [deltas["careful"][i] for i in range(n_orig)]
            warm_careful_deltas = [deltas["careful"][i] for i in range(n_orig, n_main)]
            print(f"  Careful delta stats:")
            print(f"    Orig mean={np.mean(orig_careful_deltas):.4f}  std={np.std(orig_careful_deltas):.4f}")
            print(f"    Warmth mean={np.mean(warm_careful_deltas):.4f}  std={np.std(warm_careful_deltas):.4f}")

        del model
        gc.collect()

    # Save results
    results = {"experiment": "Content split OOS validation", "date": "2026-06-28"}
    out_path = ROOT / "notes/research_cycles/tree_decomposition/content_split_oos.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
