#!/usr/bin/env python3
"""Honest pooled analysis: careful on all 90 cases, fixed method, Wilson CIs.

No per-model method selection. Just bipolar, standard framing, report CIs.
Also check if expansion battery is just easier by comparing mean accuracy
across all axes.
"""

import json, sys, gc, math
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

AXES = {
    "careful":  {"pos": ["Careful"],  "neg": ["Reckless"]},
    "thorough": {"pos": ["Thorough"], "neg": ["Superficial"]},
    "hard":     {"pos": ["Hard"],     "neg": ["Soft"]},
    "kind":     {"pos": ["Kind"],     "neg": ["Cruel"]},
    "honest":   {"pos": ["Honest"],   "neg": ["Dishonest"]},
    "helpful":  {"pos": ["Helpful"],  "neg": ["Unhelpful"]},
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "good":     {"pos": ["Good"],     "neg": ["Bad"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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

    expansion_cases = []
    for cat_file in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(cat_file))
    n_exp = len(expansion_cases)

    all_cases = main_cases + expansion_cases
    n_all = len(all_cases)

    axis_names = list(AXES.keys())

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        # FIXED: standard framing, bipolar method for ALL
        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        print(f"\n--- Bipolar, standard framing, Wilson 95% CI ---")
        print(f"{'Axis':12s}  {'Main':>6s}  {'Exp':>6s}  {'Pooled':>6s}  {'CI':>15s}  {'Sig?':>5s}")

        exp_mean_accs = []
        main_mean_accs = []

        for axis_name, anchors in AXES.items():
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

            correct = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis_unit))
                sw = float(np.dot(worse_embs[i], axis_unit))
                correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

            main_acc = np.mean(correct[:n_main])
            exp_acc = np.mean(correct[n_main:]) if n_exp > 0 else None
            pooled_acc = np.mean(correct)
            k = sum(1 for c in correct if c >= 0.5)
            lo, hi = wilson_ci(k, n_all)
            sig = "YES" if lo > 0.5 else "no"

            main_mean_accs.append(main_acc)
            if exp_acc is not None:
                exp_mean_accs.append(exp_acc)

            exp_str = f"{exp_acc:.0%}" if exp_acc is not None else "N/A"
            print(f"  {axis_name:10s}  {main_acc:5.0%}   {exp_str:>5s}   {pooled_acc:5.0%}   [{lo:.0%}, {hi:.0%}]   {sig}")

        # Is expansion easier?
        print(f"\n  Mean accuracy across all axes:")
        print(f"    Main battery ({n_main} cases): {np.mean(main_mean_accs):.1%}")
        if exp_mean_accs:
            print(f"    Expansion ({n_exp} cases):     {np.mean(exp_mean_accs):.1%}")

        # Careful specifically: pooled, balanced
        print(f"\n  === CAREFUL pooled summary ===")
        pos_emb = embed_fn(["Careful"]).mean(axis=0)
        neg_emb = embed_fn(["Reckless"]).mean(axis=0)
        axis_unit = (pos_emb - neg_emb) / (norm(pos_emb - neg_emb) + 1e-12)

        correct = []
        for i in range(n_all):
            sb = float(np.dot(better_embs[i], axis_unit))
            sw = float(np.dot(worse_embs[i], axis_unit))
            correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

        orig = np.mean(correct[:len(original)])
        warm = np.mean(correct[len(original):n_main])
        exp = np.mean(correct[n_main:]) if n_exp > 0 else None
        pooled = np.mean(correct)
        k = sum(1 for c in correct if c >= 0.5)
        lo, hi = wilson_ci(k, n_all)

        print(f"    Orig ({len(original)}): {orig:.0%}")
        print(f"    Warm ({len(warmth)}): {warm:.0%}")
        if exp is not None:
            print(f"    Expansion ({n_exp}): {exp:.0%}")
        print(f"    Pooled ({n_all}): {pooled:.0%}  CI=[{lo:.0%}, {hi:.0%}]")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
