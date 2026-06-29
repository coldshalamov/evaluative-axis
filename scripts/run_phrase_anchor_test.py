#!/usr/bin/env python3
"""Phrase anchor test: do multi-word anchors combining independent terms
outperform single-word "Careful"?

The warmth-bias analysis shows "careful" and "thorough" are the two
warmth-independent children of "good." Testing whether combining them
in a phrase creates a better single axis:
  - "Careful" (baseline)
  - "Careful and thorough"
  - "A careful, thorough response"
  - "This response is careful"
  - "Careful Thorough" (no grammar)
  - "Meticulous" (synonym at different frequency)
  - "Rigorous" (adjacent concept)
  - "Diligent" (another effort term)
  - "Precise and careful"

Also tests whether sentence-frame anchors help or hurt.
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

ANCHORS = [
    ("careful_single", ["Careful"], ["Reckless"]),
    ("careful_thorough", ["Careful and thorough"], ["Reckless and superficial"]),
    ("careful_thorough_bare", ["Careful Thorough"], ["Reckless Superficial"]),
    ("careful_sentence", ["This response is careful"], ["This response is reckless"]),
    ("careful_thorough_sentence", ["A careful, thorough response"], ["A reckless, superficial response"]),
    ("meticulous", ["Meticulous"], ["Sloppy"]),
    ("rigorous", ["Rigorous"], ["Sloppy"]),
    ("diligent", ["Diligent"], ["Negligent"]),
    ("precise_careful", ["Precise and careful"], ["Sloppy and reckless"]),
    ("conscientious", ["Conscientious"], ["Negligent"]),
    ("thorough_single", ["Thorough"], ["Superficial"]),
    ("deliberate", ["Deliberate"], ["Impulsive"]),
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

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {short}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        model_results = {}

        print(f"\n{'Anchor':30s} {'Main':>6s} {'Orig':>6s} {'Warm':>6s} {'Gap':>6s} {'Exp':>6s}")
        print("-" * 80)

        for anchor_name, pos_words, neg_words in ANCHORS:
            p = embed_fn(pos_words).mean(axis=0)
            n_ = embed_fn(neg_words).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)

            deltas = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                deltas.append(sb - sw)

            correct = [1 if d > 0 else 0 for d in deltas]

            # Main battery
            main_correct = sum(correct[:n_main])
            main_acc = main_correct / n_main
            lo, hi = wilson_ci(main_correct, n_main)

            # Orig/warmth split
            orig_correct = sum(correct[:n_orig])
            orig_acc = orig_correct / n_orig
            warm_correct = sum(correct[n_orig:n_main])
            warm_acc = warm_correct / (n_main - n_orig)
            gap = warm_acc - orig_acc

            # Expansion
            exp_correct = sum(correct[n_main:])
            exp_acc = exp_correct / n_exp

            print(f"{anchor_name:30s} {main_acc:6.0%} {orig_acc:6.0%} {warm_acc:6.0%} {gap:+6.0%} {exp_acc:6.0%}")

            # Cosine with careful_single axis
            if anchor_name == "careful_single":
                careful_axis = axis.copy()

            model_results[anchor_name] = {
                "main": main_acc, "orig": orig_acc, "warmth": warm_acc,
                "gap": gap, "expansion": exp_acc,
                "main_ci": [lo, hi]
            }

        # Cosine similarities with careful baseline
        print(f"\n--- Cosine with careful_single axis ---")
        for anchor_name, pos_words, neg_words in ANCHORS:
            if anchor_name == "careful_single":
                continue
            p = embed_fn(pos_words).mean(axis=0)
            n_ = embed_fn(neg_words).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)
            cos = float(np.dot(axis, careful_axis))
            print(f"  {anchor_name:30s}: cos={cos:.3f}")

        results[short] = model_results

        del model
        gc.collect()

    # Save results
    out_path = ROOT / "notes/research_cycles/tree_decomposition/phrase_anchor_test.json"
    out_path.write_text(json.dumps({"experiment": "Phrase anchor test", "date": "2026-06-28", "results": results}, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
