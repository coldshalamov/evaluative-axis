#!/usr/bin/env python3
"""Warmth sensitivity analysis: split cases by kind_delta to see if
good's failures correlate with warmth direction while careful stays flat.

Note: this split uses an embedding-derived variable (kind_delta) that
correlates with the axes being tested, so it re-expresses the §4.18
correlation structure rather than testing it independently. The content
split (orig vs warmth battery) is the independent test.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

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
    all_cases = original + warmth
    n = len(all_cases)

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {short}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        # Build axes
        axes = {}
        for name, pos, neg in [("good", "Good", "Bad"), ("careful", "Careful", "Reckless"),
                                ("kind", "Kind", "Cruel"), ("honest", "Honest", "Dishonest"),
                                ("thorough", "Thorough", "Superficial")]:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axes[name] = (p - n_) / (norm(p - n_) + 1e-12)

        # Score all cases on all axes
        deltas = {}
        for name, axis in axes.items():
            d = []
            for i in range(n):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                d.append(sb - sw)
            deltas[name] = d

        correct = {name: [1 if d > 0 else 0 for d in deltas[name]] for name in axes}

        # Split by kind_delta: does the warmer response win?
        kind_d = deltas["kind"]
        warmer_worse = [i for i in range(n) if kind_d[i] < 0]  # worse resp is warmer
        warmer_better = [i for i in range(n) if kind_d[i] >= 0]  # better resp is warmer/equal

        print(f"\nCases where WORSE response is warmer (kind_delta < 0): n={len(warmer_worse)}")
        print(f"Cases where BETTER response is warmer (kind_delta >= 0): n={len(warmer_better)}")

        print(f"\n--- Accuracy on 'worse-is-warmer' cases (hardest for warmth-biased axes) ---")
        for name in ["good", "careful", "thorough", "kind", "honest"]:
            if warmer_worse:
                acc = np.mean([correct[name][i] for i in warmer_worse])
                k = sum(correct[name][i] for i in warmer_worse)
                lo, hi = wilson_ci(k, len(warmer_worse))
                print(f"  {name:12s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={len(warmer_worse)}")

        print(f"\n--- Accuracy on 'better-is-warmer' cases ---")
        for name in ["good", "careful", "thorough", "kind", "honest"]:
            if warmer_better:
                acc = np.mean([correct[name][i] for i in warmer_better])
                k = sum(correct[name][i] for i in warmer_better)
                lo, hi = wilson_ci(k, len(warmer_better))
                print(f"  {name:12s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={len(warmer_better)}")

        print(f"\n--- Accuracy gap (better-is-warmer minus worse-is-warmer) ---")
        for name in ["good", "careful", "thorough", "kind", "honest"]:
            if warmer_worse and warmer_better:
                ww_acc = np.mean([correct[name][i] for i in warmer_worse])
                wb_acc = np.mean([correct[name][i] for i in warmer_better])
                gap = wb_acc - ww_acc
                print(f"  {name:12s}: gap={gap:+.0%}  (warmer-better {wb_acc:.0%} vs warmer-worse {ww_acc:.0%})")

        # Cross-tabulation: kind correctness vs good/careful correctness
        print(f"\n--- When kind gets it RIGHT, does good/careful also? ---")
        kind_right = [i for i in range(n) if correct["kind"][i] == 1]
        kind_wrong = [i for i in range(n) if correct["kind"][i] == 0]

        for name in ["good", "careful", "thorough"]:
            if kind_right:
                kr_acc = np.mean([correct[name][i] for i in kind_right])
            else:
                kr_acc = float('nan')
            if kind_wrong:
                kw_acc = np.mean([correct[name][i] for i in kind_wrong])
            else:
                kw_acc = float('nan')
            print(f"  {name:12s}: kind-right={kr_acc:.0%} (n={len(kind_right)})  "
                  f"kind-wrong={kw_acc:.0%} (n={len(kind_wrong)})")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
