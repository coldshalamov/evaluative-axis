#!/usr/bin/env python3
"""Cross-model verification of tree decomposition correlations.

The original run_tree_correlations.py only ran on Nomic. This script runs
the good→child score-delta correlations on all 3 models to verify whether
"careful is uncorrelated with good while other children correlate strongly"
holds cross-model.
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

L1_TERMS = {
    "good":        ("Good",        "Bad"),
    "careful":     ("Careful",     "Reckless"),
    "honest":      ("Honest",      "Dishonest"),
    "kind":        ("Kind",        "Cruel"),
    "wise":        ("Wise",        "Foolish"),
    "helpful":     ("Helpful",     "Unhelpful"),
    "thorough":    ("Thorough",    "Superficial"),
    "fair":        ("Fair",        "Unfair"),
    "responsible": ("Responsible", "Irresponsible"),
    "clear":       ("Clear",      "Confusing"),
    "respectful":  ("Respectful", "Disrespectful"),
}

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

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

        deltas = {}
        axes = {}
        for name, (pos_word, neg_word) in L1_TERMS.items():
            pos = embed_fn([pos_word]).mean(axis=0)
            neg = embed_fn([neg_word]).mean(axis=0)
            axis = (pos - neg) / (norm(pos - neg) + 1e-12)
            axes[name] = axis
            d = []
            for i in range(n):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                d.append(sb - sw)
            deltas[name] = np.array(d)

        # Good -> L1 child score-delta correlations
        print(f"\n--- Good -> L1 child score-delta correlations ---")
        children = [k for k in L1_TERMS if k != "good"]
        for child in sorted(children):
            r = np.corrcoef(deltas["good"], deltas[child])[0, 1]
            t_stat = r * math.sqrt((n - 2) / (1 - r**2 + 1e-12))
            print(f"  good -> {child:12s}: r={r:+.3f}  t={t_stat:+.2f}  (n={n})")

        # Axis vector cosines: good vs each child
        print(f"\n--- Axis vector cosines (good vs child) ---")
        for child in sorted(children):
            cos = float(np.dot(axes["good"], axes[child]))
            print(f"  good vs {child:12s}: cos={cos:+.3f}")

        # Accuracy of each term
        print(f"\n--- Per-term accuracy ---")
        for name in ["good"] + sorted(children):
            acc = np.mean([1 if d > 0 else (0.5 if d == 0 else 0) for d in deltas[name]])
            print(f"  {name:12s}: {acc:.1%}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
