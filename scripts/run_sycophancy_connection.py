#!/usr/bin/env python3
"""Per-category accuracy for good vs careful — testing whether good's warmth
bias maps directly to sycophancy failure.

Hypothesis: "good" should fail hardest on anti-sycophancy and persona-honesty
cases (where the better response pushes back or refuses to pretend), because
the worse response in these categories is the WARMER one.
"""

import json, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    all_cases = orig + warmth

    categories = defaultdict(list)
    for i, c in enumerate(orig):
        cat = c.get("category", "unknown")
        categories[cat].append(i)

    for i, c in enumerate(warmth):
        categories["warmth"].append(len(orig) + i)

    print(f"Categories: {dict((k, len(v)) for k, v in categories.items())}")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        axes = {}
        for name, pos, neg in [("good", "Good", "Bad"), ("careful", "Careful", "Reckless"),
                                ("kind", "Kind", "Cruel"), ("thorough", "Thorough", "Superficial")]:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axes[name] = (p - n_) / (norm(p - n_) + 1e-12)

        correct = {}
        for name, axis in axes.items():
            c = []
            for i in range(len(all_cases)):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                c.append(1 if sb - sw > 0 else 0)
            correct[name] = c

        print(f"\n{'Category':25s} {'n':>3s}  {'good':>5s}  {'careful':>7s}  {'kind':>5s}  {'thorough':>8s}  {'good picks sycoph':>17s}")
        print("-" * 90)

        cat_order = ["anti_sycophancy", "persona_honesty", "truthfulness",
                     "reasoning_rigor", "harm_reduction", "context_binding",
                     "helpfulness", "mixed", "warmth"]

        for cat in cat_order:
            if cat not in categories:
                continue
            indices = categories[cat]
            n = len(indices)
            good_acc = sum(correct["good"][i] for i in indices) / n
            careful_acc = sum(correct["careful"][i] for i in indices) / n
            kind_acc = sum(correct["kind"][i] for i in indices) / n
            thorough_acc = sum(correct["thorough"][i] for i in indices) / n
            good_wrong = sum(1 for i in indices if not correct["good"][i])
            print(f"{cat:25s} {n:3d}  {good_acc:5.0%}  {careful_acc:7.0%}  {kind_acc:5.0%}  {thorough_acc:8.0%}  "
                  f"{good_wrong}/{n} wrong")

        # Anti-sycophancy detail
        print(f"\n--- Anti-sycophancy detail ---")
        for i in categories.get("anti_sycophancy", []):
            case = all_cases[i]
            g = "Y" if correct["good"][i] else "N"
            c = "Y" if correct["careful"][i] else "N"
            prompt_short = case["prompt"][:70]
            print(f"  good={g} careful={c}  {prompt_short}...")

        # Persona honesty detail
        print(f"\n--- Persona honesty detail ---")
        for i in categories.get("persona_honesty", []):
            case = all_cases[i]
            g = "Y" if correct["good"][i] else "N"
            c = "Y" if correct["careful"][i] else "N"
            prompt_short = case["prompt"][:70]
            print(f"  good={g} careful={c}  {prompt_short}...")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
