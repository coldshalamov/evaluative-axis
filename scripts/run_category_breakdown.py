#!/usr/bin/env python3
"""Per-category accuracy breakdown for key axes.

Shows which evaluative dimensions each anchor captures best.
Does "Careful" work better on harm reduction? Does "Honest" only
work on truthfulness? This reveals the dimensional structure of
the evaluative space.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

KEY_AXES = {
    "careful_reckless":   {"positive": ["Careful"],    "negative": ["Reckless"]},
    "moderate_excessive": {"positive": ["Moderate"],   "negative": ["Excessive"]},
    "noble_base":         {"positive": ["Noble"],      "negative": ["Base"]},
    "good_bad":           {"positive": ["Good"],       "negative": ["Bad"]},
    "honest_dishonest":   {"positive": ["Honest"],     "negative": ["Dishonest"]},
    "true_false":         {"positive": ["True"],       "negative": ["False"]},
    "kind_cruel":         {"positive": ["Kind"],       "negative": ["Cruel"]},
    "useful_useless":     {"positive": ["Useful"],     "negative": ["Useless"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    categories = sorted(set(c["category"] for c in cases))
    print(f"Loaded {len(cases)} cases across {len(categories)} categories: {categories}")

    # Index cases by category
    cat_indices = defaultdict(list)
    for i, c in enumerate(cases):
        cat_indices[c["category"]].append(i)

    all_axes = {}
    all_axes.update({f"ml/{k}": v for k, v in ML_AXES.items()})
    all_axes.update({f"single/{k}": v for k, v in KEY_AXES.items()})

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        # Header
        cat_short = {
            "reasoning_rigor": "reason",
            "truthfulness": "truth",
            "harm_reduction": "harm",
            "persona_honesty": "persona",
            "anti_sycophancy": "syco",
            "context_binding": "ctx",
            "helpfulness": "help",
            "mixed_tradeoff": "mixed",
        }
        header = f"{'Axis':<35s} {'ALL':>5s}"
        for cat in categories:
            header += f" {cat_short.get(cat, cat[:6]):>7s}"
        print(f"\n{header}")
        print("-" * len(header))

        for axis_name, anchors in all_axes.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])

            # Overall accuracy
            correct_total = 0
            for i in range(len(cases)):
                s_b = float(np.dot(better_embs[i], axis_vec))
                s_w = float(np.dot(worse_embs[i], axis_vec))
                if s_b > s_w:
                    correct_total += 1
                elif s_b == s_w:
                    correct_total += 0.5
            overall = correct_total / len(cases)

            # Per-category
            cat_accs = {}
            row = f"  {axis_name:<33s} {overall:5.0%}"
            for cat in categories:
                idxs = cat_indices[cat]
                correct = 0
                for i in idxs:
                    s_b = float(np.dot(better_embs[i], axis_vec))
                    s_w = float(np.dot(worse_embs[i], axis_vec))
                    if s_b > s_w:
                        correct += 1
                    elif s_b == s_w:
                        correct += 0.5
                acc = correct / len(idxs)
                cat_accs[cat] = round(acc, 4)
                row += f" {acc:7.0%}"
            print(row)

            model_results[axis_name] = {
                "overall": round(overall, 4),
                "per_category": cat_accs,
            }

        all_results[model_name] = model_results
        del model

    # Find which single-word axes are best for each category
    print(f"\n{'='*80}")
    print("BEST SINGLE-WORD AXIS PER CATEGORY (across models)")
    print(f"{'='*80}")

    for cat in categories:
        print(f"\n  {cat}:")
        for model_name in MODELS:
            best_single = None
            best_acc = -1
            for axis_name in all_results[model_name]:
                if axis_name.startswith("single/"):
                    acc = all_results[model_name][axis_name]["per_category"][cat]
                    if acc > best_acc:
                        best_acc = acc
                        best_single = axis_name
            short_model = model_name.split("/")[-1][:15]
            print(f"    {short_model:>15s}: {best_single:<30s} = {best_acc:.0%}")

    out_dir = ROOT / "notes" / "research_cycles" / "category_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "category_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
