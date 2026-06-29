#!/usr/bin/env python3
"""Search for the universal axis set that works across ALL three models.

Strategy: instead of optimizing per-model, find the set whose WORST model
performance is highest (maximin across models). This confirmatory search uses
the fixed default recipe: standard User/Assistant framing and bipolar scoring
for every model and axis.
"""

import json, sys, gc, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scoring_recipe import (
    DEFAULT_SCORING_METHOD,
    framed_response,
    normalized_bipolar_axis,
    recipe_metadata,
)

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = {
    "careful":  {"pos": ["Careful"],  "neg": ["Reckless"]},
    "honest":   {"pos": ["Honest"],   "neg": ["Dishonest"]},
    "kind":     {"pos": ["Kind"],     "neg": ["Cruel"]},
    "thorough": {"pos": ["Thorough"], "neg": ["Superficial"]},
    "good":     {"pos": ["Good"],     "neg": ["Bad"]},
    "hard":     {"pos": ["Hard"],     "neg": ["Soft"]},
    "helpful":  {"pos": ["Helpful"],  "neg": ["Unhelpful"]},
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
    "wise":     {"pos": ["Wise"],     "neg": ["Foolish"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "warm":     {"pos": ["Warm"],     "neg": ["Cold"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "responsible": {"pos": ["Responsible"], "neg": ["Irresponsible"]},
    "clear":    {"pos": ["Clear"],    "neg": ["Confusing"]},
    "gentle":   {"pos": ["Gentle"],   "neg": ["Harsh"]},
    "constructive": {"pos": ["Constructive"], "neg": ["Destructive"]},
    "diligent": {"pos": ["Diligent"], "neg": ["Lazy"]},
    "supportive": {"pos": ["Supportive"], "neg": ["Dismissive"]},
    "patient":  {"pos": ["Patient"],  "neg": ["Impatient"]},
    "empathetic": {"pos": ["Empathetic"], "neg": ["Indifferent"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    all_cases = original + warmth
    n_orig = len(original)
    n_total = len(all_cases)
    axis_names = list(AXES.keys())
    results = {
        "search_completed": True,
        "n_axes": len(AXES),
        "models": MODELS,
        "scoring_recipe": recipe_metadata("default_universal_search"),
        "best_majority": {},
        "best_sum_margins": {},
        "per_model_best_3axis": {},
    }

    # Collect per-case correctness for all models × axes × methods
    all_model_data = {}

    for model_name in MODELS:
        print(f"Loading {model_name}...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        all_better = embed_fn([framed_response(c, "better") for c in all_cases])
        all_worse = embed_fn([framed_response(c, "worse") for c in all_cases])

        bipolar_correct = np.zeros((len(AXES), n_total))
        bipolar_margins = np.zeros((len(AXES), n_total))

        for ai, (axis_name, anchors) in enumerate(AXES.items()):
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_unit = normalized_bipolar_axis(pos_emb, neg_emb)

            for i in range(n_total):
                sb_bp = float(np.dot(all_better[i], axis_unit))
                sw_bp = float(np.dot(all_worse[i], axis_unit))
                bipolar_margins[ai, i] = sb_bp - sw_bp
                bipolar_correct[ai, i] = 1 if sb_bp > sw_bp else (0.5 if sb_bp == sw_bp else 0)

        all_model_data[model_name] = {
            "bipolar_correct": bipolar_correct,
            "bipolar_margins": bipolar_margins,
        }
        del model
        gc.collect()

    print(f"\n{'='*70}")
    print("CROSS-MODEL UNIVERSAL SEARCH")
    print(f"{'='*70}")

    print(f"Default method: {DEFAULT_SCORING_METHOD}")

    # Default recipe: fixed bipolar for every axis and model.
    best_per_model = {}
    for model_name in MODELS:
        d = all_model_data[model_name]
        best_per_model[model_name] = d["bipolar_correct"]

    # Search combos: for each combo, compute majority vote on each model,
    # then take the min across models as the "universal" score
    # Also require balanced (>50% on both splits per model)
    print(f"\n--- Best universal combos (maximin across models) ---")

    for n_axes in [3, 5]:
        best_universal = 0
        best_combo_info = None

        total_combos = 0
        for combo in itertools.combinations(range(len(AXES)), n_axes):
            total_combos += 1
            threshold = n_axes / 2
            min_balanced = 1.0

            model_accs = {}
            for model_name in MODELS:
                correct = best_per_model[model_name]
                votes = correct[list(combo)].sum(axis=0)
                orig_acc = float(np.mean(votes[:n_orig] > threshold))
                warm_acc = float(np.mean(votes[n_orig:] > threshold))
                comb_acc = float(np.mean(votes > threshold))
                balanced = min(orig_acc, warm_acc)
                min_balanced = min(min_balanced, balanced)
                model_accs[model_name] = {
                    "orig": round(orig_acc, 4), "warm": round(warm_acc, 4),
                    "comb": round(comb_acc, 4), "balanced": round(balanced, 4),
                }

            if min_balanced > best_universal:
                best_universal = min_balanced
                best_combo_info = {
                    "axes": [axis_names[i] for i in combo],
                    "per_model": model_accs,
                    "universal_balanced_min": round(min_balanced, 4),
                }

        print(f"\n  Best universal {n_axes}-axis set (searched {total_combos} combos):")
        print(f"    Axes: {best_combo_info['axes']}")
        print(f"    Universal balanced min: {best_combo_info['universal_balanced_min']:.1%}")
        for m, a in best_combo_info['per_model'].items():
            short = m.split("/")[-1]
            print(f"      {short:25s}  orig={a['orig']:.1%}  warm={a['warm']:.1%}  comb={a['comb']:.1%}  balanced={a['balanced']:.1%}")
        results["best_majority"][str(n_axes)] = best_combo_info

    # Also search for universal using SUM-OF-MARGINS instead of majority vote
    print(f"\n--- Sum-of-margins universal search ---")

    for n_axes in [3, 5]:
        best_universal = 0
        best_combo_info = None

        for combo in itertools.combinations(range(len(AXES)), n_axes):
            min_balanced = 1.0
            model_accs = {}

            for model_name in MODELS:
                d = all_model_data[model_name]
                margins_sum = np.zeros(n_total)
                for ai in combo:
                    margins_sum += d["bipolar_margins"][ai]

                orig_acc = float(np.mean(margins_sum[:n_orig] > 0))
                warm_acc = float(np.mean(margins_sum[n_orig:] > 0))
                comb_acc = float(np.mean(margins_sum > 0))
                balanced = min(orig_acc, warm_acc)
                min_balanced = min(min_balanced, balanced)
                model_accs[model_name] = {
                    "orig": round(orig_acc, 4), "warm": round(warm_acc, 4),
                    "comb": round(comb_acc, 4), "balanced": round(balanced, 4),
                }

            if min_balanced > best_universal:
                best_universal = min_balanced
                best_combo_info = {
                    "axes": [axis_names[i] for i in combo],
                    "per_model": model_accs,
                    "universal_balanced_min": round(min_balanced, 4),
                }

        print(f"\n  Best SOM universal {n_axes}-axis set:")
        print(f"    Axes: {best_combo_info['axes']}")
        print(f"    Universal balanced min: {best_combo_info['universal_balanced_min']:.1%}")
        for m, a in best_combo_info['per_model'].items():
            short = m.split("/")[-1]
            print(f"      {short:25s}  orig={a['orig']:.1%}  warm={a['warm']:.1%}  comb={a['comb']:.1%}  balanced={a['balanced']:.1%}")
        results["best_sum_margins"][str(n_axes)] = best_combo_info

    # Finally: what if we use DIFFERENT axis sets per model?
    # Print the gap between universal best and per-model best
    print(f"\n--- Per-model best vs universal best (3-axis) ---")
    for model_name in MODELS:
        d = all_model_data[model_name]
        correct = best_per_model[model_name]
        best_model_bal = 0
        best_model_combo = None

        for combo in itertools.combinations(range(len(AXES)), 3):
            votes = correct[list(combo)].sum(axis=0)
            threshold = 1.5
            orig_acc = float(np.mean(votes[:n_orig] > threshold))
            warm_acc = float(np.mean(votes[n_orig:] > threshold))
            balanced = min(orig_acc, warm_acc)
            if balanced > best_model_bal:
                best_model_bal = balanced
                best_model_combo = [axis_names[i] for i in combo]

        short = model_name.split("/")[-1]
        print(f"  {short:25s}  per-model best: {best_model_combo} bal={best_model_bal:.1%}")
        results["per_model_best_3axis"][model_name] = {
            "axes": best_model_combo,
            "balanced": round(float(best_model_bal), 4),
        }

    # Save the auditable summary
    out_dir = ROOT / "notes/research_cycles/universal_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "search_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
