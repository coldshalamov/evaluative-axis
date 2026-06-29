#!/usr/bin/env python3
"""Warmth-bias prediction test: can we predict case-level "good" outcomes
from case category alone?

The warmth-bias mechanism predicts: "good" succeeds when warmth aligns with
correctness (warmth cases) and fails when they conflict (firmness cases).

If this prediction holds at case-level across all models, the mechanism
is genuinely predictive — we can tell in advance whether "good" will get
a specific case right or wrong, just from knowing the case type.

We also test: can "careful" be predicted the same way? (It shouldn't,
because it's warmth-independent.)
"""

import json, gc, math
from pathlib import Path
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

AXES = [
    ("good", "Good", "Bad"),
    ("careful", "Careful", "Reckless"),
    ("restrained", "Restrained", "Unrestrained"),
    ("kind", "Kind", "Cruel"),
    ("thorough", "Thorough", "Superficial"),
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

    battery = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    all_cases = battery + warmth

    # Label each case
    labels = []
    for c in battery:
        if c["category"] == "anti_sycophancy":
            labels.append("sycophancy")
        else:
            labels.append("firmness")
    for c in warmth:
        labels.append("warmth")

    n = len(all_cases)
    print(f"Cases: {n}")
    print(f"  firmness: {labels.count('firmness')}")
    print(f"  warmth: {labels.count('warmth')}")
    print(f"  sycophancy: {labels.count('sycophancy')}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        for axis_name, pos, neg in AXES:
            p_emb = embed_fn([pos])[0]
            n_emb = embed_fn([neg])[0]
            axis = (p_emb - n_emb) / (norm(p_emb - n_emb) + 1e-12)

            b_scores = [float(np.dot(better_embs[i], axis)) for i in range(n)]
            w_scores = [float(np.dot(worse_embs[i], axis)) for i in range(n)]

            # Case-level outcomes
            outcomes = [b_scores[i] > w_scores[i] for i in range(n)]

            # Warmth-bias prediction: "good" succeeds on warmth cases, fails on firmness/sycophancy
            # For warmth-biased axes: predict SUCCESS on warmth, FAILURE on firmness+sycophancy
            warmth_pred = [labels[i] == "warmth" for i in range(n)]
            # For firmness-biased axes: predict SUCCESS on firmness, FAILURE on warmth
            firm_pred = [labels[i] == "firmness" for i in range(n)]

            # How well does the warmth-bias prediction predict actual outcomes?
            warmth_pred_correct = sum(1 for i in range(n) if warmth_pred[i] == outcomes[i])
            warmth_pred_acc = warmth_pred_correct / n

            # How well does firmness-bias prediction predict actual outcomes?
            firm_pred_correct = sum(1 for i in range(n) if firm_pred[i] == outcomes[i])
            firm_pred_acc = firm_pred_correct / n

            # Correlation between case label and outcome
            # Encode: warmth=1, firmness=0, sycophancy=0
            warmth_label = np.array([1.0 if labels[i] == "warmth" else 0.0 for i in range(n)])
            outcome_arr = np.array([1.0 if o else 0.0 for o in outcomes])
            if np.std(warmth_label) > 0 and np.std(outcome_arr) > 0:
                corr = float(np.corrcoef(warmth_label, outcome_arr)[0, 1])
            else:
                corr = 0.0

            acc = sum(outcomes) / n
            lo, hi = wilson_ci(sum(outcomes), n)

            model_results[axis_name] = {
                "accuracy": round(acc, 3),
                "warmth_pred_acc": round(warmth_pred_acc, 3),
                "firm_pred_acc": round(firm_pred_acc, 3),
                "warmth_outcome_corr": round(corr, 3),
            }

        # Print summary
        print(f"\n{'Axis':12s} {'Acc':>6s} {'WPred':>6s} {'FPred':>6s} {'Corr':>6s}  Interpretation")
        print("-" * 80)

        for axis_name, _, _ in AXES:
            r = model_results[axis_name]
            # Determine which prediction is better
            best_pred = max(r["warmth_pred_acc"], r["firm_pred_acc"])
            pred_type = "warmth" if r["warmth_pred_acc"] > r["firm_pred_acc"] else "firmness"

            if best_pred > 0.65:
                interp = f"strongly {pred_type}-biased (predictable)"
            elif best_pred > 0.55:
                interp = f"weakly {pred_type}-biased"
            else:
                interp = "unpredictable from case type"

            print(f"{axis_name:12s} {r['accuracy']:6.0%} {r['warmth_pred_acc']:6.0%} "
                  f"{r['firm_pred_acc']:6.0%} {r['warmth_outcome_corr']:+6.3f}  {interp}")

        all_results[short] = model_results

        del model
        gc.collect()

    # Cross-model consistency
    print(f"\n{'='*80}")
    print("CROSS-MODEL WARMTH PREDICTION CONSISTENCY")
    print(f"{'='*80}")
    print()
    print("Warmth-bias prediction: axis succeeds on warmth cases, fails on firmness/sycophancy")
    print("If this prediction accuracy is HIGH (>65%) on all 3 models, the mechanism is robust.\n")

    print(f"{'Axis':12s}", end="")
    for m in all_results:
        print(f" {m:>25s}", end="")
    print()
    print("-" * 90)

    for axis_name, _, _ in AXES:
        print(f"{axis_name:12s}", end="")
        for m in all_results:
            r = all_results[m][axis_name]
            print(f" {r['warmth_pred_acc']:25.0%}", end="")
        print()

    print("\n--- Key question: Is 'good' predictable and 'careful' not? ---")
    for axis_name in ["good", "careful"]:
        wpreds = [all_results[m][axis_name]["warmth_pred_acc"] for m in all_results]
        mean_wpred = np.mean(wpreds)
        print(f"  {axis_name}: warmth-bias prediction accuracy = {[f'{w:.0%}' for w in wpreds]}, mean = {mean_wpred:.0%}")

    # Save
    out = ROOT / "notes/research_cycles/warmth_prediction_test"
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / "warmth_prediction_results.json"
    outfile.write_text(json.dumps({
        "experiment": "Warmth-bias prediction test",
        "date": "2026-06-28",
        "n_cases": n,
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
