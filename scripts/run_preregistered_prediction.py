#!/usr/bin/env python3
"""Pre-registered prediction test: can the warmth-bias theory predict novel terms?

PREDICTIONS (declared before running):
  Warmth-INDEPENDENT (caution/risk register, predicted |r_good| < 0.3):
    prudent/reckless, vigilant/negligent, scrupulous/careless, measured/impulsive
  Warmth-BIASED (praise/admiration register, predicted r_good > 0.4):
    exemplary/terrible, superb/awful, commendable/deplorable, outstanding/abysmal
  Warmth-BIASED (directly warm, predicted r_good > 0.4):
    gracious/rude, benevolent/malicious

Success criterion: >= 8/10 predictions correct on >= 2/3 models.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

REFERENCE_AXES = [
    ("good", "Good", "Bad"),
    ("careful", "Careful", "Reckless"),
]

PREDICTED_INDEPENDENT = [
    ("prudent", "Prudent", "Reckless"),
    ("vigilant", "Vigilant", "Negligent"),
    ("scrupulous", "Scrupulous", "Careless"),
    ("measured", "Measured", "Impulsive"),
]

PREDICTED_BIASED = [
    ("exemplary", "Exemplary", "Terrible"),
    ("superb", "Superb", "Awful"),
    ("commendable", "Commendable", "Deplorable"),
    ("outstanding", "Outstanding", "Abysmal"),
    ("gracious", "Gracious", "Rude"),
    ("benevolent", "Benevolent", "Malicious"),
]

ALL_TEST_AXES = PREDICTED_INDEPENDENT + PREDICTED_BIASED

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

    orig = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    cases = orig + warmth
    n = len(cases)
    n_orig = len(orig)

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases])

        axis_deltas = {}
        axis_correct = {}

        for name, pos, neg in REFERENCE_AXES + ALL_TEST_AXES:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)
            deltas = []
            correct = []
            for i in range(n):
                d = float(np.dot(better_embs[i], axis) - np.dot(worse_embs[i], axis))
                deltas.append(d)
                correct.append(1 if d > 0 else 0)
            axis_deltas[name] = deltas
            axis_correct[name] = correct

        good_deltas = axis_deltas["good"]

        print(f"\n{'Term':15s} {'Acc':>5s} {'Orig':>5s} {'Warm':>5s} {'r_good':>7s} {'p':>7s} {'Prediction':>12s} {'Correct?':>8s}")
        print("-" * 80)

        model_results = {}
        for name, pos, neg in ALL_TEST_AXES:
            acc_k = sum(axis_correct[name])
            acc = acc_k / n
            lo, hi = wilson_ci(acc_k, n)

            orig_acc = sum(axis_correct[name][i] for i in range(n_orig)) / n_orig
            warm_acc = sum(axis_correct[name][i] for i in range(n_orig, n)) / (n - n_orig)

            r, p = pearsonr(axis_deltas[name], good_deltas)

            is_independent = name in [x[0] for x in PREDICTED_INDEPENDENT]
            predicted = "independent" if is_independent else "biased"

            if is_independent:
                prediction_correct = abs(r) < 0.3
            else:
                prediction_correct = r > 0.4

            mark = "YES" if prediction_correct else "NO"

            print(f"{name:15s} {acc:5.0%} {orig_acc:5.0%} {warm_acc:5.0%} {r:+7.2f} {p:7.3f} {predicted:>12s} {mark:>8s}")

            model_results[name] = {
                "accuracy": round(acc, 3),
                "orig_accuracy": round(orig_acc, 3),
                "warmth_accuracy": round(warm_acc, 3),
                "r_good": round(float(r), 3),
                "p_value": round(float(p), 4),
                "predicted": predicted,
                "prediction_correct": bool(prediction_correct),
            }

        n_correct = sum(1 for v in model_results.values() if v["prediction_correct"])
        print(f"\nPrediction accuracy: {n_correct}/10")

        # Also show careful's correlation as reference
        r_careful, p_careful = pearsonr(axis_deltas["careful"], good_deltas)
        careful_acc = sum(axis_correct["careful"]) / n
        print(f"\nReference: careful acc={careful_acc:.0%}, r_good={r_careful:+.2f}")

        # Content split for each test axis
        print(f"\n{'Term':15s} {'Orig gap':>10s} {'Warmth gap':>10s}")
        print("-" * 40)
        for name, pos, neg in ALL_TEST_AXES:
            orig_acc = sum(axis_correct[name][i] for i in range(n_orig)) / n_orig
            warm_acc = sum(axis_correct[name][i] for i in range(n_orig, n)) / (n - n_orig)
            gap = warm_acc - orig_acc
            bias = "warmth-biased" if gap > 0.15 else ("firmness-biased" if gap < -0.15 else "balanced")
            print(f"{name:15s} {orig_acc:10.0%} {warm_acc:10.0%}  gap={gap:+.0%}  [{bias}]")

        all_results[short] = model_results

        del model
        gc.collect()

    # Summary across models
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*70}")

    for name, pos, neg in ALL_TEST_AXES:
        is_independent = name in [x[0] for x in PREDICTED_INDEPENDENT]
        predicted = "independent" if is_independent else "biased"
        models_correct = sum(1 for m in all_results if all_results[m][name]["prediction_correct"])
        rs = [all_results[m][name]["r_good"] for m in all_results]
        accs = [all_results[m][name]["accuracy"] for m in all_results]
        print(f"  {name:15s} pred={predicted:12s}  r_good=[{', '.join(f'{r:+.2f}' for r in rs)}]  "
              f"acc=[{', '.join(f'{a:.0%}' for a in accs)}]  correct on {models_correct}/3 models")

    total_correct = sum(
        1 for name, _, _ in ALL_TEST_AXES
        if sum(1 for m in all_results if all_results[m][name]["prediction_correct"]) >= 2
    )
    print(f"\nOverall: {total_correct}/10 predictions correct on >= 2/3 models")
    print(f"Success criterion: >= 8/10")

    out = ROOT / "notes/research_cycles/tree_decomposition/preregistered_prediction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "Pre-registered warmth-bias prediction test",
        "date": "2026-06-28",
        "predictions": {
            "independent": [x[0] for x in PREDICTED_INDEPENDENT],
            "biased": [x[0] for x in PREDICTED_BIASED],
            "criterion_independent": "|r_good| < 0.3",
            "criterion_biased": "r_good > 0.4",
        },
        "results": all_results,
        "total_correct_on_2_of_3": total_correct,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
