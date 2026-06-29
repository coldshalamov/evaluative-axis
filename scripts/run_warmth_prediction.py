#!/usr/bin/env python3
"""Test whether the warmth-bias theory predicts individual case outcomes.

Prediction: "good" should get a case right when the better response is warmer
(kind axis positive), and wrong when the better response is less warm. If so,
we can predict good's behavior from warmth alone.

Also tests: when good and careful disagree, does kind always side with good?
This would confirm that the disagreement is specifically about warmth.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
EXPANSION = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = [
    ("good", "Good", "Bad"),
    ("careful", "Careful", "Reckless"),
    ("kind", "Kind", "Cruel"),
    ("thorough", "Thorough", "Superficial"),
    ("honest", "Honest", "Dishonest"),
    ("helpful", "Helpful", "Unhelpful"),
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

    orig = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    cases = orig + warmth
    n = len(cases)

    exp_cases = []
    for f in sorted(EXPANSION.glob("*.jsonl")):
        exp_cases.extend(read_jsonl(f))
    all_cases = cases + exp_cases
    n_all = len(all_cases)

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\nProcessing {short}...")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        axis_correct = {}
        axis_deltas = {}
        for name, pos, neg in AXES:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)
            deltas = []
            correct = []
            for i in range(n_all):
                d = float(np.dot(better_embs[i], axis) - np.dot(worse_embs[i], axis))
                deltas.append(d)
                correct.append(1 if d > 0 else 0)
            axis_correct[name] = correct
            axis_deltas[name] = deltas

        # ============================================================
        # 1. WARMTH PREDICTS GOOD'S BEHAVIOR
        # ============================================================
        print(f"\n{'='*60}")
        print(f"1. WARMTH (kind) PREDICTS GOOD'S BEHAVIOR ({short}, n={n})")
        print(f"{'='*60}")

        # When kind says better is warmer, does good agree?
        kind_pos = [i for i in range(n) if axis_deltas["kind"][i] > 0]
        kind_neg = [i for i in range(n) if axis_deltas["kind"][i] <= 0]

        good_when_kind_pos = sum(axis_correct["good"][i] for i in kind_pos) / len(kind_pos) if kind_pos else 0
        good_when_kind_neg = sum(axis_correct["good"][i] for i in kind_neg) / len(kind_neg) if kind_neg else 0

        print(f"  When better response IS warmer (kind>0, n={len(kind_pos)}):")
        print(f"    good accuracy: {good_when_kind_pos:.0%}")
        print(f"  When better response is NOT warmer (kind<=0, n={len(kind_neg)}):")
        print(f"    good accuracy: {good_when_kind_neg:.0%}")
        print(f"  Gap: {good_when_kind_pos - good_when_kind_neg:+.0%}")

        # Same for careful (should NOT show the pattern)
        careful_when_kind_pos = sum(axis_correct["careful"][i] for i in kind_pos) / len(kind_pos) if kind_pos else 0
        careful_when_kind_neg = sum(axis_correct["careful"][i] for i in kind_neg) / len(kind_neg) if kind_neg else 0

        print(f"\n  Careful (should NOT track warmth):")
        print(f"    When warmer better (n={len(kind_pos)}): careful={careful_when_kind_pos:.0%}")
        print(f"    When warmer worse (n={len(kind_neg)}): careful={careful_when_kind_neg:.0%}")
        print(f"    Gap: {careful_when_kind_pos - careful_when_kind_neg:+.0%}")

        # ============================================================
        # 2. GOOD-CAREFUL DISAGREEMENTS: DOES KIND SIDE WITH GOOD?
        # ============================================================
        print(f"\n{'='*60}")
        print(f"2. WHEN GOOD AND CAREFUL DISAGREE, DOES KIND SIDE WITH GOOD?")
        print(f"{'='*60}")

        disagree = [i for i in range(n) if axis_correct["good"][i] != axis_correct["careful"][i]]
        good_right = [i for i in disagree if axis_correct["good"][i] == 1]
        careful_right = [i for i in disagree if axis_correct["careful"][i] == 1]

        print(f"  Total disagreements: {len(disagree)}/{n}")

        if good_right:
            kind_sides_good_r = sum(1 for i in good_right if axis_correct["kind"][i] == 1) / len(good_right)
            print(f"  When good is right (n={len(good_right)}): kind also right {kind_sides_good_r:.0%}")
        if careful_right:
            kind_sides_good_w = sum(1 for i in careful_right if axis_correct["kind"][i] == 0) / len(careful_right)
            print(f"  When careful is right (n={len(careful_right)}): kind sides with good (wrong) {kind_sides_good_w:.0%}")

        # Detailed: what axis agrees with whom during disagreements
        print(f"\n  Alignment during good-careful disagreements:")
        for axis_name in ["kind", "helpful", "honest", "thorough"]:
            if axis_name in axis_correct:
                sides_good = sum(1 for i in disagree if axis_correct[axis_name][i] == axis_correct["good"][i])
                sides_careful = sum(1 for i in disagree if axis_correct[axis_name][i] == axis_correct["careful"][i])
                print(f"    {axis_name:10s}: sides with good {sides_good}/{len(disagree)} ({sides_good/len(disagree):.0%}), "
                      f"sides with careful {sides_careful}/{len(disagree)} ({sides_careful/len(disagree):.0%})")

        # ============================================================
        # 3. CAN WE PREDICT GOOD'S ACCURACY PER-CASE FROM KIND DELTA?
        # ============================================================
        print(f"\n{'='*60}")
        print(f"3. BINARY PREDICTION: kind_delta > 0 => good_correct")
        print(f"{'='*60}")

        # Use kind delta as a predictor of good's correctness
        pred_correct = sum(1 for i in range(n)
                         if (axis_deltas["kind"][i] > 0 and axis_correct["good"][i] == 1) or
                            (axis_deltas["kind"][i] <= 0 and axis_correct["good"][i] == 0))
        pred_acc = pred_correct / n
        lo, hi = wilson_ci(pred_correct, n)
        print(f"  Prediction accuracy: {pred_correct}/{n} = {pred_acc:.0%} [{lo:.0%}, {hi:.0%}]")

        # Same test: use helpful delta
        pred_h = sum(1 for i in range(n)
                    if (axis_deltas["helpful"][i] > 0 and axis_correct["good"][i] == 1) or
                       (axis_deltas["helpful"][i] <= 0 and axis_correct["good"][i] == 0))
        print(f"  Using helpful as predictor: {pred_h}/{n} = {pred_h/n:.0%}")

        # Control: can kind predict careful's behavior? (should be ~50%)
        pred_careful = sum(1 for i in range(n)
                         if (axis_deltas["kind"][i] > 0 and axis_correct["careful"][i] == 1) or
                            (axis_deltas["kind"][i] <= 0 and axis_correct["careful"][i] == 0))
        print(f"  Control: kind predicts careful: {pred_careful}/{n} = {pred_careful/n:.0%} (should be ~50%)")

        # ============================================================
        # 4. EXPANSION BATTERY (out-of-sample)
        # ============================================================
        print(f"\n{'='*60}")
        print(f"4. OOS TEST: WARMTH PREDICTION ON EXPANSION BATTERY (n={len(exp_cases)})")
        print(f"{'='*60}")

        n_exp = len(exp_cases)
        exp_range = range(n, n_all)

        kind_pos_e = [i for i in exp_range if axis_deltas["kind"][i] > 0]
        kind_neg_e = [i for i in exp_range if axis_deltas["kind"][i] <= 0]

        if kind_pos_e:
            good_e_pos = sum(axis_correct["good"][i] for i in kind_pos_e) / len(kind_pos_e)
            print(f"  Warmer better (n={len(kind_pos_e)}): good={good_e_pos:.0%}")
        if kind_neg_e:
            good_e_neg = sum(axis_correct["good"][i] for i in kind_neg_e) / len(kind_neg_e)
            print(f"  Warmer worse (n={len(kind_neg_e)}): good={good_e_neg:.0%}")

        pred_e = sum(1 for i in exp_range
                    if (axis_deltas["kind"][i] > 0 and axis_correct["good"][i] == 1) or
                       (axis_deltas["kind"][i] <= 0 and axis_correct["good"][i] == 0))
        print(f"  Prediction accuracy: {pred_e}/{n_exp} = {pred_e/n_exp:.0%}")

        # ============================================================
        # 5. QUADRANT ANALYSIS: good x careful x kind
        # ============================================================
        print(f"\n{'='*60}")
        print(f"5. QUADRANT: good_correct x kind_positive")
        print(f"{'='*60}")

        quad = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
        for i in range(n):
            g_ok = axis_correct["good"][i]
            kp = 1 if axis_deltas["kind"][i] > 0 else 0
            key = ("T" if g_ok else "F") + ("T" if kp else "F")
            quad[key] += 1

        print(f"  good_right + kind_positive (TT): {quad['TT']}")
        print(f"  good_right + kind_negative (TF): {quad['TF']}")
        print(f"  good_wrong + kind_positive (FT): {quad['FT']}")
        print(f"  good_wrong + kind_negative (FF): {quad['FF']}")

        # Chi-squared (2x2)
        a, b, c, d = quad["TT"], quad["TF"], quad["FT"], quad["FF"]
        total = a + b + c + d
        if total > 0 and (a+b)*(c+d)*(a+c)*(b+d) > 0:
            chi2 = (total * (a*d - b*c)**2) / ((a+b)*(c+d)*(a+c)*(b+d))
            print(f"  Chi-squared: {chi2:.1f} (>3.84 = p<0.05)")

        all_results[short] = {
            "kind_pos_good_acc": round(good_when_kind_pos, 3),
            "kind_neg_good_acc": round(good_when_kind_neg, 3),
            "kind_pos_careful_acc": round(careful_when_kind_pos, 3),
            "kind_neg_careful_acc": round(careful_when_kind_neg, 3),
            "prediction_acc": round(pred_acc, 3),
            "n_disagree": len(disagree),
        }

        del model
        gc.collect()

    # Save
    out = ROOT / "notes/research_cycles/tree_decomposition/warmth_prediction.json"
    out.write_text(json.dumps({"experiment": "Warmth predicts good's behavior",
                               "date": "2026-06-28",
                               "results": all_results}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
