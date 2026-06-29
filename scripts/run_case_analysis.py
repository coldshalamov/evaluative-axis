#!/usr/bin/env python3
"""Look at actual cases: what fails, what succeeds, and WHY.

For each case, print the prompt, both responses, and scores.
Group by outcome to see patterns.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "BAAI/bge-m3",  # only model with real signal
]

TERMS = ["Good", "Enthusiastic", "Nice", "Flattering", "Helpful", "Honest",
         "Pleasant", "Careful", "Banana"]


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    from sentence_transformers import SentenceTransformer

    battery = read_jsonl(BATTERY)
    warmth_cases = read_jsonl(WARMTH)
    all_cases = battery + warmth_cases
    n = len(all_cases)

    labels = []
    for c in battery:
        labels.append("sycophancy" if c["category"] == "anti_sycophancy" else "firmness")
    for c in warmth_cases:
        labels.append("warmth")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"MODEL: {short}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        for i in range(n):
            better_embs[i] = better_embs[i] / (norm(better_embs[i]) + 1e-12)
            worse_embs[i] = worse_embs[i] / (norm(worse_embs[i]) + 1e-12)

        # Embed all terms
        term_embs = {}
        for t in TERMS:
            e = embed([t])[0]
            term_embs[t] = e / (norm(e) + 1e-12)

        # For each case, compute scores
        case_data = []
        for i in range(n):
            scores = {}
            for t in TERMS:
                b_cos = float(np.dot(better_embs[i], term_embs[t]))
                w_cos = float(np.dot(worse_embs[i], term_embs[t]))
                scores[t] = {"better": b_cos, "worse": w_cos, "margin": b_cos - w_cos}

            good_correct = scores["Good"]["margin"] > 0
            penalty_score_b = scores["Good"]["better"] - scores["Enthusiastic"]["better"]
            penalty_score_w = scores["Good"]["worse"] - scores["Enthusiastic"]["worse"]
            penalty_correct = penalty_score_b > penalty_score_w

            case_data.append({
                "idx": i,
                "id": all_cases[i].get("id", f"case_{i}"),
                "label": labels[i],
                "category": all_cases[i].get("category", ""),
                "prompt": all_cases[i]["prompt"],
                "better": all_cases[i]["better"],
                "worse": all_cases[i]["worse"],
                "scores": scores,
                "good_correct": good_correct,
                "penalty_correct": penalty_correct,
                "good_margin": scores["Good"]["margin"],
                "better_len": len(all_cases[i]["better"]),
                "worse_len": len(all_cases[i]["worse"]),
            })

        # Summary
        good_right = [c for c in case_data if c["good_correct"]]
        good_wrong = [c for c in case_data if not c["good_correct"]]
        pen_right = [c for c in case_data if c["penalty_correct"]]
        pen_wrong = [c for c in case_data if not c["penalty_correct"]]

        print(f"\ncos(Good) correct: {len(good_right)}/{n}")
        print(f"cos(Good)-cos(Enthusiastic) correct: {len(pen_right)}/{n}")

        # Cases where good fails
        print(f"\n{'='*80}")
        print(f"CASES WHERE cos(Good) GETS IT WRONG ({len(good_wrong)} cases)")
        print(f"{'='*80}")

        for c in sorted(good_wrong, key=lambda x: x["good_margin"]):
            print(f"\n  [{c['label']:10s}] {c['id']}  (margin={c['good_margin']:+.6f})")
            print(f"  Category: {c['category']}")
            print(f"  Prompt: {c['prompt'][:120]}")
            print(f"  BETTER ({c['better_len']} chars): {c['better'][:150]}")
            print(f"  WORSE  ({c['worse_len']} chars): {c['worse'][:150]}")
            print(f"  Scores:  Good  better={c['scores']['Good']['better']:.4f}  worse={c['scores']['Good']['worse']:.4f}")
            print(f"           Enth  better={c['scores']['Enthusiastic']['better']:.4f}  worse={c['scores']['Enthusiastic']['worse']:.4f}")
            print(f"           Help  better={c['scores']['Helpful']['better']:.4f}  worse={c['scores']['Helpful']['worse']:.4f}")
            print(f"           Nice  better={c['scores']['Nice']['better']:.4f}  worse={c['scores']['Nice']['worse']:.4f}")
            print(f"           Bana  better={c['scores']['Banana']['better']:.4f}  worse={c['scores']['Banana']['worse']:.4f}")
            print(f"  Penalty (Good-Enth): {'FIXED' if c['penalty_correct'] else 'STILL WRONG'}")

        # LENGTH ANALYSIS: is the model just picking longer responses?
        print(f"\n{'='*80}")
        print(f"LENGTH ANALYSIS")
        print(f"{'='*80}")

        better_longer = sum(1 for c in case_data if c["better_len"] > c["worse_len"])
        worse_longer = sum(1 for c in case_data if c["worse_len"] > c["better_len"])
        equal_len = sum(1 for c in case_data if c["better_len"] == c["worse_len"])
        print(f"  Better response longer: {better_longer}/{n}")
        print(f"  Worse response longer:  {worse_longer}/{n}")
        print(f"  Equal length: {equal_len}/{n}")

        # Does the model prefer the longer response?
        longer_wins_good = 0
        shorter_wins_good = 0
        for c in case_data:
            if c["better_len"] > c["worse_len"]:
                if c["good_correct"]:
                    longer_wins_good += 1
                else:
                    shorter_wins_good += 1
            elif c["worse_len"] > c["better_len"]:
                if not c["good_correct"]:
                    longer_wins_good += 1
                else:
                    shorter_wins_good += 1

        print(f"  cos(Good) picks the longer response: {longer_wins_good}/{n}")
        print(f"  cos(Good) picks the shorter response: {shorter_wins_good}/{n}")

        # Correlation: good margin vs length difference
        len_diffs = [c["better_len"] - c["worse_len"] for c in case_data]
        good_margins = [c["good_margin"] for c in case_data]
        corr = np.corrcoef(len_diffs, good_margins)[0, 1]
        print(f"  Correlation(length_diff, good_margin): {corr:.4f}")

        # By split
        for label in ["firmness", "warmth", "sycophancy"]:
            subset = [c for c in case_data if c["label"] == label]
            ld = [c["better_len"] - c["worse_len"] for c in subset]
            gm = [c["good_margin"] for c in subset]
            cr = np.corrcoef(ld, gm)[0, 1] if len(ld) > 1 else 0
            bl = sum(1 for c in subset if c["better_len"] > c["worse_len"])
            print(f"  {label:12s}: better_longer={bl}/{len(subset)}  corr={cr:.4f}  mean_len_diff={np.mean(ld):+.0f} chars")

        # COSINE TO MULTIPLE TERMS: are there patterns in what the model "sees"?
        print(f"\n{'='*80}")
        print(f"SCORE PATTERNS: mean cos() by term and outcome")
        print(f"{'='*80}")
        print(f"  {'Term':15s} | {'Better (right)':>15s} {'Better (wrong)':>15s} | {'Worse (right)':>15s} {'Worse (wrong)':>15s}")
        print(f"  {'-'*75}")
        for t in TERMS:
            b_right = np.mean([c["scores"][t]["better"] for c in good_right]) if good_right else 0
            b_wrong = np.mean([c["scores"][t]["better"] for c in good_wrong]) if good_wrong else 0
            w_right = np.mean([c["scores"][t]["worse"] for c in good_right]) if good_right else 0
            w_wrong = np.mean([c["scores"][t]["worse"] for c in good_wrong]) if good_wrong else 0
            print(f"  {t:15s} | {b_right:15.4f} {b_wrong:15.4f} | {w_right:15.4f} {w_wrong:15.4f}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
