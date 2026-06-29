#!/usr/bin/env python3
"""Check whether the content split (firmness vs warmth cases) has a lexical
confound similar to the anti-sycophancy demonstration.

The content split is the paper's independent evidence for the warmth-bias
mechanism. If the warmth battery cases have systematically warmer "better"
responses (more positive words) than the firmness cases, the split could be
explained by lexical properties rather than the evaluative mechanism.
"""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

POSITIVE_WORDS = {
    "amazing", "impressive", "incredible", "smart", "strong", "bold", "lucky",
    "best", "great", "wonderful", "excellent", "brilliant", "genius", "perfect",
    "love", "passionate", "dedicated", "wise", "clever", "talented", "admire",
    "admirable", "remarkable", "exceptional", "outstanding", "superb", "champion",
    "sophisticated", "proactive", "strategic", "diligent", "committed", "proud",
    "fantastic", "awesome", "beautiful", "inspiring", "courageous", "brave",
    "deserve", "success", "successful", "winning", "triumph", "hero", "heroic",
    "congratulations", "bravo", "marvelous", "magnificent", "splendid",
    "wonderful", "clearly", "obviously", "exactly",
    "kind", "caring", "gentle", "supportive", "understanding", "compassionate",
    "empathetic", "warm", "friendly", "welcoming", "thoughtful", "generous",
    "gracious", "sweet", "lovely", "nice", "good", "happy", "glad", "pleased",
    "grateful", "thankful", "appreciative",
}

NEGATIVE_WORDS = {
    "no", "stop", "not", "don't", "cannot", "never", "wrong", "risk", "risky",
    "danger", "dangerous", "fail", "failure", "unlikely", "harm", "harmful",
    "damage", "loss", "lose", "losing", "bad", "worse", "worst", "terrible",
    "horrible", "awful", "mistake", "error", "problem", "warning", "caution",
    "careful", "avoid", "extremely", "seriously", "fraud", "illegal",
    "unfortunately", "however", "but", "despite", "although",
}

def count_words(text, word_set):
    return sum(text.lower().split().count(w) for w in word_set)

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def valence_score(text):
    return count_words(text, POSITIVE_WORDS) - count_words(text, NEGATIVE_WORDS)

def main():
    battery = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)

    firmness = [c for c in battery if c["category"] != "anti_sycophancy"]
    syc = [c for c in battery if c["category"] == "anti_sycophancy"]

    print("=== Valence word analysis of content split ===\n")

    for split_name, cases in [("firmness (n=45)", firmness), ("warmth (n=20)", warmth), ("sycophancy (n=5)", syc)]:
        better_pos = [count_words(c["better"], POSITIVE_WORDS) for c in cases]
        better_neg = [count_words(c["better"], NEGATIVE_WORDS) for c in cases]
        worse_pos = [count_words(c["worse"], POSITIVE_WORDS) for c in cases]
        worse_neg = [count_words(c["worse"], NEGATIVE_WORDS) for c in cases]
        better_val = [p - n for p, n in zip(better_pos, better_neg)]
        worse_val = [p - n for p, n in zip(worse_pos, worse_neg)]

        # Valence baseline accuracy
        correct = sum(1 for b, w in zip(better_val, worse_val) if b > w)
        n = len(cases)

        print(f"--- {split_name} ---")
        print(f"  Better response: mean {np.mean(better_pos):.1f} pos, {np.mean(better_neg):.1f} neg, valence {np.mean(better_val):+.1f}")
        print(f"  Worse response:  mean {np.mean(worse_pos):.1f} pos, {np.mean(worse_neg):.1f} neg, valence {np.mean(worse_val):+.1f}")
        print(f"  Valence baseline: {correct}/{n} = {correct/n:.0%}")
        print(f"  Better-has-higher-valence: {correct}/{n}")
        print()

    # For the content split to be a valid test, the WITHIN-CASE valence gap
    # (better_valence - worse_valence) should NOT be systematically different
    # between firmness and warmth cases
    print("=== Cross-split comparison ===\n")
    firm_gaps = [valence_score(c["better"]) - valence_score(c["worse"]) for c in firmness]
    warm_gaps = [valence_score(c["better"]) - valence_score(c["worse"]) for c in warmth]

    print(f"Firmness cases: mean valence gap (better-worse) = {np.mean(firm_gaps):+.2f} (std {np.std(firm_gaps):.2f})")
    print(f"Warmth cases:   mean valence gap (better-worse) = {np.mean(warm_gaps):+.2f} (std {np.std(warm_gaps):.2f})")
    print()

    # Also check: are the "better" responses in warmth cases WARMER than
    # "better" responses in firmness cases?
    firm_better_val = [valence_score(c["better"]) for c in firmness]
    warm_better_val = [valence_score(c["better"]) for c in warmth]
    print(f"Better-response valence: firmness mean={np.mean(firm_better_val):+.2f}, warmth mean={np.mean(warm_better_val):+.2f}")

    # And the key question: does the embedding "good" axis succeed on warmth
    # cases BECAUSE the better response in warmth cases is lexically warmer
    # (higher positive valence)?
    firm_worse_val = [valence_score(c["worse"]) for c in firmness]
    warm_worse_val = [valence_score(c["worse"]) for c in warmth]
    print(f"Worse-response valence:  firmness mean={np.mean(firm_worse_val):+.2f}, warmth mean={np.mean(warm_worse_val):+.2f}")

    print()
    print("If the valence baseline ALSO shows the content-split pattern")
    print("(high accuracy on warmth, low on firmness), the content split has")
    print("the same lexical confound as the anti-sycophancy demonstration.")
    print()

    firm_valence_acc = sum(1 for c in firmness if valence_score(c["better"]) > valence_score(c["worse"])) / len(firmness)
    warm_valence_acc = sum(1 for c in warmth if valence_score(c["better"]) > valence_score(c["worse"])) / len(warmth)
    print(f"Valence baseline on firmness: {firm_valence_acc:.0%}")
    print(f"Valence baseline on warmth:   {warm_valence_acc:.0%}")
    print(f"Gap (warmth - firmness): {warm_valence_acc - firm_valence_acc:+.0%}")
    print()

    # For comparison, the embedding results from §4.19:
    print("For reference, embedding 'good' accuracy from §4.19:")
    print("  BGE-M3: firmness=16%, warmth=85% (gap=+69 points)")
    print("  Nomic:  firmness=12%, warmth=80% (gap=+68 points)")
    print()
    print(f"Valence baseline shows gap of {(warm_valence_acc - firm_valence_acc)*100:+.0f} points")
    print("If this is much smaller than the embedding gap, the content split")
    print("is capturing something beyond lexical valence.")


if __name__ == "__main__":
    main()
