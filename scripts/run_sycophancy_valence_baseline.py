#!/usr/bin/env python3
"""Valence-word-count baseline for anti-sycophancy cases.

Check whether a trivial positive-word counter can replicate the embedding
result. If it can, the embedding finding is lexical, not geometric.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
EXPANSION = ROOT / "notes/research_cycles/battery_expansion/anti_sycophancy_expansion.jsonl"

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
}

NEGATIVE_WORDS = {
    "no", "stop", "not", "don't", "cannot", "never", "wrong", "risk", "risky",
    "danger", "dangerous", "fail", "failure", "unlikely", "harm", "harmful",
    "damage", "loss", "lose", "losing", "bad", "worse", "worst", "terrible",
    "horrible", "awful", "mistake", "error", "problem", "warning", "caution",
    "careful", "avoid", "extremely", "seriously", "fraud", "illegal",
}

def count_words(text, word_set):
    text_lower = text.lower()
    count = 0
    for word in word_set:
        count += text_lower.split().count(word)
    return count

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    all_battery = read_jsonl(BATTERY)
    orig_syc = [c for c in all_battery if c["category"] == "anti_sycophancy"]
    new_syc = read_jsonl(EXPANSION)
    all_syc = orig_syc + new_syc

    print(f"Positive word set: {len(POSITIVE_WORDS)} words")
    print(f"Negative word set: {len(NEGATIVE_WORDS)} words")
    print()

    # For each case, count positive/negative words in better vs worse
    print(f"{'ID':30s} {'B+':>3s} {'B-':>3s} {'W+':>3s} {'W-':>3s} {'Valence':>8s} {'Correct':>8s}")
    print("-" * 70)

    valence_correct = 0
    total = len(all_syc)

    for i, c in enumerate(all_syc):
        better = c["better"]
        worse = c["worse"]

        bp = count_words(better, POSITIVE_WORDS)
        bn = count_words(better, NEGATIVE_WORDS)
        wp = count_words(worse, POSITIVE_WORDS)
        wn = count_words(worse, NEGATIVE_WORDS)

        # Valence score: more positive words = warmer
        # Better should have MORE negative, FEWER positive (pushback)
        # Worse should have MORE positive (sycophantic)
        better_valence = bp - bn
        worse_valence = wp - wn

        # Valence baseline: pick the response with LOWER positive valence
        # (less sycophantic = probably the pushback)
        valence_picks_better = better_valence < worse_valence
        correct = "Y" if valence_picks_better else "."
        if valence_picks_better:
            valence_correct += 1

        src = "orig" if i < len(orig_syc) else "new"
        print(f"{c['id']:30s} {bp:3d} {bn:3d} {wp:3d} {wn:3d} {better_valence:+4d}/{worse_valence:+4d} {correct:>8s}  [{src}]")

    print(f"\nValence baseline accuracy: {valence_correct}/{total} = {valence_correct/total:.0%}")

    # Also check just original vs new
    orig_correct = sum(1 for i in range(len(orig_syc))
                      if count_words(orig_syc[i]["better"], POSITIVE_WORDS) - count_words(orig_syc[i]["better"], NEGATIVE_WORDS) <
                         count_words(orig_syc[i]["worse"], POSITIVE_WORDS) - count_words(orig_syc[i]["worse"], NEGATIVE_WORDS))
    new_correct = valence_correct - orig_correct

    print(f"  Original 5: {orig_correct}/5 = {orig_correct/5:.0%}")
    print(f"  New 15: {new_correct}/15 = {new_correct/15:.0%}")

    # Length baseline too
    print(f"\n--- Length baseline ---")
    length_correct = 0
    for c in all_syc:
        if len(c["better"]) > len(c["worse"]):
            length_correct += 1
    print(f"Length baseline (longer=better): {length_correct}/{total} = {length_correct/total:.0%}")

    # Net positive word count difference
    print(f"\n--- Positive word count stats ---")
    better_pos = [count_words(c["better"], POSITIVE_WORDS) for c in all_syc]
    worse_pos = [count_words(c["worse"], POSITIVE_WORDS) for c in all_syc]
    better_neg = [count_words(c["better"], NEGATIVE_WORDS) for c in all_syc]
    worse_neg = [count_words(c["worse"], NEGATIVE_WORDS) for c in all_syc]

    import numpy as np
    print(f"Better response: mean {np.mean(better_pos):.1f} pos, {np.mean(better_neg):.1f} neg")
    print(f"Worse response:  mean {np.mean(worse_pos):.1f} pos, {np.mean(worse_neg):.1f} neg")
    print(f"Positive gap (worse-better): {np.mean(worse_pos) - np.mean(better_pos):+.1f}")
    print(f"Negative gap (better-worse): {np.mean(better_neg) - np.mean(worse_neg):+.1f}")


if __name__ == "__main__":
    main()
