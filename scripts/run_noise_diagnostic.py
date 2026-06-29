#!/usr/bin/env python3
"""Noise diagnostic: are our results distinguishable from chance?

Three tests:
1. RANDOM BASELINES: cos(Good) - cos(Banana) etc. If random words
   score 60%, then "enthusiastic" at 70% means nothing.
2. WORD FORM SENSITIVITY: honest vs honesty vs honestly.
   If results change dramatically, the signal is fragile.
3. BOOTSTRAP CONFIDENCE INTERVALS: resample 10000 times to get
   real variance estimates on all our key results.
4. PERMUTATION TEST: shuffle labels to get null distribution.
"""

import json, gc, math, random
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

# Random unrelated words (no evaluative meaning)
RANDOM_WORDS = [
    "Banana", "Purple", "Tuesday", "Chair", "Mountain",
    "Eleven", "Sandwich", "Guitar", "Penguin", "Umbrella",
    "Bicycle", "Tornado", "Muffin", "Telescope", "Marble",
    "Cactus", "Lightning", "Blanket", "Volcano", "Scissors",
    "Keyboard", "Elevator", "Calendar", "Dinosaur", "Passport",
    "Mushroom", "Satellite", "Goldfish", "Butterfly", "Hammock",
]

# Word form variants
WORD_FORMS = {
    "honest": ["Honest", "Honesty", "Honestly", "Dishonest", "Dishonesty"],
    "careful": ["Careful", "Carefully", "Carefulness", "Careless", "Carelessly"],
    "helpful": ["Helpful", "Helpfully", "Helpfulness", "Unhelpful", "Help"],
    "thorough": ["Thorough", "Thoroughly", "Thoroughness"],
    "nice": ["Nice", "Nicely", "Niceness"],
    "good": ["Good", "Goodness", "Well"],
    "enthusiastic": ["Enthusiastic", "Enthusiastically", "Enthusiasm"],
    "flattering": ["Flattering", "Flattery", "Flatter", "Flattered"],
    "pleasant": ["Pleasant", "Pleasantly", "Pleasantness", "Please", "Pleased", "Pleasing"],
    "skeptical": ["Skeptical", "Skeptically", "Skepticism", "Skeptic"],
}

# Our "best" penalty terms from the sweep
BEST_PENALTIES = ["Enthusiastic", "Pleasing", "Flattering", "Nice", "Positive", "Charming"]


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


def bootstrap_ci(correct_array, n_boot=10000, ci=0.95):
    n = len(correct_array)
    arr = np.array(correct_array, dtype=float)
    boot_means = np.array([np.mean(np.random.choice(arr, size=n, replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha))


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

    firm_idx = [i for i in range(n) if labels[i] == "firmness"]
    warm_idx = [i for i in range(n) if labels[i] == "warmth"]
    syc_idx = [i for i in range(n) if labels[i] == "sycophancy"]

    print(f"Dataset: {n} cases ({len(firm_idx)} firmness, {len(warm_idx)} warmth, {len(syc_idx)} sycophancy)")
    print(f"With n={n}, chance=50%, Wilson 95% CI for 50% = [{wilson_ci(35,70)[0]:.0%}, {wilson_ci(35,70)[1]:.0%}]")
    print(f"With n={n}, any accuracy below {wilson_ci(35,70)[1]:.0%} is NOT distinguishable from coin flip")
    print()

    np.random.seed(42)

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        for i in range(n):
            better_embs[i] = better_embs[i] / (norm(better_embs[i]) + 1e-12)
            worse_embs[i] = worse_embs[i] / (norm(worse_embs[i]) + 1e-12)

        good_emb = embed(["Good"])[0]
        good_emb = good_emb / (norm(good_emb) + 1e-12)

        b_good = [float(np.dot(better_embs[i], good_emb)) for i in range(n)]
        w_good = [float(np.dot(worse_embs[i], good_emb)) for i in range(n)]
        base_correct = [b_good[i] > w_good[i] for i in range(n)]
        base_acc = sum(base_correct) / n
        base_lo, base_hi = bootstrap_ci(base_correct)

        print(f"\n  BASELINE: cos(Good) = {base_acc:.1%}  bootstrap 95% CI = [{base_lo:.1%}, {base_hi:.1%}]")
        bf = sum(base_correct[i] for i in firm_idx) / len(firm_idx)
        bw = sum(base_correct[i] for i in warm_idx) / len(warm_idx)
        bs = sum(base_correct[i] for i in syc_idx) / len(syc_idx)
        print(f"  Splits: firm={bf:.0%} (n={len(firm_idx)})  warm={bw:.0%} (n={len(warm_idx)})  syc={bs:.0%} (n={len(syc_idx)})")

        # ============================================================
        # TEST 1: RANDOM WORD PENALTIES
        # ============================================================
        print(f"\n  TEST 1: RANDOM WORD PENALTIES (cos(Good) - cos(Random))")
        print(f"  If random words score well, our 'best' terms are noise.")
        print(f"  {'Word':20s} {'Acc':>5s} {'Firm':>5s} {'Warm':>5s} {'Syc':>5s} {'Boot CI':>15s}")
        print(f"  {'-'*70}")

        random_accs = []
        for word in RANDOM_WORDS:
            w_emb = embed([word])[0]
            w_emb = w_emb / (norm(w_emb) + 1e-12)
            b_pen = [float(np.dot(better_embs[i], w_emb)) for i in range(n)]
            w_pen = [float(np.dot(worse_embs[i], w_emb)) for i in range(n)]
            b_score = [b_good[i] - b_pen[i] for i in range(n)]
            w_score = [w_good[i] - w_pen[i] for i in range(n)]
            correct = [b_score[i] > w_score[i] for i in range(n)]
            acc = sum(correct) / n
            f_acc = sum(correct[i] for i in firm_idx) / len(firm_idx)
            w_acc = sum(correct[i] for i in warm_idx) / len(warm_idx)
            s_acc = sum(correct[i] for i in syc_idx) / len(syc_idx)
            lo, hi = bootstrap_ci(correct)
            random_accs.append(acc)
            print(f"  {word:20s} {acc:4.0%}  {f_acc:4.0%}  {w_acc:4.0%}  {s_acc:4.0%}  [{lo:.0%}, {hi:.0%}]")

        print(f"\n  RANDOM WORD SUMMARY:")
        print(f"    Mean accuracy: {np.mean(random_accs):.1%}")
        print(f"    Std:  {np.std(random_accs):.1%}")
        print(f"    Min:  {np.min(random_accs):.1%}  Max: {np.max(random_accs):.1%}")
        print(f"    Range: {np.max(random_accs) - np.min(random_accs):.1%}")

        # ============================================================
        # TEST 2: OUR BEST PENALTIES (for comparison)
        # ============================================================
        print(f"\n  TEST 2: OUR BEST PENALTIES (for comparison)")
        print(f"  {'Word':20s} {'Acc':>5s} {'Firm':>5s} {'Warm':>5s} {'Syc':>5s} {'Boot CI':>15s}")
        print(f"  {'-'*70}")

        best_accs = []
        for word in BEST_PENALTIES:
            w_emb = embed([word])[0]
            w_emb = w_emb / (norm(w_emb) + 1e-12)
            b_pen = [float(np.dot(better_embs[i], w_emb)) for i in range(n)]
            w_pen = [float(np.dot(worse_embs[i], w_emb)) for i in range(n)]
            b_score = [b_good[i] - b_pen[i] for i in range(n)]
            w_score = [w_good[i] - w_pen[i] for i in range(n)]
            correct = [b_score[i] > w_score[i] for i in range(n)]
            acc = sum(correct) / n
            f_acc = sum(correct[i] for i in firm_idx) / len(firm_idx)
            w_acc = sum(correct[i] for i in warm_idx) / len(warm_idx)
            s_acc = sum(correct[i] for i in syc_idx) / len(syc_idx)
            lo, hi = bootstrap_ci(correct)
            best_accs.append(acc)
            print(f"  {word:20s} {acc:4.0%}  {f_acc:4.0%}  {w_acc:4.0%}  {s_acc:4.0%}  [{lo:.0%}, {hi:.0%}]")

        print(f"\n  BEST PENALTY SUMMARY:")
        print(f"    Mean accuracy: {np.mean(best_accs):.1%}")
        print(f"    Do best terms beat random? best_mean={np.mean(best_accs):.1%} vs random_mean={np.mean(random_accs):.1%}")
        print(f"    Best term max={np.max(best_accs):.1%} vs random max={np.max(random_accs):.1%}")

        # ============================================================
        # TEST 3: WORD FORM SENSITIVITY
        # ============================================================
        print(f"\n  TEST 3: WORD FORM SENSITIVITY")
        print(f"  If 'honest' vs 'honesty' changes results by 20%, signal is fragile.")
        for root, forms in WORD_FORMS.items():
            print(f"\n    {root} variants:")
            # As raw cosine similarity targets
            for word in forms:
                w_emb = embed([word])[0]
                w_emb = w_emb / (norm(w_emb) + 1e-12)
                b_cos = [float(np.dot(better_embs[i], w_emb)) for i in range(n)]
                w_cos = [float(np.dot(worse_embs[i], w_emb)) for i in range(n)]
                raw_correct = [b_cos[i] > w_cos[i] for i in range(n)]
                raw_acc = sum(raw_correct) / n
                # As penalty (cos(Good) - cos(form))
                b_score = [b_good[i] - b_cos[i] for i in range(n)]
                w_score = [w_good[i] - w_cos[i] for i in range(n)]
                pen_correct = [b_score[i] > w_score[i] for i in range(n)]
                pen_acc = sum(pen_correct) / n
                # Cosine to good
                cos_g = float(np.dot(good_emb, w_emb))
                print(f"      {word:20s}  raw={raw_acc:4.0%}  penalty={pen_acc:4.0%}  cos(good)={cos_g:.4f}")

        # ============================================================
        # TEST 4: PERMUTATION TEST
        # ============================================================
        print(f"\n  TEST 4: PERMUTATION TEST (null distribution)")
        print(f"  Shuffle better/worse labels 5000 times, compute cos(Good) accuracy each time.")
        n_perm = 5000
        perm_accs = []
        for _ in range(n_perm):
            perm_correct = []
            for i in range(n):
                if random.random() < 0.5:
                    perm_correct.append(b_good[i] > w_good[i])
                else:
                    perm_correct.append(w_good[i] > b_good[i])
            perm_accs.append(sum(perm_correct) / n)

        perm_accs = np.array(perm_accs)
        p_value = np.mean(perm_accs >= base_acc)
        print(f"    Null distribution: mean={np.mean(perm_accs):.1%}, std={np.std(perm_accs):.1%}")
        print(f"    Observed accuracy: {base_acc:.1%}")
        print(f"    p-value (fraction of null >= observed): {p_value:.4f}")
        print(f"    {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

        # Permutation test for best penalty
        print(f"\n  PERMUTATION TEST for cos(Good) - cos(Enthusiastic):")
        enth_emb = embed(["Enthusiastic"])[0]
        enth_emb = enth_emb / (norm(enth_emb) + 1e-12)
        b_enth = [float(np.dot(better_embs[i], enth_emb)) for i in range(n)]
        w_enth = [float(np.dot(worse_embs[i], enth_emb)) for i in range(n)]
        b_pen_score = [b_good[i] - b_enth[i] for i in range(n)]
        w_pen_score = [w_good[i] - w_enth[i] for i in range(n)]
        pen_correct_actual = [b_pen_score[i] > w_pen_score[i] for i in range(n)]
        pen_acc_actual = sum(pen_correct_actual) / n

        perm_pen_accs = []
        for _ in range(n_perm):
            perm_correct = []
            for i in range(n):
                if random.random() < 0.5:
                    perm_correct.append(b_pen_score[i] > w_pen_score[i])
                else:
                    perm_correct.append(w_pen_score[i] > b_pen_score[i])
            perm_pen_accs.append(sum(perm_correct) / n)
        perm_pen_accs = np.array(perm_pen_accs)
        p_pen = np.mean(perm_pen_accs >= pen_acc_actual)
        print(f"    Null: mean={np.mean(perm_pen_accs):.1%}, std={np.std(perm_pen_accs):.1%}")
        print(f"    Observed: {pen_acc_actual:.1%}")
        print(f"    p-value: {p_pen:.4f}")
        print(f"    {'SIGNIFICANT' if p_pen < 0.05 else 'NOT SIGNIFICANT'}")

        # ============================================================
        # TEST 5: RAW MARGIN ANALYSIS
        # ============================================================
        print(f"\n  TEST 5: RAW SCORE MARGINS")
        print(f"  How MUCH do scores differ between better/worse? Tiny margins = noise.")
        margins = [b_good[i] - w_good[i] for i in range(n)]
        print(f"    cos(Good) score margins (better - worse):")
        print(f"    Mean: {np.mean(margins):+.6f}")
        print(f"    Median: {np.median(margins):+.6f}")
        print(f"    Std: {np.std(margins):.6f}")
        print(f"    Min: {np.min(margins):+.6f}  Max: {np.max(margins):+.6f}")
        print(f"    Margins < 0.001: {sum(1 for m in margins if abs(m) < 0.001)}/{n}")
        print(f"    Margins < 0.01:  {sum(1 for m in margins if abs(m) < 0.01)}/{n}")
        print(f"    Margins < 0.05:  {sum(1 for m in margins if abs(m) < 0.05)}/{n}")

        # By split
        for label, idx in [("firmness", firm_idx), ("warmth", warm_idx), ("sycophancy", syc_idx)]:
            split_margins = [margins[i] for i in idx]
            print(f"    {label:12s}: mean={np.mean(split_margins):+.6f}  median={np.median(split_margins):+.6f}  n={len(idx)}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
