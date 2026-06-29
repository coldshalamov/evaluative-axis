#!/usr/bin/env python3
"""Deep investigation of the centroid quality direction.

Experiments:
1. Learning curve: how few labeled pairs do you need? (5..70)
2. Bootstrap stability: resample 100x, measure direction consistency
3. Proper k-fold cross-validation on battery
4. Per-category OOS breakdown with confidence margins
5. Direction interpretation: what words have highest cosine with it?
6. Error analysis: which cases fail and what do they look like?
7. Permutation test: is the OOS accuracy significantly above chance?
8. Margin analysis: how confident are correct vs incorrect predictions?
"""

import json, gc, sys
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

PROBE_VOCAB = [
    "good", "bad", "careful", "reckless", "honest", "dishonest",
    "helpful", "unhelpful", "thorough", "superficial", "restrained",
    "unrestrained", "safe", "dangerous", "correct", "incorrect",
    "clear", "confusing", "respectful", "rude", "thoughtful",
    "thoughtless", "precise", "vague", "reasonable", "unreasonable",
    "responsible", "irresponsible", "ethical", "unethical",
    "kind", "cruel", "warm", "cold", "professional", "unprofessional",
    "competent", "incompetent", "rigorous", "sloppy", "measured",
    "impulsive", "patient", "impatient", "nuanced", "simplistic",
    "constructive", "destructive", "informative", "misleading",
    "balanced", "biased", "empathetic", "dismissive", "direct",
    "evasive", "firm", "pushover", "accurate", "inaccurate",
    "appropriate", "inappropriate", "substantive", "shallow",
    "well-reasoned", "irrational", "cautious", "careless",
]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def score_pairs(better_embs, worse_embs, direction):
    margins = []
    for i in range(len(better_embs)):
        b = float(np.dot(better_embs[i], direction))
        w = float(np.dot(worse_embs[i], direction))
        margins.append(b - w)
    return margins


def accuracy_from_margins(margins):
    return sum(1 for m in margins if m > 0) / len(margins)


def binomial_ci(k, n, alpha=0.05):
    if n == 0:
        return 0, 1
    lo = stats.binom.ppf(alpha / 2, n, k / n) / n if k > 0 else 0
    hi = stats.binom.ppf(1 - alpha / 2, n, k / n) / n if k < n else 1
    return float(lo), float(hi)


def main():
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    orig_cases = load_cases(BATTERY_50)
    warmth_cases = load_cases(WARMTH_20)

    expansion = []
    exp_files = sorted(EXPANSION_DIR.glob("*.jsonl"))
    for f in exp_files:
        expansion.extend(load_cases(f))

    print(f"Battery: {len(battery)} | Expansion: {len(expansion)}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        bat_better = model.encode([c["better"] for c in battery], convert_to_numpy=True, batch_size=32)
        bat_worse = model.encode([c["worse"] for c in battery], convert_to_numpy=True, batch_size=32)
        for i in range(len(battery)):
            bat_better[i] /= norm(bat_better[i]) + 1e-12
            bat_worse[i] /= norm(bat_worse[i]) + 1e-12

        exp_better = model.encode([c["better"] for c in expansion], convert_to_numpy=True, batch_size=32)
        exp_worse = model.encode([c["worse"] for c in expansion], convert_to_numpy=True, batch_size=32)
        for i in range(len(expansion)):
            exp_better[i] /= norm(exp_better[i]) + 1e-12
            exp_worse[i] /= norm(exp_worse[i]) + 1e-12

        model_results = {}

        # =====================================================================
        # 1. LEARNING CURVE
        # =====================================================================
        print(f"\n--- 1. Learning Curve ---")
        print(f"  {'n_pairs':>8s} {'bat_acc':>8s} {'exp_acc':>8s} {'exp_CI':>14s}")

        sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70]
        n_repeats = 50
        learning_curve = {}

        for n_train in sizes:
            exp_accs = []
            bat_accs = []
            for _ in range(n_repeats):
                idx = np.random.choice(len(battery), size=n_train, replace=False)
                direction = bat_better[idx].mean(axis=0) - bat_worse[idx].mean(axis=0)
                direction /= norm(direction) + 1e-12

                bat_margins = score_pairs(bat_better, bat_worse, direction)
                exp_margins = score_pairs(exp_better, exp_worse, direction)
                bat_accs.append(accuracy_from_margins(bat_margins))
                exp_accs.append(accuracy_from_margins(exp_margins))

            mean_bat = np.mean(bat_accs)
            mean_exp = np.mean(exp_accs)
            std_exp = np.std(exp_accs)
            lo_exp = np.percentile(exp_accs, 2.5)
            hi_exp = np.percentile(exp_accs, 97.5)
            print(f"  {n_train:8d} {mean_bat:7.1%}  {mean_exp:7.1%}  [{lo_exp:.0%}, {hi_exp:.0%}]")
            learning_curve[n_train] = {
                "mean_battery": float(mean_bat),
                "mean_expansion": float(mean_exp),
                "std_expansion": float(std_exp),
                "ci_lo": float(lo_exp),
                "ci_hi": float(hi_exp),
            }

        model_results["learning_curve"] = learning_curve

        # =====================================================================
        # 2. BOOTSTRAP STABILITY
        # =====================================================================
        print(f"\n--- 2. Bootstrap Direction Stability ---")

        n_boot = 200
        directions = []
        for _ in range(n_boot):
            idx = np.random.choice(len(battery), size=len(battery), replace=True)
            d = bat_better[idx].mean(axis=0) - bat_worse[idx].mean(axis=0)
            d /= norm(d) + 1e-12
            directions.append(d)

        directions = np.array(directions)
        full_dir = bat_better.mean(axis=0) - bat_worse.mean(axis=0)
        full_dir /= norm(full_dir) + 1e-12

        cosines_to_full = [float(np.dot(directions[i], full_dir)) for i in range(n_boot)]
        pairwise_cosines = []
        for i in range(0, min(100, n_boot)):
            for j in range(i+1, min(100, n_boot)):
                pairwise_cosines.append(float(np.dot(directions[i], directions[j])))

        print(f"  Cosine of each bootstrap direction to full-data direction:")
        print(f"    Mean: {np.mean(cosines_to_full):.4f}")
        print(f"    Min:  {np.min(cosines_to_full):.4f}")
        print(f"    Std:  {np.std(cosines_to_full):.4f}")
        print(f"  Pairwise cosine between bootstrap directions:")
        print(f"    Mean: {np.mean(pairwise_cosines):.4f}")
        print(f"    Min:  {np.min(pairwise_cosines):.4f}")

        model_results["bootstrap_stability"] = {
            "cos_to_full_mean": float(np.mean(cosines_to_full)),
            "cos_to_full_min": float(np.min(cosines_to_full)),
            "cos_to_full_std": float(np.std(cosines_to_full)),
            "pairwise_mean": float(np.mean(pairwise_cosines)),
            "pairwise_min": float(np.min(pairwise_cosines)),
        }

        # =====================================================================
        # 3. K-FOLD CROSS-VALIDATION
        # =====================================================================
        print(f"\n--- 3. K-Fold Cross-Validation (battery only) ---")

        from sklearn.model_selection import KFold

        for n_folds in [5, 10]:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_accs = []
            for train_idx, test_idx in kf.split(battery):
                direction = bat_better[train_idx].mean(axis=0) - bat_worse[train_idx].mean(axis=0)
                direction /= norm(direction) + 1e-12
                margins = score_pairs(bat_better[test_idx], bat_worse[test_idx], direction)
                fold_accs.append(accuracy_from_margins(margins))
            mean_cv = np.mean(fold_accs)
            std_cv = np.std(fold_accs)
            print(f"  {n_folds}-fold CV: {mean_cv:.1%} +/- {std_cv:.1%}  (folds: {[f'{a:.0%}' for a in fold_accs]})")
            model_results[f"cv_{n_folds}fold"] = {
                "mean": float(mean_cv), "std": float(std_cv),
                "folds": [float(a) for a in fold_accs],
            }

        # =====================================================================
        # 4. PER-CATEGORY OOS WITH MARGINS
        # =====================================================================
        print(f"\n--- 4. Per-Category OOS Breakdown ---")

        full_margins = score_pairs(exp_better, exp_worse, full_dir)
        full_acc = accuracy_from_margins(full_margins)
        k = sum(1 for m in full_margins if m > 0)
        ci_lo, ci_hi = binomial_ci(k, len(expansion))
        print(f"  Overall: {full_acc:.0%} ({k}/{len(expansion)}) CI=[{ci_lo:.0%},{ci_hi:.0%}]")

        cat_detail = {}
        for i, c in enumerate(expansion):
            cat = c.get("category", "unknown")
            if cat not in cat_detail:
                cat_detail[cat] = {"correct": [], "margins": [], "cases": []}
            correct = full_margins[i] > 0
            cat_detail[cat]["correct"].append(correct)
            cat_detail[cat]["margins"].append(full_margins[i])
            cat_detail[cat]["cases"].append(i)

        print(f"\n  {'Category':35s} {'Acc':>6s} {'n':>4s} {'Mean margin':>12s}")
        for cat in sorted(cat_detail.keys()):
            d = cat_detail[cat]
            acc = sum(d["correct"]) / len(d["correct"])
            n = len(d["correct"])
            mean_m = np.mean(d["margins"])
            print(f"  {cat:35s} {acc:5.0%}  {n:3d}  {mean_m:+11.4f}")

        model_results["per_category"] = {
            cat: {
                "accuracy": sum(d["correct"]) / len(d["correct"]),
                "n": len(d["correct"]),
                "mean_margin": float(np.mean(d["margins"])),
            }
            for cat, d in cat_detail.items()
        }

        # =====================================================================
        # 5. DIRECTION INTERPRETATION
        # =====================================================================
        print(f"\n--- 5. Direction Interpretation ---")
        print(f"  Top words aligned with quality direction:")

        vocab_embs = model.encode(PROBE_VOCAB, convert_to_numpy=True)
        for i in range(len(vocab_embs)):
            vocab_embs[i] /= norm(vocab_embs[i]) + 1e-12

        word_cosines = [(PROBE_VOCAB[i], float(np.dot(vocab_embs[i], full_dir)))
                        for i in range(len(PROBE_VOCAB))]
        word_cosines.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Most ALIGNED with quality direction (high cos = scores better):")
        for w, c in word_cosines[:15]:
            print(f"    {w:25s} {c:+.4f}")

        print(f"\n  Most OPPOSED to quality direction (low cos = scores worse):")
        for w, c in word_cosines[-15:]:
            print(f"    {w:25s} {c:+.4f}")

        model_results["direction_words"] = {
            "top_aligned": word_cosines[:15],
            "top_opposed": word_cosines[-15:],
        }

        # =====================================================================
        # 6. ERROR ANALYSIS
        # =====================================================================
        print(f"\n--- 6. Error Analysis (expansion failures) ---")

        failures = []
        for i, m in enumerate(full_margins):
            if m <= 0:
                c = expansion[i]
                failures.append({
                    "id": c.get("id", i),
                    "category": c.get("category", "?"),
                    "phenomenon": c.get("phenomenon", "?"),
                    "margin": float(m),
                    "better_preview": c["better"][:120],
                    "worse_preview": c["worse"][:120],
                })

        for f in failures:
            print(f"\n  FAIL [{f['category']}] {f['phenomenon']}")
            print(f"    margin: {f['margin']:+.4f}")
            print(f"    better: {f['better_preview']}...")
            print(f"    worse:  {f['worse_preview']}...")

        model_results["errors"] = failures

        # =====================================================================
        # 7. PERMUTATION TEST
        # =====================================================================
        print(f"\n--- 7. Permutation Test (is OOS accuracy significant?) ---")

        n_perm = 1000
        observed_acc = full_acc
        perm_accs = []
        for _ in range(n_perm):
            perm_idx = np.random.permutation(len(battery))
            perm_better = bat_better[perm_idx[:len(battery)//2]]
            perm_worse = bat_worse[perm_idx[:len(battery)//2]]
            rest_better = bat_better[perm_idx[len(battery)//2:]]
            rest_worse = bat_worse[perm_idx[len(battery)//2:]]
            scrambled = np.vstack([perm_better, rest_worse])
            scrambled2 = np.vstack([perm_worse, rest_better])
            d = scrambled.mean(axis=0) - scrambled2.mean(axis=0)
            d /= norm(d) + 1e-12
            m = score_pairs(exp_better, exp_worse, d)
            perm_accs.append(accuracy_from_margins(m))

        p_value = sum(1 for a in perm_accs if a >= observed_acc) / n_perm
        print(f"  Observed OOS accuracy: {observed_acc:.1%}")
        print(f"  Permutation null (n={n_perm}): mean={np.mean(perm_accs):.1%}, max={np.max(perm_accs):.1%}")
        print(f"  p-value: {p_value:.4f}")

        model_results["permutation_test"] = {
            "observed": float(observed_acc),
            "null_mean": float(np.mean(perm_accs)),
            "null_max": float(np.max(perm_accs)),
            "p_value": float(p_value),
        }

        # =====================================================================
        # 8. MARGIN ANALYSIS
        # =====================================================================
        print(f"\n--- 8. Margin Distribution ---")

        correct_margins = [m for m in full_margins if m > 0]
        wrong_margins = [m for m in full_margins if m <= 0]

        print(f"  Correct predictions (n={len(correct_margins)}):")
        print(f"    Mean margin: {np.mean(correct_margins):+.4f}")
        print(f"    Min margin:  {np.min(correct_margins):+.4f}")
        if wrong_margins:
            print(f"  Wrong predictions (n={len(wrong_margins)}):")
            print(f"    Mean margin: {np.mean(wrong_margins):+.4f}")
            print(f"    Max margin:  {np.max(wrong_margins):+.4f}")

        # Battery margins for comparison
        bat_margins_full = score_pairs(bat_better, bat_worse, full_dir)
        bat_correct = [m for m in bat_margins_full if m > 0]
        bat_wrong = [m for m in bat_margins_full if m <= 0]
        print(f"  Battery correct (n={len(bat_correct)}): mean margin {np.mean(bat_correct):+.4f}")
        if bat_wrong:
            print(f"  Battery wrong (n={len(bat_wrong)}): mean margin {np.mean(bat_wrong):+.4f}")

        model_results["margins"] = {
            "exp_correct_mean": float(np.mean(correct_margins)) if correct_margins else 0,
            "exp_wrong_mean": float(np.mean(wrong_margins)) if wrong_margins else 0,
            "bat_correct_mean": float(np.mean(bat_correct)) if bat_correct else 0,
            "bat_wrong_mean": float(np.mean(bat_wrong)) if bat_wrong else 0,
        }

        all_results[short] = model_results

        del model
        gc.collect()

    # =====================================================================
    # CROSS-MODEL SUMMARY
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Learning curve (OOS accuracy):")
    print(f"  {'n_pairs':>8s}", end="")
    for short in all_results:
        print(f"  {short:>20s}", end="")
    print()
    for n in sizes:
        print(f"  {n:8d}", end="")
        for short, r in all_results.items():
            lc = r["learning_curve"][n]
            print(f"  {lc['mean_expansion']:19.1%}", end="")
        print()

    print(f"\n  Bootstrap stability (cosine to full direction):")
    for short, r in all_results.items():
        bs = r["bootstrap_stability"]
        print(f"  {short}: mean={bs['cos_to_full_mean']:.4f} min={bs['cos_to_full_min']:.4f}")

    print(f"\n  Permutation test p-values:")
    for short, r in all_results.items():
        pt = r["permutation_test"]
        print(f"  {short}: observed={pt['observed']:.1%} null_mean={pt['null_mean']:.1%} p={pt['p_value']:.4f}")

    with open(OUT_DIR / "centroid_deep_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'centroid_deep_results.json'}")


if __name__ == "__main__":
    main()
