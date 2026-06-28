#!/usr/bin/env python3
"""E-07: Per-Case Complementarity Analysis.

Scores the balanced battery (50 firmness + 20 warmth) with 15 single-word
axes and asks:
  1. Which axes get each case right / wrong (per-case correctness vector)?
  2. Which axis PAIRS are complementary (negative correlation of correctness
     = they cover each other's failures) vs redundant (positive correlation)?
  3. What is the minimum set of axes that correctly covers >90% of cases,
     under a SUM-OF-PROJECTIONS aggregation?

The 15 axes are the ones specified in EXPERIMENT_SPECS.md E-07:
  careful, honest, kind, wise, helpful, thorough, fair, responsible,
  clear, good, hard, gentle, patient, supportive, constructive

Rules followed (AGENTS.md / RESEARCH_CONTEXT.md):
  - All three local models.
  - Both battery splits reported separately + combined.
  - User/Assistant framing for response embeddings.
  - Axes scored independently (no averaging of axis vectors); the covering
    set uses SUM of independent projections per the no-averaging rule.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BALANCED_ORIG = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BALANCED_WARMTH = (
    ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# 15 axes specified by E-07. Word pairs chosen to be evaluatively contrastive.
AXES_15 = {
    "careful":      ("Careful",      "Reckless"),
    "honest":       ("Honest",       "Dishonest"),
    "kind":         ("Kind",         "Cruel"),
    "wise":         ("Wise",         "Foolish"),
    "helpful":      ("Helpful",      "Unhelpful"),
    "thorough":     ("Thorough",     "Superficial"),
    "fair":         ("Fair",         "Unfair"),
    "responsible":  ("Responsible",  "Irresponsible"),
    "clear":        ("Clear",        "Confusing"),
    "good":         ("Good",         "Bad"),
    "hard":         ("Hard",         "Soft"),
    "gentle":       ("Gentle",       "Harsh"),
    "patient":      ("Patient",      "Impatient"),
    "supportive":   ("Supportive",   "Unsupportive"),
    "constructive": ("Constructive", "Destructive"),
}


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos = embed_fn([positive])[0]
    neg = embed_fn([negative])[0]
    axis = pos - neg
    return axis / (np.linalg.norm(axis) + 1e-12)


def per_case_correct(better_embs, worse_embs, axis_vec):
    """Return list of 0/1 (1 = better scored higher)."""
    out = []
    for i in range(len(better_embs)):
        sb = float(np.dot(better_embs[i], axis_vec))
        sw = float(np.dot(worse_embs[i], axis_vec))
        out.append(1 if sb > sw else (0.5 if sb == sw else 0))
    return out


def sum_projection_correct(better_embs, worse_embs, axis_vecs):
    """Aggregate a SET of axes by summing independent projections (no averaging)."""
    out = []
    for i in range(len(better_embs)):
        sb = sum(float(np.dot(better_embs[i], ax)) for ax in axis_vecs)
        sw = sum(float(np.dot(worse_embs[i], ax)) for ax in axis_vecs)
        out.append(1 if sb > sw else (0.5 if sb == sw else 0))
    return out


def coverage(correct_vec):
    """Fraction of cases with correct >= 1 (i.e. the OR over the set)."""
    # For "covered" we use sum-of-projections correctness on the full set,
    # computed separately. Here 'coverage' = fraction correct in this vector.
    return sum(1 for x in correct_vec if x >= 1) / len(correct_vec)


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def min_covering_set(case_correct_by_axis, axes, n_cases, target=0.90):
    """Find minimum set of axes whose SUM-OF-PROJECTIONS correctness >= target.

    We approximate "set correctness" as the fraction of cases where AT LEAST
    ONE axis in the set is correct (logical OR coverage), which upper-bounds
    what a covering set can achieve. This is the standard complementarity
    framing: how few axes cover the battery between them.

    Returns (best_set, coverage_fraction).
    """
    # Greedy set cover
    remaining = set(range(n_cases))
    chosen = []
    chosen_names = []
    # Map axis -> set of correct case indices
    correct_sets = {a: {i for i, v in enumerate(case_correct_by_axis[a])
                        if v >= 1} for a in axes}

    while remaining and len(chosen_names) < len(axes):
        # Pick axis that covers most remaining cases
        best_ax = max(axes, key=lambda a: len(correct_sets[a] - set(chosen)
                                              & correct_sets[a]) if False
                      else len(correct_sets[a] & remaining))
        chosen.append(best_ax)
        chosen_names.append(best_ax)
        covered = set()
        for a in chosen_names:
            covered |= correct_sets[a]
        remaining = set(range(n_cases)) - covered
        cov = (n_cases - len(remaining)) / n_cases
        if cov >= target:
            return chosen_names, cov

    covered = set()
    for a in chosen_names:
        covered |= correct_sets[a]
    cov = len(covered) / n_cases
    return chosen_names, cov


def exhaustive_min_cover(case_correct_by_axis, axes, n_cases, target=0.90,
                         max_k=5):
    """For small k, find the true minimum covering set by exhaustive search.

    Returns list of (k, sets) for each k where a set meets target.
    """
    correct_sets = {a: {i for i, v in enumerate(case_correct_by_axis[a])
                        if v >= 1} for a in axes}
    results = {}
    for k in range(1, max_k + 1):
        meeting = []
        for combo in combinations(axes, k):
            covered = set()
            for a in combo:
                covered |= correct_sets[a]
            cov = len(covered) / n_cases
            if cov >= target:
                meeting.append((combo, cov))
        if meeting:
            # Keep the best coverage at this k
            best = max(meeting, key=lambda x: x[1])
            results[k] = {"sets_meeting_target": len(meeting),
                          "best_set": list(best[0]),
                          "best_coverage": round(best[1], 4)}
    return results


def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(BALANCED_ORIG)
    warmth = read_jsonl(BALANCED_WARMTH)
    balanced = orig + warmth
    n = len(balanced)
    print(f"Balanced battery: {len(orig)} firmness + {len(warmth)} warmth "
          f"= {n} cases")
    print(f"Axes: {len(AXES_15)}")

    # Split indices for per-split reporting
    n_orig = len(orig)

    all_results = {
        "metadata": {"n_cases": n, "n_firmness": n_orig,
                     "n_warmth": len(warmth), "axes": list(AXES_15.keys()),
                     "models": MODELS},
        "per_model": {},
    }

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False,
                                convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}"
                        for c in balanced]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}"
                       for c in balanced]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        # Per-axis correctness vectors
        axis_vecs = {}
        case_correct = {}
        axis_acc = {}
        for ax_name, (pos, neg) in AXES_15.items():
            v = compute_axis(embed_fn, pos, neg)
            axis_vecs[ax_name] = v
            cc = per_case_correct(better_embs, worse_embs, v)
            case_correct[ax_name] = cc
            full = sum(cc) / n
            firm = sum(cc[:n_orig]) / n_orig
            warm = sum(cc[n_orig:]) / (n - n_orig)
            axis_acc[ax_name] = {"combined": round(full, 4),
                                 "firmness": round(firm, 4),
                                 "warmth": round(warm, 4)}

        # Per-axis accuracy table
        print(f"\n{'Axis':<14s} {'Comb':>6s} {'Firm':>6s} {'Warm':>6s}")
        print("-" * 36)
        for ax in sorted(AXES_15, key=lambda a: -axis_acc[a]["combined"]):
            r = axis_acc[ax]
            mark = " *" if r["combined"] > 0.55 else ""
            print(f"  {ax:<12s} {r['combined']:5.0%} {r['firmness']:5.0%} "
                  f"{r['warmth']:5.0%}{mark}")

        # Pairwise correlation of per-case correctness
        ax_list = list(AXES_15.keys())
        corr_matrix = {}
        pair_list = []
        for a1, a2 in combinations(ax_list, 2):
            c = pearson(case_correct[a1], case_correct[a2])
            corr_matrix[f"{a1}|{a2}"] = round(c, 4)
            pair_list.append((a1, a2, c))

        # Most complementary (most negative) and most redundant (most positive)
        pair_sorted = sorted(pair_list, key=lambda x: x[2])
        print(f"\nMost COMPLEMENTARY pairs (negative correlation):")
        for a1, a2, c in pair_sorted[:8]:
            print(f"  {a1:<12s} x {a2:<12s}  r = {c:+.3f}")
        print(f"\nMost REDUNDANT pairs (positive correlation):")
        for a1, a2, c in pair_sorted[-8:][::-1]:
            print(f"  {a1:<12s} x {a2:<12s}  r = {c:+.3f}")

        # Minimum covering set (>90% via OR-coverage)
        greedy_set, greedy_cov = min_covering_set(case_correct, ax_list, n,
                                                  target=0.90)
        print(f"\nGreedy min covering set (>90% OR-coverage): "
              f"{len(greedy_set)} axes -> {greedy_cov:.0%}")
        print(f"  {greedy_set}")

        # Exhaustive true-minimum for k=1..5
        exh = exhaustive_min_cover(case_correct, ax_list, n, target=0.90,
                                   max_k=5)
        print(f"\nExhaustive search for true minimum covering set (>90%):")
        for k, info in exh.items():
            print(f"  k={k}: {info['sets_meeting_target']} sets meet target; "
                  f"best = {info['best_set']} @ {info['best_coverage']:.0%}")
        if not exh:
            print("  No set of <=5 axes reaches 90% OR-coverage on this model.")

        # Best SUM-OF-PROJECTIONS combo at each k (the actual scoring rule)
        print(f"\nBest SUM-OF-PROJECTIONS accuracy at each set size k:")
        for k in range(1, 5):
            best_combo, best_acc = None, -1
            for combo in combinations(ax_list, k):
                vecs = [axis_vecs[a] for a in combo]
                cc = sum_projection_correct(better_embs, worse_embs, vecs)
                acc = sum(cc) / n
                if acc > best_acc:
                    best_acc, best_combo = acc, combo
            print(f"  k={k}: {best_acc:.0%}  {list(best_combo)}")

        all_results["per_model"][model_name] = {
            "axis_accuracy": axis_acc,
            "correlation_matrix": corr_matrix,
            "most_complementary": [
                {"a1": a1, "a2": a2, "r": round(c, 4)}
                for a1, a2, c in pair_sorted[:10]],
            "most_redundant": [
                {"a1": a1, "a2": a2, "r": round(c, 4)}
                for a1, a2, c in pair_sorted[-10:][::-1]],
            "greedy_covering_set_90": greedy_set,
            "greedy_coverage": round(greedy_cov, 4),
            "exhaustive_min_cover_90": exh,
            "case_correct_vectors": {a: case_correct[a] for a in ax_list},
        }

        del model

    out_dir = ROOT / "notes" / "research_cycles" / "complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "complementarity_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # Cross-model consistency on complementarity
    print(f"\n{'='*70}")
    print("CROSS-MODEL: complementary pairs (r < -0.15) consistent across")
    print(">=2 of 3 models")
    print(f"{'='*70}")
    pair_consistency = {}
    for m in MODELS:
        cm = all_results["per_model"][m]["correlation_matrix"]
        for pair, r in cm.items():
            if r < -0.15:
                pair_consistency.setdefault(pair, []).append((m, r))
    found = False
    for pair, occ in sorted(pair_consistency.items(),
                            key=lambda x: -len(x[1])):
        if len(occ) >= 2:
            found = True
            models_str = ", ".join(
                f"{m.split('/')[-1]}={r:+.2f}" for m, r in occ)
            print(f"  {pair:<28s}  ({len(occ)}/3)  {models_str}")
    if not found:
        print("  None — no complementary pair (r<-0.15) appears on >=2 models.")


if __name__ == "__main__":
    main()
