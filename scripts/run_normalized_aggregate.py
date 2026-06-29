#!/usr/bin/env python3
"""The decisive test of the balanced-aggregate hypothesis.

The user's hypothesis: train toward the highest aggregate signal across a
matrix of terms that catch each other's failure modes. Sum many axes; the
ones that catch each other's biases cancel out in the aggregate.

Prior "sum of deltas" tests FAILED — but they used RAW margins, where the
largest-magnitude axis (always a warmth axis: good=0.87, helpful=0.61 on
Nomic vs thorough=0.33) drowns out the firmness axes by 2-3x. The balanced
hypothesis was never actually tested fairly.

This script tests the hypothesis with the magnitude control:
  - raw sum            (the failed approach, reproduced)
  - z-score normalized sum  (each axis standardized to mean=0, std=1)
  - sign sum           (pure vote count, magnitude fully removed)
  - rank-normalized sum (nonparametric, robust to outliers)

If the balanced-aggregate hypothesis is correct, the NORMALIZED sums should
be balanced (firm ≈ warm) AND high overall — because equalizing per-axis
contribution lets the complementary axes cancel each other's biases.

Uses existing per-case margins from diagnostic_profiles (no re-embedding
needed). This is a pure re-aggregation of already-computed data.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "notes" / "research_cycles" / "diagnostic_profiles" / "profile_results.json"
OUT = ROOT / "notes" / "research_cycles" / "normalized_aggregate" / "normalized_aggregate_results.json"

d = json.loads(PROFILE.read_text(encoding="utf-8"))
MODELS = d["models"]
AXES = list(d["axes"].keys())  # 15 axes

# Axis groupings (from the tree / neighborhood analysis)
FIRMNESS_AXES = ["hard", "careful", "thorough"]          # competence branch
WARMTH_AXES = ["good", "kind", "helpful", "honest", "supportive", "constructive", "gentle", "wise", "patient", "responsible", "fair", "clear"]
TREE5 = ["careful", "honest", "helpful", "thorough"]     # restrained not in this file; use available

def load_margins(model):
    """Return margins dict: margins[axis] = np.array of per-case margins.
    Order: firmness cases first (50), then warmth (20)."""
    aa = d["results"][model]["axis_accuracy"]
    margins = {}
    for ax in AXES:
        firm = aa["original"][ax]["margins"]
        warm = aa["warmth"][ax]["margins"]
        margins[ax] = np.array(firm + warm)
    return margins  # shape (70,) per axis

def accuracy(scores):
    """scores: array of margins for each case. Better response wins if > 0."""
    s = np.asarray(scores)
    n = len(s)
    correct = float(np.sum(s > 0))
    ties = float(np.sum(s == 0))
    return (correct + 0.5 * ties) / n

def split_acc(scores, n_firm=50):
    s = np.asarray(scores)
    return {
        "combined": accuracy(s),
        "firmness": accuracy(s[:n_firm]),
        "warmth": accuracy(s[n_firm:]),
    }

def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return x - x.mean()
    return (x - x.mean()) / sd

def ranknorm(x):
    """Convert to uniform [0,1] via ranking, then center to [-0.5, 0.5]."""
    from scipy.stats import rankdata
    r = rankdata(x) / (len(x))  # 0..1
    return r - 0.5

def aggregate(margins, axes, method):
    """Combine per-axis margins into a single per-case score."""
    if method == "raw":
        return sum(margins[a] for a in axes)
    if method == "zscore":
        return sum(zscore(margins[a]) for a in axes)
    if method == "sign":
        return sum(np.sign(margins[a]) for a in axes)
    if method == "rank":
        return sum(ranknorm(margins[a]) for a in axes)
    raise ValueError(method)

SETS = {
    "all_15": AXES,
    "tree_4_available": TREE5,
    "firmness_3": FIRMNESS_AXES,
    "warmth_8": WARMTH_AXES[:8],
}
METHODS = ["raw", "zscore", "sign", "rank"]

print("=" * 90)
print("THE DECISIVE TEST: does per-axis normalization fix the aggregate?")
print("Hypothesis: raw sum is warmth-dominated due to magnitude; normalized")
print("sums should be balanced (firm≈warm) and high if axes are complementary.")
print("=" * 90)

all_results = {"metadata": {"axes": AXES, "sets": SETS, "methods": METHODS},
               "per_model": {}}

for model in MODELS:
    ms = model.split("/")[-1]
    margins = load_margins(model)
    print(f"\n{'='*90}")
    print(f"MODEL: {ms}")
    print(f"{'='*90}")

    print(f"\n{'Set':<18s} {'Method':<8s} {'Comb':>6s} {'Firm':>6s} {'Warm':>6s} "
          f"{'balance':>8s}")
    print("-" * 60)

    model_res = {}
    for set_name, axes in SETS.items():
        model_res[set_name] = {}
        for method in METHODS:
            scores = aggregate(margins, axes, method)
            acc = split_acc(scores)
            balance = acc["firmness"] - acc["warmth"]
            model_res[set_name][method] = {k: round(v, 4) for k, v in acc.items()}
            flag = ""
            if abs(balance) < 0.10 and acc["combined"] > 0.55:
                flag = "  <-- balanced & >55%"
            print(f"  {set_name:<16s} {method:<8s} {acc['combined']:5.0%} "
                  f"{acc['firmness']:5.0%} {acc['warmth']:5.0%} "
                  f"{balance:+7.0%}{flag}")

    # Also test individual axes for reference
    print(f"\n  Single-axis reference (top 5 by combined):")
    single = [(a, split_acc(margins[a])) for a in AXES]
    single.sort(key=lambda x: -x[1]["combined"])
    for a, acc in single[:5]:
        print(f"    {a:<14s} {acc['combined']:5.0%} (f={acc['firmness']:.0%} w={acc['warmth']:.0%})")

    all_results["per_model"][model] = model_res

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
print(f"\nSaved: {OUT}")

# Cross-model summary: does zscore on all_15 beat raw and beat single-axis?
print(f"\n{'='*90}")
print("CROSS-MODEL: all_15 set, raw vs zscore vs best-single-axis")
print(f"{'='*90}")
print(f"{'Model':<28s} {'raw':>8s} {'zscore':>8s} {'sign':>8s} {'rank':>8s} "
      f"{'best1':>8s} {'best1_name':<14s}")
for model in MODELS:
    ms = model.split("/")[-1]
    r = all_results["per_model"][model]["all_15"]
    margins = load_margins(model)
    single = [(a, split_acc(margins[a])["combined"]) for a in AXES]
    best_a, best_acc = max(single, key=lambda x: x[1])
    print(f"  {ms:<26s} {r['raw']['combined']:7.0%} {r['zscore']['combined']:7.0%} "
          f"{r['sign']['combined']:7.0%} {r['rank']['combined']:7.0%} "
          f"{best_acc:7.0%} {best_a}")
    # balance check
    print(f"    balance(f-w): raw {r['raw']['firmness']-r['raw']['warmth']:+.0%} "
          f"zscore {r['zscore']['firmness']-r['zscore']['warmth']:+.0%} "
          f"sign {r['sign']['firmness']-r['sign']['warmth']:+.0%} "
          f"rank {r['rank']['firmness']-r['rank']['warmth']:+.0%}")
