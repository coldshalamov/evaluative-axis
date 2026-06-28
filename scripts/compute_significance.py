#!/usr/bin/env python3
"""Compute binomial-test p-values and bootstrap CIs for reported results."""

from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), using math.comb (no scipy)."""
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * p**i * (1 - p) ** (n - i)
    return total


def binom_test_greater(successes: int, n: int, p0: float = 0.5) -> float:
    """One-sided P(X >= successes | H0: p = p0)."""
    return 1.0 - binom_cdf(successes - 1, n, p0)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (works well even for small n)."""
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


results = {}

print("=" * 72)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 72)

# --- Battery results (Gemini, 50 cases) ---
print("\n## 50-Case Battery (Gemini Embedding 2, n=50, H0: p=0.50)")
print(f"{'Method':<35s} {'k/n':>6s} {'Acc':>7s} {'p-value':>10s} {'95% CI':>16s}")
print("-" * 76)

battery = [
    ("Raw good/bad", 13, 50),
    ("Best proxy (useful/useless)", 21, 50),
    ("Targeted: anti-sycophancy", 49, 50),
    ("Targeted: persona honesty", 48, 50),
    ("Targeted: harm reduction", 47, 50),
    ("Targeted: truthfulness", 45, 50),
    ("Combined targeted axes", 43, 50),
]

for name, k, n in battery:
    p = binom_test_greater(k, n)
    lo, hi = wilson_ci(k, n)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{name:<35s} {k:>2d}/{n:<2d} {k/n:>6.1%} {p:>10.2e} [{lo:.1%}, {hi:.1%}] {sig}")
    results[f"battery_{name}"] = {"k": k, "n": n, "acc": k / n, "p": p, "ci_lo": lo, "ci_hi": hi}

# --- Objective reranking (Gemini) ---
print("\n## Objective Reranking (Gemini Embedding 2, H0: p=1/3)")
print("  (Note: p=1/3 is random baseline for 3-way selection)")
print(f"{'Domain':<20s} {'k/n':>6s} {'Acc':>7s} {'p-value':>10s} {'95% CI':>16s}")
print("-" * 62)

# For 3-way selection, random baseline is 1/3 not 1/2
reranking = [
    ("Code (3-way)", 5, 6, 1 / 3),
    ("Math (3-way)", 8, 8, 1 / 3),
    ("Tool (3-way)", 7, 8, 1 / 3),
]

for name, k, n, p0 in reranking:
    p = binom_test_greater(k, n, p0)
    lo, hi = wilson_ci(k, n)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{name:<20s} {k:>2d}/{n:<2d} {k/n:>6.1%} {p:>10.4f} [{lo:.1%}, {hi:.1%}] {sig}")
    results[f"reranking_{name}"] = {"k": k, "n": n, "p0": p0, "acc": k / n, "p": p, "ci_lo": lo, "ci_hi": hi}

# --- Capability gradient (OSS vs Gemini on battery) ---
print("\n## Capability Gradient (best local model vs Gemini, n=50)")
print(f"{'Model':<40s} {'k/n':>6s} {'Acc':>7s} {'p-value':>10s} {'95% CI':>16s}")
print("-" * 80)

oss_models = [
    ("Snowflake Arctic-M (best local)", 37, 50),
    ("Jina v2 small", 34, 50),
    ("Jina v2 base", 32, 50),
    ("BGE-small", 31, 50),
    ("Gemini (combined targeted)", 43, 50),
    ("Gemini (best targeted axis)", 49, 50),
]

for name, k, n in oss_models:
    p = binom_test_greater(k, n)
    lo, hi = wilson_ci(k, n)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{name:<40s} {k:>2d}/{n:<2d} {k/n:>6.1%} {p:>10.2e} [{lo:.1%}, {hi:.1%}] {sig}")
    results[f"oss_{name}"] = {"k": k, "n": n, "acc": k / n, "p": p, "ci_lo": lo, "ci_hi": hi}

# --- Save ---
out_path = ROOT / "notes" / "research_cycles" / "statistical_significance.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {out_path}")
