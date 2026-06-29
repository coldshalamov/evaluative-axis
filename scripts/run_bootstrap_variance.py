#!/usr/bin/env python3
"""Bootstrap variance analysis — the question the user is right to ask.

Every accuracy number we've reported is a point estimate on 50-20 cases.
If honest is 13% on Snowflake and 80% on BGE-M3, is that signal or just
variance on tiny samples? This resamples cases with replacement (1000x)
to get 95% confidence intervals, and tells us which findings are real.

Uses per-case margins from diagnostic_profiles (15 axes x 3 models x 70 cases).
We cannot bootstrap the large_word_sweep (no per-case data stored) — so this
only covers the 15 axes. But the 15 include the key contested ones: good,
honest, careful, kind, hard.

For each axis x model x split we report:
  point estimate
  95% bootstrap CI (2.5th, 97.5th percentile)
  CI width (the real uncertainty)
  whether CI excludes 50% (the true chance rate, not the random-word floor)

Then the cross-model consistency check: does the CI on model A overlap the
CI on model B? If they don't overlap, the cross-model difference is real.
If they do overlap, "13% vs 80%" is within noise.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "notes" / "research_cycles" / "diagnostic_profiles" / "profile_results.json"
OUT = ROOT / "notes" / "research_cycles" / "bootstrap_variance" / "bootstrap_results.json"

d = json.loads(PROFILE.read_text(encoding="utf-8"))
MODELS = d["models"]
AXES = list(d["axes"].keys())
N_BOOT = 2000
rng = np.random.default_rng(42)


def acc_from_margins(margins):
    m = np.asarray(margins)
    return (np.sum(m > 0) + 0.5 * np.sum(m == 0)) / len(m)


def bootstrap_ci(margins, n_boot=N_BOOT):
    """Resample cases with replacement, return (mean, lo95, hi95)."""
    margins = np.asarray(margins)
    n = len(margins)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = acc_from_margins(margins[idx])
    return float(np.mean(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    print(f"Bootstrap CI on {len(AXES)} axes x {len(MODELS)} models")
    print(f"N_BOOT = {N_BOOT} resamples per cell")
    print(f"Firmness n=50, Warmth n=20\n")

    results = {"metadata": {"n_boot": N_BOOT, "axes": AXES, "models": MODELS,
                            "n_firmness": 50, "n_warmth": 20},
               "per_model": {}}

    for model in MODELS:
        ms = model.split("/")[-1]
        print(f"\n{'='*82}")
        print(f"MODEL: {ms}")
        print(f"{'='*82}")

        # Two tables: firmness and warmth
        for split, n_cases, label in [("original", 50, "FIRMNESS (n=50)"),
                                      ("warmth", 20, "WARMTH (n=20)")]:
            print(f"\n  {label}")
            print(f"  {'axis':<14s} {'point':>6s} {'CI95':<18s} {'width':>6s} "
                  f"{'excl50':>7s}")
            print("  " + "-" * 60)
            model_ax = {}
            for ax in AXES:
                margins = d["results"][model]["axis_accuracy"][split][ax]["margins"]
                mean, lo, hi = bootstrap_ci(margins)
                width = hi - lo
                excludes_50 = "YES" if (lo > 0.5 or hi < 0.5) else "no"
                model_ax[ax] = {"point": round(acc_from_margins(margins), 4),
                                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                "ci_width": round(width, 4),
                                "excludes_50": lo > 0.5 or hi < 0.5}
                flag = "  ***REAL" if (lo > 0.5 or hi < 0.5) else ""
                print(f"  {ax:<14s} {acc_from_margins(margins):5.0%} "
                      f"[{lo:.0%}, {hi:.0%}]    {width:5.0%} {excludes_50:>7s}{flag}")
            results["per_model"].setdefault(model, {})[split] = model_ax

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT}")

    # The cross-model overlap check — the real test of "13 vs 80"
    print(f"\n{'='*82}")
    print("CROSS-MODEL OVERLAP: do CIs overlap between models?")
    print("(if they don't overlap, the cross-model difference is REAL)")
    print(f"{'='*82}")
    for split in ["original", "warmth"]:
        print(f"\n  {split.upper()}")
        print(f"  {'axis':<14s} {'Snowflake CI':<22s} {'BGE-M3 CI':<22s} {'Nomic CI':<22s} {'verdict'}")
        for ax in AXES:
            cis = []
            for m in MODELS:
                r = results["per_model"][m][split][ax]
                cis.append((m.split("/")[-1][:8], r["ci_lo"], r["ci_hi"]))
            # Check pairwise overlap
            overlaps = []
            for i in range(len(cis)):
                for j in range(i+1, len(cis)):
                    n1, lo1, hi1 = cis[i]
                    n2, lo2, hi2 = cis[j]
                    overlap = not (hi1 < lo2 or hi2 < lo1)
                    overlaps.append((n1, n2, overlap))
            any_disjoint = any(not o for _, _, o in overlaps)
            verdict = "DISJOINT (real diff)" if any_disjoint else "all overlap"
            s = "  ".join(f"{n}:{lo:.0%}-{hi:.0%}" for n, lo, hi in cis)
            print(f"  {ax:<14s} {s}  {verdict}")


if __name__ == "__main__":
    main()
