#!/usr/bin/env python3
"""Test centroid direction on expanded evaluation set.

Original expansion: 35 cases (15 anti-syc, 5 each of 4 categories)
New expansion: 26 additional cases (10 factual, 8 creative, 8 nuance)
Total: 61 expansion cases

Tests the centroid trained on the original 70-case battery against the full
expanded set. Also tests per-category breakdown and margin analysis.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def embed_and_norm(model, texts):
    embs = model.encode(texts, convert_to_numpy=True, batch_size=32)
    for i in range(len(embs)):
        embs[i] /= norm(embs[i]) + 1e-12
    return embs


def accuracy(better_embs, worse_embs, direction):
    n = len(better_embs)
    return sum(1 for i in range(n)
               if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)) / n


def make_dir(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    d /= norm(d) + 1e-12
    return d


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)

    # Load ALL expansion files
    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    # Categorize
    exp_cats = {}
    for i, c in enumerate(expansion):
        cat = c.get("category", "unknown")
        exp_cats.setdefault(cat, []).append(i)

    print(f"Battery: {len(battery)}")
    print(f"Expansion: {len(expansion)} total")
    for cat in sorted(exp_cats.keys()):
        print(f"  {cat}: {len(exp_cats[cat])}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        bat_better = embed_and_norm(model, [c["better"] for c in battery])
        bat_worse = embed_and_norm(model, [c["worse"] for c in battery])
        exp_better = embed_and_norm(model, [c["better"] for c in expansion])
        exp_worse = embed_and_norm(model, [c["worse"] for c in expansion])

        dir_full = make_dir(bat_better, bat_worse)

        # Overall accuracy
        overall = accuracy(exp_better, exp_worse, dir_full)
        n_correct = sum(1 for i in range(len(expansion))
                        if np.dot(exp_better[i], dir_full) > np.dot(exp_worse[i], dir_full))
        lo, hi = wilson_ci(n_correct, len(expansion))
        print(f"\n  Overall OOS: {overall:.0%} ({n_correct}/{len(expansion)}) CI [{lo:.0%}, {hi:.0%}]")

        # Per-category
        print(f"\n  Per-category:")
        cat_results = {}
        for cat in sorted(exp_cats.keys()):
            idx = exp_cats[cat]
            n_cat = len(idx)
            correct = sum(1 for i in idx
                          if np.dot(exp_better[i], dir_full) > np.dot(exp_worse[i], dir_full))
            acc = correct / n_cat
            lo_c, hi_c = wilson_ci(correct, n_cat)
            print(f"    {cat:35s}: {acc:.0%} ({correct}/{n_cat}) CI [{lo_c:.0%}, {hi_c:.0%}]")
            cat_results[cat] = {"acc": float(acc), "n": n_cat, "correct": correct}

        # Logistic probe
        diff_bat = bat_better - bat_worse
        X = np.vstack([diff_bat, -diff_bat])
        y = np.concatenate([np.ones(len(battery)), np.zeros(len(battery))])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)

        diff_exp = exp_better - exp_worse
        X_exp = np.vstack([diff_exp, -diff_exp])
        y_exp = np.concatenate([np.ones(len(expansion)), np.zeros(len(expansion))])
        probe_acc = clf.score(X_exp, y_exp)
        print(f"\n  Logistic probe OOS: {probe_acc:.0%}")

        # Per-category probe
        print(f"  Per-category (probe):")
        for cat in sorted(exp_cats.keys()):
            idx = exp_cats[cat]
            n_cat = len(idx)
            correct = sum(1 for i in idx
                          if clf.predict(diff_exp[i:i+1])[0] == 1)
            print(f"    {cat:35s}: {correct/n_cat:.0%} ({correct}/{n_cat})")

        # Margin analysis
        print(f"\n  Margin analysis:")
        margins = []
        for i in range(len(expansion)):
            m = np.dot(exp_better[i], dir_full) - np.dot(exp_worse[i], dir_full)
            margins.append((m, expansion[i].get("id","?"), expansion[i].get("category","?")))

        margins.sort(key=lambda x: x[0])
        print(f"    Most confident WRONG (negative margin, should be positive):")
        for m, cid, cat in margins[:5]:
            if m < 0:
                print(f"      {m:+.4f} {cid} [{cat}]")

        print(f"    Most confident RIGHT:")
        for m, cid, cat in margins[-5:]:
            print(f"      {m:+.4f} {cid} [{cat}]")

        # Error details
        print(f"\n  Errors:")
        for i in range(len(expansion)):
            m = np.dot(exp_better[i], dir_full) - np.dot(exp_worse[i], dir_full)
            if m <= 0:
                c = expansion[i]
                print(f"    {c.get('id','?'):30s} [{c.get('category','?'):20s}] margin={m:+.6f}")

        all_results[short] = {
            "overall_acc": float(overall),
            "n_expansion": len(expansion),
            "probe_acc": float(probe_acc),
            "per_category": cat_results,
        }

        del model
        gc.collect()

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: Centroid OOS on {len(expansion)} expansion cases")
    print(f"{'='*80}")
    print(f"  {'':35s}", end="")
    for short in all_results:
        print(f"  {short:>20s}", end="")
    print()
    print(f"  {'Overall centroid':35s}", end="")
    for r in all_results.values():
        print(f"  {r['overall_acc']:19.0%}", end="")
    print()
    print(f"  {'Overall probe':35s}", end="")
    for r in all_results.values():
        print(f"  {r['probe_acc']:19.0%}", end="")
    print()

    for cat in sorted(exp_cats.keys()):
        print(f"  {cat:35s}", end="")
        for r in all_results.values():
            if cat in r['per_category']:
                print(f"  {r['per_category'][cat]['acc']:19.0%}", end="")
            else:
                print(f"  {'N/A':>19s}", end="")
        print()

    with open(OUT_DIR / "centroid_expanded_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'centroid_expanded_results.json'}")


if __name__ == "__main__":
    main()
