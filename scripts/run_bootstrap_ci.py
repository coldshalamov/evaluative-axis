#!/usr/bin/env python3
"""Bootstrap confidence intervals for key anchor axes.

Resamples the 50-case battery 10,000 times with replacement to produce
95% confidence intervals for each axis accuracy. This lets us report
which differences between axes are statistically meaningful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

KEY_AXES = {
    "careful_reckless":   {"positive": ["Careful"],    "negative": ["Reckless"]},
    "moderate_excessive": {"positive": ["Moderate"],   "negative": ["Excessive"]},
    "noble_base":         {"positive": ["Noble"],      "negative": ["Base"]},
    "good_bad":           {"positive": ["Good"],       "negative": ["Bad"]},
    "honest_dishonest":   {"positive": ["Honest"],     "negative": ["Dishonest"]},
    "true_false":         {"positive": ["True"],       "negative": ["False"]},
    "kind_cruel":         {"positive": ["Kind"],       "negative": ["Cruel"]},
    "complete_incomplete": {"positive": ["Complete"],  "negative": ["Incomplete"]},
    "useful_useless":     {"positive": ["Useful"],     "negative": ["Useless"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def pairwise_correct(better_embs, worse_embs, axis):
    """Return array of 0/1 per case."""
    results = []
    for i in range(len(better_embs)):
        s_better = float(np.dot(better_embs[i], axis))
        s_worse = float(np.dot(worse_embs[i], axis))
        if s_better > s_worse:
            results.append(1)
        elif s_better == s_worse:
            results.append(0.5)
        else:
            results.append(0)
    return np.array(results)


def bootstrap_ci(correct_array, n_boot=10000, ci=95):
    """Bootstrap CI for accuracy."""
    rng = np.random.default_rng(42)
    n = len(correct_array)
    boot_accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_accs[b] = correct_array[idx].mean()
    lo = np.percentile(boot_accs, (100 - ci) / 2)
    hi = np.percentile(boot_accs, 100 - (100 - ci) / 2)
    return float(lo), float(hi), float(correct_array.mean())


MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases, running 10,000 bootstrap resamples per axis")
    all_results = {}

    all_axes = {}
    all_axes.update({f"ml/{k}": v for k, v in ML_AXES.items()})
    all_axes.update({f"single/{k}": v for k, v in KEY_AXES.items()})

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        print(f"\n{'Axis':<35s} {'Acc':>6s}   {'95% CI':>16s}   {'Sig':>4s}")
        print("-" * 70)

        for axis_name, anchors in all_axes.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            correct = pairwise_correct(better_embs, worse_embs, axis_vec)
            lo, hi, acc = bootstrap_ci(correct)
            sig = "***" if lo > 0.50 else ("*" if lo > 0.45 else ("" if acc >= 0.50 else "inv"))
            model_results[axis_name] = {
                "accuracy": round(acc, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "significant": lo > 0.50,
            }
            print(f"  {axis_name:<33s} {acc:5.0%}   [{lo:5.1%}, {hi:5.1%}]   {sig}")

        all_results[model_name] = model_results
        del model

    # Cross-model summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY: Axes significantly above chance on all 3 models")
    print(f"{'='*70}")

    for axis_name in all_axes:
        sig_count = sum(
            1 for m in MODELS
            if all_results[m][axis_name]["significant"]
        )
        accs = [all_results[m][axis_name]["accuracy"] for m in MODELS]
        if sig_count >= 2:
            print(f"  {axis_name:<35s} sig on {sig_count}/3 models  accs={[f'{a:.0%}' for a in accs]}")

    out_dir = ROOT / "notes" / "research_cycles" / "bootstrap_ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
