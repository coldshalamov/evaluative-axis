#!/usr/bin/env python3
"""E-05: Battery Expansion.

Tests whether existing axes that previously worked on the balanced battery
(50 firmness + 20 warmth) still work, or break, on four new case types:
  - nuance / context-sensitivity (5)
  - factual accuracy without emotion (5)
  - conciseness vs completeness (5)
  - creative quality (5)

All axes from the existing harness (5 ML-jargon axes + the single-word
candidate set used in the optimal-basis search) are scored against the new
20-case battery. We report overall accuracy, per-category accuracy, and
a direct comparison against the same axes' accuracy on the balanced battery
so we can see what breaks or flips.

Rules followed (AGENTS.md / RESEARCH_CONTEXT.md):
  - All three local models.
  - User/Assistant framing for response embeddings.
  - Per-axis scoring (no averaging of axis vectors).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES  # noqa: E402
from run_optimal_basis_search import CANDIDATE_AXES as WORD_AXES  # noqa: E402

EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
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


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def axis_accuracy(better_embs, worse_embs, axis_vec):
    """Return (accuracy, per_case_correct list of 0/1, margins list)."""
    n = len(better_embs)
    correct = []
    margins = []
    for i in range(n):
        sb = float(np.dot(better_embs[i], axis_vec))
        sw = float(np.dot(worse_embs[i], axis_vec))
        if sb > sw:
            correct.append(1)
            margins.append(sb - sw)
        elif sb == sw:
            correct.append(0.5)
            margins.append(0.0)
        else:
            correct.append(0)
            margins.append(sb - sw)
    acc = sum(correct) / n
    return acc, correct, margins


def embed_cases(model, cases):
    def embed_fn(texts):
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
    return embed_fn, embed_fn(better_texts), embed_fn(worse_texts)


def main():
    from sentence_transformers import SentenceTransformer

    # Load all battery splits
    exp_cases = []
    per_category_cases = {}
    for cat_file in sorted(EXPANSION_DIR.glob("*.jsonl")):
        rows = read_jsonl(cat_file)
        exp_cases.extend(rows)
        cat = rows[0]["category"]
        per_category_cases[cat] = rows
    balanced_orig = read_jsonl(BALANCED_ORIG)
    balanced_warmth = read_jsonl(BALANCED_WARMTH)
    balanced_all = balanced_orig + balanced_warmth

    print(f"Loaded {len(exp_cases)} expansion cases across "
          f"{len(per_category_cases)} categories")
    print(f"Balanced battery: {len(balanced_orig)} firmness + "
          f"{len(balanced_warmth)} warmth = {len(balanced_all)}")

    # Merge all axis definitions. Prefix to keep them distinct.
    all_axes = {}
    for name, anchors in ML_AXES.items():
        all_axes[f"ml/{name}"] = anchors
    for name, anchors in WORD_AXES.items():
        all_axes[f"word/{name}"] = anchors

    all_results = {
        "metadata": {
            "expansion_categories": list(per_category_cases.keys()),
            "n_expansion": len(exp_cases),
            "n_balanced": len(balanced_all),
            "n_models": len(MODELS),
            "n_axes": len(all_axes),
        },
        "per_model": {},
    }

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Embed all splits once per model
        splits = {
            "expansion": exp_cases,
            "firmness": balanced_orig,
            "warmth": balanced_warmth,
        }
        split_embs = {}
        for split_name, cases in splits.items():
            embed_fn, be, we = embed_cases(model, cases)
            split_embs[split_name] = (be, we)

        # Per-category embeddings for expansion
        cat_embs = {}
        for cat, cases in per_category_cases.items():
            embed_fn, be, we = embed_cases(model, cases)
            cat_embs[cat] = (be, we)

        def embed_fn_for_anchors(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        model_axes_results = {}
        print(f"\n{'Axis':<26s} {'Expand':>7s} {'Firm':>7s} {'Warm':>7s}  "
              f"per-category (expansion)")
        print("-" * 100)

        for axis_name, anchors in all_axes.items():
            axis_vec = compute_axis(embed_fn_for_anchors, anchors["positive"],
                                    anchors["negative"])

            exp_acc, _, _ = axis_accuracy(*split_embs["expansion"], axis_vec)
            firm_acc, _, _ = axis_accuracy(*split_embs["firmness"], axis_vec)
            warm_acc, _, _ = axis_accuracy(*split_embs["warmth"], axis_vec)

            cat_accs = {}
            for cat, (be, we) in cat_embs.items():
                a, _, _ = axis_accuracy(be, we, axis_vec)
                cat_accs[cat] = round(a, 4)

            model_axes_results[axis_name] = {
                "expansion": round(exp_acc, 4),
                "firmness": round(firm_acc, 4),
                "warmth": round(warm_acc, 4),
                "per_category": cat_accs,
            }

            marker = " *" if exp_acc > 0.55 else (" !" if exp_acc < 0.45 else "")
            cat_str = "  ".join(f"{k[:4]}={v:.0%}" for k, v in cat_accs.items())
            print(f"  {axis_name:<24s} {exp_acc:6.0%} {firm_acc:6.0%} "
                  f"{warm_acc:6.0%}  {cat_str}{marker}")

        all_results["per_model"][model_name] = model_axes_results
        del model

    out_dir = ROOT / "notes" / "research_cycles" / "battery_expansion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expansion_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # Cross-model summary: which axes flip between balanced and expansion
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY: axes whose expansion accuracy diverges from")
    print("balanced (firmness+warmth mean) by >= 15 points")
    print(f"{'='*70}")
    flips = []
    for axis_name in all_axes:
        for model_name in MODELS:
            r = all_results["per_model"][model_name][axis_name]
            balanced_mean = (r["firmness"] + r["warmth"]) / 2
            delta = r["expansion"] - balanced_mean
            if abs(delta) >= 0.15:
                flips.append((axis_name, model_name, r["expansion"],
                              balanced_mean, delta))
    if not flips:
        print("  None — no axis diverges by >= 15 points on any model.")
    else:
        print(f"  {'Axis':<24s} {'Model':<12s} {'Exp':>6s} {'Bal':>6s} "
              f"{'Delta':>7s}")
        for ax, m, e, b, d in sorted(flips, key=lambda x: x[4]):
            mshort = m.split("/")[-1]
            print(f"  {ax:<24s} {mshort:<12s} {e:5.0%} {b:5.0%} {d:+6.0%}")


if __name__ == "__main__":
    main()
