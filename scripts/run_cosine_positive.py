#!/usr/bin/env python3
"""Experiment 1: Cosine-to-Positive vs Bipolar (systematic, 15 axes).

This is an ablation of the default scoring recipe, not the recipe itself.
The confirmatory default is fixed bipolar scoring with User/Assistant
framing. Cosine-to-positive (dropping the negative anchor) is tested here
because it sometimes helps BGE-M3 and sometimes hurts other models.

For each of 15 axes x 3 models x 3 splits we report:
  - bipolar accuracy (standard normalize(pos - neg), dot product)
  - cosine-to-positive accuracy (cos(response, pos))
  - divergence (>10 pts flagged)

Then we test the covering-set question:
  - For audit only, pick the per-axis-optimal method (whichever scores higher
    on combined accuracy).
  - Re-score the 3-axis set {hard, kind, X} for every X via SUM of independent
    per-axis scores, using bipolar throughout, then using per-axis-optimal.
  - Report whether post-hoc per-axis selection improves the best 3-set.

Rules (AGENTS.md / RESEARCH_CONTEXT.md):
  - All three local models, both splits + combined.
  - User/Assistant framing for response embeddings.
  - Combining multiple axes uses SUM of independent projections (no averaging).
"""

from __future__ import annotations

import gc
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scoring_recipe import (
    DEFAULT_SCORING_METHOD,
    bipolar_scores,
    cosine_positive_scores,
    framed_response,
    pairwise_accuracy,
    recipe_metadata,
)

BATTERY_ORIGINAL = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BATTERY_WARMTH = (
    ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
)
OUTPUT_PATH = (
    ROOT / "notes" / "research_cycles" / "cosine_positive" / "cosine_positive_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# 15 axes (same word pairs as E-07 complementarity).
AXES = {
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


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    n_orig = len(original)
    n_warm = len(warmth)
    combined = original + warmth

    print("Experiment 1: Cosine-to-Positive vs Bipolar (15 axes)")
    print(f"Firmness: {n_orig}, Warmth: {n_warm}, Combined: {len(combined)}")

    results = {
        "experiment": "Cosine-to-Positive vs Bipolar (15 axes)",
        "n_firmness": n_orig,
        "n_warmth": n_warm,
        "axes": {k: {"positive": v[0], "negative": v[1]} for k, v in AXES.items()},
        "models": MODELS,
        "scoring_recipe": recipe_metadata("cosine_positive_ablation"),
        "default_method": DEFAULT_SCORING_METHOD,
        "post_hoc_selection_warning": (
            "per-axis-optimal methods are audited here but are not the "
            "confirmatory default"
        ),
        "per_model": {},
        "divergences_gt_10pts": [],  # filled across all models
        "covering_set": {},
    }

    for model_name in MODELS:
        print(f"\n{'='*76}")
        print(f"MODEL: {model_name}")
        print(f"{'='*76}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False,
                                convert_to_numpy=True)

        # Embed responses once per split
        split_embs = {
            "firmness": (
                embed_fn([framed_response(c, "better") for c in original]),
                embed_fn([framed_response(c, "worse") for c in original]),
            ),
            "warmth": (
                embed_fn([framed_response(c, "better") for c in warmth]),
                embed_fn([framed_response(c, "worse") for c in warmth]),
            ),
            "combined": (
                embed_fn([framed_response(c, "better") for c in combined]),
                embed_fn([framed_response(c, "worse") for c in combined]),
            ),
        }

        # Per-axis anchor embeddings, plus cached score arrays for covering set
        axis_embs = {}
        for ax_name, (pos, neg) in AXES.items():
            pos_emb = embed_fn([pos])[0]
            neg_emb = embed_fn([neg])[0]
            axis_embs[ax_name] = (pos_emb, neg_emb)

        model_axes = {}
        print(f"\n{'Axis':<14s} {'bip/bip/bip':<20s} {'cos/cos/cos':<20s} "
              f"{'winner(comb)':<14s} {'divergence':<10s}")
        print(f"{'':14s} {'firm warm comb':<20s} {'firm warm comb':<20s}")
        print("-" * 82)

        for ax_name, (pos, neg) in AXES.items():
            pos_emb, neg_emb = axis_embs[ax_name]
            entry = {}
            for split, (be, we) in split_embs.items():
                bip = pairwise_accuracy(bipolar_scores(be, pos_emb, neg_emb),
                                        bipolar_scores(we, pos_emb, neg_emb))
                cos = pairwise_accuracy(cosine_positive_scores(be, pos_emb),
                                        cosine_positive_scores(we, pos_emb))
                entry[split] = {"bipolar": round(bip, 4),
                                "cosine_positive": round(cos, 4)}
            model_axes[ax_name] = entry

            comb_bip = entry["combined"]["bipolar"]
            comb_cos = entry["combined"]["cosine_positive"]
            winner = "cos" if comb_cos > comb_bip else (
                "bip" if comb_bip > comb_cos else "tie")
            div = comb_cos - comb_bip
            flag = "  <-- >10pt" if abs(div) >= 0.10 else ""
            print(f"  {ax_name:<12s} "
                  f"{entry['firmness']['bipolar']:.0%}/{entry['warmth']['bipolar']:.0%}/{comb_bip:.0%}      "
                  f"{entry['firmness']['cosine_positive']:.0%}/{entry['warmth']['cosine_positive']:.0%}/{comb_cos:.0%}      "
                  f"{winner:<12s} {div:+.0%}{flag}")

            if abs(div) >= 0.10:
                results["divergences_gt_10pts"].append({
                    "model": model_name, "axis": ax_name,
                    "bipolar_combined": comb_bip,
                    "cosine_combined": comb_cos,
                    "delta_cos_minus_bip": round(div, 4),
                    "winner": winner,
                })

        # ---- Covering-set test ----
        # Sum-of-projections: score each axis independently, sum the scores.
        def sum_proj_correct(be, we, ax_list, method_per_axis):
            """Return accuracy of sum-of-independent-scores over ax_list."""
            n = len(be)
            sb_total = np.zeros(n)
            sw_total = np.zeros(n)
            for ax in ax_list:
                pos_emb, neg_emb = axis_embs[ax]
                if method_per_axis[ax] == "cos":
                    sb_total += cosine_positive_scores(be, pos_emb)
                    sw_total += cosine_positive_scores(we, pos_emb)
                else:
                    sb_total += bipolar_scores(be, pos_emb, neg_emb)
                    sw_total += bipolar_scores(we, pos_emb, neg_emb)
            return pairwise_accuracy(sb_total, sw_total)

        # Per-axis-optimal method (by combined accuracy)
        optimal_method = {}
        for ax_name in AXES:
            bip = model_axes[ax_name]["combined"]["bipolar"]
            cos = model_axes[ax_name]["combined"]["cosine_positive"]
            optimal_method[ax_name] = "cos" if cos > bip else "bip"

        others = [a for a in AXES if a not in ("hard", "kind")]

        be_c, we_c = split_embs["combined"]

        print(f"\nCovering-set test: hard + kind + X (sum-of-projections, combined)")
        print(f"{'X':<14s} {'bip all':>8s} {'opt all':>8s} {'delta':>7s}")
        print("-" * 40)
        set_results = []
        for x in others:
            ax_set = ["hard", "kind", x]
            bip_all = {"hard": "bip", "kind": "bip", x: "bip"}
            opt_all = {a: optimal_method[a] for a in ax_set}
            a_bip = sum_proj_correct(be_c, we_c, ax_set, bip_all)
            a_opt = sum_proj_correct(be_c, we_c, ax_set, opt_all)
            delta = a_opt - a_bip
            flag = " *" if delta >= 0.05 else ""
            print(f"  {x:<12s} {a_bip:7.0%} {a_opt:7.0%} {delta:+6.0%}{flag}")
            set_results.append({
                "set": ax_set, "bipolar_all": round(a_bip, 4),
                "optimal_per_axis": round(a_opt, 4),
                "delta": round(delta, 4),
                "methods_used": opt_all,
            })

        # Best 3-set under each policy
        best_bip = max(set_results, key=lambda r: r["bipolar_all"])
        best_opt = max(set_results, key=lambda r: r["optimal_per_axis"])

        results["covering_set"][model_name] = {
            "optimal_method_per_axis": optimal_method,
            "three_axis_sets": set_results,
            "best_bipolar_all": best_bip,
            "best_per_axis_optimal": best_opt,
        }

        print(f"\nBest 3-set (bipolar all):     {best_bip['set']} "
              f"@ {best_bip['bipolar_all']:.0%}")
        print(f"Best 3-set (per-axis-optimal): {best_opt['set']} "
              f"@ {best_opt['optimal_per_axis']:.0%}  "
              f"(methods: {best_opt['methods_used']})")

        results["per_model"][model_name] = model_axes
        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")

    # ---- Cross-model summary ----
    print(f"\n{'='*76}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*76}")
    n_div = len(results["divergences_gt_10pts"])
    print(f"Divergences >10pts (any model/axis): {n_div}")
    by_axis = {}
    for d in results["divergences_gt_10pts"]:
        by_axis.setdefault(d["axis"], []).append(
            (d["model"].split("/")[-1], d["delta_cos_minus_bip"], d["winner"]))
    for ax, occ in sorted(by_axis.items()):
        s = ", ".join(f"{m}:{d:+.0%}({w})" for m, d, w in occ)
        print(f"  {ax:<12s} {s}")

    # Which method wins per axis, cross-model?
    print(f"\nMethod winner by axis (combined, across models):")
    for ax in AXES:
        cos_wins = 0
        parts = []
        for m in MODELS:
            r = results["per_model"][m][ax]["combined"]
            d = r["cosine_positive"] - r["bipolar"]
            if d > 0:
                cos_wins += 1
            parts.append(f"{m.split('/')[-1]}:{d:+.0%}")
        print(f"  {ax:<12s} cosine wins {cos_wins}/3  ({', '.join(parts)})")


if __name__ == "__main__":
    main()
