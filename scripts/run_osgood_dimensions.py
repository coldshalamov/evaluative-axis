#!/usr/bin/env python3
"""Test Osgood's three semantic differential dimensions.

Osgood (1957) found that human semantic judgment reduces to three
orthogonal factors: Evaluation (good/bad), Potency (strong/weak),
and Activity (active/passive). We test:
1. Each dimension individually
2. The composite of all three (averaged axis)
3. A "sum of projections" approach (score = sum of all 3 axis projections)
4. Typical adjective pairs Osgood used for each dimension

The question: does the ORTHOGONAL combination capture something
that individual evaluative terms miss?
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

# Osgood's three dimensions with representative adjective pairs
OSGOOD_INDIVIDUAL = {
    # EVALUATION dimension
    "eval_good_bad":         {"positive": ["Good"],        "negative": ["Bad"]},
    "eval_nice_awful":       {"positive": ["Nice"],        "negative": ["Awful"]},
    "eval_sweet_bitter":     {"positive": ["Sweet"],       "negative": ["Bitter"]},
    "eval_pleasant_unpleasant": {"positive": ["Pleasant"], "negative": ["Unpleasant"]},

    # POTENCY dimension
    "pot_strong_weak":       {"positive": ["Strong"],      "negative": ["Weak"]},
    "pot_hard_soft":         {"positive": ["Hard"],        "negative": ["Soft"]},
    "pot_heavy_light":       {"positive": ["Heavy"],       "negative": ["Light"]},
    "pot_large_small":       {"positive": ["Large"],       "negative": ["Small"]},

    # ACTIVITY dimension
    "act_active_passive":    {"positive": ["Active"],      "negative": ["Passive"]},
    "act_fast_slow":         {"positive": ["Fast"],        "negative": ["Slow"]},
    "act_hot_cold":          {"positive": ["Hot"],         "negative": ["Cold"]},
    "act_sharp_dull":        {"positive": ["Sharp"],       "negative": ["Dull"]},
}

# Composites: combining across Osgood dimensions
OSGOOD_COMPOSITES = {
    # Pure evaluation centroid (multiple eval pairs)
    "eval_centroid": {
        "positive": ["Good", "Nice", "Pleasant"],
        "negative": ["Bad", "Awful", "Unpleasant"],
    },
    # Pure potency centroid
    "pot_centroid": {
        "positive": ["Strong", "Hard", "Large"],
        "negative": ["Weak", "Soft", "Small"],
    },
    # Pure activity centroid
    "act_centroid": {
        "positive": ["Active", "Fast", "Sharp"],
        "negative": ["Passive", "Slow", "Dull"],
    },
    # Full Osgood EPA composite (one pair per dimension)
    "osgood_EPA_core": {
        "positive": ["Good", "Strong", "Active"],
        "negative": ["Bad", "Weak", "Passive"],
    },
    # Full Osgood EPA composite (3 pairs per dimension)
    "osgood_EPA_full": {
        "positive": ["Good", "Nice", "Pleasant", "Strong", "Hard", "Large", "Active", "Fast", "Sharp"],
        "negative": ["Bad", "Awful", "Unpleasant", "Weak", "Soft", "Small", "Passive", "Slow", "Dull"],
    },
}

# Reference single-word axes for comparison
REFERENCE = {
    "careful_reckless":   {"positive": ["Careful"],  "negative": ["Reckless"]},
    "moderate_excessive": {"positive": ["Moderate"], "negative": ["Excessive"]},
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


def score_battery(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        s_better = float(np.dot(better_embs[i], axis))
        s_worse = float(np.dot(worse_embs[i], axis))
        if s_better > s_worse:
            correct += 1
        elif s_better == s_worse:
            correct += 0.5
    return correct / len(better_embs)


def score_battery_sum(better_embs, worse_embs, axes_list):
    """Score using sum of projections across multiple axes."""
    correct = 0
    for i in range(len(better_embs)):
        s_better = sum(float(np.dot(better_embs[i], ax)) for ax in axes_list)
        s_worse = sum(float(np.dot(worse_embs[i], ax)) for ax in axes_list)
        if s_better > s_worse:
            correct += 1
        elif s_better == s_worse:
            correct += 0.5
    return correct / len(better_embs)


MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")
    all_results = {}

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

        # ML-jargon baselines
        print("\n--- ML-jargon baselines ---")
        for axis_name, anchors in ML_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"ml/{axis_name}"] = round(acc, 4)
            print(f"  ml/{axis_name:30s}: {acc:.0%}")

        # Reference
        print("\n--- Reference (best single words) ---")
        for axis_name, anchors in REFERENCE.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"ref/{axis_name}"] = round(acc, 4)
            print(f"  ref/{axis_name:29s}: {acc:.0%}")

        # Osgood individual pairs
        print("\n--- Osgood individual pairs ---")
        dim_axes = {"eval": [], "pot": [], "act": []}
        for axis_name, anchors in OSGOOD_INDIVIDUAL.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"osgood/{axis_name}"] = round(acc, 4)
            dim = axis_name.split("_")[0]
            dim_axes[dim].append(axis_vec)
            print(f"  osgood/{axis_name:26s}: {acc:.0%}")

        # Osgood composites
        print("\n--- Osgood composites ---")
        for axis_name, anchors in OSGOOD_COMPOSITES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"composite/{axis_name}"] = round(acc, 4)
            print(f"  composite/{axis_name:24s}: {acc:.0%}")

        # Sum-of-projections: add scores from E, P, A independently
        print("\n--- Sum of projections (E+P+A scored separately, summed) ---")

        # Best individual from each dimension
        eval_best = compute_axis(embed_fn, ["Good"], ["Bad"])
        pot_best = compute_axis(embed_fn, ["Strong"], ["Weak"])
        act_best = compute_axis(embed_fn, ["Active"], ["Passive"])

        acc_ep = score_battery_sum(better_embs, worse_embs, [eval_best, pot_best])
        acc_ea = score_battery_sum(better_embs, worse_embs, [eval_best, act_best])
        acc_pa = score_battery_sum(better_embs, worse_embs, [pot_best, act_best])
        acc_epa = score_battery_sum(better_embs, worse_embs, [eval_best, pot_best, act_best])

        model_results["sum/E+P"] = round(acc_ep, 4)
        model_results["sum/E+A"] = round(acc_ea, 4)
        model_results["sum/P+A"] = round(acc_pa, 4)
        model_results["sum/E+P+A"] = round(acc_epa, 4)

        print(f"  sum/E+P (good+strong):          {acc_ep:.0%}")
        print(f"  sum/E+A (good+active):           {acc_ea:.0%}")
        print(f"  sum/P+A (strong+active):         {acc_pa:.0%}")
        print(f"  sum/E+P+A (good+strong+active):  {acc_epa:.0%}")

        # Sum with centroid axes
        eval_cent = compute_axis(embed_fn, ["Good", "Nice", "Pleasant"], ["Bad", "Awful", "Unpleasant"])
        pot_cent = compute_axis(embed_fn, ["Strong", "Hard", "Large"], ["Weak", "Soft", "Small"])
        act_cent = compute_axis(embed_fn, ["Active", "Fast", "Sharp"], ["Passive", "Slow", "Dull"])

        acc_cent_epa = score_battery_sum(better_embs, worse_embs, [eval_cent, pot_cent, act_cent])
        model_results["sum/EPA_centroids"] = round(acc_cent_epa, 4)
        print(f"  sum/EPA centroids:               {acc_cent_epa:.0%}")

        # Cosine between E, P, A directions (are they actually orthogonal?)
        print("\n--- Axis cosine similarity (are E, P, A orthogonal?) ---")
        cos_ep = float(np.dot(eval_best, pot_best))
        cos_ea = float(np.dot(eval_best, act_best))
        cos_pa = float(np.dot(pot_best, act_best))
        print(f"  E·P = {cos_ep:.3f}")
        print(f"  E·A = {cos_ea:.3f}")
        print(f"  P·A = {cos_pa:.3f}")
        model_results["cos_EP"] = round(cos_ep, 4)
        model_results["cos_EA"] = round(cos_ea, 4)
        model_results["cos_PA"] = round(cos_pa, 4)

        all_results[model_name] = model_results
        del model

    out_dir = ROOT / "notes" / "research_cycles" / "osgood_dimensions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "osgood_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
