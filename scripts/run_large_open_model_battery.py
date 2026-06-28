#!/usr/bin/env python3
"""Run the full evaluative battery + random-axis null control + anchor
perturbation on a larger open embedding model via sentence-transformers.

This attacks the n=1-frontier confound: if a genuinely more capable open
model clears the noise floor, the result is not Gemini-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES, CATEGORY_AXIS_MAP
from run_anchor_perturbation import REPHRASED_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

MODELS_TO_TRY = [
    "dunzhang/stella_en_400M_v5",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_st_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    print(f"Loading {model_name} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, dim={model.get_embedding_dimension()}")

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    return embed, model_name, model.get_embedding_dimension()


def compute_axis(embed, positive: list[str], negative: list[str]) -> np.ndarray:
    pos_embs = embed(positive)
    neg_embs = embed(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def score_battery(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        s_better = np.dot(better_embs[i], axis)
        s_worse = np.dot(worse_embs[i], axis)
        if s_better > s_worse:
            correct += 1
        elif s_better == s_worse:
            correct += 0.5
    return correct / len(better_embs)


def run_for_model(model_name: str, cases: list, output_dir: Path):
    embed, name, dim = get_st_embedder(model_name)

    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]

    print(f"\nEmbedding {len(cases)} pairs ...")
    t0 = time.time()
    better_embs = embed(better_texts)
    worse_embs = embed(worse_texts)
    print(f"  Done in {time.time()-t0:.1f}s, dim={dim}")

    # === 1. Targeted axis accuracy ===
    print("\n" + "=" * 70)
    print("TARGETED AXIS ACCURACY")
    print("=" * 70)
    targeted_accs = {}
    axis_vectors = {}
    for axis_name, anchors in AXES.items():
        axis_vec = compute_axis(embed, anchors["positive"], anchors["negative"])
        axis_vectors[axis_name] = axis_vec
        acc = score_battery(better_embs, worse_embs, axis_vec)
        targeted_accs[axis_name] = acc

    for name_ax, acc in sorted(targeted_accs.items(), key=lambda x: -x[1]):
        print(f"  {name_ax:<25s} {acc:.1%}")

    # === 2. Random-axis null control (200 axes) ===
    print("\n" + "=" * 70)
    print("RANDOM AXIS NULL CONTROL (200 axes)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    random_accs = []
    for _ in range(200):
        axis = rng.standard_normal(dim)
        axis = axis / np.linalg.norm(axis)
        random_accs.append(score_battery(better_embs, worse_embs, axis))
    random_accs = np.array(random_accs)

    print(f"  Mean:   {random_accs.mean():.1%}")
    print(f"  Std:    {random_accs.std():.1%}")
    print(f"  Min:    {random_accs.min():.1%}")
    print(f"  Max:    {random_accs.max():.1%}")

    print("\n  Targeted vs random comparison:")
    for name_ax, acc in sorted(targeted_accs.items(), key=lambda x: -x[1]):
        n_beat = (random_accs >= acc).sum()
        percentile = (random_accs < acc).mean() * 100
        print(f"    {name_ax:<25s} {acc:.1%}  |  {n_beat}/200 random >= this  |  "
              f"percentile: {percentile:.0f}th")

    # === 3. Anchor perturbation ===
    print("\n" + "=" * 70)
    print("ANCHOR PERTURBATION")
    print("=" * 70)
    perturbation_results = {}
    print(f"  {'Axis':<25s} {'Orig':>6s} {'Reph':>6s} {'Delta':>7s} {'Cosine':>8s}")
    print("  " + "-" * 55)
    for axis_name in AXES:
        if axis_name not in REPHRASED_AXES:
            continue
        orig_axis = axis_vectors[axis_name]
        reph_axis = compute_axis(
            embed, REPHRASED_AXES[axis_name]["positive"],
            REPHRASED_AXES[axis_name]["negative"]
        )
        cosine = float(np.dot(orig_axis, reph_axis))
        orig_acc = targeted_accs[axis_name]
        reph_acc = score_battery(better_embs, worse_embs, reph_axis)
        delta = reph_acc - orig_acc
        print(f"  {axis_name:<25s} {orig_acc:>5.1%} {reph_acc:>5.1%} {delta:>+6.1%} {cosine:>8.4f}")
        perturbation_results[axis_name] = {
            "original_accuracy": orig_acc,
            "rephrased_accuracy": reph_acc,
            "delta": delta,
            "cosine_similarity": cosine,
        }

    cosines = [v["cosine_similarity"] for v in perturbation_results.values()]
    print(f"\n  Mean cosine: {np.mean(cosines):.4f}")

    # === 4. Cross-category transfer ===
    print("\n" + "=" * 70)
    print("CROSS-CATEGORY TRANSFER")
    print("=" * 70)
    categories: dict[str, list[int]] = {}
    for i, c in enumerate(cases):
        categories.setdefault(c["category"], []).append(i)

    axis_names = list(AXES.keys())
    cat_names = sorted(categories.keys())

    def score_subset(indices, axis):
        correct = 0
        for i in indices:
            s_b = np.dot(better_embs[i], axis)
            s_w = np.dot(worse_embs[i], axis)
            if s_b > s_w:
                correct += 1
            elif s_b == s_w:
                correct += 0.5
        return correct / len(indices) if indices else 0.0

    transfer_matrix = {}
    for cat in cat_names:
        indices = categories[cat]
        row = {}
        for an in axis_names:
            row[an] = score_subset(indices, axis_vectors[an])
        transfer_matrix[cat] = row

    on_diag = []
    off_diag = []
    for cat in cat_names:
        matched = CATEGORY_AXIS_MAP.get(cat, "general_evaluative")
        for an in axis_names:
            if an == matched:
                on_diag.append(transfer_matrix[cat][an])
            else:
                off_diag.append(transfer_matrix[cat][an])
    print(f"  Mean matched-axis accuracy: {np.mean(on_diag):.1%}")
    print(f"  Mean cross-axis accuracy:   {np.mean(off_diag):.1%}")

    # === 5. PCA ===
    axis_matrix = np.array([axis_vectors[an] for an in axis_names])
    centered = axis_matrix - axis_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = S**2 / (S**2).sum()
    pc1 = Vt[0]
    pc1_acc = score_battery(better_embs, worse_embs, pc1)
    print(f"  PC1 explains: {explained[0]:.1%}")
    print(f"  PC1 accuracy: {pc1_acc:.1%}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_name,
        "embedding_dim": dim,
        "n_cases": len(cases),
        "targeted_accs": targeted_accs,
        "random_axis_stats": {
            "mean": float(random_accs.mean()),
            "std": float(random_accs.std()),
            "min": float(random_accs.min()),
            "max": float(random_accs.max()),
        },
        "random_accs": [float(a) for a in random_accs],
        "perturbation": perturbation_results,
        "mean_perturbation_cosine": float(np.mean(cosines)),
        "transfer": {
            "on_diagonal_mean": float(np.mean(on_diag)),
            "off_diagonal_mean": float(np.mean(off_diag)),
        },
        "pca": {
            "pc1_explained": float(explained[0]),
            "pc1_accuracy": float(pc1_acc),
        },
    }
    out_file = output_dir / "large_open_model_results.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {out_file}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--model", default=None,
                        help="Override model name (default: try MODELS_TO_TRY list)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes" / "research_cycles" / "large_open_model")
    args = parser.parse_args()

    cases = read_jsonl(args.battery)
    print(f"Loaded {len(cases)} cases")

    models = [args.model] if args.model else MODELS_TO_TRY
    for model_name in models:
        try:
            run_for_model(model_name, cases, args.output)
        except Exception as e:
            print(f"\n  FAILED on {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
