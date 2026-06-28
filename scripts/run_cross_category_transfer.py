#!/usr/bin/env python3
"""Cross-category transfer: score each battery category with every axis
to test whether evaluative signal generalizes across domains.

If the evaluative direction captures a universal quality signal, axes
designed for one domain (e.g., harm_reduction) should show above-chance
accuracy on unrelated categories (e.g., reasoning_rigor). If the signal
is category-specific, only the matched axis should work.

Also computes PCA of axis geometry to test whether the 5 targeted axes
share a dominant evaluative direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES, CATEGORY_AXIS_MAP

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_embedder(backend: str, model: str | None):
    if backend == "fastembed":
        from fastembed import TextEmbedding
        model_name = model or "snowflake/snowflake-arctic-embed-m"
        embedder = TextEmbedding(model_name=model_name)
        def embed(texts: list[str]) -> np.ndarray:
            return np.array(list(embedder.embed(texts)))
        return embed, model_name
    raise ValueError(f"Unknown backend: {backend}")


def compute_axis(embed, positive: list[str], negative: list[str]) -> np.ndarray:
    pos_embs = embed(positive)
    neg_embs = embed(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def score_subset(better_embs, worse_embs, indices, axis):
    correct = 0
    for i in indices:
        s_better = np.dot(better_embs[i], axis)
        s_worse = np.dot(worse_embs[i], axis)
        if s_better > s_worse:
            correct += 1
        elif s_better == s_worse:
            correct += 0.5
    return correct / len(indices) if indices else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--backend", default="fastembed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes" / "research_cycles" / "cross_category_transfer")
    args = parser.parse_args()

    cases = read_jsonl(args.battery)
    print(f"Loaded {len(cases)} cases")

    embed, model_name = get_embedder(args.backend, args.model)

    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]

    print("Embedding response texts...")
    better_embs = embed(better_texts)
    worse_embs = embed(worse_texts)
    dim = better_embs.shape[1]
    print(f"Embedding dim: {dim}")

    # Group cases by category
    categories: dict[str, list[int]] = {}
    for i, c in enumerate(cases):
        cat = c["category"]
        categories.setdefault(cat, []).append(i)

    # Compute all axis vectors
    axis_vectors = {}
    for axis_name, anchors in AXES.items():
        axis_vectors[axis_name] = compute_axis(embed, anchors["positive"], anchors["negative"])

    axis_names = list(AXES.keys())
    cat_names = sorted(categories.keys())

    # === Cross-category transfer matrix ===
    print("\n" + "=" * 80)
    print("CROSS-CATEGORY TRANSFER MATRIX")
    print("=" * 80)
    header = f"{'Category':<22s} {'N':>3s}"
    for an in axis_names:
        header += f" {an[:12]:>12s}"
    print(header)
    print("-" * len(header))

    transfer_matrix = {}
    for cat in cat_names:
        indices = categories[cat]
        row = {}
        matched_axis = CATEGORY_AXIS_MAP.get(cat, "general_evaluative")
        line = f"{cat:<22s} {len(indices):>3d}"
        for an in axis_names:
            acc = score_subset(better_embs, worse_embs, indices, axis_vectors[an])
            row[an] = acc
            marker = "*" if an == matched_axis else " "
            line += f" {acc:>11.1%}{marker}"
        print(line)
        transfer_matrix[cat] = row

    # Compute mean off-diagonal transfer
    print("\n--- Transfer summary ---")
    on_diag_accs = []
    off_diag_accs = []
    for cat in cat_names:
        matched = CATEGORY_AXIS_MAP.get(cat, "general_evaluative")
        for an in axis_names:
            acc = transfer_matrix[cat][an]
            if an == matched:
                on_diag_accs.append(acc)
            else:
                off_diag_accs.append(acc)
    print(f"  Mean matched-axis accuracy:    {np.mean(on_diag_accs):.1%}")
    print(f"  Mean cross-axis accuracy:      {np.mean(off_diag_accs):.1%}")
    print(f"  Cross-axis > 50% count:        {sum(1 for a in off_diag_accs if a > 0.5)}/{len(off_diag_accs)}")

    # === PCA of axis geometry ===
    print("\n" + "=" * 80)
    print("PCA OF AXIS GEOMETRY")
    print("=" * 80)

    axis_matrix = np.array([axis_vectors[an] for an in axis_names])
    # Cosine similarity matrix
    cos_matrix = axis_matrix @ axis_matrix.T
    print("\nCosine similarity between axes:")
    header = f"{'':>22s}"
    for an in axis_names:
        header += f" {an[:12]:>12s}"
    print(header)
    for i, an_i in enumerate(axis_names):
        line = f"{an_i:<22s}"
        for j in range(len(axis_names)):
            line += f" {cos_matrix[i, j]:>12.3f}"
        print(line)

    # PCA via SVD
    centered = axis_matrix - axis_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = S**2 / (S**2).sum()
    cumulative = np.cumsum(explained)

    print(f"\nPCA explained variance:")
    for i in range(len(explained)):
        print(f"  PC{i+1}: {explained[i]:.1%} (cumulative: {cumulative[i]:.1%})")

    # Project axes onto PC1
    pc1 = Vt[0]
    print(f"\nAxis projections onto PC1 (the dominant evaluative direction):")
    for i, an in enumerate(axis_names):
        proj = np.dot(axis_vectors[an], pc1)
        print(f"  {an:<22s} {proj:>+.4f}")

    # Score the full battery with PC1
    pc1_acc = score_subset(better_embs, worse_embs, list(range(len(cases))), pc1)
    print(f"\nPC1 accuracy on full battery: {pc1_acc:.1%}")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_name,
        "embedding_dim": int(dim),
        "n_cases": len(cases),
        "categories": {cat: len(idx) for cat, idx in categories.items()},
        "transfer_matrix": transfer_matrix,
        "on_diagonal_mean": float(np.mean(on_diag_accs)),
        "off_diagonal_mean": float(np.mean(off_diag_accs)),
        "cosine_similarity": {
            axis_names[i]: {axis_names[j]: float(cos_matrix[i, j])
                            for j in range(len(axis_names))}
            for i in range(len(axis_names))
        },
        "pca_explained_variance": [float(e) for e in explained],
        "pc1_accuracy": float(pc1_acc),
        "pc1_axis_projections": {an: float(np.dot(axis_vectors[an], pc1))
                                  for an in axis_names},
    }
    (args.output / "cross_category_transfer.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
