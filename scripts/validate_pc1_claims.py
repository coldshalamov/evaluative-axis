#!/usr/bin/env python3
"""Validate PC1 claims with three checks:
1. Mean-axis scoring (the actual shared evaluative direction, not centered PC1)
2. PC1 sign orientation (principled via dot with mean_axis)
3. Matched null: PC1 of 5 random axes vs PC1 of 5 targeted axes
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

MODELS = [
    ("BAAI/bge-small-en-v1.5", 384),
    ("snowflake/snowflake-arctic-embed-m", None),
    ("BAAI/bge-large-en-v1.5", 1024),
    ("nomic-ai/nomic-embed-text-v1.5", 768),
]


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def score_battery(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        s_b = np.dot(better_embs[i], axis)
        s_w = np.dot(worse_embs[i], axis)
        if s_b > s_w:
            correct += 1
        elif s_b == s_w:
            correct += 0.5
    return correct / len(better_embs)


def compute_axis(embed, positive, negative):
    pos_embs = embed(positive)
    neg_embs = embed(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def run_checks(model_name, cases, use_fastembed=False):
    if use_fastembed:
        from fastembed import TextEmbedding
        embedder = TextEmbedding(model_name=model_name)
        def embed(texts):
            return np.array(list(embedder.embed(texts)))
        dim_label = "fastembed"
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        dim_label = "st"

    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]

    print(f"\nEmbedding texts for {model_name} ({dim_label})...")
    t0 = time.time()
    better_embs = embed(better_texts)
    worse_embs = embed(worse_texts)
    dim = better_embs.shape[1]
    print(f"  dim={dim}, took {time.time()-t0:.1f}s")

    # Compute targeted axis vectors
    axis_vectors = {}
    for axis_name, anchors in AXES.items():
        axis_vectors[axis_name] = compute_axis(embed, anchors["positive"], anchors["negative"])

    axis_names = list(AXES.keys())
    axis_matrix = np.array([axis_vectors[an] for an in axis_names])

    # === CHECK 1: Mean axis (actual shared direction) vs centered PC1 ===
    print("\n  CHECK 1: Mean axis vs centered PC1")
    mean_axis = axis_matrix.mean(axis=0)
    mean_axis = mean_axis / (np.linalg.norm(mean_axis) + 1e-12)
    mean_acc = score_battery(better_embs, worse_embs, mean_axis)
    print(f"    Mean axis accuracy: {mean_acc:.1%}")

    # Centered PC1 (what the previous code did)
    centered = axis_matrix - axis_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    centered_pc1 = Vt[0]
    # Check sign
    raw_centered_acc = score_battery(better_embs, worse_embs, centered_pc1)
    print(f"    Centered PC1 accuracy (raw sign): {raw_centered_acc:.1%}")
    # Principled sign orientation
    if np.dot(centered_pc1, mean_axis) < 0:
        centered_pc1 = -centered_pc1
    oriented_centered_acc = score_battery(better_embs, worse_embs, centered_pc1)
    print(f"    Centered PC1 accuracy (oriented sign): {oriented_centered_acc:.1%}")

    # Uncentered PC1
    U2, S2, Vt2 = np.linalg.svd(axis_matrix, full_matrices=False)
    uncentered_pc1 = Vt2[0]
    if np.dot(uncentered_pc1, mean_axis) < 0:
        uncentered_pc1 = -uncentered_pc1
    uncentered_acc = score_battery(better_embs, worse_embs, uncentered_pc1)
    print(f"    Uncentered PC1 accuracy (oriented): {uncentered_acc:.1%}")

    # Individual axis scores for comparison
    best_axis_acc = 0
    for an in axis_names:
        acc = score_battery(better_embs, worse_embs, axis_vectors[an])
        best_axis_acc = max(best_axis_acc, acc)
    print(f"    Best individual axis: {best_axis_acc:.1%}")

    # === CHECK 2: Sign matters? ===
    print("\n  CHECK 2: Does sign flip invert the score?")
    flipped_acc = score_battery(better_embs, worse_embs, -mean_axis)
    print(f"    Mean axis: {mean_acc:.1%}, Flipped: {flipped_acc:.1%}, Sum: {mean_acc + flipped_acc:.1%}")

    # === CHECK 3: Matched null (PC1 of 5 random axes, 200 trials) ===
    print("\n  CHECK 3: Matched null (PC1 of 5 random axes, 200 trials)")
    rng = np.random.default_rng(42)
    null_mean_accs = []
    null_pc1_accs = []
    for _ in range(200):
        random_axes = np.array([rng.standard_normal(dim) for _ in range(5)])
        random_axes = random_axes / np.linalg.norm(random_axes, axis=1, keepdims=True)
        # Mean of random axes
        rm = random_axes.mean(axis=0)
        rm = rm / (np.linalg.norm(rm) + 1e-12)
        rm_acc = score_battery(better_embs, worse_embs, rm)
        if rm_acc < 0.5:
            rm_acc = 1.0 - rm_acc  # take best sign
        null_mean_accs.append(rm_acc)
        # Centered PC1 of random axes
        rc = random_axes - random_axes.mean(axis=0)
        _, _, Vt_r = np.linalg.svd(rc, full_matrices=False)
        rpc1 = Vt_r[0]
        rpc1_acc = score_battery(better_embs, worse_embs, rpc1)
        if rpc1_acc < 0.5:
            rpc1_acc = 1.0 - rpc1_acc
        null_pc1_accs.append(rpc1_acc)

    null_mean_accs = np.array(null_mean_accs)
    null_pc1_accs = np.array(null_pc1_accs)

    # Compare targeted mean/PC1 against null
    # For fair comparison, take best sign of targeted too
    targeted_mean_best = max(mean_acc, 1.0 - mean_acc)
    targeted_pc1_best = max(oriented_centered_acc, 1.0 - oriented_centered_acc)

    n_beat_mean = (null_mean_accs >= targeted_mean_best).sum()
    n_beat_pc1 = (null_pc1_accs >= targeted_pc1_best).sum()

    print(f"    Targeted mean-axis (best sign): {targeted_mean_best:.1%}")
    print(f"    Null mean-axis: mean={null_mean_accs.mean():.1%}, max={null_mean_accs.max():.1%}")
    print(f"    Null mean-axis >= targeted: {n_beat_mean}/200")

    print(f"    Targeted centered-PC1 (best sign): {targeted_pc1_best:.1%}")
    print(f"    Null centered-PC1: mean={null_pc1_accs.mean():.1%}, max={null_pc1_accs.max():.1%}")
    print(f"    Null centered-PC1 >= targeted: {n_beat_pc1}/200")

    return {
        "model": model_name,
        "dim": dim,
        "mean_axis_acc": float(mean_acc),
        "centered_pc1_raw": float(raw_centered_acc),
        "centered_pc1_oriented": float(oriented_centered_acc),
        "uncentered_pc1_oriented": float(uncentered_acc),
        "best_individual_axis": float(best_axis_acc),
        "null_mean_axis_stats": {
            "mean": float(null_mean_accs.mean()),
            "max": float(null_mean_accs.max()),
            "n_beat_targeted": int(n_beat_mean),
        },
        "null_pc1_stats": {
            "mean": float(null_pc1_accs.mean()),
            "max": float(null_pc1_accs.max()),
            "n_beat_targeted": int(n_beat_pc1),
        },
    }


def main():
    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")

    results = {}
    for model_name, expected_dim in MODELS:
        try:
            # Try fastembed first for models it supports
            try:
                r = run_checks(model_name, cases, use_fastembed=True)
            except Exception:
                r = run_checks(model_name, cases, use_fastembed=False)
            results[model_name] = r
        except Exception as e:
            print(f"\n  FAILED: {model_name}: {e}")
            import traceback
            traceback.print_exc()

    out_dir = ROOT / "notes" / "research_cycles" / "pc1_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pc1_validation.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
