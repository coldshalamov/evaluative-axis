#!/usr/bin/env python3
"""Compute pairwise correlations between axis scores on the battery.

Shows whether different axes (ML-jargon vs single-word) measure the
same underlying signal or capture genuinely different dimensions.
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


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")

    all_axes = {}
    all_axes.update({f"ml/{k}": v for k, v in ML_AXES.items()})
    all_axes.update({f"sw/{k}": v for k, v in KEY_AXES.items()})

    axis_names = list(all_axes.keys())

    for model_name in [
        "snowflake/snowflake-arctic-embed-m",
        "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v1.5",
    ]:
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

        # Compute score deltas (better - worse) for each axis
        deltas = {}
        for axis_name, anchors in all_axes.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            d = []
            for i in range(len(cases)):
                s_b = float(np.dot(better_embs[i], axis_vec))
                s_w = float(np.dot(worse_embs[i], axis_vec))
                d.append(s_b - s_w)
            deltas[axis_name] = np.array(d)

        # Compute pairwise Pearson correlations
        n = len(axis_names)
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                corr_matrix[i, j] = np.corrcoef(deltas[axis_names[i]], deltas[axis_names[j]])[0, 1]

        # Print correlation matrix
        short = [a.replace("ml/", "").replace("sw/", "")[:12] for a in axis_names]
        header = f"{'':>13s}" + "".join(f"{s:>13s}" for s in short)
        print(f"\nPairwise Pearson r (score deltas):\n{header}")
        for i, name in enumerate(short):
            row = f"{name:>13s}"
            for j in range(n):
                r = corr_matrix[i, j]
                row += f"{r:13.2f}"
            print(row)

        # Axis direction cosine similarity (geometry, not scores)
        print(f"\nAxis direction cosine similarity:")
        axis_vecs = {}
        for axis_name, anchors in all_axes.items():
            axis_vecs[axis_name] = compute_axis(embed_fn, anchors["positive"], anchors["negative"])

        header = f"{'':>13s}" + "".join(f"{s:>13s}" for s in short)
        print(header)
        for i, (name_i, key_i) in enumerate(zip(short, axis_names)):
            row = f"{name_i:>13s}"
            for j, key_j in enumerate(axis_names):
                cos = float(np.dot(axis_vecs[key_i], axis_vecs[key_j]))
                row += f"{cos:13.2f}"
            print(row)

        del model


if __name__ == "__main__":
    main()
