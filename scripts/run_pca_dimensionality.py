#!/usr/bin/env python3
"""PCA analysis of quality difference vectors.

How many dimensions does quality actually occupy? If PC1 explains most
variance, quality is effectively 1D. If it takes 20 PCs, the centroid's
single direction is leaving most information on the table.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

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


def load_battery():
    return load_cases(BATTERY_50) + load_cases(WARMTH_20)


def load_expansion():
    cases = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        cases.extend(load_cases(f))
    return cases


def make_centroid(model, cases):
    better = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    worse = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    d = better.mean(axis=0) - worse.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, direction):
    correct = 0
    for i in range(len(better_embs)):
        if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction):
            correct += 1
    return correct / len(better_embs)


def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_battery()
    expansion = load_expansion()
    all_cases = battery + expansion
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}, Total: {len(all_cases)}")

    # Also load firmness and warmth separately
    firmness_cases = load_cases(BATTERY_50)
    warmth_cases = load_cases(WARMTH_20)

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Embed all cases
        all_better = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        all_worse = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )

        # Difference vectors
        diffs = all_better - all_worse  # shape: (131, dims)
        print(f"\nDifference vectors shape: {diffs.shape}")

        # Quality direction (centroid = mean of diffs)
        centroid_dir = diffs.mean(axis=0)
        centroid_dir = centroid_dir / (norm(centroid_dir) + 1e-12)

        # Also compute firmness and warmth directions
        firm_better = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in firmness_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        firm_worse = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in firmness_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        firm_dir = (firm_better - firm_worse).mean(axis=0)
        firm_dir = firm_dir / (norm(firm_dir) + 1e-12)

        warm_better = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in warmth_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        warm_worse = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in warmth_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        warm_dir = (warm_better - warm_worse).mean(axis=0)
        warm_dir = warm_dir / (norm(warm_dir) + 1e-12)

        # PCA
        n_components = min(50, diffs.shape[0] - 1, diffs.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(diffs)

        var_explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(var_explained)

        print(f"\nVariance explained by top components:")
        for i in range(min(20, len(var_explained))):
            print(f"  PC{i+1}: {var_explained[i]:.4f} (cumulative: {cumulative[i]:.4f})")

        # How many PCs for 50%, 80%, 90%?
        for thresh in [0.50, 0.80, 0.90]:
            n_needed = int(np.searchsorted(cumulative, thresh)) + 1
            print(f"  Components for {thresh:.0%} variance: {n_needed}")

        # Cosine between PCs and known directions
        print(f"\nCosines between PCs and known directions:")
        pcs = pca.components_  # shape: (n_components, dims)
        for i in range(min(10, len(pcs))):
            pc = pcs[i]
            cos_centroid = float(np.dot(pc, centroid_dir))
            cos_firm = float(np.dot(pc, firm_dir))
            cos_warm = float(np.dot(pc, warm_dir))
            print(f"  PC{i+1}: centroid {cos_centroid:+.3f}  firmness {cos_firm:+.3f}  warmth {cos_warm:+.3f}")

        # Accuracy using top-N PCs as classification
        print(f"\nAccuracy using top-N PCs (OOS on expansion):")
        bat_better = all_better[:len(battery)]
        bat_worse = all_worse[:len(battery)]
        exp_better = all_better[len(battery):]
        exp_worse = all_worse[len(battery):]

        for n_pcs in [1, 2, 3, 5, 10, 20]:
            if n_pcs > len(pcs):
                break
            # Use top-N PCs: project, sum projections weighted by explained variance
            combined_dir = np.zeros(diffs.shape[1])
            for j in range(n_pcs):
                combined_dir += var_explained[j] * pcs[j]
            combined_dir = combined_dir / (norm(combined_dir) + 1e-12)
            acc = pairwise_accuracy(exp_better, exp_worse, combined_dir)
            print(f"  Top-{n_pcs:2d} PCs: {acc:.1%}")

        # Plain centroid for comparison
        centroid_acc = pairwise_accuracy(exp_better, exp_worse, centroid_dir)
        print(f"  Centroid:   {centroid_acc:.1%}")

        results[model_name] = {
            "n_cases": len(all_cases),
            "n_dims": diffs.shape[1],
            "variance_explained": [round(float(v), 6) for v in var_explained[:20]],
            "cumulative_variance": [round(float(v), 6) for v in cumulative[:20]],
            "pcs_for_50pct": int(np.searchsorted(cumulative, 0.50)) + 1,
            "pcs_for_80pct": int(np.searchsorted(cumulative, 0.80)) + 1,
            "pcs_for_90pct": int(np.searchsorted(cumulative, 0.90)) + 1,
            "pc_cosines_centroid": [round(float(np.dot(pcs[i], centroid_dir)), 4) for i in range(min(10, len(pcs)))],
            "pc_cosines_firmness": [round(float(np.dot(pcs[i], firm_dir)), 4) for i in range(min(10, len(pcs)))],
            "pc_cosines_warmth": [round(float(np.dot(pcs[i], warm_dir)), 4) for i in range(min(10, len(pcs)))],
            "centroid_expansion_accuracy": round(centroid_acc, 4),
        }

        del model
        gc.collect()

    out_path = OUT_DIR / "pca_dimensionality_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
