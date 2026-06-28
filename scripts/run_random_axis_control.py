#!/usr/bin/env python3
"""Random-axis null control: score the battery with random directions
in embedding space to verify the signal is axis-specific.

If random axes score ~50% while targeted axes score well above chance,
the signal lives specifically in the evaluative direction, not in any
arbitrary projection.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--backend", default="fastembed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes" / "research_cycles" / "random_axis_control")
    args = parser.parse_args()

    cases = read_jsonl(args.battery)
    print(f"Loaded {len(cases)} cases")

    embed, model_name = get_embedder(args.backend, args.model)

    # Embed all better/worse texts
    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]

    print("Embedding texts...")
    better_embs = embed(better_texts)
    worse_embs = embed(worse_texts)
    dim = better_embs.shape[1]
    print(f"Embedding dim: {dim}")

    # Also compute the real targeted axes
    from run_cycle001_intervention import AXES

    targeted_accs = {}
    for axis_name, anchors in AXES.items():
        pos_embs = embed(anchors["positive"])
        neg_embs = embed(anchors["negative"])
        axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        correct = 0
        for i in range(len(cases)):
            score_better = np.dot(better_embs[i], axis)
            score_worse = np.dot(worse_embs[i], axis)
            if score_better > score_worse:
                correct += 1
            elif score_better == score_worse:
                correct += 0.5
        targeted_accs[axis_name] = correct / len(cases)

    print("\n--- Targeted axes ---")
    for name, acc in sorted(targeted_accs.items(), key=lambda x: -x[1]):
        print(f"  {name:<25s} {acc:.1%}")

    # Random axes
    rng = np.random.default_rng(args.seed)
    random_accs = []

    print(f"\nScoring {args.n_random} random axes...")
    for trial in range(args.n_random):
        axis = rng.standard_normal(dim)
        axis = axis / np.linalg.norm(axis)

        correct = 0
        for i in range(len(cases)):
            score_better = np.dot(better_embs[i], axis)
            score_worse = np.dot(worse_embs[i], axis)
            if score_better > score_worse:
                correct += 1
            elif score_better == score_worse:
                correct += 0.5
        random_accs.append(correct / len(cases))

    random_accs = np.array(random_accs)

    print("\n--- Random axis distribution ---")
    print(f"  Mean:   {random_accs.mean():.1%}")
    print(f"  Std:    {random_accs.std():.1%}")
    print(f"  Min:    {random_accs.min():.1%}")
    print(f"  Max:    {random_accs.max():.1%}")
    print(f"  Median: {np.median(random_accs):.1%}")

    # How many random axes beat the best targeted axis?
    best_targeted = max(targeted_accs.values())
    best_targeted_name = max(targeted_accs, key=targeted_accs.get)
    n_beat = (random_accs >= best_targeted).sum()
    print(f"\n  Random axes >= best targeted ({best_targeted_name} {best_targeted:.1%}): "
          f"{n_beat}/{args.n_random}")

    # How many random axes beat combined targeted?
    combined_acc = targeted_accs.get("general_evaluative", 0)
    for name, acc in targeted_accs.items():
        if name != "general_evaluative" and acc > combined_acc:
            pass  # just checking
    # Actually use "combined" score
    # For simplicity, report how many random axes beat each targeted axis
    print("\n--- Targeted vs random comparison ---")
    for name, acc in sorted(targeted_accs.items(), key=lambda x: -x[1]):
        n_beat = (random_accs >= acc).sum()
        percentile = (random_accs < acc).mean() * 100
        print(f"  {name:<25s} {acc:.1%}  |  {n_beat}/{args.n_random} random axes >= this  |  "
              f"percentile: {percentile:.0f}th")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_name,
        "n_cases": len(cases),
        "n_random_axes": args.n_random,
        "seed": args.seed,
        "embedding_dim": int(dim),
        "targeted_accs": targeted_accs,
        "random_axis_stats": {
            "mean": float(random_accs.mean()),
            "std": float(random_accs.std()),
            "min": float(random_accs.min()),
            "max": float(random_accs.max()),
            "median": float(np.median(random_accs)),
        },
        "random_accs": [float(a) for a in random_accs],
    }
    (args.output / "random_axis_control.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
