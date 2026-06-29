#!/usr/bin/env python3
"""Test centroid on embedding models we haven't tried before.

Proves the method isn't specific to our original 3 models.
Tests: gte-base, e5-base-v2, mxbai-embed-large-v1.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

# Original models for comparison
ORIGINAL_MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# NEW models to test
NEW_MODELS = [
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "mixedbread-ai/mxbai-embed-large-v1",
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


def make_centroid(model, cases, prefix=""):
    better = model.encode(
        [f"{prefix}User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    worse = model.encode(
        [f"{prefix}User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    d = better.mean(axis=0) - worse.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(model, cases, direction, prefix=""):
    better = model.encode(
        [f"{prefix}User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    worse = model.encode(
        [f"{prefix}User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    correct = sum(1 for i in range(len(cases))
                  if np.dot(better[i], direction) > np.dot(worse[i], direction))
    return correct / len(cases)


def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_battery()
    expansion = load_expansion()
    orig50 = load_cases(BATTERY_50)
    warmth20 = load_cases(WARMTH_20)
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}")

    results = {}

    for model_name in NEW_MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        try:
            model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            results[model_name] = {"error": str(e)}
            continue

        dims = model.get_sentence_embedding_dimension()
        print(f"  Dimensions: {dims}")

        # e5 models need "query: " prefix for queries
        prefix = ""
        if "e5" in model_name.lower():
            prefix = "query: "
            print(f"  Using prefix: '{prefix}'")

        # Train centroid on battery
        direction = make_centroid(model, battery, prefix=prefix)

        # Test on battery splits
        orig_acc = pairwise_accuracy(model, orig50, direction, prefix=prefix)
        warm_acc = pairwise_accuracy(model, warmth20, direction, prefix=prefix)
        bat_acc = pairwise_accuracy(model, battery, direction, prefix=prefix)
        exp_acc = pairwise_accuracy(model, expansion, direction, prefix=prefix)

        print(f"\n  Battery (in-sample):")
        print(f"    Original 50: {orig_acc:.1%}")
        print(f"    Warmth 20:   {warm_acc:.1%}")
        print(f"    Combined 70: {bat_acc:.1%}")
        print(f"  Expansion (OOS):")
        print(f"    61 cases:    {exp_acc:.1%}")

        # Permutation test
        print(f"  Permutation test (200 perms)...")
        rng = np.random.RandomState(0)
        real_acc = exp_acc

        better_embs = model.encode(
            [f"{prefix}User: {c['prompt']}\nAssistant: {c['better']}" for c in battery],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        worse_embs = model.encode(
            [f"{prefix}User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        exp_better = model.encode(
            [f"{prefix}User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        exp_worse = model.encode(
            [f"{prefix}User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )

        count_ge = 0
        for _ in range(200):
            swap = rng.rand(len(battery)) < 0.5
            b = better_embs.copy()
            w = worse_embs.copy()
            for i in range(len(swap)):
                if swap[i]:
                    b[i], w[i] = w[i], b[i]
            perm_dir = b.mean(axis=0) - w.mean(axis=0)
            perm_dir = perm_dir / (norm(perm_dir) + 1e-12)
            perm_correct = sum(1 for i in range(len(expansion))
                               if np.dot(exp_better[i], perm_dir) > np.dot(exp_worse[i], perm_dir))
            if perm_correct / len(expansion) >= real_acc:
                count_ge += 1
        p_val = count_ge / 200
        print(f"    p-value: {p_val:.3f}")

        results[model_name] = {
            "dimensions": dims,
            "original_50": round(orig_acc, 4),
            "warmth_20": round(warm_acc, 4),
            "combined_70": round(bat_acc, 4),
            "expansion_61_oos": round(exp_acc, 4),
            "permutation_p": round(p_val, 4),
        }

        del model
        gc.collect()

    out_path = OUT_DIR / "additional_models_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON WITH ORIGINAL MODELS")
    print(f"{'='*70}")
    print(f"{'Model':35s} {'Battery':>10s} {'OOS':>10s}")
    print("-" * 60)
    # Reference values from existing results
    print(f"  {'snowflake-arctic-embed-m':35s} {'77-80%':>10s} {'77%':>10s}  (from prior)")
    print(f"  {'bge-m3':35s} {'75-80%':>10s} {'75%':>10s}  (from prior)")
    print(f"  {'nomic-embed-text-v1.5':35s} {'66-77%':>10s} {'66%':>10s}  (from prior)")
    for mname, mr in results.items():
        if "error" in mr:
            continue
        short = mname.split("/")[-1]
        print(f"  {short:35s} {mr['combined_70']:.0%}{'':>6s} {mr['expansion_61_oos']:.0%}")


if __name__ == "__main__":
    main()
