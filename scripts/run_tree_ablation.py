#!/usr/bin/env python3
"""Ablation: remove each term from the 5-term tree one at a time.

If the tree is principled, each term should counter a specific failure mode:
- Remove "honest" -> sycophancy cases should break (honest counters sycophancy)
- Remove "careful" -> rigor/firmness cases should break
- Remove "restrained" -> excess/sycophancy cases should break
- Remove "helpful" -> warmth/utility cases should break
- Remove "thorough" -> completeness cases should break

If it's arbitrary, removal of any term degrades uniformly.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
EXP_DIR = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TREE = {
    "careful":    ("Careful", "Reckless"),
    "honest":     ("Honest", "Dishonest"),
    "helpful":    ("Helpful", "Unhelpful"),
    "thorough":   ("Thorough", "Superficial"),
    "restrained": ("Restrained", "Unrestrained"),
}


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - half), min(1, center + half)


def main():
    from sentence_transformers import SentenceTransformer

    # Load all cases
    battery = read_jsonl(BATTERY)
    warmth_cases = read_jsonl(WARMTH)
    in_sample = battery + warmth_cases
    n_in = len(in_sample)

    labels_in = []
    for c in battery:
        if c["category"] == "anti_sycophancy":
            labels_in.append("sycophancy")
        else:
            labels_in.append("firmness")
    for c in warmth_cases:
        labels_in.append("warmth")

    # OOS cases
    oos_cases = []
    for f in sorted(EXP_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                c["source_file"] = f.stem
                oos_cases.append(c)
    n_oos = len(oos_cases)

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed in-sample
        better_in = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in in_sample])
        worse_in = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in in_sample])

        # Embed OOS
        better_oos = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in oos_cases])
        worse_oos = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in oos_cases])

        # Compute all axis directions
        axes = {}
        for name, (pos, neg) in TREE.items():
            p = embed([pos])[0]
            n_ = embed([neg])[0]
            axes[name] = (p - n_) / (norm(p - n_) + 1e-12)

        # Per-axis outcomes (in-sample)
        axis_outcomes_in = {}
        for name, d in axes.items():
            axis_outcomes_in[name] = [
                float(np.dot(better_in[i], d)) > float(np.dot(worse_in[i], d))
                for i in range(n_in)
            ]

        # Per-axis outcomes (OOS)
        axis_outcomes_oos = {}
        for name, d in axes.items():
            axis_outcomes_oos[name] = [
                float(np.dot(better_oos[i], d)) > float(np.dot(worse_oos[i], d))
                for i in range(n_oos)
            ]

        # Full tree ANY
        full_any_in = [any(axis_outcomes_in[a][i] for a in axes) for i in range(n_in)]
        full_any_oos = [any(axis_outcomes_oos[a][i] for a in axes) for i in range(n_oos)]

        # Ablation: remove each term
        print(f"\n  IN-SAMPLE ABLATION (n={n_in})")
        print(f"  {'Removed':12s} {'All':>5s} {'Firm':>6s} {'Warm':>6s} {'Syc':>6s}  What broke?")
        print(f"  {'-'*65}")

        # Full tree baseline
        firm_idx = [i for i in range(n_in) if labels_in[i] == "firmness"]
        warm_idx = [i for i in range(n_in) if labels_in[i] == "warmth"]
        syc_idx = [i for i in range(n_in) if labels_in[i] == "sycophancy"]

        full_acc = sum(full_any_in) / n_in
        full_firm = sum(full_any_in[i] for i in firm_idx) / len(firm_idx)
        full_warm = sum(full_any_in[i] for i in warm_idx) / len(warm_idx)
        full_syc = sum(full_any_in[i] for i in syc_idx) / len(syc_idx)
        print(f"  {'(none)':12s} {full_acc:5.0%} {full_firm:6.0%} {full_warm:6.0%} {full_syc:6.0%}  BASELINE")

        for removed in axes:
            remaining = [a for a in axes if a != removed]
            ablated_in = [any(axis_outcomes_in[a][i] for a in remaining) for i in range(n_in)]
            abl_acc = sum(ablated_in) / n_in
            abl_firm = sum(ablated_in[i] for i in firm_idx) / len(firm_idx)
            abl_warm = sum(ablated_in[i] for i in warm_idx) / len(warm_idx)
            abl_syc = sum(ablated_in[i] for i in syc_idx) / len(syc_idx)

            # What specifically broke?
            broke_cases = [i for i in range(n_in) if full_any_in[i] and not ablated_in[i]]
            broke_firm = sum(1 for i in broke_cases if labels_in[i] == "firmness")
            broke_warm = sum(1 for i in broke_cases if labels_in[i] == "warmth")
            broke_syc = sum(1 for i in broke_cases if labels_in[i] == "sycophancy")

            delta = abl_acc - full_acc
            damage = f"lost {len(broke_cases)}: {broke_firm}f/{broke_warm}w/{broke_syc}s"
            print(f"  {removed:12s} {abl_acc:5.0%} {abl_firm:6.0%} {abl_warm:6.0%} {abl_syc:6.0%}  {damage}")

        # OOS ablation
        print(f"\n  OUT-OF-SAMPLE ABLATION (n={n_oos})")
        print(f"  {'Removed':12s} {'All':>5s}  Damage")
        print(f"  {'-'*40}")

        full_oos_acc = sum(full_any_oos) / n_oos
        print(f"  {'(none)':12s} {full_oos_acc:5.0%}  BASELINE")

        for removed in axes:
            remaining = [a for a in axes if a != removed]
            ablated_oos = [any(axis_outcomes_oos[a][i] for a in remaining) for i in range(n_oos)]
            abl_oos_acc = sum(ablated_oos) / n_oos
            broke = sum(1 for i in range(n_oos) if full_any_oos[i] and not ablated_oos[i])

            # Per-category damage
            cats = {}
            for i in range(n_oos):
                if full_any_oos[i] and not ablated_oos[i]:
                    cat = oos_cases[i]["source_file"]
                    cats[cat] = cats.get(cat, 0) + 1
            cat_str = ", ".join(f"{c}:{n}" for c, n in sorted(cats.items())) if cats else "none"
            print(f"  {removed:12s} {abl_oos_acc:5.0%}  lost {broke}: {cat_str}")

        # ALSO: what does each term UNIQUELY catch?
        print(f"\n  UNIQUE CONTRIBUTIONS (cases where ONLY this term is correct)")
        print(f"  {'Term':12s} {'Unique':>6s}  {'Types':30s}")
        for term in axes:
            others = [a for a in axes if a != term]
            unique_in = []
            for i in range(n_in):
                if axis_outcomes_in[term][i] and not any(axis_outcomes_in[a][i] for a in others):
                    unique_in.append(i)
            types = {}
            for i in unique_in:
                types[labels_in[i]] = types.get(labels_in[i], 0) + 1
            type_str = ", ".join(f"{t}:{n}" for t, n in sorted(types.items())) if types else "none"
            print(f"  {term:12s} {len(unique_in):6d}  {type_str}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
