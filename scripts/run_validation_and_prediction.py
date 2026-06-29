#!/usr/bin/env python3
"""Validation experiments: out-of-sample testing and prediction.

1. k-fold cross-validation of axis accuracy on the 70-case battery
2. Out-of-sample test on the expansion battery (20 new cases)
3. Correlate anchor geometry with cospos advantage
4. Make PREDICTIONS for expansion battery BEFORE scoring
"""

import json, sys, gc, random
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
BATTERY_EXPANSION = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = {
    "careful":  {"pos": ["Careful"],  "neg": ["Reckless"]},
    "honest":   {"pos": ["Honest"],   "neg": ["Dishonest"]},
    "kind":     {"pos": ["Kind"],     "neg": ["Cruel"]},
    "thorough": {"pos": ["Thorough"], "neg": ["Superficial"]},
    "good":     {"pos": ["Good"],     "neg": ["Bad"]},
    "hard":     {"pos": ["Hard"],     "neg": ["Soft"]},
    "helpful":  {"pos": ["Helpful"],  "neg": ["Unhelpful"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    all_cases = original + warmth
    n_orig = len(original)
    n_total = len(all_cases)
    axis_names = list(AXES.keys())

    # Load expansion battery
    expansion_cases = []
    for cat_file in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(cat_file))
    print(f"Expansion battery: {len(expansion_cases)} cases")

    out_dir = ROOT / "notes/research_cycles/validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"per_model": {}}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        # Embed main battery
        all_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        all_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        # Embed expansion battery
        if expansion_cases:
            exp_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion_cases])
            exp_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion_cases])

        model_data = {}

        # Per-axis data
        for ai, (axis_name, anchors) in enumerate(AXES.items()):
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)
            cos_between = cosine(pos_emb, neg_emb)

            # Main battery: bipolar per-case
            bp_correct = []
            cp_correct = []
            for i in range(n_total):
                sb = float(np.dot(all_better[i], axis_unit))
                sw = float(np.dot(all_worse[i], axis_unit))
                bp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

                sb_c = cosine(all_better[i], pos_emb)
                sw_c = cosine(all_worse[i], pos_emb)
                cp_correct.append(1 if sb_c > sw_c else (0.5 if sb_c == sw_c else 0))

            bp_acc = np.mean(bp_correct)
            cp_acc = np.mean(cp_correct)
            cospos_advantage = cp_acc - bp_acc

            # Expansion battery (OUT OF SAMPLE)
            exp_bp_correct = []
            exp_cp_correct = []
            if expansion_cases:
                for i in range(len(expansion_cases)):
                    sb = float(np.dot(exp_better[i], axis_unit))
                    sw = float(np.dot(exp_worse[i], axis_unit))
                    exp_bp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

                    sb_c = cosine(exp_better[i], pos_emb)
                    sw_c = cosine(exp_worse[i], pos_emb)
                    exp_cp_correct.append(1 if sb_c > sw_c else (0.5 if sb_c == sw_c else 0))

            # K-fold cross validation (5-fold)
            n_folds = 5
            indices = list(range(n_total))
            random.seed(42)
            random.shuffle(indices)
            fold_size = n_total // n_folds
            fold_accs = []
            for fold in range(n_folds):
                test_idx = indices[fold*fold_size:(fold+1)*fold_size]
                train_idx = [i for i in indices if i not in test_idx]
                train_acc = np.mean([bp_correct[i] for i in train_idx])
                test_acc = np.mean([bp_correct[i] for i in test_idx])
                fold_accs.append(test_acc)

            model_data[axis_name] = {
                "main_battery": {
                    "bipolar_combined": round(bp_acc, 4),
                    "cospos_combined": round(cp_acc, 4),
                    "cospos_advantage": round(cospos_advantage, 4),
                    "bipolar_orig": round(np.mean(bp_correct[:n_orig]), 4),
                    "bipolar_warm": round(np.mean(bp_correct[n_orig:]), 4),
                },
                "expansion_oos": {
                    "bipolar": round(np.mean(exp_bp_correct), 4) if exp_bp_correct else None,
                    "cospos": round(np.mean(exp_cp_correct), 4) if exp_cp_correct else None,
                },
                "kfold_5": {
                    "mean": round(np.mean(fold_accs), 4),
                    "std": round(np.std(fold_accs), 4),
                    "folds": [round(f, 4) for f in fold_accs],
                },
                "anchor_geometry": {
                    "cos_between_anchors": round(cos_between, 4),
                },
            }

        # Print results table
        print(f"\n{'Axis':12s}  {'Main BP':>8s}  {'Main CP':>8s}  {'CP adv':>7s}  {'Exp BP':>7s}  {'Exp CP':>7s}  {'KF mean':>7s}±{'std':>5s}  {'cos(p,n)':>8s}")
        for a in axis_names:
            d = model_data[a]
            m = d["main_battery"]
            e = d["expansion_oos"]
            k = d["kfold_5"]
            g = d["anchor_geometry"]
            print(f"  {a:10s}  {m['bipolar_combined']:7.1%}   {m['cospos_combined']:7.1%}   {m['cospos_advantage']:+6.1%}   "
                  f"{e['bipolar']:6.1%}   {e['cospos']:6.1%}   {k['mean']:6.1%}±{k['std']:4.1%}   {g['cos_between_anchors']:.4f}")

        # Correlation: cos_between_anchors vs cospos_advantage
        cos_vals = [model_data[a]["anchor_geometry"]["cos_between_anchors"] for a in axis_names]
        adv_vals = [model_data[a]["main_battery"]["cospos_advantage"] for a in axis_names]
        if len(cos_vals) > 2:
            corr = np.corrcoef(cos_vals, adv_vals)[0, 1]
            print(f"\n  Correlation(cos_between_anchors, cospos_advantage): r = {corr:.3f}")
            model_data["geometry_correlation"] = round(float(corr), 4)

        # Prediction check: does main battery accuracy predict expansion accuracy?
        main_bp = [model_data[a]["main_battery"]["bipolar_combined"] for a in axis_names]
        exp_bp = [model_data[a]["expansion_oos"]["bipolar"] for a in axis_names if model_data[a]["expansion_oos"]["bipolar"] is not None]
        if len(exp_bp) == len(main_bp):
            pred_corr = np.corrcoef(main_bp, exp_bp)[0, 1]
            print(f"  Correlation(main_battery_acc, expansion_acc): r = {pred_corr:.3f}")
            model_data["prediction_correlation"] = round(float(pred_corr), 4)

        # Does "careful" survive out-of-sample?
        print(f"\n  === CAREFUL out-of-sample check ===")
        c = model_data["careful"]
        print(f"  Main battery: bipolar={c['main_battery']['bipolar_combined']:.1%}  "
              f"(orig={c['main_battery']['bipolar_orig']:.1%}, warm={c['main_battery']['bipolar_warm']:.1%})")
        print(f"  Expansion OOS: bipolar={c['expansion_oos']['bipolar']:.1%}")
        print(f"  K-fold mean:   {c['kfold_5']['mean']:.1%} ± {c['kfold_5']['std']:.1%}")

        results["per_model"][model_name] = model_data
        del model
        gc.collect()

    out_path = out_dir / "validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
