#!/usr/bin/env python3
"""Model-level geometry test: does mean cos(pos,neg) across axes predict
a model's mean evaluative performance?

The within-model prediction failed (r=-0.11 to +0.24). But across models,
the relationship might be stronger: models with more separated anchors
might benefit more from cospos, and models with tighter geometry might
have stronger evaluative structure overall.

Tests the 3 main models on the rebalanced 70-case battery.
For each model: compute mean cos(pos,neg) across 10 axes,
mean bipolar accuracy, mean cospos accuracy, mean cospos advantage.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = {
    "careful":  {"pos": ["Careful"],  "neg": ["Reckless"]},
    "thorough": {"pos": ["Thorough"], "neg": ["Superficial"]},
    "hard":     {"pos": ["Hard"],     "neg": ["Soft"]},
    "kind":     {"pos": ["Kind"],     "neg": ["Cruel"]},
    "honest":   {"pos": ["Honest"],   "neg": ["Dishonest"]},
    "helpful":  {"pos": ["Helpful"],  "neg": ["Unhelpful"]},
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "good":     {"pos": ["Good"],     "neg": ["Bad"]},
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
    n_total = len(all_cases)
    n_orig = len(original)

    model_summary = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        cos_vals = []
        bp_accs = []
        cp_accs = []

        for axis_name, anchors in AXES.items():
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            cos_pn = cosine(pos_emb, neg_emb)
            cos_vals.append(cos_pn)

            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

            bp_correct = []
            cp_correct = []
            for i in range(n_total):
                sb = float(np.dot(better_embs[i], axis_unit))
                sw = float(np.dot(worse_embs[i], axis_unit))
                bp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

                sb_c = cosine(better_embs[i], pos_emb)
                sw_c = cosine(worse_embs[i], pos_emb)
                cp_correct.append(1 if sb_c > sw_c else (0.5 if sb_c == sw_c else 0))

            bp_acc = np.mean(bp_correct)
            cp_acc = np.mean(cp_correct)
            bp_accs.append(bp_acc)
            cp_accs.append(cp_acc)

            print(f"  {axis_name:10s}  cos(p,n)={cos_pn:.3f}  bp={bp_acc:.1%}  cp={cp_acc:.1%}  cp_adv={cp_acc-bp_acc:+.1%}")

        mean_cos = np.mean(cos_vals)
        mean_bp = np.mean(bp_accs)
        mean_cp = np.mean(cp_accs)
        mean_cp_adv = mean_cp - mean_bp
        best_bp = max(bp_accs)
        best_cp = max(cp_accs)

        # Within-model correlation
        within_corr = np.corrcoef(cos_vals, [cp - bp for cp, bp in zip(cp_accs, bp_accs)])[0, 1]

        print(f"\n  Summary:")
        print(f"    Mean cos(pos,neg): {mean_cos:.4f}")
        print(f"    Mean bipolar acc:  {mean_bp:.1%}")
        print(f"    Mean cospos acc:   {mean_cp:.1%}")
        print(f"    Mean cospos adv:   {mean_cp_adv:+.1%}")
        print(f"    Best bipolar:      {best_bp:.1%}")
        print(f"    Best cospos:       {best_cp:.1%}")
        print(f"    Within-model corr(cos, cp_adv): {within_corr:.3f}")

        model_summary.append({
            "model": model_name,
            "mean_cos": mean_cos,
            "mean_bp": mean_bp,
            "mean_cp": mean_cp,
            "mean_cp_adv": mean_cp_adv,
            "best_bp": best_bp,
            "best_cp": best_cp,
            "within_corr": within_corr,
        })

        del model
        gc.collect()

    # Cross-model analysis
    print(f"\n{'='*60}")
    print("CROSS-MODEL ANALYSIS")
    print(f"{'='*60}")

    models_short = [m["model"].split("/")[-1] for m in model_summary]
    mean_cos_all = [m["mean_cos"] for m in model_summary]
    mean_bp_all = [m["mean_bp"] for m in model_summary]
    mean_cp_adv_all = [m["mean_cp_adv"] for m in model_summary]

    print(f"\n{'Model':25s}  {'mean cos':>10s}  {'mean bp':>8s}  {'mean cp_adv':>11s}")
    for i, s in enumerate(model_summary):
        print(f"  {models_short[i]:23s}  {s['mean_cos']:10.4f}  {s['mean_bp']:7.1%}  {s['mean_cp_adv']:+10.1%}")

    if len(model_summary) >= 3:
        r_cos_bp = np.corrcoef(mean_cos_all, mean_bp_all)[0, 1]
        r_cos_cpadv = np.corrcoef(mean_cos_all, mean_cp_adv_all)[0, 1]
        print(f"\n  Cross-model corr(mean_cos, mean_bp_acc): r = {r_cos_bp:.3f}")
        print(f"  Cross-model corr(mean_cos, mean_cp_adv): r = {r_cos_cpadv:.3f}")
    else:
        print("  (need >= 3 models for cross-model correlation)")

    # Save
    out_dir = ROOT / "notes/research_cycles/model_level_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_geometry_results.json", "w") as f:
        json.dump(model_summary, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
