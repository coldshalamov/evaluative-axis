#!/usr/bin/env python3
"""Deep dive into cosine-to-positive scoring and why it helps on BGE-M3.

Tests:
1. Why does cospos help on BGE-M3 but not Snowflake/Nomic?
2. Is the negative anchor embedding CLOSER to the response embeddings
   than the positive? (contamination hypothesis)
3. Optimal voting with per-model best scoring method
4. Weighted voting using confidence (margin magnitude)
5. Sweep axis combinations systematically to find the actual best
"""

import json, sys, gc, itertools
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

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
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
    "wise":     {"pos": ["Wise"],     "neg": ["Foolish"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "warm":     {"pos": ["Warm"],     "neg": ["Cold"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "responsible": {"pos": ["Responsible"], "neg": ["Irresponsible"]},
    "clear":    {"pos": ["Clear"],    "neg": ["Confusing"]},
    "gentle":   {"pos": ["Gentle"],   "neg": ["Harsh"]},
    "constructive": {"pos": ["Constructive"], "neg": ["Destructive"]},
    "diligent": {"pos": ["Diligent"], "neg": ["Lazy"]},
    "supportive": {"pos": ["Supportive"], "neg": ["Dismissive"]},
    "patient":  {"pos": ["Patient"],  "neg": ["Impatient"]},
    "empathetic": {"pos": ["Empathetic"], "neg": ["Indifferent"]},
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

    out_dir = ROOT / "notes/research_cycles/cospos_deep"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"per_model": {}}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        all_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        all_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        model_data = {}
        axis_names = list(AXES.keys())

        # Per-case scoring matrices: [n_axes, n_cases] margin values
        bipolar_margins = np.zeros((len(AXES), n_total))
        cospos_margins = np.zeros((len(AXES), n_total))
        bipolar_correct = np.zeros((len(AXES), n_total))
        cospos_correct = np.zeros((len(AXES), n_total))

        for ai, (axis_name, anchors) in enumerate(AXES.items()):
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)
            pos_unit = pos_emb / (norm(pos_emb) + 1e-12)

            # Measure distances between anchor embeddings and response embeddings
            all_responses = np.vstack([all_better, all_worse])
            mean_cos_to_pos = np.mean([cosine(r, pos_emb) for r in all_responses])
            mean_cos_to_neg = np.mean([cosine(r, neg_emb) for r in all_responses])
            cos_between_anchors = cosine(pos_emb, neg_emb)

            # Per-case scoring
            for i in range(n_total):
                # Bipolar
                sb = float(np.dot(all_better[i], axis_unit))
                sw = float(np.dot(all_worse[i], axis_unit))
                bipolar_margins[ai, i] = sb - sw
                bipolar_correct[ai, i] = 1 if sb > sw else (0.5 if sb == sw else 0)

                # Cosine to positive
                sb_cos = cosine(all_better[i], pos_emb)
                sw_cos = cosine(all_worse[i], pos_emb)
                cospos_margins[ai, i] = sb_cos - sw_cos
                cospos_correct[ai, i] = 1 if sb_cos > sw_cos else (0.5 if sb_cos == sw_cos else 0)

            bp_orig = np.mean(bipolar_correct[ai, :n_orig])
            bp_warm = np.mean(bipolar_correct[ai, n_orig:])
            bp_comb = np.mean(bipolar_correct[ai])
            cp_orig = np.mean(cospos_correct[ai, :n_orig])
            cp_warm = np.mean(cospos_correct[ai, n_orig:])
            cp_comb = np.mean(cospos_correct[ai])

            model_data[axis_name] = {
                "bipolar": {"orig": round(bp_orig, 4), "warm": round(bp_warm, 4), "comb": round(bp_comb, 4)},
                "cospos": {"orig": round(cp_orig, 4), "warm": round(cp_warm, 4), "comb": round(cp_comb, 4)},
                "anchor_geometry": {
                    "cos_pos_to_responses": round(mean_cos_to_pos, 4),
                    "cos_neg_to_responses": round(mean_cos_to_neg, 4),
                    "cos_between_anchors": round(cos_between_anchors, 4),
                    "neg_closer_than_pos": bool(mean_cos_to_neg > mean_cos_to_pos),
                },
                "cospos_advantage": round(cp_comb - bp_comb, 4),
            }

        # Print anchor geometry analysis
        print(f"\n--- Anchor Geometry (why does cospos help?) ---")
        print(f"{'Axis':15s}  cos(pos,resp)  cos(neg,resp)  cos(pos,neg)  neg_closer?  cospos_adv")
        for a in sorted(axis_names, key=lambda x: model_data[x]["cospos_advantage"], reverse=True):
            d = model_data[a]
            ag = d["anchor_geometry"]
            print(f"  {a:13s}  {ag['cos_pos_to_responses']:.4f}         {ag['cos_neg_to_responses']:.4f}         "
                  f"{ag['cos_between_anchors']:.4f}        {'YES' if ag['neg_closer_than_pos'] else 'no ':3s}        "
                  f"{d['cospos_advantage']:+.4f}")

        # Optimal per-axis method selection
        print(f"\n--- Per-axis optimal method ---")
        per_axis_best_correct = np.zeros(n_total)
        for ai, a in enumerate(axis_names):
            if model_data[a]["cospos"]["comb"] > model_data[a]["bipolar"]["comb"]:
                per_axis_best_correct += cospos_correct[ai]
                method = "cospos"
            else:
                per_axis_best_correct += bipolar_correct[ai]
                method = "bipolar"

        # Exhaustive search over 2/3/4/5 axis combinations with optimal method per axis
        print(f"\n--- Exhaustive combo search (best 2-5 axis sets) ---")

        # Choose best method per axis
        best_correct = np.zeros((len(AXES), n_total))
        best_method = []
        for ai, a in enumerate(axis_names):
            if model_data[a]["cospos"]["comb"] >= model_data[a]["bipolar"]["comb"]:
                best_correct[ai] = cospos_correct[ai]
                best_method.append("cp")
            else:
                best_correct[ai] = bipolar_correct[ai]
                best_method.append("bp")

        for n_axes in [2, 3, 4, 5]:
            best_combo_acc = 0
            best_combo = None
            best_combo_detail = None

            for combo in itertools.combinations(range(len(AXES)), n_axes):
                votes = best_correct[list(combo)].sum(axis=0)
                threshold = n_axes / 2

                # Majority vote
                mv_acc = np.mean(votes > threshold)

                # Also check: sum-of-margins vote (use absolute correctness sum)
                orig_votes = votes[:n_orig]
                warm_votes = votes[n_orig:]
                orig_acc = np.mean(orig_votes > threshold)
                warm_acc = np.mean(warm_votes > threshold)

                # Require >50% on BOTH splits to count
                balanced_acc = min(orig_acc, warm_acc)

                if balanced_acc > best_combo_acc:
                    best_combo_acc = balanced_acc
                    best_combo = combo
                    best_combo_detail = {
                        "axes": [axis_names[i] for i in combo],
                        "methods": [best_method[i] for i in combo],
                        "orig": round(float(orig_acc), 4),
                        "warm": round(float(warm_acc), 4),
                        "combined": round(float(mv_acc), 4),
                        "balanced_min": round(float(balanced_acc), 4),
                    }

            if best_combo_detail:
                detail = best_combo_detail
                print(f"  Best {n_axes}-axis: {detail['axes']} ({detail['methods']})")
                print(f"    orig={detail['orig']:.1%}  warm={detail['warm']:.1%}  "
                      f"comb={detail['combined']:.1%}  balanced_min={detail['balanced_min']:.1%}")
                model_data[f"best_{n_axes}_axis_combo"] = best_combo_detail

        # Margin-weighted voting for best 3-axis combo
        if "best_3_axis_combo" in model_data:
            best3 = model_data["best_3_axis_combo"]["axes"]
            best3_methods = model_data["best_3_axis_combo"]["methods"]
            print(f"\n--- Margin-weighted voting for {best3} ---")

            # Get margins for best method
            weighted_scores = np.zeros(n_total)
            for a, m in zip(best3, best3_methods):
                ai = axis_names.index(a)
                if m == "cp":
                    weighted_scores += cospos_margins[ai]
                else:
                    weighted_scores += bipolar_margins[ai]

            # Score: positive = better wins
            sum_correct = np.mean(weighted_scores > 0)
            orig_sum = np.mean(weighted_scores[:n_orig] > 0)
            warm_sum = np.mean(weighted_scores[n_orig:] > 0)
            print(f"  Sum-of-margins: orig={orig_sum:.1%}  warm={warm_sum:.1%}  comb={sum_correct:.1%}")
            model_data["margin_weighted_3axis"] = {
                "axes": best3, "methods": best3_methods,
                "orig": round(float(orig_sum), 4),
                "warm": round(float(warm_sum), 4),
                "combined": round(float(sum_correct), 4),
            }

        results["per_model"][model_name] = model_data
        del model
        gc.collect()

    out_path = out_dir / "cospos_deep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
