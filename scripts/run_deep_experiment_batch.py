#!/usr/bin/env python3
"""Deep experiment batch: test overlooked axes, majority vote combinations,
per-case failure analysis, and the valence hypothesis from recent papers.

Tests:
1. "Active/Passive" scored 85% on expansion (Snowflake) - is it real?
2. Osgood EPA dimensions: Evaluation (good/bad), Potency (hard/soft), Activity (active/passive)
3. Majority vote accuracy for the minimum covering sets
4. Valence-axis words: "better/worse", "right/wrong", "correct/incorrect", "proper/improper"
5. Per-case failure patterns across best axes
"""

import json, sys, gc, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# Expanded axis set: standard + Osgood + valence + new candidates
AXES = {
    # Standard evaluative (established)
    "careful":      {"pos": ["Careful"],       "neg": ["Reckless"]},
    "honest":       {"pos": ["Honest"],        "neg": ["Dishonest"]},
    "kind":         {"pos": ["Kind"],          "neg": ["Cruel"]},
    "thorough":     {"pos": ["Thorough"],      "neg": ["Superficial"]},
    "good":         {"pos": ["Good"],          "neg": ["Bad"]},
    "hard":         {"pos": ["Hard"],          "neg": ["Soft"]},
    "helpful":      {"pos": ["Helpful"],       "neg": ["Unhelpful"]},
    "fair":         {"pos": ["Fair"],          "neg": ["Unfair"]},
    "wise":         {"pos": ["Wise"],          "neg": ["Foolish"]},
    "responsible":  {"pos": ["Responsible"],   "neg": ["Irresponsible"]},
    "clear":        {"pos": ["Clear"],         "neg": ["Confusing"]},

    # Osgood Activity dimension (the 85% outlier)
    "active":       {"pos": ["Active"],        "neg": ["Passive"]},
    "fast":         {"pos": ["Fast"],          "neg": ["Slow"]},
    "sharp":        {"pos": ["Sharp"],         "neg": ["Dull"]},
    "bold":         {"pos": ["Bold"],          "neg": ["Timid"]},
    "strong":       {"pos": ["Strong"],        "neg": ["Weak"]},
    "vigorous":     {"pos": ["Vigorous"],      "neg": ["Lethargic"]},

    # Direct valence (from the functional welfare axis paper)
    "better":       {"pos": ["Better"],        "neg": ["Worse"]},
    "right":        {"pos": ["Right"],         "neg": ["Wrong"]},
    "correct":      {"pos": ["Correct"],       "neg": ["Incorrect"]},
    "proper":       {"pos": ["Proper"],        "neg": ["Improper"]},
    "ideal":        {"pos": ["Ideal"],         "neg": ["Terrible"]},
    "excellent":    {"pos": ["Excellent"],      "neg": ["Awful"]},
    "superior":     {"pos": ["Superior"],      "neg": ["Inferior"]},

    # Warmth-specific
    "gentle":       {"pos": ["Gentle"],        "neg": ["Harsh"]},
    "patient":      {"pos": ["Patient"],       "neg": ["Impatient"]},
    "supportive":   {"pos": ["Supportive"],    "neg": ["Dismissive"]},
    "constructive": {"pos": ["Constructive"],  "neg": ["Destructive"]},
    "warm":         {"pos": ["Warm"],          "neg": ["Cold"]},
    "empathetic":   {"pos": ["Empathetic"],    "neg": ["Indifferent"]},

    # Competence-specific
    "accurate":     {"pos": ["Accurate"],      "neg": ["Inaccurate"]},
    "reliable":     {"pos": ["Reliable"],      "neg": ["Unreliable"]},
    "rigorous":     {"pos": ["Rigorous"],      "neg": ["Sloppy"]},
    "precise":      {"pos": ["Precise"],       "neg": ["Vague"]},
    "competent":    {"pos": ["Competent"],      "neg": ["Incompetent"]},
    "diligent":     {"pos": ["Diligent"],      "neg": ["Lazy"]},
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


def pairwise_accuracy_per_case(better_embs, worse_embs, axis):
    results = []
    for i in range(len(better_embs)):
        sb = float(np.dot(better_embs[i], axis))
        sw = float(np.dot(worse_embs[i], axis))
        correct = 1 if sb > sw else (0.5 if sb == sw else 0)
        margin = sb - sw
        results.append({"correct": correct, "margin": margin,
                        "score_better": sb, "score_worse": sw})
    return results


def cosine_to_positive_per_case(better_embs, worse_embs, pos_emb):
    from numpy.linalg import norm
    results = []
    pos_unit = pos_emb / (norm(pos_emb) + 1e-12)
    for i in range(len(better_embs)):
        sb = float(np.dot(better_embs[i] / (norm(better_embs[i]) + 1e-12), pos_unit))
        sw = float(np.dot(worse_embs[i] / (norm(worse_embs[i]) + 1e-12), pos_unit))
        correct = 1 if sb > sw else (0.5 if sb == sw else 0)
        results.append({"correct": correct, "margin": sb - sw})
    return results


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    all_cases = original + warmth

    out_dir = ROOT / "notes/research_cycles/deep_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"metadata": {
        "n_original": len(original), "n_warmth": len(warmth),
        "n_total": len(all_cases), "n_axes": len(AXES),
        "axes_tested": list(AXES.keys()),
        "models": MODELS,
    }, "per_model": {}}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        # Embed all responses
        all_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        all_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        orig_better = all_better[:len(original)]
        orig_worse = all_worse[:len(original)]
        warm_better = all_better[len(original):]
        warm_worse = all_worse[len(original):]

        model_results = {}

        for axis_name, anchors in AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["pos"], anchors["neg"])
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)

            # Bipolar scoring per case
            orig_cases = pairwise_accuracy_per_case(orig_better, orig_worse, axis_vec)
            warm_cases = pairwise_accuracy_per_case(warm_better, warm_worse, axis_vec)

            # Cosine-to-positive per case
            orig_cospos = cosine_to_positive_per_case(orig_better, orig_worse, pos_emb)
            warm_cospos = cosine_to_positive_per_case(warm_better, warm_worse, pos_emb)

            orig_acc = np.mean([c["correct"] for c in orig_cases])
            warm_acc = np.mean([c["correct"] for c in warm_cases])
            comb_acc = np.mean([c["correct"] for c in orig_cases + warm_cases])

            orig_cospos_acc = np.mean([c["correct"] for c in orig_cospos])
            warm_cospos_acc = np.mean([c["correct"] for c in warm_cospos])
            comb_cospos_acc = np.mean([c["correct"] for c in orig_cospos + warm_cospos])

            # Per-case correct vectors (for majority vote later)
            bipolar_correct = [c["correct"] for c in orig_cases + warm_cases]
            cospos_correct = [c["correct"] for c in orig_cospos + warm_cospos]

            # Mean margin
            mean_margin = np.mean([c["margin"] for c in orig_cases + warm_cases])
            orig_margin = np.mean([c["margin"] for c in orig_cases])
            warm_margin = np.mean([c["margin"] for c in warm_cases])

            model_results[axis_name] = {
                "bipolar": {"original": round(orig_acc, 4), "warmth": round(warm_acc, 4),
                           "combined": round(comb_acc, 4)},
                "cosine_pos": {"original": round(orig_cospos_acc, 4), "warmth": round(warm_cospos_acc, 4),
                              "combined": round(comb_cospos_acc, 4)},
                "margins": {"original": round(orig_margin, 6), "warmth": round(warm_margin, 6),
                           "combined": round(mean_margin, 6)},
                "per_case_bipolar": bipolar_correct,
                "per_case_cospos": cospos_correct,
            }

        # Majority vote analysis across axis combinations
        print(f"\n--- Majority Vote Analysis ---")

        # Build per-case correct matrix: axes × cases
        axis_names = list(AXES.keys())
        n_cases = len(all_cases)
        bipolar_matrix = np.array([model_results[a]["per_case_bipolar"] for a in axis_names])
        cospos_matrix = np.array([model_results[a]["per_case_cospos"] for a in axis_names])

        # Test various combinations
        covering_sets = {
            "hard+kind+careful": ["hard", "kind", "careful"],
            "hard+kind+good": ["hard", "kind", "good"],
            "hard+kind+thorough": ["hard", "kind", "thorough"],
            "hard+kind+clear": ["hard", "kind", "clear"],
            "careful+kind+honest": ["careful", "kind", "honest"],
            "active+kind+careful": ["active", "kind", "careful"],
            "better+kind+careful": ["better", "kind", "careful"],
            "correct+kind+careful": ["correct", "kind", "careful"],
            "right+kind+careful": ["right", "kind", "careful"],
            "top5_greedy": [],  # filled below
            "all_35": axis_names,
        }

        # Find top 5 by combined accuracy
        acc_ranking = sorted(axis_names, key=lambda a: model_results[a]["bipolar"]["combined"], reverse=True)
        covering_sets["top5_greedy"] = acc_ranking[:5]
        covering_sets["top3_greedy"] = acc_ranking[:3]
        covering_sets["top7_greedy"] = acc_ranking[:7]

        vote_results = {}
        for set_name, axes_in_set in covering_sets.items():
            if not axes_in_set:
                continue
            idxs = [axis_names.index(a) for a in axes_in_set if a in axis_names]
            if not idxs:
                continue

            # Majority vote (bipolar)
            votes = bipolar_matrix[idxs].sum(axis=0)
            threshold = len(idxs) / 2
            mv_correct = np.mean(votes > threshold)

            # Sum of margins (bipolar)
            all_margins = []
            for a in axes_in_set:
                if a not in axis_names:
                    continue
                cases_data = model_results[a]["per_case_bipolar"]
                # Reconstruct margins from per-axis data
                # For sum-of-scores, we need the actual scores, but we stored correctness
                # Use the margin data instead

            # Weighted sum: sum the per-case correctness values
            sum_correct = bipolar_matrix[idxs].sum(axis=0)
            sum_acc = np.mean(sum_correct > len(idxs) * 0.5)

            # Per-split
            orig_votes = bipolar_matrix[idxs, :len(original)].sum(axis=0)
            warm_votes = bipolar_matrix[idxs, len(original):].sum(axis=0)
            orig_mv = np.mean(orig_votes > threshold)
            warm_mv = np.mean(warm_votes > threshold)

            # Also try: ANY axis correct (union)
            any_correct = np.mean(bipolar_matrix[idxs].max(axis=0) > 0.5)

            # Also try: cosine-to-positive majority vote
            cospos_votes = cospos_matrix[idxs].sum(axis=0)
            cospos_mv = np.mean(cospos_votes > threshold)

            vote_results[set_name] = {
                "axes": axes_in_set if len(axes_in_set) <= 10 else f"{len(axes_in_set)} axes",
                "majority_vote": {"original": round(float(orig_mv), 4),
                                 "warmth": round(float(warm_mv), 4),
                                 "combined": round(float(mv_correct), 4)},
                "any_correct": round(float(any_correct), 4),
                "cospos_majority": round(float(cospos_mv), 4),
            }

            print(f"  {set_name:30s} MV={mv_correct:.1%}  orig={orig_mv:.1%}  warm={warm_mv:.1%}  any={any_correct:.1%}  cospos_mv={cospos_mv:.1%}")

        model_results["majority_vote_combos"] = vote_results

        # Rank all axes
        print(f"\n--- Axis Rankings (combined bipolar accuracy) ---")
        for rank, a in enumerate(acc_ranking[:15], 1):
            r = model_results[a]
            bp = r["bipolar"]
            cp = r["cosine_pos"]
            better_method = "cospos" if cp["combined"] > bp["combined"] else "bipolar"
            best = max(bp["combined"], cp["combined"])
            print(f"  {rank:2d}. {a:15s}  bp={bp['combined']:.1%} (o={bp['original']:.1%} w={bp['warmth']:.1%})  "
                  f"cp={cp['combined']:.1%}  best={best:.1%} [{better_method}]")

        # Per-case failure analysis: which cases does EVERY axis fail on?
        all_correct = bipolar_matrix.max(axis=0)  # best of any axis
        always_wrong = np.where(all_correct < 0.5)[0]

        failure_analysis = {"n_always_wrong": int(len(always_wrong)), "always_wrong_cases": []}
        for idx in always_wrong:
            case = all_cases[idx]
            failure_analysis["always_wrong_cases"].append({
                "case_id": case.get("id", f"case_{idx}"),
                "category": case.get("category", ""),
                "phenomenon": case.get("phenomenon", ""),
                "split": "original" if idx < len(original) else "warmth",
            })

        if always_wrong.size > 0:
            print(f"\n--- Cases ALL {len(AXES)} axes get wrong ({len(always_wrong)} cases) ---")
            for f in failure_analysis["always_wrong_cases"][:10]:
                print(f"  {f['case_id']:30s}  {f['category']:20s}  {f['phenomenon']}")

        model_results["failure_analysis"] = failure_analysis

        # Strip per-case vectors for JSON (keep just the summary)
        for a in axis_names:
            del model_results[a]["per_case_bipolar"]
            del model_results[a]["per_case_cospos"]

        results["per_model"][model_name] = model_results
        del model
        gc.collect()
        print()

    # Save
    out_path = out_dir / "deep_batch_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
