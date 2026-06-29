#!/usr/bin/env python3
"""Definitive voting test with per-model optimal methods.

Key insight from prior experiments:
- "careful" is the ONLY axis validated out-of-sample on all models
- Scoring method matters: cospos for BGE-M3, bipolar for others
- Framing matters: response_only/quality_query for BGE-M3, standard for Nomic

This test: does adding other axes via majority voting IMPROVE on
"careful" alone when using per-model optimal methods? And does it
hold on the expansion battery?
"""

import json, sys, gc
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
    "careful":    {"pos": ["Careful"],    "neg": ["Reckless"]},
    "thorough":   {"pos": ["Thorough"],   "neg": ["Superficial"]},
    "hard":       {"pos": ["Hard"],       "neg": ["Soft"]},
    "kind":       {"pos": ["Kind"],       "neg": ["Cruel"]},
    "honest":     {"pos": ["Honest"],     "neg": ["Dishonest"]},
    "helpful":    {"pos": ["Helpful"],    "neg": ["Unhelpful"]},
    "fair":       {"pos": ["Fair"],       "neg": ["Unfair"]},
    "bold":       {"pos": ["Bold"],       "neg": ["Timid"]},
    "active":     {"pos": ["Active"],     "neg": ["Passive"]},
    "good":       {"pos": ["Good"],       "neg": ["Bad"]},
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

    expansion_cases = []
    for cat_file in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(cat_file))
    n_exp = len(expansion_cases)

    axis_names = list(AXES.keys())
    results = {"per_model": {}}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        # Per-model optimal framing
        is_bge = "bge" in model_name.lower()
        if is_bge:
            frame_fn = lambda c, w: c[w]  # response_only
            frame_name = "response_only"
        else:
            frame_fn = lambda c, w: f"User: {c['prompt']}\nAssistant: {c[w]}"
            frame_name = "standard"

        print(f"  Framing: {frame_name}")

        # Embed responses
        main_better = embed_fn([frame_fn(c, "better") for c in all_cases])
        main_worse = embed_fn([frame_fn(c, "worse") for c in all_cases])
        exp_better = embed_fn([frame_fn(c, "better") for c in expansion_cases])
        exp_worse = embed_fn([frame_fn(c, "worse") for c in expansion_cases])

        # Score all axes with BOTH methods, then pick per-model best later
        bp_correct_main = np.zeros((len(AXES), n_total))
        cp_correct_main = np.zeros((len(AXES), n_total))
        bp_correct_exp = np.zeros((len(AXES), n_exp))
        cp_correct_exp = np.zeros((len(AXES), n_exp))

        for ai, (axis_name, anchors) in enumerate(AXES.items()):
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

            for i in range(n_total):
                sb = float(np.dot(main_better[i], axis_unit))
                sw = float(np.dot(main_worse[i], axis_unit))
                bp_correct_main[ai, i] = 1 if sb > sw else (0.5 if sb == sw else 0)

                sb_c = cosine(main_better[i], pos_emb)
                sw_c = cosine(main_worse[i], pos_emb)
                cp_correct_main[ai, i] = 1 if sb_c > sw_c else (0.5 if sb_c == sw_c else 0)

            for i in range(n_exp):
                sb = float(np.dot(exp_better[i], axis_unit))
                sw = float(np.dot(exp_worse[i], axis_unit))
                bp_correct_exp[ai, i] = 1 if sb > sw else (0.5 if sb == sw else 0)

                sb_c = cosine(exp_better[i], pos_emb)
                sw_c = cosine(exp_worse[i], pos_emb)
                cp_correct_exp[ai, i] = 1 if sb_c > sw_c else (0.5 if sb_c == sw_c else 0)

        # Choose best method per axis based on MAIN battery combined accuracy
        best_correct_main = np.zeros((len(AXES), n_total))
        best_correct_exp = np.zeros((len(AXES), n_exp))
        best_methods = []
        for ai in range(len(AXES)):
            bp_acc = np.mean(bp_correct_main[ai])
            cp_acc = np.mean(cp_correct_main[ai])
            if cp_acc > bp_acc:
                best_correct_main[ai] = cp_correct_main[ai]
                best_correct_exp[ai] = cp_correct_exp[ai]
                best_methods.append("cp")
            else:
                best_correct_main[ai] = bp_correct_main[ai]
                best_correct_exp[ai] = bp_correct_exp[ai]
                best_methods.append("bp")

        # Print per-axis results
        print(f"\n{'Axis':12s}  {'Method':6s}  {'Main':>6s}  {'Orig':>6s}  {'Warm':>6s}  {'OOS':>6s}")
        for ai, a in enumerate(axis_names):
            m_acc = np.mean(best_correct_main[ai])
            o_acc = np.mean(best_correct_main[ai, :n_orig])
            w_acc = np.mean(best_correct_main[ai, n_orig:])
            e_acc = np.mean(best_correct_exp[ai])
            print(f"  {a:10s}  {best_methods[ai]:6s}  {m_acc:5.0%}   {o_acc:5.0%}   {w_acc:5.0%}   {e_acc:5.0%}")

        # Now test voting combos
        print(f"\n--- Majority vote combos ---")
        import itertools

        # Test all 2-5 axis combos, report best by BALANCED main battery
        for n_axes in [1, 3, 5]:
            best_bal = 0
            best_info = None
            best_oos = 0

            for combo in itertools.combinations(range(len(AXES)), n_axes):
                threshold = n_axes / 2
                votes_main = best_correct_main[list(combo)].sum(axis=0)
                votes_exp = best_correct_exp[list(combo)].sum(axis=0)

                orig_acc = float(np.mean(votes_main[:n_orig] > threshold))
                warm_acc = float(np.mean(votes_main[n_orig:] > threshold))
                comb_acc = float(np.mean(votes_main > threshold))
                bal = min(orig_acc, warm_acc)
                oos_acc = float(np.mean(votes_exp > threshold))

                if bal > best_bal:
                    best_bal = bal
                    best_oos = oos_acc
                    best_info = {
                        "axes": [axis_names[i] for i in combo],
                        "methods": [best_methods[i] for i in combo],
                        "orig": round(orig_acc, 4),
                        "warm": round(warm_acc, 4),
                        "comb": round(comb_acc, 4),
                        "bal": round(bal, 4),
                        "oos": round(oos_acc, 4),
                    }

            if best_info:
                print(f"  Best {n_axes}-axis: {best_info['axes']} ({best_info['methods']})")
                print(f"    main: o={best_info['orig']:.0%} w={best_info['warm']:.0%} c={best_info['comb']:.0%} bal={best_info['bal']:.0%}  OOS={best_info['oos']:.0%}")

        # Also: "careful" alone with per-model optimal method
        ci = axis_names.index("careful")
        careful_main = np.mean(best_correct_main[ci])
        careful_orig = np.mean(best_correct_main[ci, :n_orig])
        careful_warm = np.mean(best_correct_main[ci, n_orig:])
        careful_oos = np.mean(best_correct_exp[ci])
        careful_bal = min(careful_orig, careful_warm)
        print(f"\n  === CAREFUL alone ({best_methods[ci]}, {frame_name}) ===")
        print(f"    main: o={careful_orig:.0%} w={careful_warm:.0%} c={careful_main:.0%} bal={careful_bal:.0%}  OOS={careful_oos:.0%}")

        # Per-case failure analysis: which cases does "careful" get wrong?
        careful_wrong_main = [i for i in range(n_total) if best_correct_main[ci, i] < 0.5]
        careful_wrong_exp = [i for i in range(n_exp) if best_correct_exp[ci, i] < 0.5]

        # How many of those are fixed by adding axes?
        if best_info and n_axes == 3:
            combo_idx = [axis_names.index(a) for a in best_info["axes"]]
            votes_main = best_correct_main[combo_idx].sum(axis=0)
            threshold = len(combo_idx) / 2

            fixed = sum(1 for i in careful_wrong_main if votes_main[i] > threshold)
            broken = sum(1 for i in range(n_total)
                        if best_correct_main[ci, i] >= 0.5 and votes_main[i] <= threshold)
            print(f"\n  Adding {best_info['axes']}: fixes {fixed}/{len(careful_wrong_main)} failures, "
                  f"breaks {broken} successes (net {'+'if fixed-broken>=0 else ''}{fixed-broken})")

        results["per_model"][model_name] = {
            "framing": frame_name,
            "careful_method": best_methods[ci],
            "careful_main": round(float(careful_main), 4),
            "careful_bal": round(float(careful_bal), 4),
            "careful_oos": round(float(careful_oos), 4),
        }

        del model
        gc.collect()

    out_dir = ROOT / "notes/research_cycles/definitive_voting"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "definitive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
