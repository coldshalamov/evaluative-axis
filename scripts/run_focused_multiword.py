#!/usr/bin/env python3
"""Focused multiword/framing test on BGE-M3 and Nomic.

Tests only the most informative conditions based on Snowflake results:
- careful alone vs multi-word combinations
- framing variations (quality_query was best on Snowflake)
- asymmetric anchors (careful vs Text)
- cospos for multi-word strings
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
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


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
    n_orig = len(original)

    expansion_cases = []
    for cat_file in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(cat_file))

    FRAMINGS = {
        "standard": lambda c, w: f"User: {c['prompt']}\nAssistant: {c[w]}",
        "response_only": lambda c, w: c[w],
        "quality_query": lambda c, w: f"Is this a good response?\n{c[w]}",
        "rate_quality": lambda c, w: f"Rate the quality of this response:\n{c[w]}",
    }

    ANCHOR_CONFIGS = {
        "careful_bp":           {"pos": ["Careful"], "neg": ["Reckless"]},
        "careful_thorough_bp":  {"pos": ["Careful thorough"], "neg": ["Reckless superficial"]},
        "careful_kind_bp":      {"pos": ["Careful kind"], "neg": ["Reckless cruel"]},
        "hard_kind_bp":         {"pos": ["Hard kind"], "neg": ["Soft cruel"]},
        "careful_vs_text":      {"pos": ["Careful"], "neg": ["Text"]},
        "thorough_vs_text":     {"pos": ["Thorough"], "neg": ["Text"]},
        "hard_vs_text":         {"pos": ["Hard"], "neg": ["Text"]},
        "careful_thorough_hard_bp": {"pos": ["Careful thorough hard"], "neg": ["Reckless superficial soft"]},
        "this_is_careful":      {"pos": ["This is a careful response"], "neg": ["This is a reckless response"]},
        "a_careful_response":   {"pos": ["A careful, thorough response"], "neg": ["A reckless, superficial response"]},
    }

    COSPOS_ANCHORS = [
        "Careful", "Hard", "Thorough", "Kind",
        "Careful thorough", "Hard kind",
        "Careful thorough hard",
        "This is a careful response",
        "A careful, thorough response",
        "Quality", "Excellent",
    ]

    results = {"per_model": {}}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        model_results = {}

        # Pre-embed all responses with standard framing
        all_cases = original + warmth
        std_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        std_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        # Also embed expansion battery
        exp_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion_cases])
        exp_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion_cases])

        # Test anchor configs with standard framing
        print(f"\n--- Anchor configs (standard framing) ---")
        for name, anchors in ANCHOR_CONFIGS.items():
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

            bp_correct = []
            for i in range(len(all_cases)):
                sb = float(np.dot(std_better[i], axis_unit))
                sw = float(np.dot(std_worse[i], axis_unit))
                bp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

            exp_correct = []
            for i in range(len(expansion_cases)):
                sb = float(np.dot(exp_better[i], axis_unit))
                sw = float(np.dot(exp_worse[i], axis_unit))
                exp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

            orig_acc = np.mean(bp_correct[:n_orig])
            warm_acc = np.mean(bp_correct[n_orig:])
            comb_acc = np.mean(bp_correct)
            bal = min(orig_acc, warm_acc)
            exp_acc = np.mean(exp_correct)

            model_results[name] = {
                "orig": round(float(orig_acc), 4), "warm": round(float(warm_acc), 4),
                "comb": round(float(comb_acc), 4), "bal": round(float(bal), 4),
                "expansion": round(float(exp_acc), 4),
            }
            marker = " ***" if bal > 0.55 else (" ** " if bal > 0.50 else "")
            print(f"  {name:30s}  o={orig_acc:.0%} w={warm_acc:.0%} c={comb_acc:.0%} bal={bal:.0%} exp={exp_acc:.0%}{marker}")

        # Cosine-to-positive
        print(f"\n--- Cosine-to-positive ---")
        for pos_text in COSPOS_ANCHORS:
            pos_emb = embed_fn([pos_text]).mean(axis=0)
            cp_correct = []
            for i in range(len(all_cases)):
                sb = cosine(std_better[i], pos_emb)
                sw = cosine(std_worse[i], pos_emb)
                cp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

            exp_cp = []
            for i in range(len(expansion_cases)):
                sb = cosine(exp_better[i], pos_emb)
                sw = cosine(exp_worse[i], pos_emb)
                exp_cp.append(1 if sb > sw else (0.5 if sb == sw else 0))

            orig_acc = np.mean(cp_correct[:n_orig])
            warm_acc = np.mean(cp_correct[n_orig:])
            comb_acc = np.mean(cp_correct)
            bal = min(orig_acc, warm_acc)
            exp_acc = np.mean(exp_cp)

            key = f"cospos_{pos_text.lower().replace(' ', '_').replace(',','')}"
            model_results[key] = {
                "orig": round(float(orig_acc), 4), "warm": round(float(warm_acc), 4),
                "comb": round(float(comb_acc), 4), "bal": round(float(bal), 4),
                "expansion": round(float(exp_acc), 4),
            }
            marker = " ***" if bal > 0.55 else (" ** " if bal > 0.50 else "")
            print(f"  cospos({pos_text:30s})  o={orig_acc:.0%} w={warm_acc:.0%} c={comb_acc:.0%} bal={bal:.0%} exp={exp_acc:.0%}{marker}")

        # Framing variations with careful/reckless
        print(f"\n--- Framing × careful/reckless ---")
        for frame_name, frame_fn in FRAMINGS.items():
            better_embs = embed_fn([frame_fn(c, "better") for c in all_cases])
            worse_embs = embed_fn([frame_fn(c, "worse") for c in all_cases])
            pos_emb = embed_fn(["Careful"]).mean(axis=0)
            neg_emb = embed_fn(["Reckless"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

            bp_correct = []
            cp_correct = []
            for i in range(len(all_cases)):
                sb = float(np.dot(better_embs[i], axis_unit))
                sw = float(np.dot(worse_embs[i], axis_unit))
                bp_correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

                sb_c = cosine(better_embs[i], pos_emb)
                sw_c = cosine(worse_embs[i], pos_emb)
                cp_correct.append(1 if sb_c > sw_c else (0.5 if sb_c == sw_c else 0))

            bp_orig = np.mean(bp_correct[:n_orig])
            bp_warm = np.mean(bp_correct[n_orig:])
            bp_comb = np.mean(bp_correct)
            cp_orig = np.mean(cp_correct[:n_orig])
            cp_warm = np.mean(cp_correct[n_orig:])
            cp_comb = np.mean(cp_correct)

            model_results[f"frame_{frame_name}_bp"] = {
                "orig": round(float(bp_orig), 4), "warm": round(float(bp_warm), 4),
                "comb": round(float(bp_comb), 4),
            }
            model_results[f"frame_{frame_name}_cp"] = {
                "orig": round(float(cp_orig), 4), "warm": round(float(cp_warm), 4),
                "comb": round(float(cp_comb), 4),
            }
            print(f"  {frame_name:20s}  bp: o={bp_orig:.0%} w={bp_warm:.0%} c={bp_comb:.0%}  "
                  f"cp: o={cp_orig:.0%} w={cp_warm:.0%} c={cp_comb:.0%}")

        results["per_model"][model_name] = model_results
        del model
        gc.collect()

    out_dir = ROOT / "notes/research_cycles/multiword_framing"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "focused_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
