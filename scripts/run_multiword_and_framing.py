#!/usr/bin/env python3
"""Test multi-word anchors and response framing variations.

Key questions:
1. Does "Careful Thorough" as a single anchor text beat the individual words?
2. Does "Careful, thorough, and honest" work as one anchor?
3. Does changing the embedding framing of responses matter?
   - "User: X\nAssistant: Y" (current standard)
   - Just "Y" (response only)
   - "Is this response careful? Y"
   - "Rate the quality: Y"
4. Asymmetric anchors: pos=evaluative word, neg=neutral word (not antonym)
"""

import json, sys, gc, itertools
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


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


def score_axis(embed_fn, cases, pos_texts, neg_texts, response_template):
    axis_pos = embed_fn(pos_texts).mean(axis=0)
    axis_neg = embed_fn(neg_texts).mean(axis=0)
    axis_vec = axis_pos - axis_neg
    axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

    better_texts = [response_template(c, "better") for c in cases]
    worse_texts = [response_template(c, "worse") for c in cases]
    better_embs = embed_fn(better_texts)
    worse_embs = embed_fn(worse_texts)

    correct = 0
    for i in range(len(cases)):
        sb = float(np.dot(better_embs[i], axis_unit))
        sw = float(np.dot(worse_embs[i], axis_unit))
        if sb > sw:
            correct += 1
        elif sb == sw:
            correct += 0.5
    return correct / len(cases)


def score_cospos(embed_fn, cases, pos_texts, response_template):
    axis_pos = embed_fn(pos_texts).mean(axis=0)
    better_texts = [response_template(c, "better") for c in cases]
    worse_texts = [response_template(c, "worse") for c in cases]
    better_embs = embed_fn(better_texts)
    worse_embs = embed_fn(worse_texts)

    correct = 0
    for i in range(len(cases)):
        sb = cosine(better_embs[i], axis_pos)
        sw = cosine(worse_embs[i], axis_pos)
        if sb > sw:
            correct += 1
        elif sb == sw:
            correct += 0.5
    return correct / len(cases)


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)

    out_dir = ROOT / "notes/research_cycles/multiword_framing"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Response framing templates
    def standard_frame(c, which):
        return f"User: {c['prompt']}\nAssistant: {c[which]}"

    def response_only(c, which):
        return c[which]

    def quality_query(c, which):
        return f"Is this a good response?\n{c[which]}"

    def careful_query(c, which):
        return f"Is this response careful and thorough?\n{c[which]}"

    def rate_frame(c, which):
        return f"Rate the quality of this response:\n{c[which]}"

    def search_frame(c, which):
        return f"search_query: {c['prompt']}\nsearch_document: {c[which]}"

    FRAMINGS = {
        "standard": standard_frame,
        "response_only": response_only,
        "quality_query": quality_query,
        "careful_query": careful_query,
        "rate_quality": rate_frame,
        "search_doc": search_frame,
    }

    # Multi-word anchor configurations
    ANCHOR_TESTS = {
        # Baselines
        "careful_alone":        {"pos": ["Careful"],              "neg": ["Reckless"]},
        "thorough_alone":       {"pos": ["Thorough"],             "neg": ["Superficial"]},
        "hard_alone":           {"pos": ["Hard"],                 "neg": ["Soft"]},
        "kind_alone":           {"pos": ["Kind"],                 "neg": ["Cruel"]},
        "honest_alone":         {"pos": ["Honest"],               "neg": ["Dishonest"]},

        # Concatenated words (no grammar)
        "careful_thorough":     {"pos": ["Careful thorough"],     "neg": ["Reckless superficial"]},
        "careful_honest":       {"pos": ["Careful honest"],       "neg": ["Reckless dishonest"]},
        "careful_kind":         {"pos": ["Careful kind"],         "neg": ["Reckless cruel"]},
        "thorough_kind":        {"pos": ["Thorough kind"],        "neg": ["Superficial cruel"]},
        "hard_kind":            {"pos": ["Hard kind"],            "neg": ["Soft cruel"]},
        "careful_thorough_hard":{"pos": ["Careful thorough hard"],"neg": ["Reckless superficial soft"]},
        "careful_kind_honest":  {"pos": ["Careful kind honest"],  "neg": ["Reckless cruel dishonest"]},

        # Grammatical combinations
        "careful_and_kind":     {"pos": ["Careful and kind"],     "neg": ["Reckless and cruel"]},
        "careful_and_thorough": {"pos": ["Careful and thorough"], "neg": ["Reckless and superficial"]},
        "firm_but_kind":        {"pos": ["Firm but kind"],        "neg": ["Weak and cruel"]},
        "honest_but_warm":      {"pos": ["Honest but warm"],      "neg": ["Dishonest and cold"]},

        # Asymmetric (evaluative vs neutral, not antonyms)
        "careful_vs_text":      {"pos": ["Careful"],              "neg": ["Text"]},
        "careful_vs_response":  {"pos": ["Careful"],              "neg": ["Response"]},
        "good_vs_text":         {"pos": ["Good"],                 "neg": ["Text"]},
        "quality_vs_text":      {"pos": ["Quality"],              "neg": ["Text"]},
        "thorough_vs_text":     {"pos": ["Thorough"],             "neg": ["Text"]},
        "excellent_vs_text":    {"pos": ["Excellent"],            "neg": ["Text"]},

        # Positive-only (cosine to positive, no axis)
        # (handled separately via score_cospos)

        # Multiple positive anchors (different words same embedding)
        "multi_pos_firmness":   {"pos": ["Careful", "Thorough", "Precise"],
                                 "neg": ["Reckless", "Superficial", "Vague"]},
        "multi_pos_warmth":     {"pos": ["Kind", "Supportive", "Patient"],
                                 "neg": ["Cruel", "Dismissive", "Impatient"]},
        "multi_pos_mixed":      {"pos": ["Careful", "Kind", "Thorough"],
                                 "neg": ["Reckless", "Cruel", "Superficial"]},

        # Single evaluative words tested as positive-only cosine
        # (computed via score_cospos below)
    }

    COSPOS_ONLY = [
        "Careful", "Thorough", "Hard", "Kind", "Honest", "Good",
        "Careful thorough", "Careful kind", "Hard kind",
        "Careful thorough hard", "Careful kind honest",
        "Quality", "Excellent", "Ideal",
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

        # Test 1: Multi-word anchors with standard framing
        print(f"\n--- Multi-word anchor tests (standard framing) ---")
        for name, anchors in ANCHOR_TESTS.items():
            orig_acc = score_axis(embed_fn, original, anchors["pos"], anchors["neg"], standard_frame)
            warm_acc = score_axis(embed_fn, warmth, anchors["pos"], anchors["neg"], standard_frame)
            comb_acc = (orig_acc * len(original) + warm_acc * len(warmth)) / (len(original) + len(warmth))
            bal = min(orig_acc, warm_acc)
            model_results[f"anchor_{name}"] = {
                "orig": round(orig_acc, 4), "warm": round(warm_acc, 4),
                "comb": round(comb_acc, 4), "balanced": round(bal, 4)
            }
            marker = " ***" if bal > 0.55 else (" ** " if bal > 0.50 else "")
            print(f"  {name:30s}  orig={orig_acc:.1%}  warm={warm_acc:.1%}  comb={comb_acc:.1%}  bal={bal:.1%}{marker}")

        # Test 2: Cosine-to-positive for select anchors
        print(f"\n--- Cosine-to-positive (no negative anchor) ---")
        for pos_text in COSPOS_ONLY:
            orig_acc = score_cospos(embed_fn, original, [pos_text], standard_frame)
            warm_acc = score_cospos(embed_fn, warmth, [pos_text], standard_frame)
            comb_acc = (orig_acc * len(original) + warm_acc * len(warmth)) / (len(original) + len(warmth))
            bal = min(orig_acc, warm_acc)
            key = f"cospos_{pos_text.lower().replace(' ', '_')}"
            model_results[key] = {
                "orig": round(orig_acc, 4), "warm": round(warm_acc, 4),
                "comb": round(comb_acc, 4), "balanced": round(bal, 4)
            }
            marker = " ***" if bal > 0.55 else (" ** " if bal > 0.50 else "")
            print(f"  cospos({pos_text:25s})  orig={orig_acc:.1%}  warm={warm_acc:.1%}  comb={comb_acc:.1%}  bal={bal:.1%}{marker}")

        # Test 3: Framing variations with best axes
        print(f"\n--- Framing variations (careful/reckless axis) ---")
        for frame_name, frame_fn in FRAMINGS.items():
            orig_acc = score_axis(embed_fn, original,
                                  ["Careful"], ["Reckless"], frame_fn)
            warm_acc = score_axis(embed_fn, warmth,
                                  ["Careful"], ["Reckless"], frame_fn)
            comb_acc = (orig_acc * len(original) + warm_acc * len(warmth)) / (len(original) + len(warmth))
            bal = min(orig_acc, warm_acc)
            model_results[f"frame_{frame_name}_careful"] = {
                "orig": round(orig_acc, 4), "warm": round(warm_acc, 4),
                "comb": round(comb_acc, 4), "balanced": round(bal, 4)
            }
            marker = " ***" if bal > 0.55 else (" ** " if bal > 0.50 else "")
            print(f"  {frame_name:20s}  orig={orig_acc:.1%}  warm={warm_acc:.1%}  comb={comb_acc:.1%}  bal={bal:.1%}{marker}")

        # Test 4: Same framing variations with hard/soft
        print(f"\n--- Framing variations (hard/soft axis) ---")
        for frame_name, frame_fn in FRAMINGS.items():
            orig_acc = score_axis(embed_fn, original,
                                  ["Hard"], ["Soft"], frame_fn)
            warm_acc = score_axis(embed_fn, warmth,
                                  ["Hard"], ["Soft"], frame_fn)
            comb_acc = (orig_acc * len(original) + warm_acc * len(warmth)) / (len(original) + len(warmth))
            model_results[f"frame_{frame_name}_hard"] = {
                "orig": round(orig_acc, 4), "warm": round(warm_acc, 4),
                "comb": round(comb_acc, 4),
            }
            print(f"  {frame_name:20s}  orig={orig_acc:.1%}  warm={warm_acc:.1%}  comb={comb_acc:.1%}")

        results["per_model"][model_name] = model_results
        del model
        gc.collect()

    out_path = out_dir / "multiword_framing_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
