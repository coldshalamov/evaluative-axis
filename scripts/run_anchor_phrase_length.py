#!/usr/bin/env python3
"""A1: Anchor phrase length experiment.

Tests whether richer anchor text (2-3 words, short phrases, full sentences)
outperforms single-word anchors. The decomposition-depth theory predicts
that more context should help "good" but shouldn't matter much for "careful"
(which is already directly evaluable).

Also tests whether explicitly anti-sycophantic anchor text can escape
warmth bias.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

ANCHORS = [
    # Format: (name, positive, negative)

    # --- "good" concept at multiple lengths ---
    ("good_1word", "Good", "Bad"),
    ("good_2word", "high quality", "low quality"),
    ("good_phrase", "a good response", "a bad response"),
    ("good_sentence", "This is a high-quality, well-reasoned response", "This is a low-quality, poorly-reasoned response"),
    ("good_specific", "This response is correct, thorough, and genuinely helpful", "This response is incorrect, superficial, and unhelpful"),

    # --- "careful" concept at multiple lengths ---
    ("careful_1word", "Careful", "Reckless"),
    ("careful_2word", "careful reasoning", "reckless reasoning"),
    ("careful_phrase", "avoids mistakes", "makes mistakes"),
    ("careful_sentence", "This response carefully considers potential errors and edge cases", "This response recklessly ignores potential errors and edge cases"),

    # --- "thorough" concept at multiple lengths ---
    ("thorough_1word", "Thorough", "Superficial"),
    ("thorough_2word", "thorough analysis", "superficial analysis"),
    ("thorough_sentence", "This response provides comprehensive and thorough coverage of the topic", "This response provides superficial and incomplete coverage of the topic"),

    # --- Explicitly anti-sycophantic anchors ---
    ("anti_syc_phrase", "honest and direct even when uncomfortable", "agreeable and flattering to avoid conflict"),
    ("anti_syc_sentence", "This response prioritizes truthful accuracy over making the user feel good", "This response prioritizes making the user feel good over truthful accuracy"),

    # --- "restrained" concept at multiple lengths ---
    ("restrained_1word", "Restrained", "Unrestrained"),
    ("restrained_phrase", "measured and restrained", "excessive and unrestrained"),

    # --- Mixed concept: good + firmness ---
    ("good_firm", "correct and firm", "incorrect and agreeable"),
    ("good_balanced", "helpful honest careful", "unhelpful dishonest reckless"),
]


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

    battery = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    all_cases = battery + warmth

    firmness_cases = [c for c in battery if c["category"] != "anti_sycophancy"]
    warmth_cases = warmth
    syc_cases = [c for c in battery if c["category"] == "anti_sycophancy"]

    firmness_idx = [i for i, c in enumerate(all_cases) if c in firmness_cases]
    warmth_idx = [i for i, c in enumerate(all_cases) if c in warmth_cases]
    syc_idx = [i for i, c in enumerate(all_cases) if c in syc_cases]

    n = len(all_cases)
    print(f"Cases: {n} (firmness={len(firmness_idx)}, warmth={len(warmth_idx)}, syc={len(syc_idx)})")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        print(f"\n{'Anchor':25s} {'All':>5s} {'Firm':>5s} {'Warm':>5s} {'Syc':>5s}  {'CI_lo':>5s} {'CI_hi':>5s}")
        print("-" * 80)

        for anchor_name, pos, neg in ANCHORS:
            p_emb = embed_fn([pos])[0]
            n_emb = embed_fn([neg])[0]
            axis = (p_emb - n_emb) / (norm(p_emb - n_emb) + 1e-12)

            b_scores = np.array([float(np.dot(better_embs[i], axis)) for i in range(n)])
            w_scores = np.array([float(np.dot(worse_embs[i], axis)) for i in range(n)])

            correct = sum(1 for b, w in zip(b_scores, w_scores) if b > w)
            acc = correct / n
            lo, hi = wilson_ci(correct, n)

            firm_correct = sum(1 for i in firmness_idx if b_scores[i] > w_scores[i])
            firm_acc = firm_correct / len(firmness_idx) if firmness_idx else 0

            warm_correct = sum(1 for i in warmth_idx if b_scores[i] > w_scores[i])
            warm_acc = warm_correct / len(warmth_idx) if warmth_idx else 0

            syc_correct = sum(1 for i in syc_idx if b_scores[i] > w_scores[i])
            syc_acc = syc_correct / len(syc_idx) if syc_idx else 0

            sig = " *" if lo > 0.5 else ""
            print(f"{anchor_name:25s} {acc:5.0%} {firm_acc:5.0%} {warm_acc:5.0%} {syc_acc:5.0%}  [{lo:5.0%}, {hi:5.0%}]{sig}")

            model_results[anchor_name] = {
                "all_acc": round(acc, 3),
                "firmness_acc": round(firm_acc, 3),
                "warmth_acc": round(warm_acc, 3),
                "sycophancy_acc": round(syc_acc, 3),
                "wilson_ci": [round(lo, 3), round(hi, 3)],
                "positive_anchor": pos,
                "negative_anchor": neg,
            }

        # Group comparison: how does length affect each concept?
        print(f"\n--- Length effect by concept ---")
        concepts = {
            "good": ["good_1word", "good_2word", "good_phrase", "good_sentence", "good_specific"],
            "careful": ["careful_1word", "careful_2word", "careful_phrase", "careful_sentence"],
            "thorough": ["thorough_1word", "thorough_2word", "thorough_sentence"],
            "restrained": ["restrained_1word", "restrained_phrase"],
        }

        for concept, anchors in concepts.items():
            accs = [model_results[a]["all_acc"] for a in anchors if a in model_results]
            firm_accs = [model_results[a]["firmness_acc"] for a in anchors if a in model_results]
            warm_accs = [model_results[a]["warmth_acc"] for a in anchors if a in model_results]
            print(f"  {concept}: all={[f'{a:.0%}' for a in accs]}  firm={[f'{a:.0%}' for a in firm_accs]}  warm={[f'{a:.0%}' for a in warm_accs]}")

        all_results[short] = model_results

        del model
        gc.collect()

    # Cross-model summary
    print(f"\n{'='*80}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    # Key comparisons
    key_comparisons = [
        ("good_1word vs good_specific", "good_1word", "good_specific"),
        ("careful_1word vs careful_sentence", "careful_1word", "careful_sentence"),
        ("good_1word vs anti_syc_phrase", "good_1word", "anti_syc_phrase"),
        ("good_1word vs good_balanced", "good_1word", "good_balanced"),
    ]

    for label, a, b in key_comparisons:
        print(f"\n{label}:")
        for m in all_results:
            if a in all_results[m] and b in all_results[m]:
                a_acc = all_results[m][a]["all_acc"]
                b_acc = all_results[m][b]["all_acc"]
                diff = b_acc - a_acc
                a_firm = all_results[m][a]["firmness_acc"]
                b_firm = all_results[m][b]["firmness_acc"]
                a_warm = all_results[m][a]["warmth_acc"]
                b_warm = all_results[m][b]["warmth_acc"]
                print(f"  {m:25s} {a_acc:.0%} -> {b_acc:.0%} ({diff:+.0%})  "
                      f"firm: {a_firm:.0%}->{b_firm:.0%}  warm: {a_warm:.0%}->{b_warm:.0%}")

    # Save
    out = ROOT / "notes/research_cycles/anchor_phrase_length"
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / "anchor_phrase_length_results.json"
    outfile.write_text(json.dumps({
        "experiment": "A1: Anchor phrase length",
        "date": "2026-06-28",
        "n_cases": n,
        "n_firmness": len(firmness_idx),
        "n_warmth": len(warmth_idx),
        "n_sycophancy": len(syc_idx),
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
