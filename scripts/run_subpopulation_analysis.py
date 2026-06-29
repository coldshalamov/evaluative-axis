#!/usr/bin/env python3
"""Subpopulation analysis: are there case types where local models
reliably work, vs types where they reliably fail?

If we can identify a sub-population where "careful" reliably succeeds,
the practical claim narrows but becomes defensible: embedding scoring
works for X-type evaluations but not Y-type.

Tests: per-category accuracy on the balanced battery, and whether
score-delta magnitude predicts correctness (high-confidence predictions
might be more reliable).
"""

import json, gc, math
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

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    expansion_cases = []
    for f in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(f))

    all_cases = original + warmth + expansion_cases
    n_orig = len(original)
    n_warm = len(warmth)
    n_main = n_orig + n_warm
    n_all = len(all_cases)

    # Classify cases by content analysis
    def classify_case(case):
        text = (case["prompt"] + " " + case["better"] + " " + case["worse"]).lower()
        tags = []

        # Firmness indicators
        firm_kw = ["reason", "logic", "evidence", "fact", "correct", "accurate",
                    "truth", "calculat", "math", "code", "error", "mistake",
                    "wrong", "fallac", "claim", "assert", "refut", "counter"]
        if any(k in text for k in firm_kw):
            tags.append("firmness")

        # Warmth indicators
        warm_kw = ["feel", "empath", "support", "kind", "comfort", "emotion",
                    "understand", "caring", "gentle", "compassion", "worry",
                    "anxious", "stress", "sad", "happy"]
        if any(k in text for k in warm_kw):
            tags.append("warmth")

        # AI persona indicators
        persona_kw = ["childhood", "dream", "favorite", "memory", "experience",
                       "i feel", "i love", "i hate", "i remember", "personally",
                       "as an ai", "i'm an ai", "i am an ai"]
        if any(k in text for k in persona_kw):
            tags.append("persona")

        # Safety indicators
        safety_kw = ["harm", "danger", "weapon", "drug", "illegal", "hack",
                      "attack", "exploit", "kill", "hurt", "violent"]
        if any(k in text for k in safety_kw):
            tags.append("safety")

        # Length disparity
        len_better = len(case["better"])
        len_worse = len(case["worse"])
        ratio = max(len_better, len_worse) / (min(len_better, len_worse) + 1)
        if ratio > 2:
            tags.append("length_diff")

        # Tone-vs-content mismatch (better response is shorter/terser)
        if len_better < len_worse * 0.6:
            tags.append("better_is_shorter")
        elif len_better > len_worse * 1.5:
            tags.append("better_is_longer")

        if not tags:
            tags.append("other")

        return tags

    case_tags = [classify_case(c) for c in all_cases]

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        pos_emb = embed_fn(["Careful"]).mean(axis=0)
        neg_emb = embed_fn(["Reckless"]).mean(axis=0)
        axis = (pos_emb - neg_emb) / (norm(pos_emb - neg_emb) + 1e-12)

        deltas = []
        correct = []
        for i in range(n_all):
            sb = float(np.dot(better_embs[i], axis))
            sw = float(np.dot(worse_embs[i], axis))
            delta = sb - sw
            deltas.append(delta)
            correct.append(1 if delta > 0 else (0.5 if delta == 0 else 0))

        # Per-tag accuracy
        print(f"\n--- Per-tag accuracy (careful/reckless) ---")
        all_tags = set()
        for t in case_tags:
            all_tags.update(t)

        for tag in sorted(all_tags):
            indices = [i for i in range(n_all) if tag in case_tags[i]]
            if not indices:
                continue
            tag_correct = [correct[i] for i in indices]
            k = sum(1 for c in tag_correct if c >= 0.5)
            acc = np.mean(tag_correct)
            lo, hi = wilson_ci(k, len(indices))
            sig = "YES" if lo > 0.5 else "no"
            print(f"  {tag:20s}  n={len(indices):3d}  acc={acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  {sig}")

        # Confidence analysis: does |delta| predict correctness?
        print(f"\n--- Confidence analysis ---")
        abs_deltas = [abs(d) for d in deltas]
        median_delta = np.median(abs_deltas)

        high_conf = [i for i in range(n_all) if abs(deltas[i]) >= median_delta]
        low_conf = [i for i in range(n_all) if abs(deltas[i]) < median_delta]

        hc_acc = np.mean([correct[i] for i in high_conf])
        lc_acc = np.mean([correct[i] for i in low_conf])
        k_hc = sum(1 for i in high_conf if correct[i] >= 0.5)
        k_lc = sum(1 for i in low_conf if correct[i] >= 0.5)
        lo_hc, hi_hc = wilson_ci(k_hc, len(high_conf))
        lo_lc, hi_lc = wilson_ci(k_lc, len(low_conf))

        print(f"  High confidence (|delta| >= {median_delta:.4f}): n={len(high_conf)}  acc={hc_acc:.0%}  CI=[{lo_hc:.0%}, {hi_hc:.0%}]")
        print(f"  Low confidence  (|delta| <  {median_delta:.4f}): n={len(low_conf)}   acc={lc_acc:.0%}  CI=[{lo_lc:.0%}, {hi_lc:.0%}]")

        # Quartile analysis
        print(f"\n  By confidence quartile:")
        quartiles = np.percentile(abs_deltas, [25, 50, 75])
        bins = [(0, quartiles[0], "Q1 (lowest)"),
                (quartiles[0], quartiles[1], "Q2"),
                (quartiles[1], quartiles[2], "Q3"),
                (quartiles[2], max(abs_deltas)+1, "Q4 (highest)")]

        for lo_q, hi_q, label in bins:
            q_idx = [i for i in range(n_all) if lo_q <= abs(deltas[i]) < hi_q]
            if not q_idx:
                continue
            q_acc = np.mean([correct[i] for i in q_idx])
            k_q = sum(1 for i in q_idx if correct[i] >= 0.5)
            lo_ci, hi_ci = wilson_ci(k_q, len(q_idx))
            print(f"    {label:15s}  n={len(q_idx):3d}  acc={q_acc:.0%}  CI=[{lo_ci:.0%}, {hi_ci:.0%}]")

        # Cross-model agreement
        print(f"\n--- Cases where all 3 models agree on careful ---")
        # (This requires all models to be run, so we'll note this for the summary)

        del model
        gc.collect()


if __name__ == "__main__":
    main()
