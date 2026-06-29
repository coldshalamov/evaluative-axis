#!/usr/bin/env python3
"""TEST 2B: Quality phrases vs quality direction.

Single words fail (max cosine < 0.30 across 10K vocabulary). Do multi-word
quality phrases do better? We test common evaluative phrases and compare
their cosine with the centroid direction to single-word cosines.

If phrases also fail (cosine < 0.30), it strengthens the claim that the
quality direction is genuinely novel — not capturable by any common
evaluative expression.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

QUALITY_PHRASES = [
    # positive quality
    "well written", "high quality", "good answer", "thorough analysis",
    "clear explanation", "helpful response", "excellent response",
    "accurate and complete", "thoughtful and nuanced", "well reasoned",
    "good job", "perfect answer", "very helpful", "great response",
    "insightful analysis", "comprehensive answer",
    # negative quality
    "poorly written", "low quality", "bad answer", "shallow analysis",
    "confusing explanation", "unhelpful response", "terrible response",
    "inaccurate and incomplete", "lazy and superficial", "poorly reasoned",
    "bad job", "awful answer", "not helpful", "useless response",
    # quality dimensions
    "factually correct", "emotionally supportive", "refuses harmful request",
    "avoids sycophancy", "honest and direct", "warm and empathetic",
    "technically precise", "safe and responsible",
    # anti-quality dimensions
    "factually wrong", "emotionally cold", "helps with harmful request",
    "sycophantic and agreeable", "dishonest and evasive", "cold and dismissive",
    "technically sloppy", "reckless and irresponsible",
]

SINGLE_WORDS = [
    "good", "bad", "helpful", "unhelpful", "accurate", "inaccurate",
    "careful", "careless", "thorough", "shallow", "warm", "cold",
    "honest", "dishonest", "safe", "unsafe", "sycophantic", "direct",
    "nuanced", "simplistic", "thoughtful", "thoughtless",
]

MODELS = [
    ("snowflake-arctic-embed-m", "Snowflake/snowflake-arctic-embed-m"),
    ("bge-m3", "BAAI/bge-m3"),
    ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
]

def load_battery():
    cases = []
    for f in [BATTERY_50, WARMTH_20]:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    cases.append(json.loads(line))
    return cases

def encode(model, texts, model_name):
    if "nomic" in model_name:
        return model.encode(["search_document: " + t for t in texts],
                            normalize_embeddings=True, show_progress_bar=False)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def make_centroid(model, battery, model_name):
    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in battery]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery]
    better_embs = encode(model, better_texts, model_name)
    worse_embs = encode(model, worse_texts, model_name)
    direction = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    direction = direction / norm(direction)
    return direction

def main():
    from sentence_transformers import SentenceTransformer

    battery = load_battery()
    print(f"Battery: {len(battery)}")
    print(f"Phrases: {len(QUALITY_PHRASES)}, Single words: {len(SINGLE_WORDS)}\n")

    all_results = {}
    for short_name, model_id in MODELS:
        print("=" * 70)
        print(f"MODEL: {short_name}")
        print("=" * 70)

        kwargs = {"trust_remote_code": True} if "nomic" in model_id else {}
        model = SentenceTransformer(model_id, **kwargs)

        centroid = make_centroid(model, battery, model_id)

        # embed phrases and words
        phrase_embs = encode(model, QUALITY_PHRASES, model_id)
        word_embs = encode(model, SINGLE_WORDS, model_id)

        # cosines
        phrase_cosines = {}
        for i, phrase in enumerate(QUALITY_PHRASES):
            cos = float(np.dot(phrase_embs[i], centroid))
            phrase_cosines[phrase] = cos

        word_cosines = {}
        for i, word in enumerate(SINGLE_WORDS):
            cos = float(np.dot(word_embs[i], centroid))
            word_cosines[word] = cos

        # stats
        phrase_abs = [abs(v) for v in phrase_cosines.values()]
        word_abs = [abs(v) for v in word_cosines.values()]

        result = {
            "phrase_cosines": phrase_cosines,
            "word_cosines": word_cosines,
            "phrase_stats": {
                "max_abs_cosine": float(max(phrase_abs)),
                "mean_abs_cosine": float(np.mean(phrase_abs)),
                "num_above_0.30": sum(1 for v in phrase_abs if v > 0.30),
                "num_above_0.20": sum(1 for v in phrase_abs if v > 0.20),
            },
            "word_stats": {
                "max_abs_cosine": float(max(word_abs)),
                "mean_abs_cosine": float(np.mean(word_abs)),
                "num_above_0.30": sum(1 for v in word_abs if v > 0.30),
                "num_above_0.20": sum(1 for v in word_abs if v > 0.20),
            },
        }
        all_results[short_name] = result

        # print
        print(f"\n--- PHRASE COSINES (sorted by |cosine|) ---")
        sorted_phrases = sorted(phrase_cosines.items(), key=lambda x: abs(x[1]), reverse=True)
        for phrase, cos in sorted_phrases[:15]:
            print(f"  {phrase:35s}: {cos:+.4f}")
        if len(sorted_phrases) > 15:
            print(f"  ... ({len(sorted_phrases) - 15} more)")

        print(f"\n--- SINGLE WORD COSINES (sorted by |cosine|) ---")
        sorted_words = sorted(word_cosines.items(), key=lambda x: abs(x[1]), reverse=True)
        for word, cos in sorted_words:
            print(f"  {word:20s}: {cos:+.4f}")

        print(f"\n--- SUMMARY ---")
        print(f"  Phrases: max |cos|={result['phrase_stats']['max_abs_cosine']:.4f}, "
              f"mean |cos|={result['phrase_stats']['mean_abs_cosine']:.4f}, "
              f">{0.30}: {result['phrase_stats']['num_above_0.30']}, "
              f">{0.20}: {result['phrase_stats']['num_above_0.20']}")
        print(f"  Words:   max |cos|={result['word_stats']['max_abs_cosine']:.4f}, "
              f"mean |cos|={result['word_stats']['mean_abs_cosine']:.4f}, "
              f">{0.30}: {result['word_stats']['num_above_0.30']}, "
              f">{0.20}: {result['word_stats']['num_above_0.20']}")

        improvement = result['phrase_stats']['max_abs_cosine'] - result['word_stats']['max_abs_cosine']
        print(f"  Phrase max vs word max: {improvement:+.4f} ({'phrases better' if improvement > 0 else 'words better'})")

        del model
        gc.collect()
        print()

    out_path = OUT_DIR / "phrase_projection_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
