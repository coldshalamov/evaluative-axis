#!/usr/bin/env python3
"""Project quality direction onto 10,000 common English words.

Tests whether ANY word in the vocabulary aligns with the centroid
quality direction. If cosine > 0.20 with any word, the "orthogonal
to all words" claim needs qualification.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_battery():
    return load_cases(BATTERY_50) + load_cases(WARMTH_20)


def get_word_list(n=10000):
    """Get top N English words by frequency."""
    try:
        import nltk
        from nltk.corpus import brown
        try:
            words = brown.words()
        except LookupError:
            nltk.download('brown', quiet=True)
            words = brown.words()

        from collections import Counter
        freq = Counter(w.lower() for w in words if w.isalpha() and len(w) > 1)
        return [w for w, _ in freq.most_common(n)]
    except ImportError:
        pass

    # Fallback: use a curated list of common words plus evaluative terms
    print("  nltk not available, using curated word list")
    base = [
        "good", "bad", "better", "worse", "best", "worst",
        "careful", "reckless", "honest", "dishonest", "kind", "cruel",
        "helpful", "unhelpful", "thorough", "superficial", "fair", "unfair",
        "clear", "confusing", "respectful", "disrespectful",
        "accurate", "inaccurate", "safe", "unsafe", "polite", "rude",
        "thoughtful", "thoughtless", "wise", "foolish", "patient", "impatient",
        "gentle", "harsh", "encouraging", "discouraging", "supportive", "dismissive",
        "precise", "sloppy", "truthful", "deceptive", "sincere", "insincere",
        "competent", "incompetent", "creative", "boring", "detailed", "vague",
        "professional", "amateur", "excellent", "terrible", "outstanding", "mediocre",
        "brilliant", "stupid", "insightful", "shallow", "nuanced", "simplistic",
        "comprehensive", "incomplete", "elegant", "clumsy", "efficient", "wasteful",
        "responsible", "irresponsible", "reliable", "unreliable",
        "the", "a", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can",
        "not", "no", "yes", "very", "also", "just", "more", "most",
        "about", "after", "again", "all", "an", "and", "any", "as", "at",
        "back", "because", "before", "between", "both", "but", "by",
        "come", "could", "day", "did", "do", "down", "each", "even",
        "find", "first", "for", "from", "get", "give", "go", "great",
        "hand", "he", "her", "here", "him", "his", "how", "I",
        "if", "in", "into", "it", "its", "know", "large", "last",
        "leave", "life", "like", "little", "long", "look", "make", "man",
        "many", "me", "might", "most", "much", "my", "new", "next",
        "no", "not", "now", "number", "of", "off", "old", "on", "one",
        "only", "or", "other", "our", "out", "over", "own", "part",
        "people", "place", "point", "right", "same", "say", "see", "she",
        "show", "small", "so", "some", "something", "state", "still",
        "take", "tell", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "thing", "think", "this", "those",
        "through", "time", "to", "too", "turn", "two", "under", "up",
        "us", "use", "want", "way", "we", "well", "what", "when",
        "where", "which", "while", "who", "why", "with", "work",
        "world", "would", "write", "year", "you", "your",
        "answer", "question", "response", "reply", "explanation",
        "analysis", "argument", "reasoning", "logic", "evidence",
        "fact", "opinion", "truth", "lie", "mistake", "error",
        "correct", "wrong", "right", "false", "true",
        "help", "harm", "benefit", "risk", "danger", "safety",
        "important", "trivial", "significant", "relevant", "irrelevant",
        "appropriate", "inappropriate", "suitable", "unsuitable",
        "effective", "ineffective", "useful", "useless",
        "positive", "negative", "neutral", "objective", "subjective",
        "formal", "informal", "casual", "serious", "funny", "sad",
        "happy", "angry", "calm", "anxious", "confident", "uncertain",
        "simple", "complex", "easy", "difficult", "hard", "soft",
        "warm", "cold", "hot", "cool", "firm", "flexible",
        "strong", "weak", "powerful", "powerless",
        "smart", "clever", "intelligent", "dumb", "ignorant",
        "beautiful", "ugly", "pretty", "handsome", "plain",
        "fast", "slow", "quick", "rapid", "gradual",
        "big", "small", "large", "tiny", "huge", "enormous",
        "rich", "poor", "wealthy", "expensive", "cheap",
        "clean", "dirty", "pure", "contaminated",
        "open", "closed", "shut", "locked", "free",
        "love", "hate", "fear", "hope", "trust", "doubt",
    ]
    return list(dict.fromkeys(base))[:n]


def make_centroid(model, cases):
    better = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    worse = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    d = better.mean(axis=0) - worse.mean(axis=0)
    return d / (norm(d) + 1e-12)


def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_battery()
    words = get_word_list(10000)
    print(f"Battery: {len(battery)} pairs")
    print(f"Word list: {len(words)} words")

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        direction = make_centroid(model, battery)

        # Embed all words (raw, no User/Assistant framing)
        word_embs = model.encode(words, convert_to_numpy=True, batch_size=64,
                                 show_progress_bar=True)

        cosines = []
        for i, w in enumerate(words):
            emb = word_embs[i]
            cos = float(np.dot(emb, direction) / (norm(emb) * norm(direction) + 1e-12))
            cosines.append((w, cos))

        # Sort by absolute cosine
        cosines.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\nTop 50 words by |cosine| with quality direction:")
        print(f"{'Word':20s} {'Cosine':>8s}")
        print("-" * 30)
        flagged = []
        for w, cos in cosines[:50]:
            flag = " ***" if abs(cos) > 0.20 else ""
            print(f"  {w:20s} {cos:+.4f}{flag}")
            if abs(cos) > 0.20:
                flagged.append((w, cos))

        if flagged:
            print(f"\n  WARNING: {len(flagged)} words with |cosine| > 0.20:")
            for w, cos in flagged:
                print(f"    {w}: {cos:+.4f}")
        else:
            print(f"\n  No words with |cosine| > 0.20")

        # Statistics
        abs_cos = [abs(c) for _, c in cosines]
        print(f"\n  Mean |cosine|: {np.mean(abs_cos):.4f}")
        print(f"  Max |cosine|: {np.max(abs_cos):.4f}")
        print(f"  Median |cosine|: {np.median(abs_cos):.4f}")
        print(f"  Words with |cosine| > 0.10: {sum(1 for c in abs_cos if c > 0.10)}")
        print(f"  Words with |cosine| > 0.15: {sum(1 for c in abs_cos if c > 0.15)}")
        print(f"  Words with |cosine| > 0.20: {sum(1 for c in abs_cos if c > 0.20)}")

        results[model_name] = {
            "n_words": len(words),
            "top_50": [{"word": w, "cosine": round(cos, 6)} for w, cos in cosines[:50]],
            "flagged_above_020": [{"word": w, "cosine": round(cos, 6)} for w, cos in flagged],
            "mean_abs_cosine": round(float(np.mean(abs_cos)), 6),
            "max_abs_cosine": round(float(np.max(abs_cos)), 6),
            "median_abs_cosine": round(float(np.median(abs_cos)), 6),
            "count_above_010": sum(1 for c in abs_cos if c > 0.10),
            "count_above_015": sum(1 for c in abs_cos if c > 0.15),
            "count_above_020": sum(1 for c in abs_cos if c > 0.20),
        }

        del model
        gc.collect()

    out_path = OUT_DIR / "vocabulary_projection_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
