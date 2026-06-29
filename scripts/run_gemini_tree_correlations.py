#!/usr/bin/env python3
"""Tree decomposition correlations on Gemini Embedding.

Tests whether "careful is independent of good while other children correlate
strongly" replicates at frontier scale. Also tests accuracy of all L1 terms.
"""

import json, os, time, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local")

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

L1_TERMS = {
    "good":        ("Good",        "Bad"),
    "careful":     ("Careful",     "Reckless"),
    "honest":      ("Honest",      "Dishonest"),
    "kind":        ("Kind",        "Cruel"),
    "wise":        ("Wise",        "Foolish"),
    "helpful":     ("Helpful",     "Unhelpful"),
    "thorough":    ("Thorough",    "Superficial"),
    "fair":        ("Fair",        "Unfair"),
    "responsible": ("Responsible", "Irresponsible"),
    "clear":       ("Clear",      "Confusing"),
    "respectful":  ("Respectful", "Disrespectful"),
}

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

def embed_gemini(texts, batch_size=20, delay=4.0, max_retries=3):
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                )
                results.extend([np.array(e.values) for e in resp.embeddings])
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        if i + batch_size < len(texts):
            time.sleep(delay)
        print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.array(results)

def main():
    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    all_cases = original + warmth
    n = len(all_cases)

    print(f"Testing {n} cases on Gemini Embedding")

    # Embed all responses
    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases]

    print("Embedding better responses...")
    better_embs = embed_gemini(better_texts)
    print("Embedding worse responses...")
    worse_embs = embed_gemini(worse_texts)

    # Embed all anchor words
    all_pos = [info[0] for info in L1_TERMS.values()]
    all_neg = [info[1] for info in L1_TERMS.values()]
    print("Embedding anchors...")
    pos_embs = embed_gemini(all_pos, delay=0.5)
    neg_embs = embed_gemini(all_neg, delay=0.5)

    # Build axes and score
    deltas = {}
    axes = {}
    for idx, (name, (pos_word, neg_word)) in enumerate(L1_TERMS.items()):
        axis = (pos_embs[idx] - neg_embs[idx])
        axis = axis / (norm(axis) + 1e-12)
        axes[name] = axis
        d = []
        for i in range(n):
            sb = float(np.dot(better_embs[i], axis))
            sw = float(np.dot(worse_embs[i], axis))
            d.append(sb - sw)
        deltas[name] = np.array(d)

    # Per-term accuracy
    print(f"\n{'='*60}")
    print("GEMINI EMBEDDING — TREE DECOMPOSITION L1")
    print(f"{'='*60}")

    print(f"\n--- Per-term accuracy (n={n}) ---")
    for name in ["good"] + sorted([k for k in L1_TERMS if k != "good"]):
        correct = [1 if d > 0 else 0 for d in deltas[name]]
        acc = np.mean(correct)
        k = sum(correct)
        lo, hi = wilson_ci(k, n)
        sig = "YES" if lo > 0.5 else "no"
        print(f"  {name:12s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  {sig}")

    # Good -> L1 child correlations
    print(f"\n--- Good -> L1 child score-delta correlations ---")
    children = [k for k in L1_TERMS if k != "good"]
    for child in sorted(children):
        r = np.corrcoef(deltas["good"], deltas[child])[0, 1]
        t_stat = r * math.sqrt((n - 2) / (1 - r**2 + 1e-12))
        print(f"  good -> {child:12s}: r={r:+.3f}  t={t_stat:+.2f}")

    # Axis vector cosines
    print(f"\n--- Axis vector cosines (good vs child) ---")
    for child in sorted(children):
        cos = float(np.dot(axes["good"], axes[child]))
        print(f"  good vs {child:12s}: cos={cos:+.3f}")

    # Orig vs warmth split
    print(f"\n--- Orig/Warmth split ---")
    n_orig = len(original)
    for name in ["good", "careful", "thorough", "kind", "honest"]:
        orig_acc = np.mean([1 if d > 0 else 0 for d in deltas[name][:n_orig]])
        warm_acc = np.mean([1 if d > 0 else 0 for d in deltas[name][n_orig:]])
        print(f"  {name:12s}: orig={orig_acc:.0%}  warmth={warm_acc:.0%}")


if __name__ == "__main__":
    main()
