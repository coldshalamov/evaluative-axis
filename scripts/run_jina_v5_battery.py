#!/usr/bin/env python3
"""Test Jina Embeddings v5 on the balanced battery.

Jina v5 is an API-based model (like Gemini). This tells us whether it
performs like Gemini (86-98%) or like local models (50-74%), which helps
answer the question of whether the frontier gap is Gemini-specific or
an API-model advantage.

Also tests with and without task parameter, and at different dimensions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

load_dotenv(ROOT / ".env.local")

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL = "jina-embeddings-v5-text-small"

BATTERY_ORIGINAL = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BATTERY_WARMTH = (
    ROOT / "notes" / "research_cycles" / "battery_rebalancing"
    / "warmth_cases.jsonl"
)

SINGLE_WORD_AXES = {
    "good":     {"positive": ["Good"],     "negative": ["Bad"]},
    "careful":  {"positive": ["Careful"],   "negative": ["Reckless"]},
    "honest":   {"positive": ["Honest"],    "negative": ["Dishonest"]},
    "kind":     {"positive": ["Kind"],      "negative": ["Cruel"]},
    "thorough": {"positive": ["Thorough"],  "negative": ["Superficial"]},
    "hard":     {"positive": ["Hard"],      "negative": ["Soft"]},
    "wise":     {"positive": ["Wise"],      "negative": ["Foolish"]},
    "fair":     {"positive": ["Fair"],      "negative": ["Unfair"]},
    "helpful":  {"positive": ["Helpful"],   "negative": ["Unhelpful"]},
    "clear":    {"positive": ["Clear"],     "negative": ["Confusing"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def jina_embed(texts, task=None, dimensions=None, batch_size=25):
    """Embed texts via Jina API. Returns numpy array of embeddings."""
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        payload = {
            "model": MODEL,
            "normalized": True,
            "input": batch,
        }
        if task:
            payload["task"] = task
        if dimensions:
            payload["dimensions"] = dimensions

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        }

        for attempt in range(3):
            resp = requests.post(JINA_URL, json=payload, headers=headers,
                                 timeout=60)
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Jina API failed after 3 attempts: {resp.status_code}")

        data = resp.json()
        batch_embs = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embs)

        if start + batch_size < len(texts):
            time.sleep(0.5)

    return np.array(all_embeddings)


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, axis):
    correct = 0
    margins = []
    for i in range(len(better_embs)):
        sb = float(np.dot(better_embs[i], axis))
        sw = float(np.dot(worse_embs[i], axis))
        margin = sb - sw
        margins.append(margin)
        if sb > sw:
            correct += 1
        elif sb == sw:
            correct += 0.5
    return correct / len(better_embs), margins


def run_battery(embed_fn, label, original, warmth):
    """Run all axes on both battery splits. Returns results dict."""
    print(f"\n{'='*70}")
    print(f"CONFIG: {label}")
    print(f"{'='*70}")

    # Embed responses
    print("  Embedding original battery...")
    orig_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}"
                            for c in original])
    orig_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}"
                           for c in original])

    print("  Embedding warmth battery...")
    warm_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}"
                            for c in warmth])
    warm_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}"
                           for c in warmth])

    comb_better = np.vstack([orig_better, warm_better])
    comb_worse = np.vstack([orig_worse, warm_worse])

    results = {}

    # Single-word axes
    print(f"\n  {'Axis':<20s} {'Original':>10s} {'Warmth':>10s} {'Combined':>10s}")
    print(f"  {'-'*55}")

    print("  --- Single-word axes ---")
    for name, anchors in SINGLE_WORD_AXES.items():
        print(f"  Computing axis: {name}...")
        ax = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
        orig_acc, _ = pairwise_accuracy(orig_better, orig_worse, ax)
        warm_acc, _ = pairwise_accuracy(warm_better, warm_worse, ax)
        comb_acc, _ = pairwise_accuracy(comb_better, comb_worse, ax)
        print(f"  {name:<20s} {orig_acc:>9.0%} {warm_acc:>9.0%} {comb_acc:>9.0%}")
        results[name] = {
            "original": round(orig_acc, 4),
            "warmth": round(warm_acc, 4),
            "combined": round(comb_acc, 4),
        }

    # ML-jargon axes
    print("\n  --- ML-jargon axes ---")
    for name, anchors in ML_AXES.items():
        print(f"  Computing axis: ml/{name}...")
        ax = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
        orig_acc, _ = pairwise_accuracy(orig_better, orig_worse, ax)
        warm_acc, _ = pairwise_accuracy(warm_better, warm_worse, ax)
        comb_acc, _ = pairwise_accuracy(comb_better, comb_worse, ax)
        print(f"  ml/{name:<17s} {orig_acc:>9.0%} {warm_acc:>9.0%} {comb_acc:>9.0%}")
        results[f"ml/{name}"] = {
            "original": round(orig_acc, 4),
            "warmth": round(warm_acc, 4),
            "combined": round(comb_acc, 4),
        }

    # Sum of top single-word axes
    print("\n  --- Sum-of-projections (top axes) ---")
    for combo in [
        ("careful", "kind"),
        ("careful", "honest", "kind"),
        ("careful", "honest", "kind", "thorough"),
        ("careful", "honest", "kind", "thorough", "fair"),
    ]:
        axes_list = []
        for term in combo:
            anchors = SINGLE_WORD_AXES[term]
            axes_list.append(compute_axis(embed_fn, anchors["positive"],
                                          anchors["negative"]))
        # Sum of projections
        correct_orig = correct_warm = correct_comb = 0
        for i in range(len(original)):
            sb = sum(float(np.dot(orig_better[i], ax)) for ax in axes_list)
            sw = sum(float(np.dot(orig_worse[i], ax)) for ax in axes_list)
            correct_orig += 1 if sb > sw else (0.5 if sb == sw else 0)
        for i in range(len(warmth)):
            sb = sum(float(np.dot(warm_better[i], ax)) for ax in axes_list)
            sw = sum(float(np.dot(warm_worse[i], ax)) for ax in axes_list)
            correct_warm += 1 if sb > sw else (0.5 if sb == sw else 0)
        for i in range(len(original) + len(warmth)):
            sb = sum(float(np.dot(comb_better[i], ax)) for ax in axes_list)
            sw = sum(float(np.dot(comb_worse[i], ax)) for ax in axes_list)
            correct_comb += 1 if sb > sw else (0.5 if sb == sw else 0)

        orig_acc = correct_orig / len(original)
        warm_acc = correct_warm / len(warmth)
        comb_acc = correct_comb / (len(original) + len(warmth))

        combo_str = "+".join(combo)
        print(f"  sum({combo_str})")
        print(f"  {'':20s} {orig_acc:>9.0%} {warm_acc:>9.0%} {comb_acc:>9.0%}")
        results[f"sum/{combo_str}"] = {
            "original": round(orig_acc, 4),
            "warmth": round(warm_acc, 4),
            "combined": round(comb_acc, 4),
        }

    return results


def main():
    if not JINA_API_KEY:
        print("ERROR: JINA_API_KEY not set in .env.local")
        sys.exit(1)

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    print(f"Loaded {len(original)} original + {len(warmth)} warmth cases")

    all_results = {}

    # Test 1: Default (no task parameter)
    def embed_default(texts):
        return jina_embed(texts)
    all_results["default"] = run_battery(embed_default, "Jina v5 (no task)",
                                          original, warmth)

    # Test 2: With text-matching task
    def embed_matching(texts):
        return jina_embed(texts, task="text-matching")
    all_results["text_matching"] = run_battery(embed_matching,
                                                "Jina v5 (text-matching)",
                                                original, warmth)

    # Test 3: With classification task
    def embed_classify(texts):
        return jina_embed(texts, task="classification")
    all_results["classification"] = run_battery(embed_classify,
                                                 "Jina v5 (classification)",
                                                 original, warmth)

    # Save results
    out_dir = ROOT / "notes" / "research_cycles" / "jina_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jina_v5_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Jina v5 vs local models vs Gemini")
    print(f"{'='*70}")
    print("(Local models from battery rebalancing: Snowflake/BGE-M3/Nomic)")
    print("(Gemini from prior experiments: 86-98% on targeted axes)")
    print()
    for config, res in all_results.items():
        print(f"Config: {config}")
        for axis_name in ["good", "careful", "honest", "kind", "thorough"]:
            if axis_name in res:
                r = res[axis_name]
                print(f"  {axis_name:<15s}: orig={r['original']:.0%}  "
                      f"warm={r['warmth']:.0%}  comb={r['combined']:.0%}")
        for axis_name in ["ml/anti_sycophancy", "ml/general_evaluative",
                          "ml/truthfulness"]:
            if axis_name in res:
                r = res[axis_name]
                print(f"  {axis_name:<15s}: orig={r['original']:.0%}  "
                      f"warm={r['warmth']:.0%}  comb={r['combined']:.0%}")
        print()


if __name__ == "__main__":
    main()
