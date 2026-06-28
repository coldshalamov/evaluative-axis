#!/usr/bin/env python3
"""Vocabulary depth experiment on Gemini embeddings.

Tests NSM-backed universal terms, character projection, and synonym
clusters against the current ML-jargon anchors on Gemini.
Estimated API calls: ~270 (well under 1000 daily free-tier limit).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES as ML_AXES
from run_gemini_rerun import GeminiEmbedder, load_local_env, get_api_key

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

SINGLE_WORD_AXES = {
    "good_bad":       {"positive": ["Good"],        "negative": ["Bad"]},
    "true_false":     {"positive": ["True"],        "negative": ["False"]},
    "honest":         {"positive": ["Honest"],      "negative": ["Dishonest"]},
    "just":           {"positive": ["Just"],        "negative": ["Unjust"]},
    "fair":           {"positive": ["Fair"],        "negative": ["Unfair"]},
    "kind":           {"positive": ["Kind"],        "negative": ["Cruel"]},
    "wise":           {"positive": ["Wise"],        "negative": ["Foolish"]},
    "useful":         {"positive": ["Useful"],      "negative": ["Useless"]},
    "helpful":        {"positive": ["Helpful"],     "negative": ["Harmful"]},
    "brave":          {"positive": ["Brave"],       "negative": ["Cowardly"]},
    "clear":          {"positive": ["Clear"],       "negative": ["Confused"]},
    "capable":        {"positive": ["Capable"],     "negative": ["Incapable"]},
    "right_wrong":    {"positive": ["Right"],       "negative": ["Wrong"]},
    "noble":          {"positive": ["Noble"],       "negative": ["Base"]},
    "careful":        {"positive": ["Careful"],     "negative": ["Reckless"]},
    "accurate":       {"positive": ["Accurate"],    "negative": ["Inaccurate"]},
    "virtuous":       {"positive": ["Virtuous"],    "negative": ["Wicked"]},
    "trustworthy":    {"positive": ["Trustworthy"], "negative": ["Untrustworthy"]},
}

CHARACTER_AXES = {
    "good_bad":    {"positive": ["A good person said this"],
                    "negative": ["A bad person said this"]},
    "true_false":  {"positive": ["A truthful person said this"],
                    "negative": ["A deceptive person said this"]},
    "honest":      {"positive": ["An honest person said this"],
                    "negative": ["A dishonest person said this"]},
    "just":        {"positive": ["A just person said this"],
                    "negative": ["An unjust person said this"]},
    "fair":        {"positive": ["A fair person said this"],
                    "negative": ["An unfair person said this"]},
    "kind":        {"positive": ["A kind person said this"],
                    "negative": ["A cruel person said this"]},
    "wise":        {"positive": ["A wise person said this"],
                    "negative": ["A foolish person said this"]},
    "useful":      {"positive": ["A useful response to the question"],
                    "negative": ["A useless response to the question"]},
    "helpful":     {"positive": ["A helpful person said this"],
                    "negative": ["A harmful person said this"]},
    "brave":       {"positive": ["A brave person said this"],
                    "negative": ["A cowardly person said this"]},
    "clear":       {"positive": ["A clear answer to the question"],
                    "negative": ["A confused answer to the question"]},
    "capable":     {"positive": ["A capable person said this"],
                    "negative": ["An incapable person said this"]},
    "right_wrong": {"positive": ["The right thing to say"],
                    "negative": ["The wrong thing to say"]},
    "careful":     {"positive": ["A careful person said this"],
                    "negative": ["A reckless person said this"]},
    "accurate":    {"positive": ["An accurate response"],
                    "negative": ["An inaccurate response"]},
    "virtuous":    {"positive": ["A virtuous person said this"],
                    "negative": ["A wicked person said this"]},
    "trustworthy": {"positive": ["A trustworthy person said this"],
                    "negative": ["An untrustworthy person said this"]},
    "wise_short":  {"positive": ["This is wise"],
                    "negative": ["This is foolish"]},
}

CLUSTER_AXES = {
    "good_bad":   {"positive": ["Good", "Excellent", "Admirable"],
                   "negative": ["Bad", "Terrible", "Contemptible"]},
    "true_false": {"positive": ["True", "Accurate", "Correct"],
                   "negative": ["False", "Inaccurate", "Incorrect"]},
    "honest":     {"positive": ["Honest", "Truthful", "Sincere"],
                   "negative": ["Dishonest", "Deceptive", "Insincere"]},
    "just":       {"positive": ["Just", "Fair", "Equitable"],
                   "negative": ["Unjust", "Unfair", "Biased"]},
    "kind":       {"positive": ["Kind", "Compassionate", "Benevolent"],
                   "negative": ["Cruel", "Heartless", "Malevolent"]},
    "wise":       {"positive": ["Wise", "Prudent", "Discerning"],
                   "negative": ["Foolish", "Reckless", "Obtuse"]},
    "useful":     {"positive": ["Useful", "Practical", "Effective"],
                   "negative": ["Useless", "Impractical", "Ineffective"]},
    "helpful":    {"positive": ["Helpful", "Beneficial", "Constructive"],
                   "negative": ["Harmful", "Destructive", "Detrimental"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis_from_embs(pos_embs, neg_embs):
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return axis
    return axis / norm


def score_battery(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        s_better = float(np.dot(better_embs[i], axis))
        s_worse = float(np.dot(worse_embs[i], axis))
        if s_better > s_worse:
            correct += 1
        elif s_better == s_worse:
            correct += 0.5
    return correct / len(better_embs)


def main():
    load_local_env()
    api_key, source = get_api_key()
    print(f"API key from: {source}")

    embedder = GeminiEmbedder(api_key, model=None,
                               max_workers=1, batch_size=20, sleep_between=0.2)
    embedder.probe_model()
    print(f"Using model: {embedder.model}")

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")

    # Collect ALL texts to embed in one pass to minimize API calls
    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]

    # Collect all anchor texts
    anchor_labels = []  # (set_name, axis_name, side, index)
    anchor_texts = []

    for axis_name, anchors in ML_AXES.items():
        for t in anchors["positive"]:
            anchor_labels.append(("ml_jargon", axis_name, "positive", len(anchor_texts)))
            anchor_texts.append(t)
        for t in anchors["negative"]:
            anchor_labels.append(("ml_jargon", axis_name, "negative", len(anchor_texts)))
            anchor_texts.append(t)

    for axis_name, anchors in SINGLE_WORD_AXES.items():
        for t in anchors["positive"]:
            anchor_labels.append(("single_word", axis_name, "positive", len(anchor_texts)))
            anchor_texts.append(t)
        for t in anchors["negative"]:
            anchor_labels.append(("single_word", axis_name, "negative", len(anchor_texts)))
            anchor_texts.append(t)

    for axis_name, anchors in CHARACTER_AXES.items():
        for t in anchors["positive"]:
            anchor_labels.append(("character_projection", axis_name, "positive", len(anchor_texts)))
            anchor_texts.append(t)
        for t in anchors["negative"]:
            anchor_labels.append(("character_projection", axis_name, "negative", len(anchor_texts)))
            anchor_texts.append(t)

    for axis_name, anchors in CLUSTER_AXES.items():
        for t in anchors["positive"]:
            anchor_labels.append(("synonym_cluster", axis_name, "positive", len(anchor_texts)))
            anchor_texts.append(t)
        for t in anchors["negative"]:
            anchor_labels.append(("synonym_cluster", axis_name, "negative", len(anchor_texts)))
            anchor_texts.append(t)

    total_calls = len(better_texts) + len(worse_texts) + len(anchor_texts)
    print(f"Total texts to embed: {total_calls}")
    print(f"  Battery: {len(better_texts)} better + {len(worse_texts)} worse")
    print(f"  Anchors: {len(anchor_texts)}")

    # Embed battery
    print("\nEmbedding battery responses ...")
    better_embs = embedder.encode(better_texts, label="better")
    worse_embs = embedder.encode(worse_texts, label="worse")

    # Embed all anchors at once
    print("Embedding anchor texts ...")
    all_anchor_embs = embedder.encode(anchor_texts, label="anchors")

    # Reconstruct per-axis embeddings
    def get_axis_vec(set_name, axis_name):
        pos_indices = [l[3] for l in anchor_labels
                       if l[0] == set_name and l[1] == axis_name and l[2] == "positive"]
        neg_indices = [l[3] for l in anchor_labels
                       if l[0] == set_name and l[1] == axis_name and l[2] == "negative"]
        pos_embs = all_anchor_embs[pos_indices]
        neg_embs = all_anchor_embs[neg_indices]
        return compute_axis_from_embs(pos_embs, neg_embs)

    results = {"model": "gemini/text-embedding-004", "n_cases": len(cases),
               "api_calls_used": total_calls}

    # Score each set
    for set_name, axes_dict, label in [
        ("ml_jargon", ML_AXES, "ML-JARGON ANCHORS (baseline)"),
        ("single_word", SINGLE_WORD_AXES, "SINGLE WORD PAIRS (NSM-backed)"),
        ("character_projection", CHARACTER_AXES, "CHARACTER PROJECTION"),
        ("synonym_cluster", CLUSTER_AXES, "SYNONYM CLUSTERS"),
    ]:
        print(f"\n=== {label} ===")
        set_results = {}
        for axis_name in axes_dict:
            axis_vec = get_axis_vec(set_name, axis_name)
            acc = score_battery(better_embs, worse_embs, axis_vec)
            set_results[axis_name] = round(acc, 4)
            print(f"  {axis_name:25s}: {acc:.0%}")
        results[set_name] = set_results

    # Summary
    ml_results = results["ml_jargon"]
    all_universal = {}
    for name_set in ["single_word", "character_projection", "synonym_cluster"]:
        for ax, acc in results[name_set].items():
            all_universal[f"{name_set}/{ax}"] = acc

    best_universal = max(all_universal.items(), key=lambda x: x[1])
    best_ml = max(ml_results.items(), key=lambda x: x[1])
    results["best_universal"] = {"axis": best_universal[0], "accuracy": best_universal[1]}
    results["best_ml_jargon"] = {"axis": best_ml[0], "accuracy": best_ml[1]}

    for n in [3, 5, 7, 10]:
        top_n = sorted(all_universal.values(), reverse=True)[:n]
        results[f"top_{n}_universal_mean"] = round(sum(top_n) / len(top_n), 4)

    top_ml = sorted(ml_results.values(), reverse=True)
    results["top_3_ml_mean"] = round(sum(top_ml[:3]) / 3, 4)
    results["top_5_ml_mean"] = round(sum(top_ml[:5]) / 5, 4)

    out_dir = ROOT / "notes" / "research_cycles" / "vocabulary_depth_gemini"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vocabulary_depth_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
    print(f"Total API calls: {total_calls}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best ML-jargon axis:  {best_ml[0]:35s} = {best_ml[1]:.0%}")
    print(f"Best universal axis:  {best_universal[0]:35s} = {best_universal[1]:.0%}")
    print(f"Top-5 ML mean:        {results['top_5_ml_mean']:.0%}")
    print(f"Top-5 universal mean: {results['top_5_universal_mean']:.0%}")
    print(f"Top-10 universal mean:{results['top_10_universal_mean']:.0%}")


if __name__ == "__main__":
    main()
