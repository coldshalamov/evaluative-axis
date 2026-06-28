#!/usr/bin/env python3
"""Vocabulary Depth Experiment: NSM-backed universal terms vs ML-jargon anchors.

Tests three anchor framings against the frozen 50-case battery:
  1. Single universal word pairs (e.g., "Honest" / "Dishonest")
  2. Character projection phrases (e.g., "An honest person said this")
  3. Current multi-sentence ML-jargon anchors (baseline)

Hypothesis: culturally universal terms with deep training-data
representation will produce stronger, more robust evaluative axes
than domain-specific ML vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cycle001_intervention import AXES as ML_AXES, CATEGORY_AXIS_MAP

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

# --- NSM-backed universal word pairs ---
# Selected based on: cross-linguistic universality (NSM semantic primes),
# temporal stability, frequency in training corpora, and polarity clarity.

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
    "sincere":        {"positive": ["Sincere"],     "negative": ["Insincere"]},
    "careful":        {"positive": ["Careful"],     "negative": ["Reckless"]},
    "strong":         {"positive": ["Strong"],      "negative": ["Weak"]},
    "accurate":       {"positive": ["Accurate"],    "negative": ["Inaccurate"]},
    "virtuous":       {"positive": ["Virtuous"],    "negative": ["Wicked"]},
    "trustworthy":    {"positive": ["Trustworthy"], "negative": ["Untrustworthy"]},
}

# --- Character projection framing ---
# Same concepts but framed as character judgments.

CHARACTER_AXES = {
    "good_bad":   {"positive": ["A good person said this"],
                   "negative": ["A bad person said this"]},
    "true_false": {"positive": ["A truthful person said this"],
                   "negative": ["A deceptive person said this"]},
    "honest":     {"positive": ["An honest person said this"],
                   "negative": ["A dishonest person said this"]},
    "just":       {"positive": ["A just person said this"],
                   "negative": ["An unjust person said this"]},
    "fair":       {"positive": ["A fair person said this"],
                   "negative": ["An unfair person said this"]},
    "kind":       {"positive": ["A kind person said this"],
                   "negative": ["A cruel person said this"]},
    "wise":       {"positive": ["A wise person said this"],
                   "negative": ["A foolish person said this"]},
    "useful":     {"positive": ["A useful response to the question"],
                   "negative": ["A useless response to the question"]},
    "helpful":    {"positive": ["A helpful person said this"],
                   "negative": ["A harmful person said this"]},
    "brave":      {"positive": ["A brave person said this"],
                   "negative": ["A cowardly person said this"]},
    "clear":      {"positive": ["A clear answer to the question"],
                   "negative": ["A confused answer to the question"]},
    "capable":    {"positive": ["A capable person said this"],
                   "negative": ["An incapable person said this"]},
    "right_wrong": {"positive": ["The right thing to say"],
                    "negative": ["The wrong thing to say"]},
    "noble":      {"positive": ["A noble person said this"],
                   "negative": ["A base person said this"]},
    "sincere":    {"positive": ["A sincere person said this"],
                   "negative": ["An insincere person said this"]},
    "careful":    {"positive": ["A careful person said this"],
                   "negative": ["A reckless person said this"]},
    "strong":     {"positive": ["A strong answer to the question"],
                   "negative": ["A weak answer to the question"]},
    "accurate":   {"positive": ["An accurate response"],
                   "negative": ["An inaccurate response"]},
    "virtuous":   {"positive": ["A virtuous person said this"],
                   "negative": ["A wicked person said this"]},
    "trustworthy": {"positive": ["A trustworthy person said this"],
                    "negative": ["An untrustworthy person said this"]},
}

# --- Synonym cluster anchors (thicker representation) ---

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
    "brave":      {"positive": ["Brave", "Courageous", "Bold"],
                   "negative": ["Cowardly", "Timid", "Craven"]},
    "clear":      {"positive": ["Clear", "Lucid", "Direct"],
                   "negative": ["Confused", "Vague", "Evasive"]},
}

DEFAULT_MODELS = [
    "snowflake/snowflake-arctic-embed-m",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    print(f"Loading {model_name} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, dim={model.get_embedding_dimension()}")

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    return embed, model_name, model.get_embedding_dimension()


def compute_axis(embed, positive: list[str], negative: list[str]) -> np.ndarray:
    pos_embs = embed(positive)
    neg_embs = embed(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


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


def run_experiment(model_name: str, cases: list, output_dir: Path):
    embed, name, dim = get_embedder(model_name)

    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]

    print("Embedding battery responses ...")
    better_embs = embed(better_texts)
    worse_embs = embed(worse_texts)

    results = {"model": model_name, "dim": dim, "n_cases": len(cases)}

    # --- 1. Current ML-jargon anchors (baseline) ---
    print("\n=== ML-JARGON ANCHORS (current) ===")
    ml_results = {}
    for axis_name, anchors in ML_AXES.items():
        axis_vec = compute_axis(embed, anchors["positive"], anchors["negative"])
        acc = score_battery(better_embs, worse_embs, axis_vec)
        ml_results[axis_name] = round(acc, 4)
        print(f"  {axis_name:25s}: {acc:.0%}")
    results["ml_jargon"] = ml_results

    # --- 2. Single universal word pairs ---
    print("\n=== SINGLE WORD PAIRS (NSM-backed) ===")
    single_results = {}
    for axis_name, anchors in SINGLE_WORD_AXES.items():
        axis_vec = compute_axis(embed, anchors["positive"], anchors["negative"])
        acc = score_battery(better_embs, worse_embs, axis_vec)
        single_results[axis_name] = round(acc, 4)
        print(f"  {axis_name:25s}: {acc:.0%}")
    results["single_word"] = single_results

    # --- 3. Character projection ---
    print("\n=== CHARACTER PROJECTION ===")
    char_results = {}
    for axis_name, anchors in CHARACTER_AXES.items():
        axis_vec = compute_axis(embed, anchors["positive"], anchors["negative"])
        acc = score_battery(better_embs, worse_embs, axis_vec)
        char_results[axis_name] = round(acc, 4)
        print(f"  {axis_name:25s}: {acc:.0%}")
    results["character_projection"] = char_results

    # --- 4. Synonym clusters ---
    print("\n=== SYNONYM CLUSTERS ===")
    cluster_results = {}
    for axis_name, anchors in CLUSTER_AXES.items():
        axis_vec = compute_axis(embed, anchors["positive"], anchors["negative"])
        acc = score_battery(better_embs, worse_embs, axis_vec)
        cluster_results[axis_name] = round(acc, 4)
        print(f"  {axis_name:25s}: {acc:.0%}")
    results["synonym_cluster"] = cluster_results

    # --- 5. Best-of across all universal terms ---
    all_universal = {}
    for name_set, src in [("single_word", single_results),
                          ("character_projection", char_results),
                          ("synonym_cluster", cluster_results)]:
        for ax, acc in src.items():
            key = f"{name_set}/{ax}"
            all_universal[key] = acc

    best_universal = max(all_universal.items(), key=lambda x: x[1])
    best_ml = max(ml_results.items(), key=lambda x: x[1])
    results["best_universal"] = {"axis": best_universal[0], "accuracy": best_universal[1]}
    results["best_ml_jargon"] = {"axis": best_ml[0], "accuracy": best_ml[1]}

    # --- 6. Top-N combined score (best N universal axes averaged) ---
    for n in [3, 5, 7, 10]:
        top_n = sorted(all_universal.values(), reverse=True)[:n]
        results[f"top_{n}_universal_mean"] = round(sum(top_n) / len(top_n), 4)

    top_ml = sorted(ml_results.values(), reverse=True)
    results["top_3_ml_mean"] = round(sum(top_ml[:3]) / 3, 4)
    results["top_5_ml_mean"] = round(sum(top_ml[:5]) / 5, 4)

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "vocabulary_depth_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best ML-jargon axis:  {best_ml[0]:30s} = {best_ml[1]:.0%}")
    print(f"Best universal axis:  {best_universal[0]:30s} = {best_universal[1]:.0%}")
    print(f"Top-5 ML mean:        {results['top_5_ml_mean']:.0%}")
    print(f"Top-5 universal mean: {results['top_5_universal_mean']:.0%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Vocabulary depth experiment")
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--model", type=str, default=DEFAULT_MODELS[0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes" / "research_cycles"
                        / "vocabulary_depth_experiment")
    args = parser.parse_args()

    cases = read_jsonl(args.battery)
    print(f"Loaded {len(cases)} cases from {args.battery.name}")

    run_experiment(args.model, cases, args.output)


if __name__ == "__main__":
    main()
