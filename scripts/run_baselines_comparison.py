#!/usr/bin/env python3
"""Compare evaluative axis scoring against simple baselines.

Baselines tested:
1. Prompt-response cosine similarity (relevance)
2. Response length (word count)
3. Response embedding norm
4. Cosine similarity to "This is a good response"
5. Cosine similarity to "This is helpful"

If any baseline matches the evaluative axis, the axis isn't adding value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def main():
    from sentence_transformers import SentenceTransformer

    cases = read_jsonl(DEFAULT_BATTERY)
    print(f"Loaded {len(cases)} cases")

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed responses with prompt context
        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        # Embed prompts alone
        prompt_embs = embed_fn([c["prompt"] for c in cases])

        # Embed reference phrases
        ref_phrases = embed_fn([
            "This is a good response",
            "This is helpful",
            "This is a high quality answer",
        ])

        model_results = {}

        # Baseline 1: Prompt-response cosine similarity
        correct = 0
        for i in range(len(cases)):
            sim_better = cosine_sim(better_embs[i], prompt_embs[i])
            sim_worse = cosine_sim(worse_embs[i], prompt_embs[i])
            if sim_better > sim_worse:
                correct += 1
            elif sim_better == sim_worse:
                correct += 0.5
        model_results["prompt_cosine"] = round(correct / len(cases), 4)

        # Baseline 2: Response length (word count of response only)
        correct = 0
        for c in cases:
            len_b = len(c["better"].split())
            len_w = len(c["worse"].split())
            if len_b > len_w:
                correct += 1
            elif len_b == len_w:
                correct += 0.5
        model_results["length_bias"] = round(correct / len(cases), 4)

        # Baseline 3: Embedding norm
        correct = 0
        for i in range(len(cases)):
            norm_b = float(np.linalg.norm(better_embs[i]))
            norm_w = float(np.linalg.norm(worse_embs[i]))
            if norm_b > norm_w:
                correct += 1
            elif norm_b == norm_w:
                correct += 0.5
        model_results["embedding_norm"] = round(correct / len(cases), 4)

        # Baseline 4-6: Cosine to reference phrases
        for j, label in enumerate(["cos_good_response", "cos_helpful", "cos_high_quality"]):
            correct = 0
            for i in range(len(cases)):
                sim_b = cosine_sim(better_embs[i], ref_phrases[j])
                sim_w = cosine_sim(worse_embs[i], ref_phrases[j])
                if sim_b > sim_w:
                    correct += 1
                elif sim_b == sim_w:
                    correct += 0.5
            model_results[label] = round(correct / len(cases), 4)

        # Axis-based scoring
        for axis_name, anchors in ML_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            correct = 0
            for i in range(len(cases)):
                s_b = float(np.dot(better_embs[i], axis_vec))
                s_w = float(np.dot(worse_embs[i], axis_vec))
                if s_b > s_w:
                    correct += 1
                elif s_b == s_w:
                    correct += 0.5
            model_results[f"axis/{axis_name}"] = round(correct / len(cases), 4)

        # Best single word
        careful_axis = compute_axis(embed_fn, ["Careful"], ["Reckless"])
        correct = 0
        for i in range(len(cases)):
            s_b = float(np.dot(better_embs[i], careful_axis))
            s_w = float(np.dot(worse_embs[i], careful_axis))
            if s_b > s_w:
                correct += 1
            elif s_b == s_w:
                correct += 0.5
        model_results["axis/careful_reckless"] = round(correct / len(cases), 4)

        all_results[model_name] = model_results

        # Print
        print(f"\n{'Method':<35s} {'Accuracy':>8s}")
        print("-" * 45)
        for method, acc in sorted(model_results.items(), key=lambda x: -x[1]):
            marker = " ***" if acc > 0.55 else (" **" if acc > 0.50 else "")
            print(f"  {method:<33s} {acc:7.0%}{marker}")

        del model

    out_dir = ROOT / "notes" / "research_cycles" / "baselines_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
