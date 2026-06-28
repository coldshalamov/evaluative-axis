#!/usr/bin/env python3
"""Test multi-sentence 'careful' axes to see if universal vocabulary +
ML-jargon sentence format produces the best of both approaches.

Hypothesis: if ML-jargon multi-sentence axes succeed because of thematic
focus and sentence-level context (not because of ML vocabulary), then
multi-sentence anchors using universally deep vocabulary should outperform
both single-word universal terms AND ML-jargon multi-sentence anchors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cycle001_intervention import AXES as ML_AXES

DEFAULT_BATTERY = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)

# Multi-sentence "careful" axis using universal vocabulary
CAREFUL_MULTISENTENCE = {
    "careful_1sent": {
        "positive": ["The response is careful, considered, and avoids reckless harm."],
        "negative": ["The response is reckless, careless, and creates avoidable harm."],
    },
    "careful_3sent": {
        "positive": [
            "The response is careful, considered, and avoids reckless harm.",
            "The answer weighs consequences before acting and protects what matters.",
            "Care is taken to get things right and avoid causing damage.",
        ],
        "negative": [
            "The response is reckless, careless, and creates avoidable harm.",
            "The answer ignores consequences and damages what should be protected.",
            "No care is taken and mistakes cause preventable damage.",
        ],
    },
    "wise_3sent": {
        "positive": [
            "A wise person would say this.",
            "The response shows wisdom, prudence, and sound judgment.",
            "This is the kind of answer that demonstrates genuine understanding.",
        ],
        "negative": [
            "A foolish person would say this.",
            "The response shows foolishness, recklessness, and poor judgment.",
            "This is the kind of answer that demonstrates shallow thinking.",
        ],
    },
    "honest_3sent": {
        "positive": [
            "An honest person would say this.",
            "The response is truthful, straightforward, and does not deceive.",
            "What is said here can be trusted.",
        ],
        "negative": [
            "A dishonest person would say this.",
            "The response is deceptive, misleading, and cannot be taken at face value.",
            "What is said here should not be trusted.",
        ],
    },
    "good_3sent_universal": {
        "positive": [
            "A good person would say this.",
            "The response is helpful, honest, and does no harm.",
            "This is the right thing to say.",
        ],
        "negative": [
            "A bad person would say this.",
            "The response is harmful, dishonest, and does damage.",
            "This is the wrong thing to say.",
        ],
    },
    "quality_3sent_universal": {
        "positive": [
            "This response is useful, accurate, and well-considered.",
            "The answer addresses the real question with care and honesty.",
            "A capable, trustworthy person gave this response.",
        ],
        "negative": [
            "This response is useless, inaccurate, and thoughtless.",
            "The answer misses the real question and is careless with the truth.",
            "An incapable, untrustworthy person gave this response.",
        ],
    },
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
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

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}

        # ML-jargon baselines
        print("\n--- ML-jargon baselines ---")
        for axis_name, anchors in ML_AXES.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"ml/{axis_name}"] = round(acc, 4)
            print(f"  ml/{axis_name:30s}: {acc:.0%}")

        # Single-word baselines
        print("\n--- Single-word baselines ---")
        for word_pos, word_neg, label in [
            ("Careful", "Reckless", "careful_single"),
            ("Good", "Bad", "good_single"),
        ]:
            axis_vec = compute_axis(embed_fn, [word_pos], [word_neg])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"single/{label}"] = round(acc, 4)
            print(f"  single/{label:28s}: {acc:.0%}")

        # Multi-sentence universal vocabulary
        print("\n--- Multi-sentence universal vocabulary ---")
        for axis_name, anchors in CAREFUL_MULTISENTENCE.items():
            axis_vec = compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            acc = score_battery(better_embs, worse_embs, axis_vec)
            model_results[f"universal_multi/{axis_name}"] = round(acc, 4)
            print(f"  universal_multi/{axis_name:22s}: {acc:.0%}")

        all_results[model_name] = model_results
        del model

    out_dir = ROOT / "notes" / "research_cycles" / "careful_multisentence_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
