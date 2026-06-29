#!/usr/bin/env python3
"""E-03: Word Length and Phrase Complexity.

Tests the SAME evaluative concept at five lengths to see whether anchor
length matters and whether there is a sweet spot.

For each of four concepts (careful, honest, kind, thorough) we define five
length variants that preserve the evaluative meaning but scale from a single
word to two sentences:

  1_word   : the bare antonym pair
  2_word   : a two-word intensifier pair
  phrase   : a gerund phrase
  clause   : "The response is X"
  sentence : one sentence
  two_sent : two sentences

We score each variant on the balanced battery (50 firmness + 20 warmth),
on all three local models, reporting firmness / warmth / combined.

Rules followed (AGENTS.md / RESEARCH_CONTEXT.md):
  - All three local models.
  - Both battery splits reported separately + combined.
  - User/Assistant framing for response embeddings.
  - Each length variant scored independently (no averaging of variants).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BALANCED_ORIG = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BALANCED_WARMTH = (
    ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

LENGTH_ORDER = ["1_word", "2_word", "phrase", "clause", "sentence", "two_sent"]

# Each concept has 6 length variants. Positives and negatives are lists so we
# can use the same compute_axis (mean-of-anchors) code path.
LENGTH_VARIANTS = {
    "careful": {
        "1_word":   (["Careful"], ["Reckless"]),
        "2_word":   (["Very careful"], ["Very reckless"]),
        "phrase":   (["Being careful"], ["Being reckless"]),
        "clause":   (["The response is careful"], ["The response is reckless"]),
        "sentence": (["This is a careful and considered response"],
                     ["This is a reckless and unconsidered response"]),
        "two_sent": (["This response is careful and considered. It avoids "
                      "reckless or sloppy mistakes."],
                     ["This response is reckless and unconsidered. It makes "
                      "careless and avoidable mistakes."]),
    },
    "honest": {
        "1_word":   (["Honest"], ["Dishonest"]),
        "2_word":   (["Truly honest"], ["Truly dishonest"]),
        "phrase":   (["Being honest"], ["Being dishonest"]),
        "clause":   (["The response is honest"], ["The response is dishonest"]),
        "sentence": (["This response demonstrates genuine honesty"],
                     ["This response demonstrates genuine dishonesty"]),
        "two_sent": (["This response is honest and truthful. It does not "
                      "fabricate or mislead the reader."],
                     ["This response is dishonest and misleading. It "
                      "fabricates claims and deceives the reader."]),
    },
    "kind": {
        "1_word":   (["Kind"], ["Cruel"]),
        "2_word":   (["Genuinely kind"], ["Genuinely cruel"]),
        "phrase":   (["Being kind"], ["Being cruel"]),
        "clause":   (["The response is kind"], ["The response is cruel"]),
        "sentence": (["This response shows genuine kindness and warmth"],
                     ["This response shows genuine cruelty and coldness"]),
        "two_sent": (["This response is kind and warm. It treats the reader "
                      "with genuine compassion and respect."],
                     ["This response is cruel and cold. It treats the reader "
                      "with contempt and dismissiveness."]),
    },
    "thorough": {
        "1_word":   (["Thorough"], ["Superficial"]),
        "2_word":   (["Remarkably thorough"], ["Remarkably superficial"]),
        "phrase":   (["Being thorough"], ["Being superficial"]),
        "clause":   (["The response is thorough"], ["The response is superficial"]),
        "sentence": (["This response is thorough and complete in its coverage"],
                     ["This response is superficial and incomplete in its "
                      "coverage"]),
        "two_sent": (["This response is thorough and complete. It covers the "
                      "relevant details without leaving gaps."],
                     ["This response is superficial and incomplete. It skips "
                      "relevant details and leaves obvious gaps."]),
    },
}


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_axis(embed_fn, positive, negative):
    pos = embed_fn(positive)
    neg = embed_fn(negative)
    axis = pos.mean(axis=0) - neg.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)


def axis_accuracy(better_embs, worse_embs, axis_vec):
    n = len(better_embs)
    correct = 0
    for i in range(n):
        sb = float(np.dot(better_embs[i], axis_vec))
        sw = float(np.dot(worse_embs[i], axis_vec))
        if sb > sw:
            correct += 1
        elif sb == sw:
            correct += 0.5
    return correct / n


def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(BALANCED_ORIG)
    warmth = read_jsonl(BALANCED_WARMTH)
    n_orig = len(orig)
    n_warm = len(warmth)
    print(f"Balanced battery: {n_orig} firmness + {n_warm} warmth")
    print(f"Concepts: {list(LENGTH_VARIANTS.keys())}")
    print(f"Length variants: {LENGTH_ORDER}")

    all_results = {
        "metadata": {
            "concepts": list(LENGTH_VARIANTS.keys()),
            "length_variants": LENGTH_ORDER,
            "n_firmness": n_orig,
            "n_warmth": n_warm,
            "models": MODELS,
        },
        "per_model": {},
    }

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False,
                                convert_to_numpy=True)

        # Embed all response texts once (firmness + warmth)
        all_cases = orig + warmth
        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}"
                        for c in all_cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}"
                       for c in all_cases]
        better_embs = embed_fn(better_texts)
        worse_embs = embed_fn(worse_texts)

        model_results = {}
        for concept, variants in LENGTH_VARIANTS.items():
            print(f"\n  CONCEPT: {concept}")
            print(f"  {'Length':<12s} {'Comb':>6s} {'Firm':>6s} {'Warm':>6s}")
            print("  " + "-" * 34)
            model_results[concept] = {}
            for length in LENGTH_ORDER:
                pos, neg = variants[length]
                axis_vec = compute_axis(embed_fn, pos, neg)

                combined = axis_accuracy(better_embs, worse_embs, axis_vec)
                firm = axis_accuracy(better_embs[:n_orig],
                                     worse_embs[:n_orig], axis_vec)
                warm = axis_accuracy(better_embs[n_orig:],
                                     worse_embs[n_orig:], axis_vec)
                model_results[concept][length] = {
                    "combined": round(combined, 4),
                    "firmness": round(firm, 4),
                    "warmth": round(warm, 4),
                }
                mark = " *" if combined > 0.55 else ""
                print(f"  {length:<12s} {combined:5.0%} {firm:5.0%} "
                      f"{warm:5.0%}{mark}")

        all_results["per_model"][model_name] = model_results
        del model

    out_dir = ROOT / "notes" / "research_cycles" / "word_length"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "length_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # Cross-model: does the length pattern hold across models?
    print(f"\n{'='*70}")
    print("CROSS-MODEL: combined accuracy by length, averaged across concepts")
    print(f"{'='*70}")
    print(f"  {'Length':<12s}", end="")
    for m in MODELS:
        print(f" {m.split('/')[-1]:<16s}", end="")
    print("  mean")
    print("  " + "-" * 70)
    for length in LENGTH_ORDER:
        print(f"  {length:<12s}", end="")
        vals = []
        for m in MODELS:
            accs = [all_results["per_model"][m][c][length]["combined"]
                    for c in LENGTH_VARIANTS]
            mean_c = sum(accs) / len(accs)
            vals.append(mean_c)
            print(f" {mean_c:15.0%}", end="")
        print(f"  {sum(vals)/len(vals):.0%}")

    # Best length per concept per model
    print(f"\nBEST length per concept (by combined accuracy):")
    for concept in LENGTH_VARIANTS:
        line = f"  {concept:<10s}"
        for m in MODELS:
            best = max(LENGTH_ORDER,
                       key=lambda L: all_results["per_model"][m][concept][L]["combined"])
            acc = all_results["per_model"][m][concept][best]["combined"]
            line += f"  {m.split('/')[-1]}={best}@{acc:.0%}"
        print(line)


if __name__ == "__main__":
    main()
