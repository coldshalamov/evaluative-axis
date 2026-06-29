#!/usr/bin/env python3
"""Two tests:

1. SUBTRACTION: Remove warmth from the good/bad direction.
   good_residual = good_direction - projection_onto_warmth
   Does the residual capture quality better than raw good?

2. PRINCIPLED TREE: 5 terms chosen to cover what "good" should mean:
   - careful (competence/rigor)
   - honest (integrity/truth)
   - helpful (utility/addressing needs)
   - thorough (completeness)
   - restrained (discipline/avoiding excess)

   Score each independently. A response is "good" if ANY of these say yes.
   This is the tree functioning as a decomposed good signal.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TREE = [
    ("careful", "Careful", "Reckless"),
    ("honest", "Honest", "Dishonest"),
    ("helpful", "Helpful", "Unhelpful"),
    ("thorough", "Thorough", "Superficial"),
    ("restrained", "Restrained", "Unrestrained"),
]

# Words to subtract from "good" direction
WARMTH_PAIRS = [
    ("Kind", "Cruel"),
    ("Warm", "Cold"),
    ("Friendly", "Hostile"),
    ("Gentle", "Harsh"),
    ("Supportive", "Unsupportive"),
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


def project_out(vec, direction):
    """Remove the component of vec along direction."""
    direction_norm = direction / (norm(direction) + 1e-12)
    projection = np.dot(vec, direction_norm) * direction_norm
    return vec - projection


def main():
    from sentence_transformers import SentenceTransformer

    battery = read_jsonl(BATTERY)
    warmth_cases = read_jsonl(WARMTH)
    all_cases = battery + warmth_cases
    n = len(all_cases)

    labels = []
    for c in battery:
        if c["category"] == "anti_sycophancy":
            labels.append("sycophancy")
        else:
            labels.append("firmness")
    for c in warmth_cases:
        labels.append("warmth")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases]
        better_embs = embed(better_texts)
        worse_embs = embed(worse_texts)

        # === RAW GOOD ===
        good_emb = embed(["Good"])[0]
        bad_emb = embed(["Bad"])[0]
        good_dir = good_emb - bad_emb
        good_dir = good_dir / (norm(good_dir) + 1e-12)

        good_outcomes = [float(np.dot(better_embs[i], good_dir)) > float(np.dot(worse_embs[i], good_dir))
                         for i in range(n)]

        # === WARMTH DIRECTION (average of several warmth pairs) ===
        warmth_dirs = []
        for pos, neg in WARMTH_PAIRS:
            p = embed([pos])[0]
            n_ = embed([neg])[0]
            d = p - n_
            warmth_dirs.append(d / (norm(d) + 1e-12))
        warmth_avg = np.mean(warmth_dirs, axis=0)
        warmth_avg = warmth_avg / (norm(warmth_avg) + 1e-12)

        # How aligned is good with warmth?
        good_warmth_cos = float(np.dot(good_dir, warmth_avg))
        print(f"\n  cos(good_dir, warmth_dir) = {good_warmth_cos:.3f}")

        # === DEBIASED GOOD: subtract warmth from good ===
        good_residual = project_out(good_emb - bad_emb, warmth_avg)
        good_residual = good_residual / (norm(good_residual) + 1e-12)
        print(f"  |residual| / |original| = {norm(good_emb - bad_emb - np.dot(good_emb - bad_emb, warmth_avg / norm(warmth_avg)) * warmth_avg / norm(warmth_avg)) / norm(good_emb - bad_emb):.3f}")

        debiased_outcomes = [float(np.dot(better_embs[i], good_residual)) > float(np.dot(worse_embs[i], good_residual))
                             for i in range(n)]

        # === SUBTRACT MULTIPLE WARMTH DIRECTIONS ===
        # Project out each warmth pair direction independently, then each emotion direction
        multi_subtract = good_emb - bad_emb
        emotion_words = [("Happy", "Sad"), ("Pleasant", "Unpleasant"), ("Positive", "Negative")]
        all_subtract = WARMTH_PAIRS + emotion_words
        for pos, neg in all_subtract:
            p = embed([pos])[0]
            n_ = embed([neg])[0]
            d = p - n_
            multi_subtract = project_out(multi_subtract, d)
        multi_residual = multi_subtract / (norm(multi_subtract) + 1e-12)
        print(f"  |multi_residual| / |original| = {norm(multi_subtract) / norm(good_emb - bad_emb):.3f}")

        multi_outcomes = [float(np.dot(better_embs[i], multi_residual)) > float(np.dot(worse_embs[i], multi_residual))
                          for i in range(n)]

        # === TREE: 5 principled terms ===
        tree_outcomes_per_axis = {}
        for axis_name, pos, neg in TREE:
            p = embed([pos])[0]
            n_ = embed([neg])[0]
            d = (p - n_) / (norm(p - n_) + 1e-12)
            outcomes = [float(np.dot(better_embs[i], d)) > float(np.dot(worse_embs[i], d))
                        for i in range(n)]
            tree_outcomes_per_axis[axis_name] = outcomes

        # Tree ANY (response is "good" if ANY tree term says yes)
        tree_any = [any(tree_outcomes_per_axis[a][i] for a in tree_outcomes_per_axis) for i in range(n)]
        # Tree MAJORITY (3 of 5 say yes)
        tree_maj = [sum(tree_outcomes_per_axis[a][i] for a in tree_outcomes_per_axis) >= 3 for i in range(n)]
        # Tree 2-of-5
        tree_2of5 = [sum(tree_outcomes_per_axis[a][i] for a in tree_outcomes_per_axis) >= 2 for i in range(n)]

        # === REPORT ===
        def report(name, outcomes):
            acc = sum(outcomes) / n
            firm_idx = [i for i in range(n) if labels[i] == "firmness"]
            warm_idx = [i for i in range(n) if labels[i] == "warmth"]
            syc_idx = [i for i in range(n) if labels[i] == "sycophancy"]
            firm_acc = sum(outcomes[i] for i in firm_idx) / len(firm_idx) if firm_idx else 0
            warm_acc = sum(outcomes[i] for i in warm_idx) / len(warm_idx) if warm_idx else 0
            syc_acc = sum(outcomes[i] for i in syc_idx) / len(syc_idx) if syc_idx else 0
            lo, hi = wilson_ci(sum(outcomes), n)
            print(f"  {name:25s} {acc:5.0%} [{lo:.0%}-{hi:.0%}]  firm={firm_acc:.0%}  warm={warm_acc:.0%}  syc={syc_acc:.0%}")

        print(f"\n  {'Strategy':25s} {'All':>5s} {'CI':>10s}  {'Firm':>5s}  {'Warm':>5s}  {'Syc':>5s}")
        print(f"  {'-'*70}")
        report("raw good/bad", good_outcomes)
        report("good - warmth", debiased_outcomes)
        report("good - warmth - emotion", multi_outcomes)
        print()
        for axis_name in tree_outcomes_per_axis:
            report(f"tree: {axis_name}", tree_outcomes_per_axis[axis_name])
        print()
        report("tree: ANY (1 of 5)", tree_any)
        report("tree: 2 of 5", tree_2of5)
        report("tree: MAJORITY (3 of 5)", tree_maj)

        del model
        gc.collect()


if __name__ == "__main__":
    main()
