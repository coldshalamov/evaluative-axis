#!/usr/bin/env python3
"""Characterize the failure ceiling: what cases does "Careful" get wrong
on the models where it performs best (Nomic, Gemini-via-local-proxy)?

For each model, identify cases that fail on ALL models and cases where
"Careful" succeeds — look for patterns in the failure cases.
Also: does ANY combination of our 10 axes solve the universally-failed cases?
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
BATTERY_EXPANSION = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = {
    "careful":  {"pos": ["Careful"],  "neg": ["Reckless"]},
    "thorough": {"pos": ["Thorough"], "neg": ["Superficial"]},
    "hard":     {"pos": ["Hard"],     "neg": ["Soft"]},
    "kind":     {"pos": ["Kind"],     "neg": ["Cruel"]},
    "honest":   {"pos": ["Honest"],   "neg": ["Dishonest"]},
    "helpful":  {"pos": ["Helpful"],  "neg": ["Unhelpful"]},
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "good":     {"pos": ["Good"],     "neg": ["Bad"]},
}


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    main_cases = original + warmth
    n_orig = len(original)
    n_total = len(main_cases)

    expansion_cases = []
    for cat_file in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(cat_file))

    all_cases = main_cases + expansion_cases
    n_all = len(all_cases)
    axis_names = list(AXES.keys())

    # Track per-case, per-axis, per-model correctness
    all_correct = {}  # model -> axis -> [correct_0, correct_1, ...]

    for model_name in MODELS:
        print(f"\nLoading {model_name}...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        all_correct[model_name] = {}

        for axis_name, anchors in AXES.items():
            pos_emb = embed_fn(anchors["pos"]).mean(axis=0)
            neg_emb = embed_fn(anchors["neg"]).mean(axis=0)
            axis_vec = pos_emb - neg_emb
            axis_unit = axis_vec / (norm(axis_vec) + 1e-12)

            correct = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis_unit))
                sw = float(np.dot(worse_embs[i], axis_unit))
                correct.append(1 if sb > sw else (0.5 if sb == sw else 0))

            all_correct[model_name][axis_name] = correct

        del model
        gc.collect()

    # Analysis 1: Cases that fail "careful" on ALL 3 models
    print(f"\n{'='*60}")
    print("CASES FAILING 'CAREFUL' ON ALL 3 MODELS")
    print(f"{'='*60}")

    careful_fail_all = []
    careful_succeed_all = []
    for i in range(n_all):
        votes = sum(1 for m in MODELS if all_correct[m]["careful"][i] >= 0.5)
        if votes == 0:
            careful_fail_all.append(i)
        elif votes == 3:
            careful_succeed_all.append(i)

    split_label = lambda i: "ORIG" if i < n_orig else ("WARM" if i < n_total else "EXP")

    print(f"\n{len(careful_fail_all)} cases fail 'careful' on ALL 3 models:")
    for i in careful_fail_all:
        c = all_cases[i]
        # Check if ANY axis gets it right on ALL models
        any_axis_all = []
        for a in axis_names:
            if all(all_correct[m][a][i] >= 0.5 for m in MODELS):
                any_axis_all.append(a)

        rescue_str = f"  Rescued by: {', '.join(any_axis_all)}" if any_axis_all else "  NO axis rescues this"
        print(f"\n  [{split_label(i)}] Case {i}")
        print(f"    Prompt: {c['prompt'][:100]}...")
        print(f"    Better: {c['better'][:80]}...")
        print(f"    Worse:  {c['worse'][:80]}...")
        print(rescue_str)

    # Analysis 2: Universal failures — no axis gets it right on all models
    print(f"\n{'='*60}")
    print("TRULY UNIVERSAL FAILURES (no axis works on all 3 models)")
    print(f"{'='*60}")

    truly_universal = []
    for i in range(n_all):
        any_axis_universal = False
        for a in axis_names:
            if all(all_correct[m][a][i] >= 0.5 for m in MODELS):
                any_axis_universal = True
                break
        if not any_axis_universal:
            truly_universal.append(i)

    print(f"\n{len(truly_universal)}/{n_all} cases have NO axis correct on all 3 models")
    orig_count = sum(1 for i in truly_universal if i < n_orig)
    warm_count = sum(1 for i in truly_universal if n_orig <= i < n_total)
    exp_count = sum(1 for i in truly_universal if i >= n_total)
    print(f"  Orig: {orig_count}/{n_orig}, Warm: {warm_count}/{len(warmth)}, Exp: {exp_count}/{len(expansion_cases)}")

    for i in truly_universal[:15]:
        c = all_cases[i]
        # Show which axes got closest
        best_votes = 0
        best_axes = []
        for a in axis_names:
            votes = sum(1 for m in MODELS if all_correct[m][a][i] >= 0.5)
            if votes > best_votes:
                best_votes = votes
                best_axes = [a]
            elif votes == best_votes:
                best_axes.append(a)

        print(f"\n  [{split_label(i)}] Case {i} (best: {best_votes}/3 models via {', '.join(best_axes[:3])})")
        print(f"    Prompt: {c['prompt'][:100]}...")
        print(f"    Better: {c['better'][:80]}...")
        print(f"    Worse:  {c['worse'][:80]}...")

    # Analysis 3: Which axes are most complementary to "careful"?
    print(f"\n{'='*60}")
    print("COMPLEMENTARITY: which axis rescues careful failures?")
    print(f"{'='*60}")

    for m in MODELS:
        short = m.split("/")[-1]
        careful_wrong = [i for i in range(n_all) if all_correct[m]["careful"][i] < 0.5]
        print(f"\n  {short}: {len(careful_wrong)} careful failures")

        for a in axis_names:
            if a == "careful":
                continue
            rescues = sum(1 for i in careful_wrong if all_correct[m][a][i] >= 0.5)
            breaks = sum(1 for i in range(n_all) if all_correct[m]["careful"][i] >= 0.5 and all_correct[m][a][i] < 0.5)
            net = rescues - breaks
            print(f"    {a:10s}: rescues {rescues:2d}, breaks {breaks:2d}, net {'+' if net>=0 else ''}{net:d}")

    # Analysis 4: Category analysis on rebalanced battery
    print(f"\n{'='*60}")
    print("CATEGORY PATTERNS IN FAILURES")
    print(f"{'='*60}")

    # Classify cases by what dimension they test
    # Simple heuristic: look for keywords in prompt/better/worse
    firmness_keywords = ["reason", "logic", "evidence", "fact", "correct", "accurate", "truth",
                         "calculat", "math", "code", "error", "mistake", "wrong", "fallac"]
    warmth_keywords = ["feel", "empath", "support", "kind", "comfort", "emotion",
                        "understand", "caring", "gentle", "compassion"]

    for i in range(n_total):
        c = all_cases[i]
        text = (c["prompt"] + " " + c["better"] + " " + c["worse"]).lower()
        is_firmness = any(k in text for k in firmness_keywords)
        is_warmth = any(k in text for k in warmth_keywords)

        careful_votes = sum(1 for m in MODELS if all_correct[m]["careful"][i] >= 0.5)
        hard_votes = sum(1 for m in MODELS if all_correct[m]["hard"][i] >= 0.5)
        kind_votes = sum(1 for m in MODELS if all_correct[m]["kind"][i] >= 0.5)

        if i < 5 or careful_votes == 0:
            dim = "firmness" if is_firmness and not is_warmth else ("warmth" if is_warmth and not is_firmness else "mixed")
            print(f"  Case {i} [{split_label(i)}] dim={dim}: careful={careful_votes}/3 hard={hard_votes}/3 kind={kind_votes}/3")

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total cases: {n_all}")
    print(f"  Cases failing careful on all 3 models: {len(careful_fail_all)}")
    print(f"  Truly universal failures (no axis works): {len(truly_universal)}")
    print(f"  Cases succeeding on all 3 models via careful: {len(careful_succeed_all)}")
    pct_succeed = len(careful_succeed_all) / n_all
    pct_fail = len(careful_fail_all) / n_all
    pct_universal = len(truly_universal) / n_all
    print(f"  Success rate (careful on all 3): {pct_succeed:.0%}")
    print(f"  Failure rate (careful on all 3): {pct_fail:.0%}")
    print(f"  Universal failure rate: {pct_universal:.0%}")


if __name__ == "__main__":
    main()
