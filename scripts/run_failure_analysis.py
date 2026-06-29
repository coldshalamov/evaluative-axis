#!/usr/bin/env python3
"""Failure analysis: what do the cases that "careful" gets wrong have in
common across models?

For each case, we know:
1. Whether careful gets it right on each of the 3 local models
2. The category tags
3. The delta magnitude (confidence)
4. Whether good/kind/thorough get it right

We want to find:
- Cases that ALL 3 models get wrong on careful → structural failures
- Cases that only 1-2 models get right → fragile cases
- Surface features that predict failure
"""

import json, gc, math
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

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    main_cases = original + warmth
    n = len(main_cases)

    expansion_cases = []
    for f in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        cases = read_jsonl(f)
        cat = f.stem
        for c in cases:
            c["_category"] = cat
        expansion_cases.extend(cases)

    all_cases = main_cases + expansion_cases
    n_all = len(all_cases)

    model_correct = {}
    model_deltas = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"Processing {short}...")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        model_correct[short] = {}
        model_deltas[short] = {}

        for name, pos, neg in [("careful", "Careful", "Reckless"), ("good", "Good", "Bad"),
                                ("kind", "Kind", "Cruel"), ("thorough", "Thorough", "Superficial")]:
            p = embed_fn([pos]).mean(axis=0)
            n_ = embed_fn([neg]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)

            deltas = []
            correct = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                d = sb - sw
                deltas.append(d)
                correct.append(1 if d > 0 else 0)

            model_correct[short][name] = correct
            model_deltas[short][name] = deltas

        del model
        gc.collect()

    models = list(model_correct.keys())

    print(f"\n{'='*60}")
    print("CAREFUL FAILURE ANALYSIS (main battery, n={})".format(n))
    print(f"{'='*60}")

    careful_votes = []
    for i in range(n):
        votes = sum(model_correct[m]["careful"][i] for m in models)
        careful_votes.append(votes)

    n_all_right = sum(1 for v in careful_votes if v == 3)
    n_two_right = sum(1 for v in careful_votes if v == 2)
    n_one_right = sum(1 for v in careful_votes if v == 1)
    n_none_right = sum(1 for v in careful_votes if v == 0)

    print(f"\nCross-model agreement on 'careful':")
    print(f"  All 3 correct: {n_all_right} ({n_all_right/n:.0%})")
    print(f"  Exactly 2 correct: {n_two_right} ({n_two_right/n:.0%})")
    print(f"  Exactly 1 correct: {n_one_right} ({n_one_right/n:.0%})")
    print(f"  None correct: {n_none_right} ({n_none_right/n:.0%})")

    print(f"\n--- Universal careful failures (0/3 correct) ---")
    failure_cases = []
    for i in range(n):
        if careful_votes[i] == 0:
            case = main_cases[i]
            tags = case.get("tags", [])
            source = "orig" if i < 50 else "warmth"
            good_votes = sum(model_correct[m]["good"][i] for m in models)
            kind_votes = sum(model_correct[m]["kind"][i] for m in models)
            thor_votes = sum(model_correct[m]["thorough"][i] for m in models)
            prompt_short = case["prompt"][:100]
            print(f"  [{source}] tags={tags}")
            print(f"    prompt: {prompt_short}...")
            print(f"    good={good_votes}/3  kind={kind_votes}/3  thorough={thor_votes}/3")
            for m in models:
                d = model_deltas[m]["careful"][i]
                print(f"    {m}: careful_delta={d:.4f}")
            failure_cases.append({
                "index": i, "source": source, "tags": tags,
                "prompt_preview": prompt_short,
                "good_votes": good_votes, "kind_votes": kind_votes,
                "thorough_votes": thor_votes,
                "careful_deltas": {m: model_deltas[m]["careful"][i] for m in models}
            })
            print()

    # --- ANY axis gets it? ---
    print(f"\n--- For universal careful failures: does ANY axis get it? ---")
    for fc in failure_cases:
        i = fc["index"]
        any_correct = {}
        for term in ["careful", "good", "kind", "thorough"]:
            for m in models:
                if model_correct[m][term][i]:
                    any_correct.setdefault(term, []).append(m)
        rescued = {t: ms for t, ms in any_correct.items() if ms}
        if rescued:
            print(f"  Case {i}: rescued by {rescued}")
        else:
            print(f"  Case {i}: UNIVERSAL FAILURE — no axis on any model")

    # --- Tag distribution ---
    print(f"\n--- Tag distribution among failures vs successes ---")
    all_tags = set()
    for case in main_cases:
        for tag in case.get("tags", []):
            all_tags.add(tag)

    for tag in sorted(all_tags):
        tag_cases = [i for i in range(n) if tag in main_cases[i].get("tags", [])]
        if not tag_cases:
            continue
        tag_fail_rate = sum(1 for i in tag_cases if careful_votes[i] == 0) / len(tag_cases)
        tag_robust_rate = sum(1 for i in tag_cases if careful_votes[i] == 3) / len(tag_cases)
        if len(tag_cases) >= 3:
            print(f"  {tag:30s}: n={len(tag_cases):3d}  "
                  f"all-fail={tag_fail_rate:.0%}  all-correct={tag_robust_rate:.0%}")

    # --- Length analysis ---
    print(f"\n--- Response length vs careful success ---")
    better_lens = [len(main_cases[i]["better"]) for i in range(n)]
    worse_lens = [len(main_cases[i]["worse"]) for i in range(n)]
    len_diffs = [better_lens[i] - worse_lens[i] for i in range(n)]

    for category, indices in [
        ("all-correct", [i for i in range(n) if careful_votes[i] == 3]),
        ("all-fail", [i for i in range(n) if careful_votes[i] == 0]),
        ("fragile (1-2)", [i for i in range(n) if careful_votes[i] in [1, 2]]),
    ]:
        if not indices:
            continue
        mean_len_diff = np.mean([len_diffs[i] for i in indices])
        mean_better = np.mean([better_lens[i] for i in indices])
        mean_worse = np.mean([worse_lens[i] for i in indices])
        longer_better = sum(1 for i in indices if better_lens[i] > worse_lens[i])
        pct_longer = longer_better / len(indices)
        print(f"  {category:15s}: n={len(indices):3d}  "
              f"better_longer={pct_longer:.0%}  "
              f"mean_len_diff={mean_len_diff:+.0f}  "
              f"avg_better={mean_better:.0f}  avg_worse={mean_worse:.0f}")

    # --- Cross-model correlation of correctness ---
    print(f"\n--- Cross-model correlation on careful ---")
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            agree = sum(1 for j in range(n)
                       if model_correct[m1]["careful"][j] == model_correct[m2]["careful"][j])
            print(f"  {m1} vs {m2}: agree={agree}/{n} ({agree/n:.0%})")

    # --- Expansion failures ---
    print(f"\n{'='*60}")
    print("EXPANSION BATTERY CAREFUL FAILURES")
    print(f"{'='*60}")

    n_exp = len(expansion_cases)
    exp_votes = []
    for i in range(n_exp):
        idx = n + i
        votes = sum(model_correct[m]["careful"][idx] for m in models)
        exp_votes.append(votes)

    exp_all_right = sum(1 for v in exp_votes if v == 3)
    exp_none_right = sum(1 for v in exp_votes if v == 0)
    print(f"All 3 correct: {exp_all_right}/{n_exp}")
    print(f"None correct: {exp_none_right}/{n_exp}")

    for i in range(n_exp):
        idx = n + i
        if exp_votes[i] == 0:
            case = expansion_cases[i]
            cat = case.get("_category", "?")
            prompt_short = case["prompt"][:80]
            good_v = sum(model_correct[m]["good"][idx] for m in models)
            print(f"\n  [{cat}] good={good_v}/3")
            print(f"    prompt: {prompt_short}...")

    # Save
    summary = {
        "experiment": "Failure analysis",
        "date": "2026-06-28",
        "main_battery": {
            "all_correct": n_all_right,
            "two_correct": n_two_right,
            "one_correct": n_one_right,
            "none_correct": n_none_right,
            "total": n
        },
        "expansion_battery": {
            "all_correct": exp_all_right,
            "none_correct": exp_none_right,
            "total": n_exp
        },
        "universal_failures": failure_cases
    }
    out_path = ROOT / "notes/research_cycles/tree_decomposition/failure_analysis.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
