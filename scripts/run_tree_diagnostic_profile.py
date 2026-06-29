#!/usr/bin/env python3
"""Test: does the competence-branch tree give DIAGNOSTIC profiles?

The user's "tree of individual scores" idea: don't reduce to one number.
Keep the per-axis scores as a vector. Does that vector tell you something
useful about what KIND of response failure you're looking at?

Specifically: score every case on warmth-branch words (kind, helpful) and
competence-branch words (careful, thorough, restrained) independently.
Do the profiles differ between firmness vs warmth cases in a predictable way?

Also test: if you ONLY use the competence branch (careful + thorough + restrained),
scored independently and majority-voted, does it beat careful alone?
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

WARMTH_BRANCH = [
    ("kind", "Kind", "Cruel"),
    ("helpful", "Helpful", "Unhelpful"),
    ("honest", "Honest", "Dishonest"),
    ("fair", "Fair", "Unfair"),
    ("respectful", "Respectful", "Disrespectful"),
]

COMPETENCE_BRANCH = [
    ("careful", "Careful", "Reckless"),
    ("thorough", "Thorough", "Superficial"),
    ("restrained", "Restrained", "Unrestrained"),
]

ALL_AXES = [("good", "Good", "Bad")] + WARMTH_BRANCH + COMPETENCE_BRANCH


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


def main():
    from sentence_transformers import SentenceTransformer

    battery = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    all_cases = battery + warmth
    n = len(all_cases)

    labels = []
    for c in battery:
        if c["category"] == "anti_sycophancy":
            labels.append("sycophancy")
        else:
            labels.append("firmness")
    for c in warmth:
        labels.append("warmth")

    print(f"Cases: {n}")
    print(f"  firmness: {labels.count('firmness')}")
    print(f"  warmth: {labels.count('warmth')}")
    print(f"  sycophancy: {labels.count('sycophancy')}")

    results = {}

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

        axis_outcomes = {}  # axis -> list of booleans (correct or not)

        for axis_name, pos, neg in ALL_AXES:
            p_emb = embed([pos])[0]
            n_emb = embed([neg])[0]
            axis = (p_emb - n_emb) / (norm(p_emb - n_emb) + 1e-12)

            outcomes = [float(np.dot(better_embs[i], axis)) > float(np.dot(worse_embs[i], axis))
                        for i in range(n)]
            axis_outcomes[axis_name] = outcomes

        # Per-split accuracy for each axis
        print(f"\n{'Axis':12s} {'All':>5s} {'Firm':>5s} {'Warm':>5s} {'Syc':>5s}")
        print("-" * 40)
        for axis_name, _, _ in ALL_AXES:
            outcomes = axis_outcomes[axis_name]
            all_acc = sum(outcomes) / n
            firm_acc = sum(outcomes[i] for i in range(n) if labels[i] == "firmness") / labels.count("firmness")
            warm_acc = sum(outcomes[i] for i in range(n) if labels[i] == "warmth") / labels.count("warmth")
            syc_acc = sum(outcomes[i] for i in range(n) if labels[i] == "sycophancy") / labels.count("sycophancy")
            print(f"{axis_name:12s} {all_acc:5.0%} {firm_acc:5.0%} {warm_acc:5.0%} {syc_acc:5.0%}")

        # COMBINATION STRATEGIES
        print(f"\n--- Combination strategies ---")

        # Strategy 1: Competence branch majority vote
        comp_names = [a[0] for a in COMPETENCE_BRANCH]
        comp_vote = []
        for i in range(n):
            votes = sum(1 for a in comp_names if axis_outcomes[a][i])
            comp_vote.append(votes > len(comp_names) / 2)
        comp_acc = sum(comp_vote) / n
        comp_firm = sum(comp_vote[i] for i in range(n) if labels[i] == "firmness") / labels.count("firmness")
        comp_warm = sum(comp_vote[i] for i in range(n) if labels[i] == "warmth") / labels.count("warmth")
        comp_syc = sum(comp_vote[i] for i in range(n) if labels[i] == "sycophancy") / labels.count("sycophancy")
        print(f"{'comp_vote':12s} {comp_acc:5.0%} {comp_firm:5.0%} {comp_warm:5.0%} {comp_syc:5.0%}")

        # Strategy 2: Warmth branch majority vote
        warm_names = [a[0] for a in WARMTH_BRANCH]
        warm_vote = []
        for i in range(n):
            votes = sum(1 for a in warm_names if axis_outcomes[a][i])
            warm_vote.append(votes > len(warm_names) / 2)
        warm_acc = sum(warm_vote) / n
        warm_firm = sum(warm_vote[i] for i in range(n) if labels[i] == "firmness") / labels.count("firmness")
        warm_warm = sum(warm_vote[i] for i in range(n) if labels[i] == "warmth") / labels.count("warmth")
        warm_syc = sum(warm_vote[i] for i in range(n) if labels[i] == "sycophancy") / labels.count("sycophancy")
        print(f"{'warm_vote':12s} {warm_acc:5.0%} {warm_firm:5.0%} {warm_warm:5.0%} {warm_syc:5.0%}")

        # Strategy 3: Both branches independently, require BOTH to agree
        both_and = [comp_vote[i] and warm_vote[i] for i in range(n)]
        both_acc = sum(both_and) / n
        both_firm = sum(both_and[i] for i in range(n) if labels[i] == "firmness") / labels.count("firmness")
        both_warm = sum(both_and[i] for i in range(n) if labels[i] == "warmth") / labels.count("warmth")
        both_syc = sum(both_and[i] for i in range(n) if labels[i] == "sycophancy") / labels.count("sycophancy")
        print(f"{'both_AND':12s} {both_acc:5.0%} {both_firm:5.0%} {both_warm:5.0%} {both_syc:5.0%}")

        # Strategy 4: Either branch says yes
        both_or = [comp_vote[i] or warm_vote[i] for i in range(n)]
        or_acc = sum(both_or) / n
        or_firm = sum(both_or[i] for i in range(n) if labels[i] == "firmness") / labels.count("firmness")
        or_warm = sum(both_or[i] for i in range(n) if labels[i] == "warmth") / labels.count("warmth")
        or_syc = sum(both_or[i] for i in range(n) if labels[i] == "sycophancy") / labels.count("sycophancy")
        print(f"{'either_OR':12s} {or_acc:5.0%} {or_firm:5.0%} {or_warm:5.0%} {or_syc:5.0%}")

        # DIAGNOSTIC PROFILES: Show per-case patterns
        print(f"\n--- Diagnostic profiles (first 5 firmness, first 5 warmth) ---")
        print(f"{'Case':35s} ", end="")
        for a, _, _ in ALL_AXES:
            print(f"{a[:4]:>5s}", end="")
        print("  type")

        shown_firm = 0
        shown_warm = 0
        for i in range(n):
            if labels[i] == "firmness" and shown_firm < 5:
                cat = all_cases[i].get("category", "?")
                prompt = all_cases[i]["prompt"][:30]
                print(f"{prompt:35s} ", end="")
                for a, _, _ in ALL_AXES:
                    mark = "+" if axis_outcomes[a][i] else "-"
                    print(f"    {mark}", end="")
                print(f"  FIRM/{cat}")
                shown_firm += 1
            elif labels[i] == "warmth" and shown_warm < 5:
                prompt = all_cases[i]["prompt"][:30]
                print(f"{prompt:35s} ", end="")
                for a, _, _ in ALL_AXES:
                    mark = "+" if axis_outcomes[a][i] else "-"
                    print(f"    {mark}", end="")
                print(f"  WARM")
                shown_warm += 1

        # Agreement analysis: when competence and warmth branches DISAGREE
        disagree = [(i, comp_vote[i], warm_vote[i]) for i in range(n) if comp_vote[i] != warm_vote[i]]
        print(f"\n--- Branch disagreements ({len(disagree)} cases) ---")
        print(f"  Comp=yes, Warm=no: {sum(1 for _, c, w in disagree if c and not w)}")
        print(f"  Comp=no, Warm=yes: {sum(1 for _, c, w in disagree if not c and w)}")

        # When they disagree, which branch is right?
        comp_right = sum(1 for i, c, w in disagree if c and not w and
                         (labels[i] in ("firmness", "sycophancy")))
        comp_right += sum(1 for i, c, w in disagree if not c and w and
                          labels[i] == "warmth")
        # This is wrong - let me do it properly
        # When they disagree, who predicted the correct case type?
        comp_yes_warm_no = [(i, labels[i]) for i, c, w in disagree if c and not w]
        comp_no_warm_yes = [(i, labels[i]) for i, c, w in disagree if not c and w]

        print(f"\n  When comp=yes, warm=no ({len(comp_yes_warm_no)} cases):")
        for lbl in ["firmness", "warmth", "sycophancy"]:
            ct = sum(1 for _, l in comp_yes_warm_no if l == lbl)
            if ct > 0:
                print(f"    {lbl}: {ct}")

        print(f"  When comp=no, warm=yes ({len(comp_no_warm_yes)} cases):")
        for lbl in ["firmness", "warmth", "sycophancy"]:
            ct = sum(1 for _, l in comp_no_warm_yes if l == lbl)
            if ct > 0:
                print(f"    {lbl}: {ct}")

        results[short] = {
            "careful_alone": sum(axis_outcomes["careful"]) / n,
            "comp_vote": comp_acc,
            "warm_vote": warm_acc,
            "both_AND": both_acc,
            "either_OR": or_acc,
            "good": sum(axis_outcomes["good"]) / n,
        }

        del model
        gc.collect()

    # Cross-model summary
    print(f"\n{'='*80}")
    print("CROSS-MODEL SUMMARY: Does the tree help?")
    print(f"{'='*80}")
    print(f"\n{'Strategy':15s}", end="")
    for m in results:
        print(f" {m[:12]:>12s}", end="")
    print()
    print("-" * 55)
    for strat in ["good", "careful_alone", "comp_vote", "warm_vote", "both_AND", "either_OR"]:
        print(f"{strat:15s}", end="")
        for m in results:
            print(f" {results[m][strat]:12.0%}", end="")
        print()

    # Save
    out = ROOT / "notes/research_cycles/good_neighborhood"
    outfile = out / "tree_diagnostic_results.json"
    outfile.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
