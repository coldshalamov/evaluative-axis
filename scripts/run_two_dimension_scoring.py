#!/usr/bin/env python3
"""Two-dimension scoring: can we combine careful (rigor) and kind (warmth)
in a principled way?

Naive voting/summing fails because warmth-biased terms outvote independent
ones. But what about principled combinations that treat the two dimensions
as complementary rather than redundant?

Tests:
1. Min(careful, kind) — better response must be better on BOTH dimensions
2. Careful-then-kind — use careful as primary, break ties with kind
3. Weighted sum — tune weight on main battery, test on expansion
4. OR rule — better response wins on EITHER dimension
5. Selective: only predict when careful and kind agree, abstain otherwise
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

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    main_cases = original + warmth
    n_main = len(main_cases)

    expansion_cases = []
    for f in sorted(BATTERY_EXPANSION.glob("*.jsonl")):
        expansion_cases.extend(read_jsonl(f))
    n_exp = len(expansion_cases)

    all_cases = main_cases + expansion_cases
    n_all = len(all_cases)

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {short}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

        # Build axes for multiple terms
        term_pairs = [
            ("careful", "Careful", "Reckless"),
            ("kind", "Kind", "Cruel"),
            ("thorough", "Thorough", "Superficial"),
            ("honest", "Honest", "Dishonest"),
            ("good", "Good", "Bad"),
        ]

        deltas = {}
        for name, pos_w, neg_w in term_pairs:
            p = embed_fn([pos_w]).mean(axis=0)
            n_ = embed_fn([neg_w]).mean(axis=0)
            axis = (p - n_) / (norm(p - n_) + 1e-12)
            d = []
            for i in range(n_all):
                sb = float(np.dot(better_embs[i], axis))
                sw = float(np.dot(worse_embs[i], axis))
                d.append(sb - sw)
            deltas[name] = d

        # Baselines
        print(f"\n--- Baselines ---")
        for name in ["careful", "kind", "thorough", "good"]:
            for label, start, end in [("main", 0, n_main), ("expansion", n_main, n_all)]:
                subset = deltas[name][start:end]
                nn = len(subset)
                correct = sum(1 for d in subset if d > 0)
                acc = correct / nn
                lo, hi = wilson_ci(correct, nn)
                print(f"  {name:12s} {label:10s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={nn}")

        # Combination strategies
        print(f"\n--- Two-dimension combinations (careful + kind) ---")

        strategies = {}

        # 1. Both agree (AND rule) — only predict when both say same thing
        both_agree_correct = 0
        both_agree_n = 0
        both_agree_correct_exp = 0
        both_agree_n_exp = 0
        for i in range(n_all):
            c_pred = 1 if deltas["careful"][i] > 0 else 0
            k_pred = 1 if deltas["kind"][i] > 0 else 0
            if c_pred == k_pred:
                if i < n_main:
                    both_agree_n += 1
                    both_agree_correct += c_pred
                else:
                    both_agree_n_exp += 1
                    both_agree_correct_exp += c_pred
        if both_agree_n:
            acc = both_agree_correct / both_agree_n
            lo, hi = wilson_ci(both_agree_correct, both_agree_n)
            coverage = both_agree_n / n_main
            print(f"  AND (agree)  main: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  coverage={coverage:.0%} ({both_agree_n}/{n_main})")
        if both_agree_n_exp:
            acc = both_agree_correct_exp / both_agree_n_exp
            lo, hi = wilson_ci(both_agree_correct_exp, both_agree_n_exp)
            coverage = both_agree_n_exp / n_exp
            print(f"  AND (agree)  exp:  {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  coverage={coverage:.0%} ({both_agree_n_exp}/{n_exp})")

        # 2. OR rule — predict positive if either says positive
        for i in range(n_all):
            c_pred = 1 if deltas["careful"][i] > 0 else 0
            k_pred = 1 if deltas["kind"][i] > 0 else 0
            or_pred = 1 if (c_pred or k_pred) else 0
        # Actually compute this properly
        for label, start, end in [("main", 0, n_main), ("exp", n_main, n_all)]:
            correct = 0
            nn = end - start
            for i in range(start, end):
                c_pred = 1 if deltas["careful"][i] > 0 else 0
                k_pred = 1 if deltas["kind"][i] > 0 else 0
                or_pred = 1 if (c_pred or k_pred) else 0
                correct += or_pred
            acc = correct / nn
            lo, hi = wilson_ci(correct, nn)
            print(f"  OR rule      {label:4s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={nn}")

        # 3. Sum of deltas (equal weight)
        for label, start, end in [("main", 0, n_main), ("exp", n_main, n_all)]:
            correct = 0
            nn = end - start
            for i in range(start, end):
                combined = deltas["careful"][i] + deltas["kind"][i]
                correct += 1 if combined > 0 else 0
            acc = correct / nn
            lo, hi = wilson_ci(correct, nn)
            print(f"  Sum (c+k)    {label:4s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={nn}")

        # 4. Min-max: better response wins if it has higher min(careful, kind)
        for label, start, end in [("main", 0, n_main), ("exp", n_main, n_all)]:
            correct = 0
            nn = end - start
            for i in range(start, end):
                c_b = float(np.dot(better_embs[i], embed_fn(["Careful"]).mean(axis=0)))
                c_w = float(np.dot(worse_embs[i], embed_fn(["Careful"]).mean(axis=0)))
                # Actually this is too expensive, use deltas
                pass
            # Simpler: just check if delta of min is positive
            # skip this one, it requires raw scores not deltas

        # 5. Careful + thorough (both independent)
        for label, start, end in [("main", 0, n_main), ("exp", n_main, n_all)]:
            correct = 0
            nn = end - start
            for i in range(start, end):
                combined = deltas["careful"][i] + deltas["thorough"][i]
                correct += 1 if combined > 0 else 0
            acc = correct / nn
            lo, hi = wilson_ci(correct, nn)
            print(f"  Sum (c+t)    {label:4s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={nn}")

        # 6. Careful primary, kind tiebreaker (use kind only when |careful_delta| is small)
        for threshold_pct in [25, 50]:
            main_abs_careful = [abs(deltas["careful"][i]) for i in range(n_main)]
            threshold = np.percentile(main_abs_careful, threshold_pct)
            for label, start, end in [("main", 0, n_main), ("exp", n_main, n_all)]:
                correct = 0
                nn = end - start
                for i in range(start, end):
                    if abs(deltas["careful"][i]) >= threshold:
                        pred = 1 if deltas["careful"][i] > 0 else 0
                    else:
                        pred = 1 if deltas["kind"][i] > 0 else 0
                    correct += pred
                acc = correct / nn
                lo, hi = wilson_ci(correct, nn)
                print(f"  C-then-K(p{threshold_pct}) {label:4s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={nn}")

        # 7. Majority vote: careful, kind, thorough
        for label, start, end in [("main", 0, n_main), ("exp", n_main, n_all)]:
            correct = 0
            nn = end - start
            for i in range(start, end):
                votes = sum(1 for name in ["careful", "kind", "thorough"]
                           if deltas[name][i] > 0)
                pred = 1 if votes >= 2 else 0
                correct += pred
            acc = correct / nn
            lo, hi = wilson_ci(correct, nn)
            print(f"  Vote(c,k,t)  {label:4s}: {acc:.0%}  CI=[{lo:.0%}, {hi:.0%}]  n={nn}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
