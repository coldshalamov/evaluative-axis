#!/usr/bin/env python3
"""Test the tree-based training signal:
    reward = cos(good) - alpha * cos(failure_mode)

The semantic tree shows that good's actual failure modes are:
- BGE-M3: flattering (0.695 close to good), obsequious, overwhelming
- Nomic: placating (0.531), servile, misleading
- Snowflake: vague (0.945), dismissive, extreme

NOT "sycophantic" (which is far from good on all models).

Test:
1. good - flattering (the actual nearest failure mode)
2. good - placating
3. good - penalty_cluster (mean of top failure modes)
4. Hierarchical: good_children scores minus their failure modes
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"
EXP_DIR = ROOT / "notes/research_cycles/battery_expansion"

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

    battery = read_jsonl(BATTERY)
    warmth_cases = read_jsonl(WARMTH)
    all_cases = battery + warmth_cases
    n = len(all_cases)

    labels = []
    for c in battery:
        labels.append("sycophancy" if c["category"] == "anti_sycophancy" else "firmness")
    for c in warmth_cases:
        labels.append("warmth")

    firm_idx = [i for i in range(n) if labels[i] == "firmness"]
    warm_idx = [i for i in range(n) if labels[i] == "warmth"]
    syc_idx = [i for i in range(n) if labels[i] == "sycophancy"]

    # OOS
    oos = []
    for f in sorted(EXP_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                oos.append(json.loads(line))
    n_oos = len(oos)

    # All terms we need embeddings for
    TERMS = [
        "Good", "Bad",
        # Failure modes (from tree analysis)
        "Flattering", "Placating", "Obsequious", "Sycophantic",
        "Vague", "Dismissive", "Extreme", "Servile", "Misleading",
        "Overwhelming", "Manipulative", "Superficial", "Simplistic",
        "Ingratiating", "Fawning", "Submissive", "Presumptuous",
        # Good's children (positive)
        "Excellent", "Pleasant", "Friendly", "Useful", "Quality",
        "Helpful", "Honest", "Thoughtful", "Careful", "Reliable",
        # Children's positive siblings
        "Supportive", "Kind", "Constructive",
        # Anti-failure terms (what we WANT instead of the failure modes)
        "Genuine", "Authentic", "Principled", "Candid",
    ]

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed responses
        better_embs = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
        worse_embs = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])
        better_oos_embs = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in oos])
        worse_oos_embs = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in oos])

        # Embed all terms
        term_embs = {}
        for t in TERMS:
            term_embs[t] = embed([t])[0]
            term_embs[t] = term_embs[t] / (norm(term_embs[t]) + 1e-12)

        # For each response, compute cosine similarity to each term
        def cos_scores(response_embs, term_name):
            t = term_embs[term_name]
            return [float(np.dot(response_embs[i] / (norm(response_embs[i]) + 1e-12), t))
                    for i in range(len(response_embs))]

        # Scoring strategies using SCORE SUBTRACTION (not direction subtraction)
        def test_strategy(name, better_scores, worse_scores, n_cases, case_labels=None, idx_groups=None):
            correct = [better_scores[i] > worse_scores[i] for i in range(n_cases)]
            acc = sum(correct) / n_cases
            lo, hi = wilson_ci(sum(correct), n_cases)

            if idx_groups:
                parts = []
                for lbl, idx in idx_groups:
                    if idx:
                        sub_acc = sum(correct[i] for i in idx) / len(idx)
                        parts.append(f"{lbl[0]}={sub_acc:.0%}")
                detail = "  ".join(parts)
                print(f"  {name:50s} {acc:4.0%} [{lo:.0%}-{hi:.0%}]  {detail}")
            else:
                print(f"  {name:50s} {acc:4.0%} [{lo:.0%}-{hi:.0%}]")

        groups = [("firm", firm_idx), ("warm", warm_idx), ("syc", syc_idx)]

        print(f"\n  IN-SAMPLE (n={n})")
        print(f"  {'Strategy':50s} {'Acc':>4s} {'CI':>10s}  Details")
        print(f"  {'-'*85}")

        # Baseline: raw good cosine
        b_good = cos_scores(better_embs, "Good")
        w_good = cos_scores(worse_embs, "Good")
        test_strategy("raw cos(Good)", b_good, w_good, n, idx_groups=groups)

        # Strategy 1: good - flattering
        for alpha in [0.2, 0.5, 0.8, 1.0]:
            b_flat = cos_scores(better_embs, "Flattering")
            w_flat = cos_scores(worse_embs, "Flattering")
            b_score = [b_good[i] - alpha * b_flat[i] for i in range(n)]
            w_score = [w_good[i] - alpha * w_flat[i] for i in range(n)]
            test_strategy(f"cos(Good) - {alpha}*cos(Flattering)", b_score, w_score, n, idx_groups=groups)

        print()
        # Strategy 2: good - placating
        for alpha in [0.5, 1.0]:
            b_plac = cos_scores(better_embs, "Placating")
            w_plac = cos_scores(worse_embs, "Placating")
            b_score = [b_good[i] - alpha * b_plac[i] for i in range(n)]
            w_score = [w_good[i] - alpha * w_plac[i] for i in range(n)]
            test_strategy(f"cos(Good) - {alpha}*cos(Placating)", b_score, w_score, n, idx_groups=groups)

        # Strategy 3: good - mean(top failure modes)
        print()
        failure_cluster = ["Flattering", "Placating", "Obsequious", "Sycophantic"]
        for alpha in [0.5, 1.0]:
            b_fail = [np.mean([cos_scores(better_embs, t)[i] for t in failure_cluster])
                      for i in range(n)]
            w_fail = [np.mean([cos_scores(worse_embs, t)[i] for t in failure_cluster])
                      for i in range(n)]
            b_score = [b_good[i] - alpha * b_fail[i] for i in range(n)]
            w_score = [w_good[i] - alpha * w_fail[i] for i in range(n)]
            test_strategy(f"cos(Good) - {alpha}*mean(flatter+plac+obseq+syc)",
                         b_score, w_score, n, idx_groups=groups)

        # Strategy 4: good - failure + genuine
        print()
        for alpha in [0.5, 1.0]:
            b_flat = cos_scores(better_embs, "Flattering")
            w_flat = cos_scores(worse_embs, "Flattering")
            b_gen = cos_scores(better_embs, "Genuine")
            w_gen = cos_scores(worse_embs, "Genuine")
            b_score = [b_good[i] + alpha * b_gen[i] - alpha * b_flat[i] for i in range(n)]
            w_score = [w_good[i] + alpha * w_gen[i] - alpha * w_flat[i] for i in range(n)]
            test_strategy(f"cos(Good) + {alpha}*Genuine - {alpha}*Flattering",
                         b_score, w_score, n, idx_groups=groups)

        # Strategy 5: hierarchical tree scoring
        # Level 1: score each child of good, penalize each child's failure mode
        print()
        # good's children: pleasant, helpful, honest, careful, reliable
        # their failure modes (from tree analysis): flattering, manipulative, dishonest, reckless, unreliable
        children = {
            "Pleasant": "Flattering",
            "Helpful": "Manipulative",
            "Honest": "Misleading",
            "Careful": "Dismissive",
        }
        for alpha in [0.3, 0.5, 1.0]:
            b_tree = []
            w_tree = []
            for i in range(n):
                b_s = cos_scores(better_embs, "Good")[i]
                w_s = cos_scores(worse_embs, "Good")[i]
                for child, failure in children.items():
                    b_child = cos_scores(better_embs, child)[i]
                    w_child = cos_scores(worse_embs, child)[i]
                    b_fail = cos_scores(better_embs, failure)[i]
                    w_fail = cos_scores(worse_embs, failure)[i]
                    b_s += b_child - alpha * b_fail
                    w_s += w_child - alpha * w_fail
                b_tree.append(b_s)
                w_tree.append(w_s)
            test_strategy(f"tree(Good+4 children - {alpha}*failures)",
                         b_tree, w_tree, n, idx_groups=groups)

        # Strategy 6: Multi-level penalization from ACTUAL tree paths
        # good - flattering - placating - obsequious (all the sycophancy path terms)
        # PLUS good + genuine + principled + candid (anti-sycophancy terms)
        print()
        sycophancy_path = ["Flattering", "Placating", "Obsequious", "Sycophantic",
                           "Fawning", "Submissive", "Ingratiating", "Servile"]
        anti_syc = ["Genuine", "Authentic", "Principled", "Candid"]
        for alpha in [0.3, 0.5]:
            b_score = []
            w_score = []
            for i in range(n):
                bs = cos_scores(better_embs, "Good")[i]
                ws = cos_scores(worse_embs, "Good")[i]
                for t in sycophancy_path:
                    bs -= alpha * cos_scores(better_embs, t)[i] / len(sycophancy_path)
                    ws -= alpha * cos_scores(worse_embs, t)[i] / len(sycophancy_path)
                for t in anti_syc:
                    bs += alpha * cos_scores(better_embs, t)[i] / len(anti_syc)
                    ws += alpha * cos_scores(worse_embs, t)[i] / len(anti_syc)
                b_score.append(bs)
                w_score.append(ws)
            test_strategy(f"good - {alpha}*mean(8 syc_path) + {alpha}*mean(4 anti_syc)",
                         b_score, w_score, n, idx_groups=groups)

        # OOS for best strategies
        print(f"\n  OUT-OF-SAMPLE (n={n_oos})")

        b_good_oos = cos_scores(better_oos_embs, "Good")
        w_good_oos = cos_scores(worse_oos_embs, "Good")
        test_strategy("raw cos(Good)", b_good_oos, w_good_oos, n_oos)

        for alpha in [0.5, 1.0]:
            b_flat_oos = cos_scores(better_oos_embs, "Flattering")
            w_flat_oos = cos_scores(worse_oos_embs, "Flattering")
            b_s = [b_good_oos[i] - alpha * b_flat_oos[i] for i in range(n_oos)]
            w_s = [w_good_oos[i] - alpha * w_flat_oos[i] for i in range(n_oos)]
            test_strategy(f"cos(Good) - {alpha}*cos(Flattering)", b_s, w_s, n_oos)

        for alpha in [0.5, 1.0]:
            b_plac_oos = cos_scores(better_oos_embs, "Placating")
            w_plac_oos = cos_scores(worse_oos_embs, "Placating")
            b_s = [b_good_oos[i] - alpha * b_plac_oos[i] for i in range(n_oos)]
            w_s = [w_good_oos[i] - alpha * w_plac_oos[i] for i in range(n_oos)]
            test_strategy(f"cos(Good) - {alpha}*cos(Placating)", b_s, w_s, n_oos)

        del model
        gc.collect()


if __name__ == "__main__":
    main()
