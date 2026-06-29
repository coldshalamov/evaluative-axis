#!/usr/bin/env python3
"""Out-of-sample confidence calibration test.

Tests whether the confidence pattern (high |delta| = more accurate) holds
on the 20 expansion cases. Also tests whether a threshold learned on the
70-case main battery predicts well on expansion.
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

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {short}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        pos_emb = embed_fn(["Careful"]).mean(axis=0)
        neg_emb = embed_fn(["Reckless"]).mean(axis=0)
        axis = (pos_emb - neg_emb) / (norm(pos_emb - neg_emb) + 1e-12)

        # Score main battery
        main_deltas = []
        main_correct = []
        for c in main_cases:
            be = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}"])[0]
            we = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}"])[0]
            d = float(np.dot(be, axis)) - float(np.dot(we, axis))
            main_deltas.append(d)
            main_correct.append(1 if d > 0 else 0)

        # Score expansion battery
        exp_deltas = []
        exp_correct = []
        for c in expansion_cases:
            be = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}"])[0]
            we = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}"])[0]
            d = float(np.dot(be, axis)) - float(np.dot(we, axis))
            exp_deltas.append(d)
            exp_correct.append(1 if d > 0 else 0)

        # 1. Confidence on main battery itself (replicate prior result)
        main_abs = [abs(d) for d in main_deltas]
        main_median = np.median(main_abs)

        hi_main = [i for i in range(n_main) if abs(main_deltas[i]) >= main_median]
        lo_main = [i for i in range(n_main) if abs(main_deltas[i]) < main_median]

        hi_acc = np.mean([main_correct[i] for i in hi_main])
        lo_acc = np.mean([main_correct[i] for i in lo_main])
        k_hi = sum(main_correct[i] for i in hi_main)
        k_lo = sum(main_correct[i] for i in lo_main)
        lo_ci_hi, hi_ci_hi = wilson_ci(k_hi, len(hi_main))
        lo_ci_lo, hi_ci_lo = wilson_ci(k_lo, len(lo_main))

        print(f"\n--- Main battery confidence split (median |delta| = {main_median:.5f}) ---")
        print(f"  High conf: n={len(hi_main)}  acc={hi_acc:.0%}  CI=[{lo_ci_hi:.0%}, {hi_ci_hi:.0%}]")
        print(f"  Low conf:  n={len(lo_main)}  acc={lo_acc:.0%}  CI=[{lo_ci_lo:.0%}, {hi_ci_lo:.0%}]")

        # 2. Apply main-battery median to expansion cases (OOS test)
        hi_exp = [i for i in range(n_exp) if abs(exp_deltas[i]) >= main_median]
        lo_exp = [i for i in range(n_exp) if abs(exp_deltas[i]) < main_median]

        print(f"\n--- Expansion battery using main-battery threshold ---")
        if hi_exp:
            hi_exp_acc = np.mean([exp_correct[i] for i in hi_exp])
            k = sum(exp_correct[i] for i in hi_exp)
            lo_ci, hi_ci = wilson_ci(k, len(hi_exp))
            print(f"  High conf: n={len(hi_exp)}  acc={hi_exp_acc:.0%}  CI=[{lo_ci:.0%}, {hi_ci:.0%}]")
        else:
            print(f"  High conf: n=0")
        if lo_exp:
            lo_exp_acc = np.mean([exp_correct[i] for i in lo_exp])
            k = sum(exp_correct[i] for i in lo_exp)
            lo_ci, hi_ci = wilson_ci(k, len(lo_exp))
            print(f"  Low conf:  n={len(lo_exp)}  acc={lo_exp_acc:.0%}  CI=[{lo_ci:.0%}, {hi_ci:.0%}]")
        else:
            print(f"  Low conf:  n=0")

        # 3. Overall expansion accuracy
        exp_acc = np.mean(exp_correct)
        k = sum(exp_correct)
        lo_ci, hi_ci = wilson_ci(k, n_exp)
        print(f"\n  Expansion overall: n={n_exp}  acc={exp_acc:.0%}  CI=[{lo_ci:.0%}, {hi_ci:.0%}]")

        # 4. Point-biserial correlation between |delta| and correctness
        from scipy.stats import pointbiserialr
        r_main, p_main = pointbiserialr(main_correct, main_abs)
        print(f"\n--- Point-biserial r(|delta|, correct) ---")
        print(f"  Main battery: r={r_main:.3f}  p={p_main:.3f}")

        if n_exp > 5:
            exp_abs = [abs(d) for d in exp_deltas]
            r_exp, p_exp = pointbiserialr(exp_correct, exp_abs)
            print(f"  Expansion:    r={r_exp:.3f}  p={p_exp:.3f}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
