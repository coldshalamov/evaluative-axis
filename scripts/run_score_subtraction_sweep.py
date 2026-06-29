#!/usr/bin/env python3
"""Score-subtraction sweep: cos(base) - alpha * cos(penalty).

The user's tree-penalty idea: keep "good" (or another base) as the primary
reward, and subtract a weighted score of a *failure-mode* word (flattering,
placating, sycophantic, etc.) to penalize the specific contamination path.

This is SCORE subtraction (independent projections, then combine scalars),
NOT direction subtraction (vector arithmetic). Prior direction-subtraction
failed; this tests the alternative the tree architecture predicts works.

Sweep:
  - BASE words: good, helpful, kind, warm, honest, careful  (try each as root)
  - PENALTY words: the failure-mode cluster + a few controls
  - ALPHA values: 0.0 (baseline), 0.25, 0.5, 1.0, 1.5, 2.0
  - All 3 models, all 7 battery categories (105 cases)

Reports the best (base, penalty, alpha) per model and per category, and
whether the subtraction beats raw good consistently across models.

Exploratory: we report what helps and what doesn't, without overclaiming.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "research_cycles" / "score_subtraction" / "subtraction_results.json"

ORIG = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARM = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXP_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

BASES = ["good", "helpful", "kind", "warm", "honest", "careful"]
PENALTIES = [
    "flattering", "placating", "sycophantic", "obsequious", "fawning",
    "gushing", "pandering", "saccharine", "verbose", "vague",
    # controls: penalty words that SHOULD NOT help (random-ish / wrong direction)
    "blue", "however", "therefore",
]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def framed(c, k):
    return f"User: {c['prompt']}\nAssistant: {c[k]}"


def cosine_to(embs, anchor):
    an = np.linalg.norm(anchor) + 1e-12
    rn = np.linalg.norm(embs, axis=1) + 1e-12
    return (embs @ anchor) / (rn * an)


def acc(better_scores, worse_scores):
    m = np.asarray(better_scores) - np.asarray(worse_scores)
    return (np.sum(m > 0) + 0.5 * np.sum(m == 0)) / len(m)


def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(ORIG)
    warm = read_jsonl(WARM)
    exp_cases = []
    for f in sorted(EXP_DIR.glob("*.jsonl")):
        exp_cases.extend(read_jsonl(f))
    cat_cases = {"firmness": orig, "warmth": warm}
    for c in exp_cases:
        cat_cases.setdefault(c["category"], []).append(c)
    cats = list(cat_cases.keys())
    n_per = {k: len(v) for k, v in cat_cases.items()}
    print(f"Batteries: {n_per}  total={sum(n_per.values())}")

    results = {
        "metadata": {"bases": BASES, "penalties": PENALTIES, "alphas": ALPHAS,
                     "categories": n_per, "models": MODELS},
        "per_model": {},
    }

    for model_name in MODELS:
        ms = model_name.split("/")[-1]
        print(f"\n{'='*84}")
        print(f"MODEL: {ms}")
        print(f"{'='*84}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        def emb(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        cat_embs = {}
        for cat, cases in cat_cases.items():
            be = emb([framed(c, "better") for c in cases])
            we = emb([framed(c, "worse") for c in cases])
            cat_embs[cat] = (be, we)

        # Embed all base + penalty words once
        word_embs = {w: emb([w])[0] for w in set(BASES) | set(PENALTIES)}

        model_res = {}
        # For each base: best penalty+alpha per category, and combined
        print(f"\n{'base':<10s} {'alpha':<7s} {'comb':>6s} {'firm':>6s} {'warm':>6s} "
              f"{'syc':>6s}   best_penalty")
        print("-" * 80)

        for base in BASES:
            bp = word_embs[base]
            base_res = {"penalties": {}}
            # baseline (alpha=0) is same regardless of penalty
            baseline = {}
            for cat in cats:
                be, we = cat_embs[cat]
                baseline[cat] = acc(cosine_to(be, bp), cosine_to(we, bp))
            baseline_comb = sum(baseline.values()) / len(cats)

            best_overall = None  # (comb, penalty, alpha, per_cat)
            for penalty in PENALTIES:
                pen_v = word_embs[penalty]
                pen_res = {}
                for alpha in ALPHAS:
                    per_cat = {}
                    for cat in cats:
                        be, we = cat_embs[cat]
                        sb = cosine_to(be, bp) - alpha * cosine_to(be, pen_v)
                        sw = cosine_to(we, bp) - alpha * cosine_to(we, pen_v)
                        per_cat[cat] = acc(sb, sw)
                    comb = sum(per_cat.values()) / len(cats)
                    pen_res[alpha] = {"combined": round(comb, 4),
                                      "per_category": {k: round(v, 4) for k, v in per_cat.items()}}
                    if best_overall is None or comb > best_overall[0]:
                        best_overall = (comb, penalty, alpha, per_cat)
                base_res["penalties"][penalty] = pen_res

            base_res["baseline_combined"] = round(baseline_comb, 4)
            base_res["best"] = {
                "penalty": best_overall[1], "alpha": best_overall[2],
                "combined": round(best_overall[0], 4),
                "delta_vs_baseline": round(best_overall[0] - baseline_comb, 4),
                "per_category": {k: round(v, 4) for k, v in best_overall[3].items()},
            }
            model_res[base] = base_res

            b = base_res["best"]
            print(f"  {base:<10s} base={baseline_comb:.0%} -> best={b['combined']:.0%} "
                  f"(pen={b['penalty']}, a={b['alpha']}, +{b['delta_vs_baseline']:.0%})")
            pc = b["per_category"]
            print(f"            firm={pc['firmness']:.0%} warm={pc['warmth']:.0%} "
                  f"syc={pc.get('anti_sycophancy',0):.0%}")

        results["per_model"][model_name] = model_res
        del model
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT}")

    # Cross-model: does penalty help consistently for base=good?
    print(f"\n{'='*84}")
    print("CROSS-MODEL: base=good. Does subtracting a penalty beat raw good?")
    print(f"{'='*84}")
    for penalty in PENALTIES:
        line = f"  {penalty:<14s}  "
        for m in MODELS:
            r = results["per_model"][m]["good"]["penalties"][penalty]
            # best alpha for this penalty on this model
            best_a = max(ALPHAS, key=lambda a: r[a]["combined"])
            best_comb = r[best_a]["combined"]
            base = results["per_model"][m]["good"]["baseline_combined"]
            delta = best_comb - base
            line += f"{m.split('/')[-1]}: {base:.0%}->{best_comb:.0%}(a={best_a},{delta:+.0%})  "
        print(line)


if __name__ == "__main__":
    main()
