#!/usr/bin/env python3
"""Comprehensive analysis of ALL tested terms across 3 models.

Compiles ~35 unique evaluative terms from tree decomposition, prediction v1,
and prediction v2 experiments. For each term, measures:
  - accuracy (overall, orig, warmth)
  - r_good (correlation with good axis deltas)
  - cosine similarity between axis vectors
  - embedding properties (cosine between positive and negative anchors)

Goal: find what geometrically or semantically distinguishes the few
warmth-independent terms from the many biased ones.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

ALL_TERMS = [
    # Tree L0
    ("good", "Good", "Bad"),
    # Tree L1
    ("careful", "Careful", "Reckless"),
    ("honest", "Honest", "Dishonest"),
    ("kind", "Kind", "Cruel"),
    ("wise", "Wise", "Foolish"),
    ("helpful", "Helpful", "Unhelpful"),
    ("thorough", "Thorough", "Superficial"),
    ("fair", "Fair", "Unfair"),
    ("responsible", "Responsible", "Irresponsible"),
    ("clear", "Clear", "Confusing"),
    ("respectful", "Respectful", "Disrespectful"),
    # Tree L2 (careful children)
    ("deliberate", "Deliberate", "Impulsive"),
    ("attentive", "Attentive", "Inattentive"),
    ("precise", "Precise", "Sloppy"),
    ("cautious", "Cautious", "Careless"),
    ("methodical", "Methodical", "Haphazard"),
    # Tree L2 (honest children)
    ("truthful", "Truthful", "Deceptive"),
    ("transparent", "Transparent", "Opaque"),
    ("sincere", "Sincere", "Insincere"),
    ("forthright", "Forthright", "Evasive"),
    ("candid", "Candid", "Misleading"),
    # Tree L2 (kind children)
    ("compassionate", "Compassionate", "Indifferent"),
    ("patient", "Patient", "Impatient"),
    ("gentle", "Gentle", "Harsh"),
    ("encouraging", "Encouraging", "Discouraging"),
    ("supportive", "Supportive", "Dismissive"),
    # Prediction v1
    ("prudent", "Prudent", "Reckless"),
    ("vigilant", "Vigilant", "Negligent"),
    ("scrupulous", "Scrupulous", "Careless"),
    ("measured", "Measured", "Impulsive"),
    ("exemplary", "Exemplary", "Terrible"),
    ("superb", "Superb", "Awful"),
    ("commendable", "Commendable", "Deplorable"),
    ("outstanding", "Outstanding", "Abysmal"),
    ("gracious", "Gracious", "Rude"),
    ("benevolent", "Benevolent", "Malicious"),
    # Prediction v2
    ("systematic", "Systematic", "Disorganized"),
    ("rigorous", "Rigorous", "Lax"),
    ("stringent", "Stringent", "Lenient"),
    ("accurate", "Accurate", "Inaccurate"),
    ("logical", "Logical", "Illogical"),
    ("analytical", "Analytical", "Muddled"),
    ("admirable", "Admirable", "Contemptible"),
    ("noble", "Noble", "Ignoble"),
    ("generous", "Generous", "Selfish"),
    ("worthy", "Worthy", "Unworthy"),
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

    orig = read_jsonl(BATTERY)
    warmth = read_jsonl(WARMTH)
    cases = orig + warmth
    n = len(cases)
    n_orig = len(orig)

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases])
        worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases])

        axis_vectors = {}
        axis_deltas = {}
        axis_correct = {}
        anchor_cosines = {}

        for name, pos, neg in ALL_TERMS:
            p_emb = embed_fn([pos])[0]
            n_emb = embed_fn([neg])[0]

            anchor_cos = float(np.dot(p_emb, n_emb) / (norm(p_emb) * norm(n_emb) + 1e-12))
            anchor_cosines[name] = anchor_cos

            axis = (p_emb - n_emb) / (norm(p_emb - n_emb) + 1e-12)
            axis_vectors[name] = axis

            deltas = []
            correct = []
            for i in range(n):
                d = float(np.dot(better_embs[i], axis) - np.dot(worse_embs[i], axis))
                deltas.append(d)
                correct.append(1 if d > 0 else 0)
            axis_deltas[name] = deltas
            axis_correct[name] = correct

        good_deltas = axis_deltas["good"]
        good_vec = axis_vectors["good"]

        # Compute metrics for each term
        print(f"\n{'Term':15s} {'Acc':>5s} {'Orig':>5s} {'Warm':>5s} {'r_good':>7s} "
              f"{'cos_gd':>7s} {'anch_cos':>8s} {'Status':>12s}")
        print("-" * 85)

        model_results = {}
        for name, pos, neg in ALL_TERMS:
            acc_k = sum(axis_correct[name])
            acc = acc_k / n

            orig_acc = sum(axis_correct[name][i] for i in range(n_orig)) / n_orig
            warm_acc = sum(axis_correct[name][i] for i in range(n_orig, n)) / (n - n_orig)

            r, p_val = pearsonr(axis_deltas[name], good_deltas)

            cos_with_good = float(np.dot(axis_vectors[name], good_vec))

            if abs(r) < 0.3:
                status = "INDEPENDENT"
            elif r > 0.4:
                status = "warmth-biased"
            else:
                status = "borderline"

            print(f"{name:15s} {acc:5.0%} {orig_acc:5.0%} {warm_acc:5.0%} {r:+7.2f} "
                  f"{cos_with_good:+7.2f} {anchor_cosines[name]:8.3f} {status:>12s}")

            model_results[name] = {
                "accuracy": round(float(acc), 3),
                "orig_accuracy": round(float(orig_acc), 3),
                "warmth_accuracy": round(float(warm_acc), 3),
                "r_good": round(float(r), 3),
                "p_value": round(float(p_val), 4),
                "cos_with_good_axis": round(float(cos_with_good), 3),
                "anchor_cosine": round(float(anchor_cosines[name]), 3),
                "status": status,
            }

        # Summary: independent vs biased
        independent = [name for name in model_results if model_results[name]["status"] == "INDEPENDENT"]
        biased = [name for name in model_results if model_results[name]["status"] == "warmth-biased"]

        print(f"\nIndependent terms ({len(independent)}): {', '.join(independent)}")
        print(f"Biased terms ({len(biased)}): {', '.join(biased)}")

        if independent and biased:
            ind_cos = np.mean([model_results[n]["cos_with_good_axis"] for n in independent])
            bias_cos = np.mean([model_results[n]["cos_with_good_axis"] for n in biased])
            ind_anch = np.mean([model_results[n]["anchor_cosine"] for n in independent])
            bias_anch = np.mean([model_results[n]["anchor_cosine"] for n in biased])
            ind_acc = np.mean([model_results[n]["accuracy"] for n in independent])
            bias_acc = np.mean([model_results[n]["accuracy"] for n in biased])

            print(f"\nGroup comparison:")
            print(f"  {'':15s} {'cos_good':>10s} {'anchor_cos':>10s} {'accuracy':>10s}")
            print(f"  {'Independent':15s} {ind_cos:+10.3f} {ind_anch:10.3f} {ind_acc:10.0%}")
            print(f"  {'Biased':15s} {bias_cos:+10.3f} {bias_anch:10.3f} {bias_acc:10.0%}")

        all_results[short] = model_results

        del model
        gc.collect()

    # Cross-model consistency
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL ANALYSIS")
    print(f"{'='*80}")

    print(f"\n{'Term':15s} {'r_good (3 models)':>30s}  {'Independent on':>15s}")
    print("-" * 70)

    term_independence = {}
    for name, pos, neg in ALL_TERMS:
        rs = [all_results[m][name]["r_good"] for m in all_results]
        n_indep = sum(1 for r in rs if abs(r) < 0.3)
        accs = [all_results[m][name]["accuracy"] for m in all_results]
        mean_acc = np.mean(accs)
        print(f"{name:15s} [{', '.join(f'{r:+.2f}' for r in rs)}]  {n_indep}/3 models  "
              f"mean_acc={mean_acc:.0%}")
        term_independence[name] = n_indep

    # Terms independent on 2+ models
    print(f"\n--- Terms independent on >= 2/3 models ---")
    for name, pos, neg in ALL_TERMS:
        if term_independence[name] >= 2:
            rs = [all_results[m][name]["r_good"] for m in all_results]
            accs = [all_results[m][name]["accuracy"] for m in all_results]
            print(f"  {name:15s} r_good=[{', '.join(f'{r:+.2f}' for r in rs)}]  "
                  f"acc=[{', '.join(f'{a:.0%}' for a in accs)}]")

    # Save results
    out = ROOT / "notes/research_cycles/tree_decomposition/comprehensive_term_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "Comprehensive analysis of all tested evaluative terms",
        "date": "2026-06-28",
        "n_terms": len(ALL_TERMS),
        "n_cases": n,
        "results": all_results,
        "cross_model_independence": {name: term_independence[name] for name, _, _ in ALL_TERMS},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
