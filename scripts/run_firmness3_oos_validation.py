#!/usr/bin/env python3
"""Out-of-sample validation of the firmness_3 aggregate (hard, careful, thorough).

The in-sample finding (from run_normalized_aggregate.py):
  - The "balanced matrix" hypothesis FAILS: adding warmth axes to firmness
    axes makes things worse, not better.
  - The reason: asymmetry. Firmness axes are GENERALISTS (balanced on both
    splits). Warmth axes are SPECIALISTS (fail on firmness). Warmth axes
    don't catch firmness failures because firmness axes already handle those.
  - Best config: firmness_3 raw sum = hard + careful + thorough.
    Beats careful alone on 2 of 3 models, balanced firm≈warm.

This script validates firmness_3 on UNSEEN cases:
  - 20 battery_expansion cases (nuance, factual, conciseness, creative)
  - plus the balanced battery for in-sample reference

If firmness_3 holds at ~55-60% balanced OOS, it's a real, publishable
configuration: three competence-branch words that together self-regularize
without needing a warmth counterweight.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

EXPANSION = ROOT / "notes" / "research_cycles" / "battery_expansion"
ORIG = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
OUT = ROOT / "notes" / "research_cycles" / "normalized_aggregate" / "firmness3_oos_validation.json"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

FIRMNESS_3 = [("Hard", "Soft"), ("Careful", "Reckless"), ("Thorough", "Superficial")]
SINGLE_REFS = [("Careful", "Reckless"), ("Hard", "Soft"), ("Thorough", "Superficial")]

def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def framed(c, k):
    return f"User: {c['prompt']}\nAssistant: {c[k]}"

def main():
    from sentence_transformers import SentenceTransformer

    # Load all OOS cases (4 expansion categories)
    oos_cases = []
    for f in sorted(EXPANSION.glob("*.jsonl")):
        oos_cases.extend(read_jsonl(f))
    orig = read_jsonl(ORIG)
    warm = read_jsonl(WARMTH)

    print(f"OOS cases: {len(oos_cases)} (expansion, never seen)")
    print(f"In-sample: {len(orig)} firmness + {len(warm)} warmth")

    results = {"metadata": {"firmness_3": FIRMNESS_3, "n_oos": len(oos_cases)},
               "per_model": {}}

    for model_name in MODELS:
        ms = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {ms}")
        print(f"{'='*80}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        def emb(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Axis vectors
        axis_vecs = []
        for pos, neg in FIRMNESS_3:
            pe = emb([pos])[0]; ne = emb([neg])[0]
            v = pe - ne; v = v / (np.linalg.norm(v) + 1e-12)
            axis_vecs.append(v)
        careful_v = axis_vecs[1]  # Careful alone for reference

        def score_cases(cases, method):
            be = emb([framed(c, "better") for c in cases])
            we = emb([framed(c, "worse") for c in cases])
            if method == "careful_alone":
                sb = be @ careful_v; sw = we @ careful_v
            elif method == "firmness3_raw":
                sb = sum(be @ v for v in axis_vecs)
                sw = sum(we @ v for v in axis_vecs)
            margins = sb - sw
            return margins

        def acc(margins):
            m = np.asarray(margins)
            return (np.sum(m > 0) + 0.5 * np.sum(m == 0)) / len(m)

        model_res = {}
        for label, cases, n_firm in [
            ("IN-SAMPLE combined", orig + warm, len(orig)),
            ("  firmness split", orig, len(orig)),
            ("  warmth split", warm, len(warm)),
            ("OOS expansion", oos_cases, None),
        ]:
            row = {}
            for method in ["careful_alone", "firmness3_raw"]:
                margins = score_cases(cases, method)
                a = acc(margins)
                row[method] = round(a, 4)
            model_res[label] = row
            if "split" in label:
                indent = "    "
            else:
                indent = "  "
            print(f"{indent}{label:<24s} careful={row['careful_alone']:.0%}  "
                  f"firmness3={row['firmness3_raw']:.0%}  "
                  f"delta={row['firmness3_raw']-row['careful_alone']:+.0%}")

        # Per-category OOS breakdown
        print(f"  OOS by category:")
        cats = {}
        for c in oos_cases:
            cats.setdefault(c["category"], []).append(c)
        for cat, ccases in sorted(cats.items()):
            m_c = score_cases(ccases, "careful_alone")
            m_f = score_cases(ccases, "firmness3_raw")
            print(f"    {cat:<26s} careful={acc(m_c):.0%}  "
                  f"firmness3={acc(m_f):.0%}")

        results["per_model"][model_name] = model_res
        del model

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT}")

    # Cross-model OOS verdict
    print(f"\n{'='*80}")
    print("CROSS-MODEL OOS VERDICT (firmness3 vs careful alone)")
    print(f"{'='*80}")
    print(f"{'Model':<28s} {'careful':>8s} {'firmness3':>10s} {'delta':>7s} {'holds?':>7s}")
    for m in MODELS:
        ms = m.split("/")[-1]
        r = results["per_model"][m]["OOS expansion"]
        d = r["firmness3_raw"] - r["careful_alone"]
        holds = "YES" if r["firmness3_raw"] >= 0.55 else "no"
        print(f"  {ms:<26s} {r['careful_alone']:7.0%} {r['firmness3_raw']:9.0%} "
              f"{d:+6.0%} {holds:>7s}")

if __name__ == "__main__":
    main()
