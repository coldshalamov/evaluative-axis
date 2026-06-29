#!/usr/bin/env python3
"""Experiment 2: Cross-Concept Compound Anchors.

The tree decomposition (E-01) tested child terms of the SAME parent scored
independently. This tests a different combination: putting two terms from
DIFFERENT branches into ONE anchor text string, used as a single anchor pair.

The question: does combining a firmness term with a warmth term IN THE SAME
ANCHOR TEXT create a single axis that covers both branches, or does the
combination dilute both signals (the same failure mode as compositing axis
vectors by averaging)?

For each compound anchor we compare it against:
  - the component single words scored INDEPENDENTLY and SUMMED (the
    no-averaging baseline from AGENTS.md), and
  - the better of the two component single words alone.

This is a direct dilution test: compound-in-one-string vs independent-sum.

Note: these compound anchor strings are scored with BOTH methods from
Experiment 1 (bipolar and cosine-to-positive), because cosine-to-positive
just won on 14/15 single-word axes. Reporting both lets us see whether the
dilution story changes under the better scoring method.

Rules (AGENTS.md / RESEARCH_CONTEXT.md):
  - All three local models, both splits + combined.
  - User/Assistant framing for response embeddings.
  - Independent-sum baseline uses SUM of independent per-axis scores
    (no averaging of axis vectors).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BATTERY_ORIGINAL = (
    ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BATTERY_WARMTH = (
    ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
)
OUTPUT_PATH = (
    ROOT / "notes" / "research_cycles" / "compound_anchors" / "compound_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# Compound anchor texts and the single-word components they are built from.
# Each compound's components are scored independently and summed for the
# dilution comparison. "all_top3" components are the three words it lists.
COMPOUND_ANCHORS = {
    "careful_kind":        (["Careful and kind"],         ["Reckless and cruel"],
                            [("Careful", "Reckless"), ("Kind", "Cruel")]),
    "honest_gentle":       (["Honest and gentle"],        ["Dishonest and harsh"],
                            [("Honest", "Dishonest"), ("Gentle", "Harsh")]),
    "thorough_supportive": (["Thorough and supportive"],  ["Superficial and dismissive"],
                            [("Thorough", "Superficial"), ("Supportive", "Unsupportive")]),
    "wise_patient":        (["Wise and patient"],         ["Foolish and impatient"],
                            [("Wise", "Foolish"), ("Patient", "Impatient")]),
    "clear_fair":          (["Clear and fair"],           ["Confusing and unfair"],
                            [("Clear", "Confusing"), ("Fair", "Unfair")]),
    "firm_kind":           (["Firm but kind"],            ["Weak and cruel"],
                            [("Hard", "Soft"), ("Kind", "Cruel")]),
    "honest_warm":         (["Honest but warm"],          ["Dishonest and cold"],
                            [("Honest", "Dishonest"), ("Kind", "Cruel")]),
    "precise_encouraging": (["Precise and encouraging"],  ["Sloppy and discouraging"],
                            [("Precise", "Sloppy"), ("Supportive", "Unsupportive")]),
    "careful_kind_honest": (["Careful, kind, and honest"],["Reckless, cruel, and dishonest"],
                            [("Careful", "Reckless"), ("Kind", "Cruel"), ("Honest", "Dishonest")]),
    "all_top3":            (["Hard, kind, and careful"],  ["Soft, cruel, and reckless"],
                            [("Hard", "Soft"), ("Kind", "Cruel"), ("Careful", "Reckless")]),
}


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def framed(case, key):
    return f"User: {case['prompt']}\nAssistant: {case[key]}"


def cosine_to_anchor(response_embs, anchor):
    anchor_norm = np.linalg.norm(anchor) + 1e-12
    response_norms = np.linalg.norm(response_embs, axis=1) + 1e-12
    return (response_embs @ anchor) / (response_norms * anchor_norm)


def accuracy(better_scores, worse_scores):
    margins = np.asarray(better_scores) - np.asarray(worse_scores)
    correct = float(np.sum(margins > 0))
    ties = float(np.sum(margins == 0))
    return (correct + 0.5 * ties) / len(margins)


def bipolar_axis(pos_emb, neg_emb):
    axis = pos_emb - neg_emb
    return axis / (np.linalg.norm(axis) + 1e-12)


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    n_orig = len(original)
    combined = original + warmth

    print("Experiment 2: Cross-Concept Compound Anchors")
    print(f"Firmness: {n_orig}, Warmth: {len(warmth)}, Combined: {len(combined)}")
    print(f"Compounds: {len(COMPOUND_ANCHORS)}")

    results = {
        "experiment": "Cross-Concept Compound Anchors",
        "n_firmness": n_orig,
        "n_warmth": len(warmth),
        "compound_anchors": {
            k: {"pos": v[0], "neg": v[1], "components": v[2]}
            for k, v in COMPOUND_ANCHORS.items()
        },
        "models": MODELS,
        "per_model": {},
        "dilution_summary": [],
    }

    # Collect every single-word component used, to embed once per model.
    all_components = set()
    for _, _, comps in COMPOUND_ANCHORS.values():
        for pos, neg in comps:
            all_components.add(pos)
            all_components.add(neg)
    all_components = sorted(all_components)

    for model_name in MODELS:
        print(f"\n{'='*88}")
        print(f"MODEL: {model_name}")
        print(f"{'='*88}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        def embed_fn(texts):
            return model.encode(texts, show_progress_bar=False,
                                convert_to_numpy=True)

        # Embed responses per split
        split_embs = {
            "firmness": (
                embed_fn([framed(c, "better") for c in original]),
                embed_fn([framed(c, "worse") for c in original]),
            ),
            "warmth": (
                embed_fn([framed(c, "better") for c in warmth]),
                embed_fn([framed(c, "worse") for c in warmth]),
            ),
            "combined": (
                embed_fn([framed(c, "better") for c in combined]),
                embed_fn([framed(c, "worse") for c in combined]),
            ),
        }

        # Embed single-word components once
        comp_embs = {w: embed_fn([w])[0] for w in all_components}

        # Embed compound anchor strings once
        compound_pos = {k: embed_fn(v[0])[0] for k, v in COMPOUND_ANCHORS.items()}
        compound_neg = {k: embed_fn(v[1])[0] for k, v in COMPOUND_ANCHORS.items()}

        model_results = {}
        print(f"\n{'Compound':<24s} {'method':<10s} {'compound':>9s} "
              f"{'sum-comp':>9s} {'best-comp':>9s} {'dilute?':>8s}")
        print("-" * 80)

        for cname, (pos_list, neg_list, comps) in COMPOUND_ANCHORS.items():
            pos_emb = compound_pos[cname]
            neg_emb = compound_neg[cname]
            comp_axis_vecs = []
            comp_cos_pos = []
            for pos, neg in comps:
                pe = comp_embs[pos]
                ne = comp_embs[neg]
                comp_axis_vecs.append((pos, neg, bipolar_axis(pe, ne), pe))
                comp_cos_pos.append(pe)

            entry = {}
            for method in ("bipolar", "cosine"):
                split_res = {}
                for split, (be, we) in split_embs.items():
                    # Compound (single anchor string)
                    if method == "bipolar":
                        ax = bipolar_axis(pos_emb, neg_emb)
                        c_b = be @ ax
                        c_w = we @ ax
                    else:
                        c_b = cosine_to_anchor(be, pos_emb)
                        c_w = cosine_to_anchor(we, pos_emb)
                    comp_acc = accuracy(c_b, c_w)

                    # Independent-sum of component single words (no averaging)
                    sum_b = np.zeros(len(be))
                    sum_w = np.zeros(len(we))
                    for pos, neg, axv, pe in comp_axis_vecs:
                        if method == "bipolar":
                            sum_b += be @ axv
                            sum_w += we @ axv
                        else:
                            sum_b += cosine_to_anchor(be, pe)
                            sum_w += cosine_to_anchor(we, pe)
                    sum_acc = accuracy(sum_b, sum_w)

                    # Best single component alone
                    comp_only_accs = []
                    for pos, neg, axv, pe in comp_axis_vecs:
                        if method == "bipolar":
                            a = accuracy(be @ axv, we @ axv)
                        else:
                            a = accuracy(cosine_to_anchor(be, pe),
                                         cosine_to_anchor(we, pe))
                        comp_only_accs.append(a)
                    best_comp = max(comp_only_accs)

                    split_res[split] = {
                        "compound": round(comp_acc, 4),
                        "independent_sum": round(sum_acc, 4),
                        "best_component": round(best_comp, 4),
                        "dilution": round(comp_acc - sum_acc, 4),
                    }
                entry[method] = split_res

            model_results[cname] = entry

            # Print using cosine (the better method from Exp 1) for the headline
            m = "cosine"
            r = entry[m]["combined"]
            dil = r["compound"] - r["independent_sum"]
            flag = "DILUTE" if dil < -0.05 else ("boost" if dil > 0.05 else "~")
            print(f"  {cname:<22s} {m:<10s} {r['compound']:8.0%} "
                  f"{r['independent_sum']:8.0%} {r['best_component']:8.0%} "
                  f"{dil:+7.0%} {flag}")
            m = "bipolar"
            r = entry[m]["combined"]
            dil = r["compound"] - r["independent_sum"]
            flag = "DILUTE" if dil < -0.05 else ("boost" if dil > 0.05 else "~")
            print(f"  {'':22s} {m:<10s} {r['compound']:8.0%} "
                  f"{r['independent_sum']:8.0%} {r['best_component']:8.0%} "
                  f"{dil:+7.0%} {flag}")

        results["per_model"][model_name] = model_results
        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")

    # ---- Cross-model dilution summary ----
    print(f"\n{'='*88}")
    print("CROSS-MODEL SUMMARY: compound vs independent-sum (combined, cosine)")
    print("Negative dilution = compound-in-one-string is WORSE than summing")
    print("the components independently (the compositing failure mode).")
    print(f"{'='*88}")
    print(f"{'Compound':<24s} {'Snowflake':>10s} {'BGE-M3':>10s} {'Nomic':>10s} "
          f"{'verdict':>10s}")
    print("-" * 70)
    for cname in COMPOUND_ANCHORS:
        vals = []
        for m in MODELS:
            r = results["per_model"][m][cname]["cosine"]["combined"]
            vals.append(r["compound"] - r["independent_sum"])
        agree = sum(1 for v in vals if v < -0.05)
        verdict = "DILUTE" if agree >= 2 else ("boost" if sum(1 for v in vals if v > 0.05) >= 2 else "mixed")
        print(f"  {cname:<22s} {vals[0]:+9.0%} {vals[1]:+9.0%} {vals[2]:+9.0%} "
              f"{verdict:>10s}")
        results["dilution_summary"].append({
            "compound": cname, "deltas_cosine_combined": vals, "verdict": verdict,
        })

    # Does any compound beat the best component on >=2 models?
    print(f"\nDoes any compound BEAT its best single component (combined, cosine)?")
    any_beat = False
    for cname in COMPOUND_ANCHORS:
        beats = 0
        for m in MODELS:
            r = results["per_model"][m][cname]["cosine"]["combined"]
            if r["compound"] > r["best_component"]:
                beats += 1
        if beats >= 2:
            any_beat = True
            print(f"  {cname}: beats best component on {beats}/3 models")
    if not any_beat:
        print("  None — no compound beats its best single component on >=2 models.")
        print("  Combining terms in one anchor string does not create a")
        print("  super-additive axis; it dilutes or ties.")


if __name__ == "__main__":
    main()
