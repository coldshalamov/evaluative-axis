#!/usr/bin/env python3
"""Correlation analysis for tree decomposition: do same-branch children
correlate more than cross-branch children?

Tests the key prediction of the semantic tree theory on Nomic (best model).
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

TREE = {
    "good":        {"pos": "Good",        "neg": "Bad",         "branch": "root",    "depth": 0},
    "careful":     {"pos": "Careful",     "neg": "Reckless",    "branch": "effort",  "depth": 1},
    "honest":      {"pos": "Honest",      "neg": "Dishonest",   "branch": "truth",   "depth": 1},
    "kind":        {"pos": "Kind",        "neg": "Cruel",       "branch": "warmth",  "depth": 1},
    "wise":        {"pos": "Wise",        "neg": "Foolish",     "branch": "judgment", "depth": 1},
    "helpful":     {"pos": "Helpful",     "neg": "Unhelpful",   "branch": "utility", "depth": 1},
    "thorough":    {"pos": "Thorough",    "neg": "Superficial", "branch": "effort",  "depth": 1},
    "fair":        {"pos": "Fair",        "neg": "Unfair",      "branch": "truth",   "depth": 1},
    "responsible": {"pos": "Responsible", "neg": "Irresponsible","branch": "effort", "depth": 1},
    "clear":       {"pos": "Clear",       "neg": "Confusing",   "branch": "utility", "depth": 1},
    "respectful":  {"pos": "Respectful",  "neg": "Disrespectful","branch": "warmth", "depth": 1},
    "deliberate":  {"pos": "Deliberate",  "neg": "Impulsive",   "branch": "effort",  "depth": 2, "parent": "careful"},
    "attentive":   {"pos": "Attentive",   "neg": "Inattentive", "branch": "effort",  "depth": 2, "parent": "careful"},
    "precise":     {"pos": "Precise",     "neg": "Sloppy",      "branch": "effort",  "depth": 2, "parent": "careful"},
    "cautious":    {"pos": "Cautious",    "neg": "Careless",    "branch": "effort",  "depth": 2, "parent": "careful"},
    "methodical":  {"pos": "Methodical",  "neg": "Haphazard",   "branch": "effort",  "depth": 2, "parent": "careful"},
    "truthful":    {"pos": "Truthful",    "neg": "Deceptive",   "branch": "truth",   "depth": 2, "parent": "honest"},
    "transparent": {"pos": "Transparent", "neg": "Opaque",      "branch": "truth",   "depth": 2, "parent": "honest"},
    "sincere":     {"pos": "Sincere",     "neg": "Insincere",   "branch": "truth",   "depth": 2, "parent": "honest"},
    "forthright":  {"pos": "Forthright",  "neg": "Evasive",     "branch": "truth",   "depth": 2, "parent": "honest"},
    "candid":      {"pos": "Candid",      "neg": "Misleading",  "branch": "truth",   "depth": 2, "parent": "honest"},
    "compassionate":{"pos":"Compassionate","neg":"Indifferent",  "branch": "warmth",  "depth": 2, "parent": "kind"},
    "patient":     {"pos": "Patient",     "neg": "Impatient",   "branch": "warmth",  "depth": 2, "parent": "kind"},
    "gentle":      {"pos": "Gentle",      "neg": "Harsh",       "branch": "warmth",  "depth": 2, "parent": "kind"},
    "encouraging": {"pos": "Encouraging", "neg": "Discouraging","branch": "warmth",  "depth": 2, "parent": "kind"},
    "supportive":  {"pos": "Supportive",  "neg": "Dismissive",  "branch": "warmth",  "depth": 2, "parent": "kind"},
}


def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    all_cases = original + warmth
    n = len(all_cases)

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    embed_fn = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    better_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases])
    worse_embs = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases])

    deltas = {}
    for name, info in TREE.items():
        pos = embed_fn([info["pos"]]).mean(axis=0)
        neg = embed_fn([info["neg"]]).mean(axis=0)
        axis = (pos - neg) / (norm(pos - neg) + 1e-12)
        d = []
        for i in range(n):
            sb = float(np.dot(better_embs[i], axis))
            sw = float(np.dot(worse_embs[i], axis))
            d.append(sb - sw)
        deltas[name] = np.array(d)

    L1 = [k for k, v in TREE.items() if v["depth"] == 1]
    L2 = [k for k, v in TREE.items() if v["depth"] == 2]

    # L1 same vs cross branch
    print("=== L1 SAME vs CROSS BRANCH (Nomic) ===")
    same, cross = [], []
    for i, a in enumerate(L1):
        for j, b in enumerate(L1):
            if i >= j:
                continue
            r = np.corrcoef(deltas[a], deltas[b])[0, 1]
            if TREE[a]["branch"] == TREE[b]["branch"]:
                same.append((a, b, r))
            else:
                cross.append((a, b, r))

    print(f"Same-branch pairs ({len(same)}):")
    for a, b, r in same:
        print(f"  {a:12s} - {b:12s}: r={r:.3f}")
    if same:
        print(f"  Mean: {np.mean([r for _,_,r in same]):.3f}")

    print(f"\nCross-branch pairs ({len(cross)}, showing top 10 by |r|):")
    sorted_cross = sorted(cross, key=lambda x: -abs(x[2]))[:10]
    for a, b, r in sorted_cross:
        print(f"  {a:12s} - {b:12s}: r={r:.3f}")
    print(f"  Full mean: {np.mean([r for _,_,r in cross]):.3f}")

    # L2 same vs cross branch
    print("\n=== L2 SAME vs CROSS BRANCH ===")
    same2, cross2 = [], []
    for i, a in enumerate(L2):
        for j, b in enumerate(L2):
            if i >= j:
                continue
            r = np.corrcoef(deltas[a], deltas[b])[0, 1]
            if TREE[a]["branch"] == TREE[b]["branch"]:
                same2.append(r)
            else:
                cross2.append(r)
    print(f"Same-branch mean r: {np.mean(same2):.3f} (n={len(same2)} pairs)")
    print(f"Cross-branch mean r: {np.mean(cross2):.3f} (n={len(cross2)} pairs)")
    print(f"Difference: {np.mean(same2) - np.mean(cross2):.3f}")

    # Parent-child correlations
    print("\n=== PARENT-CHILD CORRELATIONS ===")
    for child in L2:
        parent = TREE[child].get("parent", "")
        if parent:
            r = np.corrcoef(deltas[child], deltas[parent])[0, 1]
            print(f"  {parent:12s} -> {child:15s}: r={r:.3f}")

    # Good -> L1 child correlations
    print("\n=== GOOD -> L1 CHILD CORRELATIONS ===")
    for child in L1:
        r = np.corrcoef(deltas["good"], deltas[child])[0, 1]
        print(f"  good -> {child:12s}: r={r:.3f}  (branch: {TREE[child]['branch']})")

    # Axis vector cosine similarities (geometric structure)
    print("\n=== AXIS VECTOR COSINE SIMILARITIES ===")
    axes = {}
    for name, info in TREE.items():
        pos = embed_fn([info["pos"]]).mean(axis=0)
        neg = embed_fn([info["neg"]]).mean(axis=0)
        axes[name] = (pos - neg) / (norm(pos - neg) + 1e-12)

    print("Good vs L1 children (axis vector cosine):")
    for child in L1:
        cos = float(np.dot(axes["good"], axes[child]))
        print(f"  good vs {child:12s}: cos={cos:.3f}  (branch: {TREE[child]['branch']})")

    print("\nSame-branch L1 axis cosines:")
    for i, a in enumerate(L1):
        for j, b in enumerate(L1):
            if i >= j:
                continue
            if TREE[a]["branch"] == TREE[b]["branch"]:
                cos = float(np.dot(axes[a], axes[b]))
                print(f"  {a:12s} vs {b:12s}: cos={cos:.3f}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
