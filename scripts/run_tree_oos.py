#!/usr/bin/env python3
"""Out-of-sample validation: run the 5-term tree on expansion cases."""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "notes/research_cycles/battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

TREE = [
    ("careful", "Careful", "Reckless"),
    ("honest", "Honest", "Dishonest"),
    ("helpful", "Helpful", "Unhelpful"),
    ("thorough", "Thorough", "Superficial"),
    ("restrained", "Restrained", "Unrestrained"),
]


def main():
    from sentence_transformers import SentenceTransformer

    cases = []
    for f in sorted(EXP_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                c["source_file"] = f.stem
                cases.append(c)

    n = len(cases)
    print(f"Expansion cases: {n}")
    cats = {}
    for c in cases:
        cat = c["source_file"]
        cats[cat] = cats.get(cat, 0) + 1
    for cat, ct in cats.items():
        print(f"  {cat}: {ct}")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"OUT-OF-SAMPLE: {short}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_texts = []
        worse_texts = []
        for c in cases:
            better_texts.append(f"User: {c['prompt']}\nAssistant: {c['better']}")
            worse_texts.append(f"User: {c['prompt']}\nAssistant: {c['worse']}")
        better_embs = embed(better_texts)
        worse_embs = embed(worse_texts)

        # Raw good
        g = embed(["Good"])[0]
        b = embed(["Bad"])[0]
        gd = (g - b) / (norm(g - b) + 1e-12)
        good_out = [float(np.dot(better_embs[i], gd)) > float(np.dot(worse_embs[i], gd))
                    for i in range(n)]

        # Tree axes
        tree_out = {}
        for name, pos, neg in TREE:
            p = embed([pos])[0]
            nn = embed([neg])[0]
            d = (p - nn) / (norm(p - nn) + 1e-12)
            tree_out[name] = [float(np.dot(better_embs[i], d)) > float(np.dot(worse_embs[i], d))
                              for i in range(n)]

        tree_any = [any(tree_out[a][i] for a in tree_out) for i in range(n)]
        tree_2of5 = [sum(tree_out[a][i] for a in tree_out) >= 2 for i in range(n)]
        tree_maj = [sum(tree_out[a][i] for a in tree_out) >= 3 for i in range(n)]

        print(f"\n  {'Strategy':25s} {'Acc':>5s}  (n={n})")
        print(f"  {'-'*40}")
        print(f"  {'raw good/bad':25s} {sum(good_out)/n:5.0%}")
        for name in tree_out:
            print(f"  {'tree: '+name:25s} {sum(tree_out[name])/n:5.0%}")
        print(f"  {'tree: ANY (1 of 5)':25s} {sum(tree_any)/n:5.0%}")
        print(f"  {'tree: 2 of 5':25s} {sum(tree_2of5)/n:5.0%}")
        print(f"  {'tree: MAJORITY (3/5)':25s} {sum(tree_maj)/n:5.0%}")

        print(f"\n  Per-category:")
        for cat in sorted(cats):
            idx = [i for i, c in enumerate(cases) if c["source_file"] == cat]
            if idx:
                good_acc = sum(good_out[i] for i in idx) / len(idx)
                any_acc = sum(tree_any[i] for i in idx) / len(idx)
                t2_acc = sum(tree_2of5[i] for i in idx) / len(idx)
                print(f"    {cat:30s}  good={good_acc:.0%}  ANY={any_acc:.0%}  2of5={t2_acc:.0%}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
