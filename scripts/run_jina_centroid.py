#!/usr/bin/env python3
"""Centroid experiment on Jina Embeddings via API.

Same protocol as Gemini: train centroid on 70-pair battery, test on 61
expansion cases. Permutation test, per-category breakdown, learning curve.
"""

import json, os, sys, io, time
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

load_dotenv(ROOT / ".env.local")

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/embeddings"

MODELS_TO_TEST = [
    "jina-embeddings-v3",
    "jina-embeddings-v5-text-small",
]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def jina_embed(texts, model, task=None, batch_size=25):
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {
            "model": model,
            "normalized": True,
            "input": batch,
        }
        if task:
            payload["task"] = task

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        }

        for attempt in range(3):
            resp = requests.post(JINA_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:300]}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Jina API failed after 3 attempts: {resp.status_code}")

        data = resp.json()
        batch_embs = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embs)

        if start + batch_size < len(texts):
            time.sleep(0.3)

    return np.array(all_embeddings)


def make_centroid(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, direction):
    correct = sum(1 for i in range(len(better_embs))
                  if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction))
    return correct / len(better_embs)


def permutation_test(train_better, train_worse, test_better, test_worse, observed_acc, n_perms=1000):
    n = len(train_better)
    null_accs = []
    for _ in range(n_perms):
        mask = np.random.randint(0, 2, size=n).astype(bool)
        perm_better = np.where(mask[:, None], train_better, train_worse)
        perm_worse = np.where(mask[:, None], train_worse, train_better)
        d = make_centroid(perm_better, perm_worse)
        acc = pairwise_accuracy(test_better, test_worse, d)
        null_accs.append(acc)
    null_accs = np.array(null_accs)
    p = (null_accs >= observed_acc).mean()
    return p, null_accs.mean(), null_accs.max()


def run_model(model_name, battery, expansion, expansion_categories):
    short = model_name.split("/")[-1] if "/" in model_name else model_name
    print(f"\n{'='*70}")
    print(f"MODEL: {short}")
    print(f"{'='*70}")

    fmt = lambda c: f"User: {c['prompt']}\nAssistant: {c['better']}"
    fmt_w = lambda c: f"User: {c['prompt']}\nAssistant: {c['worse']}"

    print("  Embedding battery (70 pairs)...")
    bat_better = jina_embed([fmt(c) for c in battery], model_name)
    bat_worse = jina_embed([fmt_w(c) for c in battery], model_name)

    print("  Embedding expansion (61 pairs)...")
    exp_better = jina_embed([fmt(c) for c in expansion], model_name)
    exp_worse = jina_embed([fmt_w(c) for c in expansion], model_name)

    dim = bat_better.shape[1]
    print(f"  Embedding dimension: {dim}")

    # Centroid
    centroid = make_centroid(bat_better, bat_worse)
    oos_acc = pairwise_accuracy(exp_better, exp_worse, centroid)
    insample_acc = pairwise_accuracy(bat_better, bat_worse, centroid)
    print(f"\n  Centroid in-sample (battery):  {insample_acc:.1%}")
    print(f"  Centroid OOS (61 expansion):   {oos_acc:.1%}")

    # LOO on battery
    loo_correct = 0
    for i in range(len(battery)):
        mask = np.ones(len(battery), dtype=bool)
        mask[i] = False
        d = make_centroid(bat_better[mask], bat_worse[mask])
        if np.dot(bat_better[i], d) > np.dot(bat_worse[i], d):
            loo_correct += 1
    loo_acc = loo_correct / len(battery)
    print(f"  LOO on battery:                {loo_acc:.1%}")

    # Word baselines
    print("\n  Word baselines on expansion:")
    for word_pair in [("Good", "Bad"), ("Careful", "Reckless"), ("Honest", "Dishonest")]:
        pos_emb = jina_embed([word_pair[0]], model_name)
        neg_emb = jina_embed([word_pair[1]], model_name)
        word_dir = (pos_emb[0] - neg_emb[0])
        word_dir = word_dir / (norm(word_dir) + 1e-12)
        w_acc = pairwise_accuracy(exp_better, exp_worse, word_dir)
        print(f"    {word_pair[0]}/{word_pair[1]}: {w_acc:.1%}")

    # Permutation test: shuffle battery labels, test on expansion
    print(f"\n  Permutation test (1000 shuffles, train on battery, test on 61 expansion)...")
    p_val, null_mean, null_max = permutation_test(bat_better, bat_worse, exp_better, exp_worse, oos_acc)
    print(f"    Observed: {oos_acc:.1%}, Null mean: {null_mean:.1%}, Null max: {null_max:.1%}, p={p_val:.4f}")

    # Per-category breakdown
    print(f"\n  Per-category breakdown:")
    cat_results = {}
    for cat, indices in expansion_categories.items():
        if not indices:
            continue
        cat_better = exp_better[indices]
        cat_worse = exp_worse[indices]
        cat_acc = pairwise_accuracy(cat_better, cat_worse, centroid)
        print(f"    {cat:<25s}: {cat_acc:>6.0%} (n={len(indices)})")
        cat_results[cat] = {"accuracy": round(cat_acc, 4), "n": len(indices)}

    # Learning curve (tested on 61 cases)
    print(f"\n  Learning curve (OOS on 61 expansion cases):")
    lc_results = {}
    for n_train in [5, 10, 20, 30, 50, 70]:
        accs = []
        n_samples = 50
        for _ in range(n_samples):
            idx = np.random.choice(len(battery), size=min(n_train, len(battery)), replace=False)
            d = make_centroid(bat_better[idx], bat_worse[idx])
            accs.append(pairwise_accuracy(exp_better, exp_worse, d))
        mean_acc = np.mean(accs)
        print(f"    {n_train:>3d} pairs: {mean_acc:.1%}")
        lc_results[str(n_train)] = round(float(mean_acc), 4)

    # Logistic probe (both orientations for two classes)
    from sklearn.linear_model import LogisticRegression
    diff_pos = bat_better - bat_worse
    diff_neg = bat_worse - bat_better
    diff_train = np.vstack([diff_pos, diff_neg])
    labels_train = np.array([1]*len(battery) + [0]*len(battery))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(diff_train, labels_train)
    diff_test = exp_better - exp_worse
    probe_acc = (clf.predict(diff_test) == 1).mean()
    print(f"\n  Logistic probe OOS (61 cases): {probe_acc:.1%}")

    # Cosine between centroid and probe direction
    probe_dir = clf.coef_[0] / (norm(clf.coef_[0]) + 1e-12)
    cos_cp = float(np.dot(centroid, probe_dir))
    print(f"  Cosine(centroid, probe):       {cos_cp:.4f}")

    return {
        "model": model_name,
        "dimension": dim,
        "centroid_insample": round(insample_acc, 4),
        "centroid_oos_61": round(oos_acc, 4),
        "loo_battery": round(loo_acc, 4),
        "probe_oos_61": round(float(probe_acc), 4),
        "cosine_centroid_probe": round(cos_cp, 4),
        "permutation_p": round(float(p_val), 4),
        "permutation_null_mean": round(float(null_mean), 4),
        "permutation_null_max": round(float(null_max), 4),
        "per_category": cat_results,
        "learning_curve": lc_results,
    }


def main():
    if not JINA_API_KEY:
        print("ERROR: JINA_API_KEY not set")
        sys.exit(1)

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}")

    # Build category index
    categories = {}
    for i, case in enumerate(expansion):
        cat = case.get("category", "unknown")
        categories.setdefault(cat, []).append(i)
    print(f"Categories: {', '.join(f'{k}({len(v)})' for k, v in categories.items())}")

    all_results = {}
    for model_name in MODELS_TO_TEST:
        try:
            result = run_model(model_name, battery, expansion, categories)
            all_results[model_name] = result
        except Exception as e:
            print(f"\nERROR on {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "jina_centroid_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("JINA CENTROID SUMMARY")
    print(f"{'='*70}")
    for mname, mr in all_results.items():
        if "error" in mr:
            print(f"  {mname}: ERROR - {mr['error']}")
            continue
        print(f"  {mname} ({mr['dimension']}d):")
        print(f"    Centroid OOS (61): {mr['centroid_oos_61']:.0%}")
        print(f"    Probe OOS (61):    {mr['probe_oos_61']:.0%}")
        print(f"    Permutation p:     {mr['permutation_p']:.4f}")
        print(f"    LOO battery:       {mr['loo_battery']:.0%}")


if __name__ == "__main__":
    main()
