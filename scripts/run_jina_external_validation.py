#!/usr/bin/env python3
"""Test centroid on external datasets with Jina models + strict filtering.

Key question: does Jina v5 (our best model at 80% OOS) succeed where
local models failed on SHP/UltraFeedback?

Also tests strict SHP filtering (score_ratio >= 5, >= 10) to see if
cleaner labels improve OOS accuracy.
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
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

load_dotenv(ROOT / ".env.local")

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL = "jina-embeddings-v5-text-small"


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def jina_embed(texts, batch_size=25):
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {"model": MODEL, "normalized": True, "input": batch}
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
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Jina API failed: {resp.status_code}")
        data = resp.json()
        all_embeddings.extend([item["embedding"] for item in data["data"]])
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


def permutation_test(train_b, train_w, test_b, test_w, observed, n_perms=500):
    n = len(train_b)
    null_accs = []
    for _ in range(n_perms):
        mask = np.random.randint(0, 2, size=n).astype(bool)
        pb = np.where(mask[:, None], train_b, train_w)
        pw = np.where(mask[:, None], train_w, train_b)
        d = make_centroid(pb, pw)
        null_accs.append(pairwise_accuracy(test_b, test_w, d))
    null_accs = np.array(null_accs)
    return (null_accs >= observed).mean(), null_accs.mean()


def load_shp(min_score_ratio=2.0, n_max=1000, split="test"):
    from datasets import load_dataset
    print(f"Loading SHP ({split}, score_ratio >= {min_score_ratio})...")
    ds = load_dataset("stanfordnlp/SHP", split=split)
    pairs = []
    for row in ds:
        sr = row.get("score_ratio") or 0
        if sr < min_score_ratio:
            continue
        if row["labels"] == 1:
            better, worse = row["human_ref_A"], row["human_ref_B"]
        else:
            better, worse = row["human_ref_B"], row["human_ref_A"]
        prompt = row.get("history", "") or ""
        if len(better) < 20 or len(worse) < 20:
            continue
        if len(better) > 2000 or len(worse) > 2000:
            continue
        pairs.append({"prompt": prompt[:1500], "better": better, "worse": worse,
                       "score_ratio": sr})
    print(f"  {len(pairs)} pairs meet criteria")
    rng = np.random.RandomState(42)
    if len(pairs) > n_max:
        idx = rng.choice(len(pairs), n_max, replace=False)
        pairs = [pairs[i] for i in idx]
    return pairs


def load_ultrafeedback(n_max=1000):
    from datasets import load_dataset
    print("Loading UltraFeedback binarized...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    except Exception:
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned",
                          split="test")
    pairs = []
    for row in ds:
        prompt = row.get("prompt", "") or ""
        chosen = row.get("chosen", [])
        rejected = row.get("rejected", [])
        ct = next((m["content"] for m in chosen if m["role"] == "assistant"), None)
        rt = next((m["content"] for m in rejected if m["role"] == "assistant"), None)
        if not ct or not rt or len(ct) < 20 or len(rt) < 20:
            continue
        if len(ct) > 2000 or len(rt) > 2000:
            continue
        pairs.append({"prompt": prompt[:1500], "better": ct, "worse": rt})
    print(f"  {len(pairs)} pairs")
    rng = np.random.RandomState(42)
    if len(pairs) > n_max:
        idx = rng.choice(len(pairs), n_max, replace=False)
        pairs = [pairs[i] for i in idx]
    return pairs


def run_within_dataset(name, pairs, n_train=70, n_repeats=10):
    """Train on n_train pairs, test on rest. Repeat with different splits."""
    print(f"\n{'='*70}")
    print(f"DATASET: {name} ({len(pairs)} pairs)")
    print(f"{'='*70}")

    fmt_b = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in pairs]
    fmt_w = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in pairs]

    print("  Embedding all pairs...")
    all_better = jina_embed(fmt_b)
    all_worse = jina_embed(fmt_w)

    # In-sample centroid (all data)
    full_dir = make_centroid(all_better, all_worse)
    full_acc = pairwise_accuracy(all_better, all_worse, full_dir)
    print(f"  Full in-sample accuracy: {full_acc:.1%}")

    # Multiple random train/test splits
    rng = np.random.RandomState(42)
    oos_accs = []
    for r in range(n_repeats):
        perm = rng.permutation(len(pairs))
        tr, te = perm[:n_train], perm[n_train:]
        d = make_centroid(all_better[tr], all_worse[tr])
        acc = pairwise_accuracy(all_better[te], all_worse[te], d)
        oos_accs.append(acc)
    mean_oos = np.mean(oos_accs)
    std_oos = np.std(oos_accs)
    print(f"  OOS accuracy ({n_train}-train, {n_repeats} splits): {mean_oos:.1%} ± {std_oos:.1%}")
    print(f"    Range: {min(oos_accs):.1%} – {max(oos_accs):.1%}")

    # Permutation test on best split
    best_split_idx = np.argmax(oos_accs)
    perm = np.random.RandomState(42 + best_split_idx).permutation(len(pairs))
    tr, te = perm[:n_train], perm[n_train:]
    obs_acc = oos_accs[best_split_idx]
    p_val, null_mean = permutation_test(all_better[tr], all_worse[tr],
                                         all_better[te], all_worse[te],
                                         obs_acc, n_perms=500)
    print(f"  Permutation (best split): p={p_val:.4f}, null_mean={null_mean:.1%}")

    # Also try larger training sets
    for n_tr in [70, 200, 500]:
        if n_tr >= len(pairs):
            continue
        accs = []
        for r in range(n_repeats):
            perm = rng.permutation(len(pairs))
            tr, te = perm[:n_tr], perm[n_tr:]
            d = make_centroid(all_better[tr], all_worse[tr])
            accs.append(pairwise_accuracy(all_better[te], all_worse[te], d))
        print(f"  {n_tr}-train OOS: {np.mean(accs):.1%} ± {np.std(accs):.1%} (test n={len(pairs)-n_tr})")

    return {
        "n_pairs": len(pairs),
        "full_insample": round(full_acc, 4),
        "oos_mean": round(float(mean_oos), 4),
        "oos_std": round(float(std_oos), 4),
        "oos_min": round(float(min(oos_accs)), 4),
        "oos_max": round(float(max(oos_accs)), 4),
        "permutation_p": round(float(p_val), 4),
    }


def main():
    if not JINA_API_KEY:
        print("ERROR: JINA_API_KEY not set")
        sys.exit(1)

    results = {}

    # SHP with different filtering thresholds
    for ratio in [2.0, 5.0, 10.0]:
        try:
            pairs = load_shp(min_score_ratio=ratio, n_max=1000)
            if len(pairs) < 100:
                print(f"  Too few pairs at ratio {ratio}, skipping")
                continue
            results[f"SHP_ratio{ratio}"] = run_within_dataset(
                f"SHP (ratio>={ratio})", pairs)
        except Exception as e:
            print(f"  ERROR loading SHP ratio {ratio}: {e}")

    # UltraFeedback
    try:
        pairs = load_ultrafeedback(n_max=1000)
        results["UltraFeedback"] = run_within_dataset("UltraFeedback", pairs)
    except Exception as e:
        print(f"  ERROR loading UltraFeedback: {e}")

    # Our battery for comparison (using expansion as test)
    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    from pathlib import Path
    expansion_dir = ROOT / "notes" / "research_cycles" / "battery_expansion"
    expansion = []
    for f in sorted(expansion_dir.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    print(f"\n{'='*70}")
    print(f"COMPARISON: Our battery (70 train) -> expansion (61 test)")
    print(f"{'='*70}")

    bat_b = jina_embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in battery])
    bat_w = jina_embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery])
    exp_b = jina_embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion])
    exp_w = jina_embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion])

    our_dir = make_centroid(bat_b, bat_w)
    our_oos = pairwise_accuracy(exp_b, exp_w, our_dir)
    print(f"  Our centroid OOS (61 expansion): {our_oos:.1%}")

    results["our_battery"] = {
        "n_train": len(battery),
        "n_test": len(expansion),
        "oos_accuracy": round(our_oos, 4),
    }

    # Save
    out_path = OUT_DIR / "jina_external_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Jina v5 on external datasets vs our battery")
    print(f"{'='*70}")
    for name, r in results.items():
        if "oos_mean" in r:
            print(f"  {name}: {r['oos_mean']:.1%} ± {r['oos_std']:.1%} OOS "
                  f"(insample {r['full_insample']:.1%}, p={r['permutation_p']:.4f})")
        elif "oos_accuracy" in r:
            print(f"  {name}: {r['oos_accuracy']:.1%} OOS")


if __name__ == "__main__":
    main()
