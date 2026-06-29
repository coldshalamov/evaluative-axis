#!/usr/bin/env python3
"""Test centroid on SHP at scale with Jina v5.

The previous test used 70 training pairs (matching our battery). That's
artificially small. SHP has 9000+ clean pairs. Use them.

Also tests response-only format (no User:/Assistant: wrapper) since
meta-validity showed response-only beats full-format on 2/3 models.

This is the credibility test: does the centroid work on a real public
dataset at real scale?
"""

import json, os, sys, io, time, hashlib
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

load_dotenv(ROOT / ".env.local")

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL = "jina-embeddings-v5-text-small"

CACHE_DIR = OUT_DIR / "shp_embedding_cache"


def jina_embed_cached(texts, batch_size=25):
    """Embed texts via Jina API with disk cache to avoid re-embedding."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = [None] * len(texts)
    to_embed = []
    to_embed_idx = []

    for i, text in enumerate(texts):
        key = hashlib.md5(text.encode()).hexdigest()
        cache_path = CACHE_DIR / f"{key}.npy"
        if cache_path.exists():
            embeddings[i] = np.load(cache_path)
        else:
            to_embed.append(text)
            to_embed_idx.append(i)

    if to_embed:
        print(f"    Cache hit: {len(texts) - len(to_embed)}/{len(texts)}, embedding {len(to_embed)} new...")
        new_embs = _jina_embed_raw(to_embed, batch_size)
        for j, idx in enumerate(to_embed_idx):
            embeddings[idx] = new_embs[j]
            key = hashlib.md5(to_embed[j].encode()).hexdigest()
            np.save(CACHE_DIR / f"{key}.npy", new_embs[j])
    else:
        print(f"    All {len(texts)} from cache")

    return np.array(embeddings)


def _jina_embed_raw(texts, batch_size=25):
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {"model": MODEL, "normalized": True, "input": batch}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        for attempt in range(5):
            resp = requests.post(JINA_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait = 2 ** attempt * 3
                time.sleep(wait)
            else:
                print(f"    API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Jina API failed: {resp.status_code}")
        data = resp.json()
        all_embeddings.extend([item["embedding"] for item in data["data"]])
        if start + batch_size < len(texts):
            time.sleep(0.2)
        if (start // batch_size) % 20 == 0 and start > 0:
            print(f"    Embedded {start + len(batch)}/{len(texts)}...")
    return np.array(all_embeddings)


def make_centroid(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, direction):
    scores_b = better_embs @ direction
    scores_w = worse_embs @ direction
    return (scores_b > scores_w).mean()


def permutation_test(train_b, train_w, test_b, test_w, observed, n_perms=1000):
    n = len(train_b)
    null_accs = []
    for _ in range(n_perms):
        mask = np.random.randint(0, 2, size=n).astype(bool)
        pb = np.where(mask[:, None], train_b, train_w)
        pw = np.where(mask[:, None], train_w, train_b)
        d = make_centroid(pb, pw)
        null_accs.append(pairwise_accuracy(test_b, test_w, d))
    null_accs = np.array(null_accs)
    return float((null_accs >= observed).mean()), float(null_accs.mean())


def load_shp_all(min_score_ratio=2.0):
    from datasets import load_dataset
    print(f"Loading ALL SHP splits (score_ratio >= {min_score_ratio})...")

    pairs = []
    for split in ["train", "validation", "test"]:
        print(f"  Loading {split}...")
        ds = load_dataset("stanfordnlp/SHP", split=split)
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
            if len(better) > 3000 or len(worse) > 3000:
                continue
            pairs.append({
                "prompt": prompt[:2000],
                "better": better,
                "worse": worse,
                "score_ratio": sr,
                "subreddit": row.get("domain", ""),
            })

    print(f"  Total: {len(pairs)} pairs")
    np.random.RandomState(42).shuffle(pairs)
    return pairs


def main():
    if not JINA_API_KEY:
        print("ERROR: JINA_API_KEY not set")
        sys.exit(1)

    pairs = load_shp_all(min_score_ratio=2.0)

    # Try both formats: response-only and with-context
    formats = {
        "response_only": {
            "better": lambda c: c["better"],
            "worse": lambda c: c["worse"],
        },
        "with_context": {
            "better": lambda c: f"{c['prompt']}\n\n{c['better']}",
            "worse": lambda c: f"{c['prompt']}\n\n{c['worse']}",
        },
    }

    all_results = {}

    for fmt_name, fmt_fns in formats.items():
        print(f"\n{'='*70}")
        print(f"FORMAT: {fmt_name}")
        print(f"{'='*70}")

        # Cap at 5000 for API cost/time
        use_pairs = pairs[:5000]
        n_total = len(use_pairs)

        print(f"  Embedding {n_total} better responses...")
        all_better = jina_embed_cached([fmt_fns["better"](c) for c in use_pairs])
        print(f"  Embedding {n_total} worse responses...")
        all_worse = jina_embed_cached([fmt_fns["worse"](c) for c in use_pairs])

        # Full in-sample
        full_dir = make_centroid(all_better, all_worse)
        full_acc = pairwise_accuracy(all_better, all_worse, full_dir)
        print(f"\n  Full in-sample ({n_total} pairs): {full_acc:.1%}")

        # Learning curve: train on N, test on rest
        fmt_results = {"n_total": n_total, "full_insample": round(float(full_acc), 4)}

        rng = np.random.RandomState(42)
        perm = rng.permutation(n_total)

        # 50/50 split for the main result
        half = n_total // 2
        train_idx, test_idx = perm[:half], perm[half:]

        train_b, train_w = all_better[train_idx], all_worse[train_idx]
        test_b, test_w = all_better[test_idx], all_worse[test_idx]

        d = make_centroid(train_b, train_w)
        main_oos = pairwise_accuracy(test_b, test_w, d)
        print(f"\n  === MAIN RESULT: {half}-train / {n_total - half}-test ===")
        print(f"  OOS accuracy: {main_oos:.1%}")

        # Permutation test
        print(f"  Running permutation test (1000 shuffles)...")
        p_val, null_mean = permutation_test(train_b, train_w, test_b, test_w,
                                             main_oos, n_perms=1000)
        print(f"  Permutation: p={p_val:.4f}, null_mean={null_mean:.1%}")

        fmt_results["main_train_n"] = int(half)
        fmt_results["main_test_n"] = int(n_total - half)
        fmt_results["main_oos"] = round(float(main_oos), 4)
        fmt_results["main_perm_p"] = round(p_val, 4)

        # Learning curve
        print(f"\n  Learning curve:")
        lc = {}
        for n_train in [50, 100, 200, 500, 1000, 2000, half]:
            if n_train > n_total - 100:
                continue
            accs = []
            for rep in range(5):
                p2 = np.random.RandomState(rep).permutation(n_total)
                tr, te = p2[:n_train], p2[n_train:]
                d2 = make_centroid(all_better[tr], all_worse[tr])
                accs.append(float(pairwise_accuracy(all_better[te], all_worse[te], d2)))
            mean_a = np.mean(accs)
            std_a = np.std(accs)
            print(f"    {n_train:>5d} train -> {n_total - n_train:>5d} test: "
                  f"{mean_a:.1%} ± {std_a:.1%}")
            lc[str(n_train)] = {"mean": round(float(mean_a), 4),
                                 "std": round(float(std_a), 4)}
        fmt_results["learning_curve"] = lc

        # Multiple 50/50 splits for stability
        split_accs = []
        for rep in range(10):
            p3 = np.random.RandomState(100 + rep).permutation(n_total)
            tr, te = p3[:half], p3[half:]
            d3 = make_centroid(all_better[tr], all_worse[tr])
            split_accs.append(float(pairwise_accuracy(all_better[te], all_worse[te], d3)))
        print(f"\n  10 random 50/50 splits: {np.mean(split_accs):.1%} ± {np.std(split_accs):.1%}")
        print(f"    Range: {min(split_accs):.1%} – {max(split_accs):.1%}")
        fmt_results["splits_mean"] = round(float(np.mean(split_accs)), 4)
        fmt_results["splits_std"] = round(float(np.std(split_accs)), 4)

        all_results[fmt_name] = fmt_results

    # Save
    out_path = OUT_DIR / "shp_at_scale_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for fmt_name, r in all_results.items():
        print(f"\n  {fmt_name}:")
        print(f"    Main OOS ({r['main_train_n']}/{r['main_test_n']}): "
              f"{r['main_oos']:.1%} (p={r['main_perm_p']:.4f})")
        print(f"    10 splits: {r['splits_mean']:.1%} ± {r['splits_std']:.1%}")
        if "learning_curve" in r:
            for n, v in r["learning_curve"].items():
                print(f"    {n:>5s} train: {v['mean']:.1%}")


if __name__ == "__main__":
    main()
