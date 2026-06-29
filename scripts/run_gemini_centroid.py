#!/usr/bin/env python3
"""Test centroid direction on Gemini embedding model."""

import json, os, sys, io, time
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

load_dotenv(ROOT / ".env.local")


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def accuracy(better_embs, worse_embs, direction):
    n = len(better_embs)
    return sum(1 for i in range(n)
               if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)) / n


def make_dir(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    d /= norm(d) + 1e-12
    return d


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY found")
        return

    from google import genai

    client = genai.Client(api_key=api_key)

    # List available embedding models
    print("Available embedding models:")
    for m in client.models.list():
        if "embed" in m.name.lower():
            print(f"  {m.name}")

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    orig_cases = load_cases(BATTERY_50)
    warmth_cases = load_cases(WARMTH_20)

    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    print(f"\nBattery: {len(battery)} | Expansion: {len(expansion)}")

    # Try different model names
    model_name = "gemini-embedding-001"

    CACHE_FILE = OUT_DIR / "gemini_embedding_cache.json"
    emb_cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            emb_cache = json.load(f)
        print(f"Loaded {len(emb_cache)} cached embeddings")

    def save_cache():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(emb_cache, f)

    def embed_batch(texts, task="SEMANTIC_SIMILARITY"):
        all_embs = []
        uncached = [(i, t) for i, t in enumerate(texts) if t not in emb_cache]
        cached_count = len(texts) - len(uncached)
        if cached_count > 0:
            print(f"  {cached_count} cached, {len(uncached)} to embed")

        batch_size = 5
        for batch_start in range(0, len(uncached), batch_size):
            batch_items = uncached[batch_start:batch_start+batch_size]
            batch_texts = [t for _, t in batch_items]
            retries = 0
            while retries < 3:
                try:
                    result = client.models.embed_content(
                        model=model_name,
                        contents=batch_texts,
                        config={"task_type": task},
                    )
                    for j, emb in enumerate(result.embeddings):
                        emb_cache[batch_items[j][1]] = emb.values
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        wait = 30 * (retries + 1)
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        retries += 1
                    else:
                        print(f"  Error: {e}")
                        return None
            else:
                print(f"  Failed after 3 retries")
                save_cache()
                return None
            if batch_start + batch_size < len(uncached):
                time.sleep(8)
            if (batch_start // batch_size) % 5 == 4:
                save_cache()

        save_cache()
        for t in texts:
            v = np.array(emb_cache[t])
            v /= norm(v) + 1e-12
            all_embs.append(v)
        return np.array(all_embs)

    print(f"\nUsing model: {model_name}")
    all_texts = ([c["better"] for c in battery] + [c["worse"] for c in battery] +
                 [c["better"] for c in expansion] + [c["worse"] for c in expansion])
    print(f"Total texts to embed: {len(all_texts)}")

    print(f"Embedding battery better responses...")
    bat_better = embed_batch([c["better"] for c in battery])
    if bat_better is None:
        print("Failed on battery better")
        return
    print(f"  Shape: {bat_better.shape}")

    print(f"Embedding battery worse responses...")
    bat_worse = embed_batch([c["worse"] for c in battery])
    if bat_worse is None:
        print("Failed on battery worse")
        return

    print(f"Embedding expansion better responses...")
    exp_better = embed_batch([c["better"] for c in expansion])
    if exp_better is None:
        print("Failed on expansion better")
        return

    print(f"Embedding expansion worse responses...")
    exp_worse = embed_batch([c["worse"] for c in expansion])
    if exp_worse is None:
        print("Failed on expansion worse")
        return

    # Centroid direction
    dir_full = make_dir(bat_better, bat_worse)

    bat_acc = accuracy(bat_better, bat_worse, dir_full)
    exp_acc = accuracy(exp_better, exp_worse, dir_full)

    print(f"\n{'='*60}")
    print(f"GEMINI CENTROID RESULTS")
    print(f"{'='*60}")
    print(f"  Battery: {bat_acc:.0%}")
    print(f"  Expansion OOS: {exp_acc:.0%}")

    # Orig vs warmth split
    orig_better = bat_better[:len(orig_cases)]
    orig_worse = bat_worse[:len(orig_cases)]
    warm_better = bat_better[len(orig_cases):]
    warm_worse = bat_worse[len(orig_cases):]

    orig_acc = accuracy(orig_better, orig_worse, dir_full)
    warm_acc = accuracy(warm_better, warm_worse, dir_full)
    print(f"  Orig 50: {orig_acc:.0%}")
    print(f"  Warmth 20: {warm_acc:.0%}")

    # Per-category
    print(f"\n  Per-category OOS:")
    cat_results = {}
    for i, c in enumerate(expansion):
        cat = c.get("category", "unknown")
        correct = np.dot(exp_better[i], dir_full) > np.dot(exp_worse[i], dir_full)
        cat_results.setdefault(cat, []).append(correct)
    for cat in sorted(cat_results.keys()):
        vals = cat_results[cat]
        print(f"    {cat:35s}: {sum(vals)/len(vals):.0%} ({sum(vals)}/{len(vals)})")

    # Cross-type transfer
    dir_firm = make_dir(orig_better, orig_worse)
    dir_warm = make_dir(warm_better, warm_worse)
    cos_fw = float(np.dot(dir_firm, dir_warm))
    print(f"\n  Direction cosines:")
    print(f"    Firmness <-> Warmth: {cos_fw:+.4f}")
    print(f"    Firm on warmth: {accuracy(warm_better, warm_worse, dir_firm):.0%}")
    print(f"    Warm on firm: {accuracy(orig_better, orig_worse, dir_warm):.0%}")

    # Learning curve
    print(f"\n  Learning curve:")
    np.random.seed(42)
    for n_train in [5, 10, 20, 30, 50, 70]:
        accs = []
        for _ in range(50 if n_train < 70 else 1):
            idx = np.random.choice(len(battery), size=n_train, replace=False)
            d = make_dir(bat_better[idx], bat_worse[idx])
            accs.append(accuracy(exp_better, exp_worse, d))
        print(f"    n={n_train:3d}: {np.mean(accs):.1%} [{np.percentile(accs, 2.5):.0%}, {np.percentile(accs, 97.5):.0%}]")

    # Anchor word comparison
    print(f"\n  Anchor word comparison (expansion OOS):")
    for word in ["good", "bad", "careful", "reckless", "helpful", "thorough"]:
        time.sleep(0.3)
        w_emb = embed_batch([word])
        if w_emb is not None:
            w_dir = w_emb[0]
            w_acc = accuracy(exp_better, exp_worse, w_dir)
            cos_q = float(np.dot(w_dir, dir_full))
            print(f"    '{word}': {w_acc:.0%}  cos(quality)={cos_q:+.4f}")

    results = {
        "model": model_name,
        "dim": int(bat_better.shape[1]),
        "battery_acc": float(bat_acc),
        "expansion_acc": float(exp_acc),
        "orig_acc": float(orig_acc),
        "warm_acc": float(warm_acc),
        "cos_firm_warm": cos_fw,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "gemini_centroid_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'gemini_centroid_results.json'}")


if __name__ == "__main__":
    main()
