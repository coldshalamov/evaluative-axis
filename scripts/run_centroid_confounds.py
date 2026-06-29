#!/usr/bin/env python3
"""Check confounds that could explain the centroid direction's accuracy.

If the quality direction is secretly encoding "longer responses are better"
or "responses with more sentences are better" or just "response diversity,"
the whole finding is undermined. This tests those confounds.

Also tests Gemini Embedding 2 if the API key is available.

Checks:
1. Length confound: correlation between direction score and response length
2. Length-only baseline: can response length alone predict quality?
3. Direction after length regression: remove length from embeddings, recompute
4. Vocabulary confound: are better responses using systematically different words?
5. Gemini centroid (if API available)
"""

import json, gc, sys, io, os
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

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

load_dotenv(ROOT / ".env.local")


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def embed_and_norm(model, texts):
    embs = model.encode(texts, convert_to_numpy=True, batch_size=32)
    for i in range(len(embs)):
        embs[i] /= norm(embs[i]) + 1e-12
    return embs


def accuracy(better_embs, worse_embs, direction):
    n = len(better_embs)
    correct = sum(
        1 for i in range(n)
        if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)
    )
    return correct / n


def make_dir(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    d /= norm(d) + 1e-12
    return d


def main():
    from sentence_transformers import SentenceTransformer
    from scipy import stats

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    all_cases = battery + expansion
    all_better_texts = [c["better"] for c in all_cases]
    all_worse_texts = [c["worse"] for c in all_cases]

    # =================================================================
    # 1. LENGTH ANALYSIS (no model needed)
    # =================================================================
    print(f"{'='*80}")
    print(f"LENGTH ANALYSIS")
    print(f"{'='*80}")

    bat_better_lens = [len(c["better"]) for c in battery]
    bat_worse_lens = [len(c["worse"]) for c in battery]
    exp_better_lens = [len(c["better"]) for c in expansion]
    exp_worse_lens = [len(c["worse"]) for c in expansion]

    print(f"\n  Battery ({len(battery)} pairs):")
    print(f"    Better response: mean={np.mean(bat_better_lens):.0f} chars, median={np.median(bat_better_lens):.0f}")
    print(f"    Worse response:  mean={np.mean(bat_worse_lens):.0f} chars, median={np.median(bat_worse_lens):.0f}")
    len_diff = [b - w for b, w in zip(bat_better_lens, bat_worse_lens)]
    print(f"    Difference (better - worse): mean={np.mean(len_diff):+.0f}, median={np.median(len_diff):+.0f}")
    pct_longer = sum(1 for d in len_diff if d > 0) / len(len_diff)
    print(f"    Better is longer in {pct_longer:.0%} of cases")

    print(f"\n  Expansion ({len(expansion)} pairs):")
    exp_len_diff = [b - w for b, w in zip(exp_better_lens, exp_worse_lens)]
    print(f"    Difference (better - worse): mean={np.mean(exp_len_diff):+.0f}, median={np.median(exp_len_diff):+.0f}")
    exp_pct_longer = sum(1 for d in exp_len_diff if d > 0) / len(exp_len_diff)
    print(f"    Better is longer in {exp_pct_longer:.0%} of cases")

    # Length-only classifier: pick the longer response as "better"
    bat_length_acc = sum(1 for b, w in zip(bat_better_lens, bat_worse_lens) if b > w) / len(battery)
    exp_length_acc = sum(1 for b, w in zip(exp_better_lens, exp_worse_lens) if b > w) / len(expansion)
    print(f"\n  Length-only classifier (longer = better):")
    print(f"    Battery: {bat_length_acc:.0%}")
    print(f"    Expansion: {exp_length_acc:.0%}")

    # Word count
    bat_better_words = [len(c["better"].split()) for c in battery]
    bat_worse_words = [len(c["worse"].split()) for c in battery]
    word_diff = [b - w for b, w in zip(bat_better_words, bat_worse_words)]
    print(f"\n  Word count difference (better - worse): mean={np.mean(word_diff):+.1f}")

    all_results = {"length_analysis": {
        "bat_better_mean_len": float(np.mean(bat_better_lens)),
        "bat_worse_mean_len": float(np.mean(bat_worse_lens)),
        "bat_length_acc": float(bat_length_acc),
        "exp_length_acc": float(exp_length_acc),
        "bat_pct_better_longer": float(pct_longer),
    }}

    # =================================================================
    # PER-MODEL CONFOUND CHECKS
    # =================================================================
    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        bat_better = embed_and_norm(model, [c["better"] for c in battery])
        bat_worse = embed_and_norm(model, [c["worse"] for c in battery])
        exp_better = embed_and_norm(model, [c["better"] for c in expansion])
        exp_worse = embed_and_norm(model, [c["worse"] for c in expansion])

        dir_full = make_dir(bat_better, bat_worse)

        mr = {}

        # --- Correlation between direction score and length ---
        print(f"\n--- Correlation: direction score vs length ---")

        all_better_scores = [float(np.dot(bat_better[i], dir_full)) for i in range(len(battery))]
        all_worse_scores = [float(np.dot(bat_worse[i], dir_full)) for i in range(len(battery))]
        all_scores = all_better_scores + all_worse_scores
        all_lens = bat_better_lens + bat_worse_lens

        r, p = stats.pearsonr(all_scores, all_lens)
        print(f"  Score vs length (all battery responses): r={r:.4f}, p={p:.4f}")

        # Correlation between score MARGIN and length DIFFERENCE
        margins = [all_better_scores[i] - all_worse_scores[i] for i in range(len(battery))]
        r_m, p_m = stats.pearsonr(margins, len_diff)
        print(f"  Score margin vs length diff: r={r_m:.4f}, p={p_m:.4f}")

        mr["score_vs_length_r"] = float(r)
        mr["margin_vs_lengthdiff_r"] = float(r_m)

        # --- Remove length from embeddings ---
        print(f"\n--- Length-Regressed Direction ---")

        # Create length vector: embed responses, find the direction that correlates with length
        all_embs = np.vstack([bat_better, bat_worse])
        all_lens_arr = np.array(bat_better_lens + bat_worse_lens, dtype=float)
        all_lens_arr = (all_lens_arr - all_lens_arr.mean()) / (all_lens_arr.std() + 1e-12)

        # Length direction: weighted mean of embeddings by normalized length
        length_dir = (all_embs * all_lens_arr[:, None]).mean(axis=0)
        length_dir /= norm(length_dir) + 1e-12

        cos_quality_length = float(np.dot(dir_full, length_dir))
        print(f"  Cosine (quality dir, length dir): {cos_quality_length:+.4f}")

        # Project out length from all embeddings
        bat_better_nolen = bat_better - np.outer(bat_better @ length_dir, length_dir)
        bat_worse_nolen = bat_worse - np.outer(bat_worse @ length_dir, length_dir)
        exp_better_nolen = exp_better - np.outer(exp_better @ length_dir, length_dir)
        exp_worse_nolen = exp_worse - np.outer(exp_worse @ length_dir, length_dir)

        # Re-normalize
        for arr in [bat_better_nolen, bat_worse_nolen, exp_better_nolen, exp_worse_nolen]:
            for i in range(len(arr)):
                arr[i] /= norm(arr[i]) + 1e-12

        dir_nolen = make_dir(bat_better_nolen, bat_worse_nolen)
        bat_nolen_acc = accuracy(bat_better_nolen, bat_worse_nolen, dir_nolen)
        exp_nolen_acc = accuracy(exp_better_nolen, exp_worse_nolen, dir_nolen)

        bat_orig_acc = accuracy(bat_better, bat_worse, dir_full)
        exp_orig_acc = accuracy(exp_better, exp_worse, dir_full)

        print(f"  Original:         Battery={bat_orig_acc:.0%} Expansion={exp_orig_acc:.0%}")
        print(f"  Length-regressed:  Battery={bat_nolen_acc:.0%} Expansion={exp_nolen_acc:.0%}")

        mr["cos_quality_length"] = float(cos_quality_length)
        mr["orig_bat"] = float(bat_orig_acc)
        mr["orig_exp"] = float(exp_orig_acc)
        mr["nolen_bat"] = float(bat_nolen_acc)
        mr["nolen_exp"] = float(exp_nolen_acc)

        # --- Random direction baseline ---
        print(f"\n--- Random Direction Baseline ---")
        np.random.seed(42)
        n_random = 1000
        random_accs = []
        dim = bat_better.shape[1]
        for _ in range(n_random):
            rand_dir = np.random.randn(dim)
            rand_dir /= norm(rand_dir)
            # Flip to point toward better on training set
            if accuracy(bat_better, bat_worse, rand_dir) < 0.5:
                rand_dir = -rand_dir
            random_accs.append(accuracy(exp_better, exp_worse, rand_dir))

        print(f"  Random direction OOS: mean={np.mean(random_accs):.1%}, max={np.max(random_accs):.1%}, p95={np.percentile(random_accs, 95):.1%}")
        mr["random_dir_mean"] = float(np.mean(random_accs))
        mr["random_dir_max"] = float(np.max(random_accs))

        # --- Embedding norm analysis ---
        print(f"\n--- Embedding Norm (pre-normalization) ---")
        raw_better = model.encode([c["better"] for c in battery], convert_to_numpy=True, batch_size=32)
        raw_worse = model.encode([c["worse"] for c in battery], convert_to_numpy=True, batch_size=32)
        better_norms = [norm(raw_better[i]) for i in range(len(battery))]
        worse_norms = [norm(raw_worse[i]) for i in range(len(battery))]
        print(f"  Better response norms: mean={np.mean(better_norms):.4f} std={np.std(better_norms):.4f}")
        print(f"  Worse response norms:  mean={np.mean(worse_norms):.4f} std={np.std(worse_norms):.4f}")
        norm_diff = [b - w for b, w in zip(better_norms, worse_norms)]
        print(f"  Norm difference: mean={np.mean(norm_diff):+.4f}")

        # Norm-only classifier
        norm_acc = sum(1 for b, w in zip(better_norms, worse_norms) if b > w) / len(battery)
        print(f"  Norm-only classifier (higher norm = better): {norm_acc:.0%}")

        all_results[short] = mr

        del model
        gc.collect()

    # =================================================================
    # GEMINI (if available)
    # =================================================================
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"\n{'='*80}")
        print(f"GEMINI EMBEDDING 2")
        print(f"{'='*80}")

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            import time

            def gemini_embed(texts, task_type="SEMANTIC_SIMILARITY"):
                all_embs = []
                batch_size = 5
                for start in range(0, len(texts), batch_size):
                    batch = texts[start:start+batch_size]
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=batch,
                        task_type=task_type,
                    )
                    all_embs.extend(result["embedding"])
                    if start + batch_size < len(texts):
                        time.sleep(1)
                embs = np.array(all_embs)
                for i in range(len(embs)):
                    embs[i] /= norm(embs[i]) + 1e-12
                return embs

            print(f"  Embedding battery responses...")
            gem_bat_better = gemini_embed([c["better"] for c in battery])
            time.sleep(2)
            print(f"  Embedding battery worse...")
            gem_bat_worse = gemini_embed([c["worse"] for c in battery])
            time.sleep(2)
            print(f"  Embedding expansion responses...")
            gem_exp_better = gemini_embed([c["better"] for c in expansion])
            time.sleep(2)
            gem_exp_worse = gemini_embed([c["worse"] for c in expansion])

            dir_gem = make_dir(gem_bat_better, gem_bat_worse)
            gem_bat_acc = accuracy(gem_bat_better, gem_bat_worse, dir_gem)
            gem_exp_acc = accuracy(gem_exp_better, gem_exp_worse, dir_gem)

            print(f"\n  Gemini centroid direction:")
            print(f"    Battery: {gem_bat_acc:.0%}")
            print(f"    Expansion: {gem_exp_acc:.0%}")

            # Per-category
            print(f"\n    Per-category OOS:")
            for i, c in enumerate(expansion):
                cat = c.get("category", "unknown")
                correct = np.dot(gem_exp_better[i], dir_gem) > np.dot(gem_exp_worse[i], dir_gem)
                # Group by category
                pass  # we'll do it properly

            cat_results = {}
            for i, c in enumerate(expansion):
                cat = c.get("category", "unknown")
                correct = np.dot(gem_exp_better[i], dir_gem) > np.dot(gem_exp_worse[i], dir_gem)
                cat_results.setdefault(cat, []).append(correct)

            for cat in sorted(cat_results.keys()):
                vals = cat_results[cat]
                print(f"      {cat:35s}: {sum(vals)/len(vals):.0%} ({sum(vals)}/{len(vals)})")

            # Learning curve
            print(f"\n    Learning curve:")
            np.random.seed(42)
            for n_train in [5, 10, 20, 30, 50, 70]:
                accs = []
                for _ in range(50 if n_train < 70 else 1):
                    idx = np.random.choice(len(battery), size=n_train, replace=False)
                    d = make_dir(gem_bat_better[idx], gem_bat_worse[idx])
                    accs.append(accuracy(gem_exp_better, gem_exp_worse, d))
                print(f"      n={n_train:3d}: {np.mean(accs):.1%}")

            # Also test "careful" anchor for comparison
            careful_emb = gemini_embed(["careful"])[0]
            careful_acc = accuracy(gem_exp_better, gem_exp_worse, careful_emb)
            good_emb = gemini_embed(["good"])[0]
            good_acc = accuracy(gem_exp_better, gem_exp_worse, good_emb)
            print(f"\n    Anchor comparison on expansion:")
            print(f"      'careful': {careful_acc:.0%}")
            print(f"      'good': {good_acc:.0%}")
            print(f"      centroid: {gem_exp_acc:.0%}")

            all_results["gemini"] = {
                "battery": float(gem_bat_acc),
                "expansion": float(gem_exp_acc),
                "careful_exp": float(careful_acc),
                "good_exp": float(good_acc),
            }

        except Exception as e:
            print(f"  Gemini failed: {e}")
    else:
        print(f"\nNo GOOGLE_API_KEY found, skipping Gemini")

    # Final summary
    print(f"\n{'='*80}")
    print(f"CONFOUND CHECK SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Length baseline: battery={all_results['length_analysis']['bat_length_acc']:.0%} expansion={all_results['length_analysis']['exp_length_acc']:.0%}")

    print(f"\n  {'Model':25s} {'Original':>10s} {'No-Length':>10s} {'cos(q,len)':>12s} {'RandMax':>10s}")
    for short in MODELS:
        s = short.split("/")[-1]
        if s in all_results:
            r = all_results[s]
            print(f"  {s:25s} {r['orig_exp']:9.0%}  {r['nolen_exp']:9.0%}  {r['cos_quality_length']:+11.4f}  {r['random_dir_max']:9.0%}")

    with open(OUT_DIR / "confound_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'confound_results.json'}")


if __name__ == "__main__":
    main()
