#!/usr/bin/env python3
"""Test centroid direction on external human-preference datasets.

The single most important test for publication. If the centroid works on
data we didn't write, the finding is real. If it drops to chance, we're
detecting authoring artifacts.

Tests:
  A. Our 70-pair direction → external pairs
  B. External data's own direction → itself (in-sample sanity)
  C. External direction → our battery (reverse transfer)
  D. 70-pair subsample of external data → rest of external (OOS)
  E. Cosine between our direction and external direction
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_battery():
    return load_cases(BATTERY_50) + load_cases(WARMTH_20)


def load_shp(n_samples=500):
    from datasets import load_dataset
    print("Loading SHP dataset (test split)...")
    ds = load_dataset("stanfordnlp/SHP", split="test")
    pairs = []
    for row in ds:
        if row["score_ratio"] is not None and row["score_ratio"] >= 2.0:
            if row["labels"] == 1:
                better, worse = row["human_ref_A"], row["human_ref_B"]
            else:
                better, worse = row["human_ref_B"], row["human_ref_A"]
            prompt = row["history"] or ""
            if len(better) < 20 or len(worse) < 20:
                continue
            if len(better) > 2000 or len(worse) > 2000:
                continue
            if len(prompt) > 1500:
                prompt = prompt[:1500]
            pairs.append({"prompt": prompt, "better": better, "worse": worse})
    print(f"  {len(pairs)} clear-preference pairs (score_ratio >= 2.0)")
    rng = np.random.RandomState(42)
    if len(pairs) > n_samples:
        idx = rng.choice(len(pairs), n_samples, replace=False)
        pairs = [pairs[i] for i in idx]
    print(f"  Subsampled to {len(pairs)} pairs")
    return pairs


def load_ultrafeedback(n_samples=500):
    from datasets import load_dataset
    print("Loading UltraFeedback binarized (test_prefs)...")
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
        chosen_text = next((m["content"] for m in chosen if m["role"] == "assistant"), None)
        rejected_text = next((m["content"] for m in rejected if m["role"] == "assistant"), None)
        if not chosen_text or not rejected_text:
            continue
        if len(chosen_text) < 20 or len(rejected_text) < 20:
            continue
        if len(chosen_text) > 2000 or len(rejected_text) > 2000:
            continue
        pairs.append({"prompt": prompt, "better": chosen_text, "worse": rejected_text})
    print(f"  {len(pairs)} preference pairs")
    rng = np.random.RandomState(42)
    if len(pairs) > n_samples:
        idx = rng.choice(len(pairs), n_samples, replace=False)
        pairs = [pairs[i] for i in idx]
    print(f"  Subsampled to {len(pairs)} pairs")
    return pairs


def make_centroid(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, direction):
    correct = 0
    margins = []
    for i in range(len(better_embs)):
        sb = float(np.dot(better_embs[i], direction))
        sw = float(np.dot(worse_embs[i], direction))
        margin = sb - sw
        margins.append(margin)
        if sb > sw:
            correct += 1
        elif sb == sw:
            correct += 0.5
    return correct / len(better_embs), margins


def permutation_test(better_embs, worse_embs, direction, n_perms=1000):
    real_acc, _ = pairwise_accuracy(better_embs, worse_embs, direction)
    rng = np.random.RandomState(0)
    count_ge = 0
    for _ in range(n_perms):
        swap = rng.rand(len(better_embs)) < 0.5
        b = better_embs.copy()
        w = worse_embs.copy()
        for i in range(len(swap)):
            if swap[i]:
                b[i], w[i] = w[i], b[i]
        perm_dir = make_centroid(b, w)
        perm_acc, _ = pairwise_accuracy(better_embs, worse_embs, perm_dir)
        if perm_acc >= real_acc:
            count_ge += 1
    return count_ge / n_perms


def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_battery()
    print(f"Internal battery: {len(battery)} pairs\n")

    external = {}
    for loader, name in [(load_shp, "SHP"), (load_ultrafeedback, "UltraFeedback")]:
        try:
            external[name] = loader(500)
        except Exception as e:
            print(f"  SKIP {name}: {e}\n")

    if not external:
        print("ERROR: No external datasets loaded.")
        sys.exit(1)

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        def embed(texts):
            return model.encode(texts, convert_to_numpy=True, batch_size=32,
                                show_progress_bar=False)

        # Our battery embeddings
        bat_b = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in battery])
        bat_w = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery])
        our_dir = make_centroid(bat_b, bat_w)

        our_acc, _ = pairwise_accuracy(bat_b, bat_w, our_dir)
        print(f"\nOur battery in-sample: {our_acc:.1%} ({int(our_acc*len(battery))}/{len(battery)})")

        mr = {"model": model_name, "our_battery_insample": round(our_acc, 4)}

        for ds_name, pairs in external.items():
            print(f"\n--- {ds_name} ({len(pairs)} pairs) ---")

            ext_b = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in pairs])
            ext_w = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in pairs])

            # A: our direction -> external
            a_acc, a_margins = pairwise_accuracy(ext_b, ext_w, our_dir)
            print(f"  [A] Our dir -> {ds_name}: {a_acc:.1%} ({int(a_acc*len(pairs))}/{len(pairs)})")
            print(f"      Mean margin: {np.mean(a_margins):.4f}  Std: {np.std(a_margins):.4f}")

            # B: external direction -> itself
            ext_dir = make_centroid(ext_b, ext_w)
            b_acc, _ = pairwise_accuracy(ext_b, ext_w, ext_dir)
            print(f"  [B] {ds_name} dir -> self (in-sample): {b_acc:.1%}")

            # C: external direction -> our battery
            c_acc, _ = pairwise_accuracy(bat_b, bat_w, ext_dir)
            print(f"  [C] {ds_name} dir -> our battery: {c_acc:.1%}")

            # D: 70-pair subsample -> rest OOS
            rng = np.random.RandomState(42)
            n_train = min(70, len(pairs) // 2)
            perm = rng.permutation(len(pairs))
            tr, te = perm[:n_train], perm[n_train:]
            sub_dir = make_centroid(ext_b[tr], ext_w[tr])
            d_acc, _ = pairwise_accuracy(ext_b[te], ext_w[te], sub_dir)
            print(f"  [D] {ds_name} 70-train -> rest OOS: {d_acc:.1%} ({int(d_acc*len(te))}/{len(te)})")

            # E: cosine between directions
            cos_val = float(np.dot(our_dir, ext_dir))
            print(f"  [E] Cosine(our_dir, {ds_name}_dir): {cos_val:+.4f}")

            # Permutation test on test A (our dir -> external, 200 perms for speed)
            print(f"  Running permutation test (200 perms)...")
            p_val = permutation_test(ext_b, ext_w, our_dir, n_perms=200)
            print(f"  Permutation p-value (our dir -> {ds_name}): {p_val:.3f}")

            mr[ds_name] = {
                "n_pairs": len(pairs),
                "our_dir_to_external": round(a_acc, 4),
                "our_dir_to_external_margin_mean": round(float(np.mean(a_margins)), 6),
                "our_dir_to_external_margin_std": round(float(np.std(a_margins)), 6),
                "external_dir_to_self_insample": round(b_acc, 4),
                "external_dir_to_our_battery": round(c_acc, 4),
                "external_70train_oos": round(d_acc, 4),
                "external_70train_oos_n_test": int(len(te)),
                "cosine_directions": round(cos_val, 6),
                "permutation_p": round(p_val, 4),
            }

        results[model_name] = mr
        del model
        gc.collect()

    out_path = OUT_DIR / "external_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mname, mr in results.items():
        short = mname.split("/")[-1]
        for ds_name in external:
            if ds_name in mr:
                d = mr[ds_name]
                print(f"  {short} | {ds_name}: our_dir->{ds_name} {d['our_dir_to_external']:.1%}  "
                      f"70-train OOS {d['external_70train_oos']:.1%}  "
                      f"cos(dirs) {d['cosine_directions']:+.3f}  "
                      f"p={d['permutation_p']:.3f}")


if __name__ == "__main__":
    main()
