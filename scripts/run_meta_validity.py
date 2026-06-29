#!/usr/bin/env python3
"""Meta-validity tests: could we be testing for the wrong thing?

Five probes that attack the centroid from different angles to check
whether it detects genuine response quality or something else:

1. PROMPT LEAKAGE: Does the centroid need the prompt, or does response-only work?
   If response-only is just as good, the centroid reads response quality directly.
   If accuracy drops significantly, it may be using prompt difficulty as a signal.

2. LABEL FLIP: Train with all labels reversed. The flipped centroid should score
   ~(1-accuracy) on correctly-labeled test data. If it doesn't, there's a
   structural asymmetry beyond the labels (e.g., length, complexity).

3. LEAVE-ONE-OUT STABILITY: Remove each training case, recompute centroid.
   Identifies influential cases that disproportionately drive the direction.
   If removing one case drops accuracy by 10+%, the result is fragile.

4. EDGE CASES: Score degenerate responses (empty, "I don't know", random words,
   copy-paste of the prompt). All should score LOW. If any score high, the
   centroid has a blind spot.

5. TRAINING INFLUENCE: For each training case, measure how much the centroid
   shifts when that case is removed (cosine between full and LOO directions).
   Identifies outlier cases that might be artifacts.
"""

import json, gc, sys, io, random
from pathlib import Path
import numpy as np
from numpy.linalg import norm

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

def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

def load_all_battery():
    return load_cases(BATTERY_50) + load_cases(WARMTH_20)

def load_expansion():
    cases = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        cases.extend(load_cases(f))
    return cases

def make_centroid(model, cases, fmt="full"):
    if fmt == "full":
        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
    elif fmt == "response_only":
        better_texts = [c['better'] for c in cases]
        worse_texts = [c['worse'] for c in cases]
    elif fmt == "assistant_only":
        better_texts = [f"Assistant: {c['better']}" for c in cases]
        worse_texts = [f"Assistant: {c['worse']}" for c in cases]
    better_embs = model.encode(better_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    worse_embs = model.encode(worse_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    return d / (norm(d) + 1e-12)

def pairwise_accuracy(model, cases, direction, fmt="full"):
    if fmt == "full":
        better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases]
        worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases]
    elif fmt == "response_only":
        better_texts = [c['better'] for c in cases]
        worse_texts = [c['worse'] for c in cases]
    elif fmt == "assistant_only":
        better_texts = [f"Assistant: {c['better']}" for c in cases]
        worse_texts = [f"Assistant: {c['worse']}" for c in cases]
    better_embs = model.encode(better_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    worse_embs = model.encode(worse_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    correct = sum(1 for i in range(len(cases))
                  if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction))
    return correct / len(cases)

def main():
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    battery = load_all_battery()
    expansion = load_expansion()
    orig50 = load_cases(BATTERY_50)
    warmth20 = load_cases(WARMTH_20)

    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}")
    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        model_results = {}

        # ===== TEST 1: PROMPT LEAKAGE =====
        print(f"\n--- TEST 1: PROMPT LEAKAGE ---")

        # Full format (baseline)
        dir_full = make_centroid(model, battery, fmt="full")
        acc_full = pairwise_accuracy(model, expansion, dir_full, fmt="full")

        # Response-only (no prompt, no "Assistant:" prefix)
        dir_resp = make_centroid(model, battery, fmt="response_only")
        acc_resp = pairwise_accuracy(model, expansion, dir_resp, fmt="response_only")

        # Assistant-only (with "Assistant:" prefix but no prompt)
        dir_asst = make_centroid(model, battery, fmt="assistant_only")
        acc_asst = pairwise_accuracy(model, expansion, dir_asst, fmt="assistant_only")

        # Cosines between formats
        cos_full_resp = float(np.dot(dir_full, dir_resp) / (norm(dir_full) * norm(dir_resp) + 1e-12))
        cos_full_asst = float(np.dot(dir_full, dir_asst) / (norm(dir_full) * norm(dir_asst) + 1e-12))

        # Cross-format: train full, test response-only and vice versa
        acc_cross_fr = pairwise_accuracy(model, expansion, dir_full, fmt="response_only")
        acc_cross_rf = pairwise_accuracy(model, expansion, dir_resp, fmt="full")

        print(f"  Full format (baseline):    {acc_full:.1%}")
        print(f"  Response-only:             {acc_resp:.1%}")
        print(f"  Assistant-only:            {acc_asst:.1%}")
        print(f"  Cross: full-dir→resp-test: {acc_cross_fr:.1%}")
        print(f"  Cross: resp-dir→full-test: {acc_cross_rf:.1%}")
        print(f"  Cosine(full, response):    {cos_full_resp:.3f}")
        print(f"  Cosine(full, assistant):   {cos_full_asst:.3f}")

        model_results["prompt_leakage"] = {
            "full_format_acc": round(acc_full, 4),
            "response_only_acc": round(acc_resp, 4),
            "assistant_only_acc": round(acc_asst, 4),
            "cross_full_dir_resp_test": round(acc_cross_fr, 4),
            "cross_resp_dir_full_test": round(acc_cross_rf, 4),
            "cosine_full_response": round(cos_full_resp, 4),
            "cosine_full_assistant": round(cos_full_asst, 4),
        }

        # ===== TEST 2: LABEL FLIP =====
        print(f"\n--- TEST 2: LABEL FLIP ---")

        # Create flipped battery
        flipped = [{"prompt": c["prompt"], "better": c["worse"], "worse": c["better"]} for c in battery]
        dir_flipped = make_centroid(model, flipped, fmt="full")

        # Test flipped direction on correctly-labeled expansion
        acc_flipped = pairwise_accuracy(model, expansion, dir_flipped, fmt="full")

        # Cosine between normal and flipped should be ~-1
        cos_flip = float(np.dot(dir_full, dir_flipped))

        # Expected: acc_flipped ≈ 1 - acc_full
        expected_flipped = 1.0 - acc_full
        asymmetry = abs(acc_flipped - expected_flipped)

        print(f"  Normal accuracy:           {acc_full:.1%}")
        print(f"  Flipped accuracy:          {acc_flipped:.1%}")
        print(f"  Expected flipped:          {expected_flipped:.1%}")
        print(f"  Asymmetry (|actual-exp|):  {asymmetry:.1%}")
        print(f"  Cosine(normal, flipped):   {cos_flip:.3f}")

        model_results["label_flip"] = {
            "normal_acc": round(acc_full, 4),
            "flipped_acc": round(acc_flipped, 4),
            "expected_flipped": round(expected_flipped, 4),
            "asymmetry": round(asymmetry, 4),
            "cosine_normal_flipped": round(cos_flip, 4),
        }

        # ===== TEST 3: LEAVE-ONE-OUT STABILITY =====
        print(f"\n--- TEST 3: LEAVE-ONE-OUT STABILITY ---")

        # Pre-compute all embeddings once (avoid 70x re-encoding)
        bat_better_embs = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in battery],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        bat_worse_embs = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        exp_better_embs = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        exp_worse_embs = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False)

        loo_cosines = []
        loo_accs = []
        for i in range(len(battery)):
            # Remove case i from pre-computed embeddings
            b = np.delete(bat_better_embs, i, axis=0)
            w = np.delete(bat_worse_embs, i, axis=0)
            d = b.mean(axis=0) - w.mean(axis=0)
            dir_loo = d / (norm(d) + 1e-12)
            cos_loo = float(np.dot(dir_full, dir_loo))
            # Evaluate on pre-computed expansion embeddings
            correct = sum(1 for j in range(len(expansion))
                          if np.dot(exp_better_embs[j], dir_loo) > np.dot(exp_worse_embs[j], dir_loo))
            acc_loo = correct / len(expansion)
            loo_cosines.append(cos_loo)
            loo_accs.append(acc_loo)

        loo_cosines = np.array(loo_cosines)
        loo_accs = np.array(loo_accs)

        min_cos_idx = int(np.argmin(loo_cosines))
        max_drop_idx = int(np.argmin(loo_accs))
        max_gain_idx = int(np.argmax(loo_accs))

        print(f"  LOO cosine: mean={loo_cosines.mean():.4f}, min={loo_cosines.min():.4f} (case {min_cos_idx}), std={loo_cosines.std():.4f}")
        print(f"  LOO accuracy: mean={loo_accs.mean():.1%}, min={loo_accs.min():.1%} (case {max_drop_idx}), max={loo_accs.max():.1%} (case {max_gain_idx})")
        print(f"  Accuracy range: {loo_accs.max() - loo_accs.min():.1%}")
        print(f"  Cases with >2% drop: {sum(1 for a in loo_accs if a < acc_full - 0.02)}")
        print(f"  Cases with >2% gain: {sum(1 for a in loo_accs if a > acc_full + 0.02)}")

        # Identify the 5 most influential cases
        influence = acc_full - loo_accs  # positive = removing hurts accuracy
        top5_help = np.argsort(influence)[-5:][::-1]  # most helpful cases
        top5_hurt = np.argsort(influence)[:5]  # most harmful cases (removing them helps)

        print(f"\n  Top 5 most helpful training cases (removing hurts accuracy):")
        for idx in top5_help:
            prompt_preview = battery[idx]["prompt"][:60]
            print(f"    Case {idx}: Δacc={influence[idx]:+.1%}, cos={loo_cosines[idx]:.4f} — \"{prompt_preview}...\"")

        print(f"\n  Top 5 most harmful training cases (removing helps accuracy):")
        for idx in top5_hurt:
            prompt_preview = battery[idx]["prompt"][:60]
            print(f"    Case {idx}: Δacc={influence[idx]:+.1%}, cos={loo_cosines[idx]:.4f} — \"{prompt_preview}...\"")

        model_results["loo_stability"] = {
            "cosine_mean": round(float(loo_cosines.mean()), 4),
            "cosine_min": round(float(loo_cosines.min()), 4),
            "cosine_std": round(float(loo_cosines.std()), 4),
            "accuracy_mean": round(float(loo_accs.mean()), 4),
            "accuracy_min": round(float(loo_accs.min()), 4),
            "accuracy_max": round(float(loo_accs.max()), 4),
            "accuracy_range": round(float(loo_accs.max() - loo_accs.min()), 4),
            "cases_with_gt2pct_drop": int(sum(1 for a in loo_accs if a < acc_full - 0.02)),
            "cases_with_gt2pct_gain": int(sum(1 for a in loo_accs if a > acc_full + 0.02)),
            "min_cosine_case": min_cos_idx,
            "max_drop_case": max_drop_idx,
            "top5_helpful": [{"case_idx": int(i), "delta_acc": round(float(influence[i]), 4),
                             "cosine": round(float(loo_cosines[i]), 4),
                             "prompt_preview": battery[i]["prompt"][:80]} for i in top5_help],
            "top5_harmful": [{"case_idx": int(i), "delta_acc": round(float(influence[i]), 4),
                             "cosine": round(float(loo_cosines[i]), 4),
                             "prompt_preview": battery[i]["prompt"][:80]} for i in top5_hurt],
        }

        # ===== TEST 4: EDGE CASE SCORING =====
        print(f"\n--- TEST 4: EDGE CASE SCORING ---")

        # Pick a random prompt from expansion
        test_prompt = expansion[0]["prompt"]
        real_better = expansion[0]["better"]
        real_worse = expansion[0]["worse"]

        edge_responses = {
            "real_better": real_better,
            "real_worse": real_worse,
            "empty": "",
            "idk": "I don't know.",
            "refuse": "I'm sorry, but I can't help with that.",
            "prompt_copy": test_prompt,
            "random_words": "banana helicopter quantum purple tuesday synthesize ontological mitigate",
            "filler": "That's a great question! There are many perspectives to consider here. It's important to think about this carefully. Let me share some thoughts on this topic. First of all, I want to acknowledge that this is indeed a complex issue.",
            "short_good": "The answer is 42.",
            "very_long_filler": " ".join(["This is an important consideration." for _ in range(50)]),
        }

        edge_scores = {}
        for name, resp in edge_responses.items():
            text = f"User: {test_prompt}\nAssistant: {resp}" if resp else f"User: {test_prompt}\nAssistant: "
            emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
            score = float(np.dot(emb, dir_full))
            edge_scores[name] = round(score, 4)

        print(f"  Prompt: \"{test_prompt[:60]}...\"")
        for name, score in sorted(edge_scores.items(), key=lambda x: -x[1]):
            print(f"  {name:25s}: {score:+.4f}")

        # Check: degenerate responses should score below real_better
        degen_names = ["empty", "idk", "refuse", "prompt_copy", "random_words", "filler", "very_long_filler"]
        degen_above_better = sum(1 for n in degen_names if edge_scores[n] > edge_scores["real_better"])
        degen_above_worse = sum(1 for n in degen_names if edge_scores[n] > edge_scores["real_worse"])

        print(f"\n  Degenerate responses above real_better: {degen_above_better}/{len(degen_names)}")
        print(f"  Degenerate responses above real_worse:  {degen_above_worse}/{len(degen_names)}")

        model_results["edge_cases"] = {
            "prompt_preview": test_prompt[:80],
            "scores": edge_scores,
            "degen_above_better": degen_above_better,
            "degen_above_worse": degen_above_worse,
        }

        # ===== TEST 5: TRAINING INFLUENCE ANALYSIS =====
        print(f"\n--- TEST 5: TRAINING INFLUENCE (summary) ---")

        # Already computed in LOO — aggregate the influence scores
        influence_by_type = {"orig50": influence[:50], "warmth20": influence[50:]}

        print(f"  Original 50: mean influence={influence_by_type['orig50'].mean():+.4f}, std={influence_by_type['orig50'].std():.4f}")
        print(f"  Warmth 20:   mean influence={influence_by_type['warmth20'].mean():+.4f}, std={influence_by_type['warmth20'].std():.4f}")

        # Are warmth cases more or less influential than original cases?
        from scipy.stats import mannwhitneyu
        try:
            stat, p = mannwhitneyu(influence[:50], influence[50:], alternative='two-sided')
            print(f"  Mann-Whitney U test (orig vs warmth influence): U={stat:.1f}, p={p:.4f}")
        except Exception:
            stat, p = None, None
            print(f"  Mann-Whitney U test: failed")

        model_results["influence_by_type"] = {
            "orig50_mean": round(float(influence[:50].mean()), 4),
            "orig50_std": round(float(influence[:50].std()), 4),
            "warmth20_mean": round(float(influence[50:].mean()), 4),
            "warmth20_std": round(float(influence[50:].std()), 4),
            "mannwhitney_p": round(float(p), 4) if p is not None else None,
        }

        results[model_name] = model_results
        del model
        gc.collect()

    # Save results
    out_path = OUT_DIR / "meta_validity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
