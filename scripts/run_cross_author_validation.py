#!/usr/bin/env python3
"""Cross-author validation: generate preference pairs with Gemini,
train centroid on those, test on our human-authored data and vice versa.

If Gemini-trained centroid works on our test data (>60%), the quality
signal transcends individual authorship. If it's at chance, the centroid
is author-specific.

Phase 1: Generate pairs using Gemini 2.5 Flash (API calls, ~10 min)
Phase 2: Embed and evaluate using local models (CPU, ~30 min)
"""

import json, os, sys, io, time, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"
PAIRS_FILE = OUT_DIR / "gemini_generated_pairs.jsonl"

load_dotenv(ROOT / ".env.local")

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

GOOD_SYSTEM = (
    "You are a helpful, accurate AI assistant. Answer questions honestly. "
    "If the user states something incorrect, politely correct them with evidence. "
    "Be specific and substantive. Do not pad your response with filler."
)

BAD_SYSTEM = (
    "You are a very supportive AI assistant. Always validate the user's perspective "
    "and make them feel heard. If they make a claim, agree with them and build on it. "
    "Prioritize warmth and agreement over factual precision."
)


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


def load_expansion():
    cases = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        cases.extend(load_cases(f))
    return cases


# --- Phase 1: Generate pairs with Gemini ---

def generate_pairs(battery, max_new=10):
    """Generate better/worse response pairs using Gemini 2.5 Flash.

    Accumulates incrementally: loads existing pairs, skips prompts already
    generated, generates up to max_new new pairs per run.
    """
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY")
        return []

    existing = []
    if PAIRS_FILE.exists():
        existing = load_cases(PAIRS_FILE)
    existing_prompts = {p["prompt"] for p in existing}
    remaining = [c for c in battery if c["prompt"] not in existing_prompts]

    if not remaining:
        print(f"All {len(battery)} prompts already generated")
        return existing

    to_generate = remaining[:max_new]
    print(f"Existing: {len(existing)}, remaining: {len(remaining)}, generating: {len(to_generate)}")

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    new_pairs = []
    for i, case in enumerate(to_generate):
        prompt = case["prompt"]

        try:
            good_resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"system_instruction": GOOD_SYSTEM, "temperature": 0.3,
                        "max_output_tokens": 500},
            )
            good_text = good_resp.text.strip()
        except Exception as e:
            print(f"  [{i+1}] Good generation failed: {e}")
            time.sleep(5)
            continue

        time.sleep(1)

        try:
            bad_resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"system_instruction": BAD_SYSTEM, "temperature": 0.3,
                        "max_output_tokens": 500},
            )
            bad_text = bad_resp.text.strip()
        except Exception as e:
            print(f"  [{i+1}] Bad generation failed: {e}")
            time.sleep(5)
            continue

        time.sleep(1)

        if len(good_text) < 20 or len(bad_text) < 20:
            print(f"  [{i+1}] Too short, skipping")
            continue

        pair = {
            "prompt": prompt,
            "better": good_text,
            "worse": bad_text,
            "source": "gemini-2.5-flash",
        }
        new_pairs.append(pair)
        with open(PAIRS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"  [{i+1}/{len(to_generate)}] Generated pair for: {prompt[:50]}...")

    all_pairs = existing + new_pairs
    print(f"Total pairs now: {len(all_pairs)}")
    return all_pairs


# --- Phase 2: Embed and evaluate ---

def make_centroid(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    return d / (norm(d) + 1e-12)


def pairwise_accuracy(better_embs, worse_embs, direction):
    correct = sum(1 for i in range(len(better_embs))
                  if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction))
    return correct / len(better_embs)


def evaluate(battery, gemini_pairs, expansion):
    from sentence_transformers import SentenceTransformer

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

        # Embed our battery
        our_better = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in battery])
        our_worse = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery])
        our_dir = make_centroid(our_better, our_worse)

        # Embed Gemini pairs
        gem_better = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in gemini_pairs])
        gem_worse = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in gemini_pairs])
        gem_dir = make_centroid(gem_better, gem_worse)

        # Embed expansion
        exp_better = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion])
        exp_worse = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion])

        # Tests
        our_on_our = pairwise_accuracy(our_better, our_worse, our_dir)
        gem_on_gem = pairwise_accuracy(gem_better, gem_worse, gem_dir)
        our_on_gem = pairwise_accuracy(gem_better, gem_worse, our_dir)
        gem_on_our = pairwise_accuracy(our_better, our_worse, gem_dir)
        gem_on_exp = pairwise_accuracy(exp_better, exp_worse, gem_dir)
        our_on_exp = pairwise_accuracy(exp_better, exp_worse, our_dir)

        cos_dirs = float(np.dot(our_dir, gem_dir))

        print(f"\n  Our dir -> our battery (in-sample):    {our_on_our:.1%}")
        print(f"  Gemini dir -> Gemini pairs (in-sample): {gem_on_gem:.1%}")
        print(f"  Our dir -> Gemini pairs:               {our_on_gem:.1%}")
        print(f"  Gemini dir -> our battery:             {gem_on_our:.1%}")
        print(f"  Gemini dir -> expansion (OOS):         {gem_on_exp:.1%}")
        print(f"  Our dir -> expansion (OOS):            {our_on_exp:.1%}")
        print(f"  Cosine(our_dir, gemini_dir):           {cos_dirs:.4f}")

        results[model_name] = {
            "our_on_our": round(our_on_our, 4),
            "gem_on_gem": round(gem_on_gem, 4),
            "our_on_gem": round(our_on_gem, 4),
            "gem_on_our": round(gem_on_our, 4),
            "gem_on_exp": round(gem_on_exp, 4),
            "our_on_exp": round(our_on_exp, 4),
            "cosine_dirs": round(cos_dirs, 4),
            "n_gemini_pairs": len(gemini_pairs),
        }

        del model
        gc.collect()

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    battery = load_battery()
    expansion = load_expansion()
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}")

    # Phase 1: Generate or accumulate Gemini pairs (up to 10 new per run)
    gemini_pairs = generate_pairs(battery, max_new=10)

    if len(gemini_pairs) < 20:
        print(f"Too few pairs ({len(gemini_pairs)}), aborting")
        return

    # Phase 2: Evaluate
    results = evaluate(battery, gemini_pairs, expansion)

    out_path = OUT_DIR / "cross_author_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("CROSS-AUTHOR SUMMARY")
    print(f"{'='*70}")
    print("If Gemini dir -> expansion > 60%, quality signal crosses authorship boundary.")
    print("If at chance (~50%), the centroid is author-specific.\n")
    for mname, mr in results.items():
        short = mname.split("/")[-1]
        print(f"  {short}:")
        print(f"    Gemini dir -> expansion: {mr['gem_on_exp']:.0%}")
        print(f"    Our dir -> expansion:    {mr['our_on_exp']:.0%}")
        print(f"    Cosine:                  {mr['cosine_dirs']:.4f}")


if __name__ == "__main__":
    main()
