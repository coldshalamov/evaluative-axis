#!/usr/bin/env python3
"""Test the universal best axis set on Gemini Embedding.

Quick test: does {careful, thorough, hard} + voting maintain
Gemini's advantage, or does Gemini not need multi-axis voting?
Also test cosine-to-positive on Gemini.
"""

import json, sys, os, time
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local")

BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("No GOOGLE_API_KEY found in .env.local")
    sys.exit(1)


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


class GeminiEmbedder:
    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = "gemini-embedding-001"

    def embed(self, texts, batch_size=20, delay=1.5):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                for emb in result.embeddings:
                    all_embs.append(emb.values)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  Quota hit at batch {i}, waiting 60s...")
                    time.sleep(60)
                    try:
                        result = self.client.models.embed_content(
                            model=self.model,
                            contents=batch,
                        )
                        for emb in result.embeddings:
                            all_embs.append(emb.values)
                    except Exception as e2:
                        print(f"  Still failing: {e2}")
                        return None
                else:
                    print(f"  Error: {e}")
                    return None
            if i + batch_size < len(texts):
                time.sleep(delay)
        return np.array(all_embs)


AXES = {
    "careful":  {"pos": ["Careful"],  "neg": ["Reckless"]},
    "thorough": {"pos": ["Thorough"], "neg": ["Superficial"]},
    "hard":     {"pos": ["Hard"],     "neg": ["Soft"]},
    "kind":     {"pos": ["Kind"],     "neg": ["Cruel"]},
    "honest":   {"pos": ["Honest"],   "neg": ["Dishonest"]},
    "good":     {"pos": ["Good"],     "neg": ["Bad"]},
    "bold":     {"pos": ["Bold"],     "neg": ["Timid"]},
    "helpful":  {"pos": ["Helpful"],  "neg": ["Unhelpful"]},
    "active":   {"pos": ["Active"],   "neg": ["Passive"]},
    "fair":     {"pos": ["Fair"],     "neg": ["Unfair"]},
}


def main():
    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)
    all_cases = original + warmth
    n_orig = len(original)

    gemini = GeminiEmbedder()
    print("Embedding responses with Gemini...")

    all_better = gemini.embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases],
                               batch_size=10, delay=2.0)
    if all_better is None:
        print("Failed to embed better responses (quota?). Exiting.")
        sys.exit(1)

    all_worse = gemini.embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases],
                              batch_size=10, delay=2.0)
    if all_worse is None:
        print("Failed to embed worse responses (quota?). Exiting.")
        sys.exit(1)

    print(f"Embedded {len(all_better)} response pairs.")

    results = {}
    per_case_correct = {}

    for axis_name, anchors in AXES.items():
        pos_embs = gemini.embed(anchors["pos"], delay=1.0)
        neg_embs = gemini.embed(anchors["neg"], delay=1.0)
        if pos_embs is None or neg_embs is None:
            print(f"  Quota hit on {axis_name} anchors, skipping.")
            continue

        axis_vec = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
        axis_unit = axis_vec / (norm(axis_vec) + 1e-12)
        pos_unit = pos_embs.mean(axis=0)

        bp_correct = []
        cp_correct = []
        for i in range(len(all_cases)):
            sb_bp = float(np.dot(all_better[i], axis_unit))
            sw_bp = float(np.dot(all_worse[i], axis_unit))
            bp_correct.append(1 if sb_bp > sw_bp else (0.5 if sb_bp == sw_bp else 0))

            sb_cp = cosine(all_better[i], pos_unit)
            sw_cp = cosine(all_worse[i], pos_unit)
            cp_correct.append(1 if sb_cp > sw_cp else (0.5 if sb_cp == sw_cp else 0))

        bp_orig = np.mean(bp_correct[:n_orig])
        bp_warm = np.mean(bp_correct[n_orig:])
        bp_comb = np.mean(bp_correct)
        cp_orig = np.mean(cp_correct[:n_orig])
        cp_warm = np.mean(cp_correct[n_orig:])
        cp_comb = np.mean(cp_correct)

        results[axis_name] = {
            "bipolar": {"orig": round(bp_orig, 4), "warm": round(bp_warm, 4), "comb": round(bp_comb, 4)},
            "cospos": {"orig": round(cp_orig, 4), "warm": round(cp_warm, 4), "comb": round(cp_comb, 4)},
        }
        per_case_correct[axis_name] = {"bp": bp_correct, "cp": cp_correct}

        best_bp = max(bp_orig, bp_warm)
        best_cp = max(cp_orig, cp_warm)
        print(f"  {axis_name:12s}  bp: orig={bp_orig:.1%} warm={bp_warm:.1%} comb={bp_comb:.1%}  "
              f"cp: orig={cp_orig:.1%} warm={cp_warm:.1%} comb={cp_comb:.1%}")

    # Majority vote for best combos
    if per_case_correct:
        print(f"\n--- Majority vote (Gemini) ---")
        combos = {
            "careful+thorough+hard": ["careful", "thorough", "hard"],
            "careful+kind+thorough": ["careful", "kind", "thorough"],
            "careful+honest+hard": ["careful", "honest", "hard"],
        }
        for name, axes_list in combos.items():
            available = [a for a in axes_list if a in per_case_correct]
            if len(available) < len(axes_list):
                continue
            threshold = len(available) / 2

            bp_votes = np.array([per_case_correct[a]["bp"] for a in available]).sum(axis=0)
            cp_votes = np.array([per_case_correct[a]["cp"] for a in available]).sum(axis=0)

            bp_orig = np.mean(bp_votes[:n_orig] > threshold)
            bp_warm = np.mean(bp_votes[n_orig:] > threshold)
            bp_comb = np.mean(bp_votes > threshold)
            cp_orig = np.mean(cp_votes[:n_orig] > threshold)
            cp_warm = np.mean(cp_votes[n_orig:] > threshold)
            cp_comb = np.mean(cp_votes > threshold)

            print(f"  {name:30s}  bp: o={bp_orig:.1%} w={bp_warm:.1%} c={bp_comb:.1%}  "
                  f"cp: o={cp_orig:.1%} w={cp_warm:.1%} c={cp_comb:.1%}")

    out_dir = ROOT / "notes/research_cycles/gemini_universal"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gemini_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
