#!/usr/bin/env python3
"""Test whether the centroid score can be mechanically inflated.

If padding, hedging, or reformatting responses increases the quality score
without improving quality, the direction is gameable under optimization and
unsafe as a training signal.
"""

import json, gc, sys, io, re
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)

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

FILLER_SENTENCES = [
    "It's worth noting that this is an important topic with many dimensions to consider.",
    "There are several perspectives one could take on this matter.",
    "Let me elaborate on this point in a bit more detail.",
]

HEDGE_PHRASES = [
    ("However, ", "However, it's important to note that "),
    ("This ", "To be fair, this "),
    ("The ", "One could argue that the "),
    (". ", ". That said, "),
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


def load_expansion():
    cases = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        cases.extend(load_cases(f))
    return cases


def make_centroid(model, cases):
    better = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    worse = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32, show_progress_bar=False,
    )
    d = better.mean(axis=0) - worse.mean(axis=0)
    return d / (norm(d) + 1e-12)


# --- Mechanical modifications ---

def pad_response(text):
    """Add filler sentences at the end."""
    return text + " " + " ".join(FILLER_SENTENCES)


def hedge_response(text):
    """Insert hedging phrases throughout."""
    result = text
    for old, new in HEDGE_PHRASES:
        result = result.replace(old, new, 1)
    return result


def bulletize_response(text):
    """Convert to bullet-point format."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) < 3:
        return text
    return "\n".join(f"- {s}." if not s.endswith(".") else f"- {s}" for s in sentences)


def repeat_conclusion(text):
    """Duplicate the last sentence."""
    sentences = text.rstrip().split(". ")
    if len(sentences) < 2:
        return text + " " + text
    last = sentences[-1]
    return text + " " + last


def formalize(text):
    """Replace casual language with formal style."""
    replacements = [
        ("don't", "do not"), ("can't", "cannot"), ("won't", "will not"),
        ("it's", "it is"), ("that's", "that is"), ("I'd", "I would"),
        ("you're", "you are"), ("they're", "they are"),
        ("isn't", "is not"), ("wasn't", "was not"),
    ]
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def lengthen_response(text):
    """Double the length by expanding each sentence."""
    padding = (
        " This point deserves careful consideration given the broader context."
        " Multiple factors contribute to understanding this aspect fully."
    )
    sentences = text.split(". ")
    expanded = []
    for s in sentences:
        expanded.append(s)
        if len(s) > 20:
            expanded.append(padding.strip())
    return ". ".join(expanded)


MODIFICATIONS = {
    "padded": pad_response,
    "hedged": hedge_response,
    "bulletized": bulletize_response,
    "repeated": repeat_conclusion,
    "formalized": formalize,
    "lengthened": lengthen_response,
}


def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_battery()
    expansion = load_expansion()
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}")

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

        direction = make_centroid(model, battery)

        # Find correctly-classified expansion cases
        exp_better = embed([f"User: {c['prompt']}\nAssistant: {c['better']}" for c in expansion])
        exp_worse = embed([f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in expansion])

        correct_idx = []
        for i in range(len(expansion)):
            sb = float(np.dot(exp_better[i], direction))
            sw = float(np.dot(exp_worse[i], direction))
            if sb > sw:
                correct_idx.append(i)

        print(f"Correctly classified: {len(correct_idx)}/{len(expansion)}")
        use_idx = correct_idx[:30]
        print(f"Using {len(use_idx)} cases for gameability test\n")

        model_results = {"correct_total": len(correct_idx), "tested": len(use_idx)}

        for mod_name, mod_fn in MODIFICATIONS.items():
            print(f"  --- {mod_name} ---")
            score_increases_better = 0
            score_increases_worse = 0
            worse_flips = 0
            better_margins = []
            worse_margins = []
            original_better_scores = []
            modified_better_scores = []
            original_worse_scores = []
            modified_worse_scores = []

            for i in use_idx:
                c = expansion[i]
                orig_better_text = f"User: {c['prompt']}\nAssistant: {c['better']}"
                orig_worse_text = f"User: {c['prompt']}\nAssistant: {c['worse']}"

                mod_better_text = f"User: {c['prompt']}\nAssistant: {mod_fn(c['better'])}"
                mod_worse_text = f"User: {c['prompt']}\nAssistant: {mod_fn(c['worse'])}"

                orig_b_emb = embed([orig_better_text])[0]
                orig_w_emb = embed([orig_worse_text])[0]
                mod_b_emb = embed([mod_better_text])[0]
                mod_w_emb = embed([mod_worse_text])[0]

                orig_b_score = float(np.dot(orig_b_emb, direction))
                orig_w_score = float(np.dot(orig_w_emb, direction))
                mod_b_score = float(np.dot(mod_b_emb, direction))
                mod_w_score = float(np.dot(mod_w_emb, direction))

                original_better_scores.append(orig_b_score)
                modified_better_scores.append(mod_b_score)
                original_worse_scores.append(orig_w_score)
                modified_worse_scores.append(mod_w_score)

                if mod_b_score > orig_b_score:
                    score_increases_better += 1
                better_margins.append(mod_b_score - orig_b_score)

                if mod_w_score > orig_w_score:
                    score_increases_worse += 1
                worse_margins.append(mod_w_score - orig_w_score)

                if mod_w_score > orig_b_score:
                    worse_flips += 1

            n = len(use_idx)
            print(f"    Better score increased: {score_increases_better}/{n} ({score_increases_better/n:.0%})")
            print(f"    Worse score increased:  {score_increases_worse}/{n} ({score_increases_worse/n:.0%})")
            print(f"    Worse flipped above original better: {worse_flips}/{n} ({worse_flips/n:.0%})")
            print(f"    Mean better margin change: {np.mean(better_margins):+.4f}")
            print(f"    Mean worse margin change:  {np.mean(worse_margins):+.4f}")

            model_results[mod_name] = {
                "better_score_increased": score_increases_better,
                "worse_score_increased": score_increases_worse,
                "worse_flipped_above_better": worse_flips,
                "n_tested": n,
                "better_margin_change_mean": round(float(np.mean(better_margins)), 6),
                "better_margin_change_std": round(float(np.std(better_margins)), 6),
                "worse_margin_change_mean": round(float(np.mean(worse_margins)), 6),
                "worse_margin_change_std": round(float(np.std(worse_margins)), 6),
                "original_better_mean": round(float(np.mean(original_better_scores)), 6),
                "modified_better_mean": round(float(np.mean(modified_better_scores)), 6),
                "original_worse_mean": round(float(np.mean(original_worse_scores)), 6),
                "modified_worse_mean": round(float(np.mean(modified_worse_scores)), 6),
            }

        results[model_name] = model_results
        del model
        gc.collect()

    out_path = OUT_DIR / "gameability_probe_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("GAMEABILITY SUMMARY")
    print(f"{'='*70}")
    print("Higher = more gameable = worse for training signal safety")
    print()
    for mname, mr in results.items():
        short = mname.split("/")[-1]
        print(f"  {short}:")
        for mod_name in MODIFICATIONS:
            if mod_name in mr:
                d = mr[mod_name]
                n = d["n_tested"]
                print(f"    {mod_name:12s}: better+{d['better_score_increased']:2d}/{n}  "
                      f"worse+{d['worse_score_increased']:2d}/{n}  "
                      f"flips {d['worse_flipped_above_better']}/{n}  "
                      f"Δbetter {d['better_margin_change_mean']:+.4f}  "
                      f"Δworse {d['worse_margin_change_mean']:+.4f}")


if __name__ == "__main__":
    main()
