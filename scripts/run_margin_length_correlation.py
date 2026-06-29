#!/usr/bin/env python3
"""Test whether centroid margins correlate with response length differences.

If margin tracks length rather than quality gap magnitude, the
self-weighting claim is a length artifact.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy import stats

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


def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    battery = load_battery()
    expansion = load_expansion()
    all_cases = battery + expansion
    print(f"Battery: {len(battery)}, Expansion: {len(expansion)}, Total: {len(all_cases)}")

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        direction = make_centroid(model, battery)

        better_embs = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in all_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )
        worse_embs = model.encode(
            [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in all_cases],
            convert_to_numpy=True, batch_size=32, show_progress_bar=False,
        )

        margins = []
        len_diffs_char = []
        len_diffs_word = []
        len_ratios = []
        prompt_lens = []
        better_lens = []
        worse_lens = []
        abs_margins = []

        for i, c in enumerate(all_cases):
            sb = float(np.dot(better_embs[i], direction))
            sw = float(np.dot(worse_embs[i], direction))
            margin = sb - sw
            margins.append(margin)
            abs_margins.append(abs(margin))

            bl = len(c["better"])
            wl = len(c["worse"])
            pl = len(c["prompt"])
            better_lens.append(bl)
            worse_lens.append(wl)
            prompt_lens.append(pl)
            len_diffs_char.append(bl - wl)
            len_diffs_word.append(len(c["better"].split()) - len(c["worse"].split()))
            len_ratios.append(bl / max(wl, 1))

        margins = np.array(margins)
        abs_margins = np.array(abs_margins)
        len_diffs_char = np.array(len_diffs_char)
        len_diffs_word = np.array(len_diffs_word)
        len_ratios = np.array(len_ratios)
        prompt_lens = np.array(prompt_lens)
        better_lens = np.array(better_lens)
        worse_lens = np.array(worse_lens)

        metrics = {
            "char_length_diff": len_diffs_char,
            "word_length_diff": len_diffs_word,
            "length_ratio": len_ratios,
            "prompt_length": prompt_lens,
            "better_length": better_lens,
            "worse_length": worse_lens,
            "abs_length_diff": np.abs(len_diffs_char),
        }

        print(f"\nMargin vs length correlations (n={len(all_cases)}):")
        print(f"{'Metric':25s} {'Pearson r':>10s} {'p-value':>10s} {'Spearman r':>10s} {'p-value':>10s}")
        print("-" * 70)

        model_corrs = {}
        for mname, mvals in metrics.items():
            pr, pp = stats.pearsonr(margins, mvals)
            sr, sp = stats.spearmanr(margins, mvals)
            flag = " ***" if (abs(pr) > 0.3 and pp < 0.05) else ""
            print(f"  {mname:25s} {pr:+.4f}     {pp:.4f}     {sr:+.4f}     {sp:.4f}{flag}")
            model_corrs[mname] = {
                "pearson_r": round(float(pr), 4),
                "pearson_p": round(float(pp), 4),
                "spearman_r": round(float(sr), 4),
                "spearman_p": round(float(sp), 4),
            }

        # Also check: does |margin| correlate with |length_diff|?
        pr2, pp2 = stats.pearsonr(abs_margins, np.abs(len_diffs_char))
        sr2, sp2 = stats.spearmanr(abs_margins, np.abs(len_diffs_char))
        print(f"\n  |margin| vs |char_diff|:  Pearson {pr2:+.4f} (p={pp2:.4f})  Spearman {sr2:+.4f} (p={sp2:.4f})")

        # Correctness vs length analysis
        correct = margins > 0
        correct_len_diff = len_diffs_char[correct].mean() if correct.sum() > 0 else 0
        wrong_len_diff = len_diffs_char[~correct].mean() if (~correct).sum() > 0 else 0
        print(f"\n  Mean length diff (better-worse) for correct: {correct_len_diff:.0f} chars")
        print(f"  Mean length diff (better-worse) for errors:  {wrong_len_diff:.0f} chars")

        model_corrs["abs_margin_vs_abs_diff"] = {
            "pearson_r": round(float(pr2), 4),
            "pearson_p": round(float(pp2), 4),
            "spearman_r": round(float(sr2), 4),
            "spearman_p": round(float(sp2), 4),
        }
        model_corrs["correct_mean_len_diff"] = round(float(correct_len_diff), 1)
        model_corrs["wrong_mean_len_diff"] = round(float(wrong_len_diff), 1)
        model_corrs["n_cases"] = len(all_cases)

        results[model_name] = model_corrs
        del model
        gc.collect()

    out_path = OUT_DIR / "margin_length_correlation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
