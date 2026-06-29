#!/usr/bin/env python3
"""Decompose "good" by its 30 senses using synonym clusters.

Instead of embedding the overloaded word "good," embed the synonym words
for each sense of "good" and average them to get a "sense direction."
Test each sense direction as a quality scorer on the battery.

This tests: which MEANING of "good" actually captures response quality?
Does the quality sense (#3: excellent/superior/fine) work where the
warmth sense (#2: kind/generous/compassionate) fails?
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# The 30 senses of "good" — synonym clusters from deep research
SENSES = {
    "1_moral": ["virtuous", "ethical", "righteous", "honorable", "decent", "upright", "noble", "just", "moral"],
    "2_kind": ["kind", "generous", "compassionate", "caring", "charitable", "humane", "thoughtful"],
    "3_quality": ["excellent", "fine", "superior", "first-rate", "admirable", "impressive", "strong"],
    "4_competent": ["capable", "proficient", "talented", "adept", "skillful", "accomplished", "effective"],
    "5_effective": ["effective", "useful", "workable", "successful", "productive", "beneficial", "sound"],
    "6_beneficial": ["beneficial", "helpful", "useful", "favorable", "advantageous", "constructive"],
    "7_pleasant": ["pleasant", "enjoyable", "delightful", "satisfying", "agreeable", "nice", "lovely"],
    "8_desirable": ["desirable", "preferable", "welcome", "wanted", "favorable", "ideal", "fitting"],
    "9_suitable": ["suitable", "appropriate", "fitting", "proper", "apt", "right", "convenient"],
    "10_valid": ["valid", "sound", "legitimate", "justified", "reasonable", "defensible", "credible"],
    "11_genuine": ["genuine", "real", "authentic", "true", "legitimate"],
    "12_reliable": ["reliable", "dependable", "trustworthy", "faithful", "steady", "honest"],
    "13_healthy": ["healthy", "well", "fit", "strong", "robust", "sound"],
    "14_safe": ["safe", "secure", "protected", "stable", "harmless"],
    "15_sufficient": ["sufficient", "enough", "adequate", "ample", "satisfactory"],
    "17_thorough": ["thorough", "complete", "solid", "careful", "proper", "full"],
    "18_obedient": ["obedient", "disciplined", "compliant", "dutiful"],
    "19_respectable": ["respectable", "decent", "proper", "reputable", "acceptable", "presentable"],
    "20_fortunate": ["fortunate", "lucky", "auspicious", "favorable", "blessed"],
    "24_holy": ["holy", "sacred", "godly", "blessed", "pious", "divine"],
}

# Group senses by the five root ideas from the research
ROOT_IDEAS = {
    "moral_rightness": ["1_moral", "24_holy"],
    "quality": ["3_quality", "4_competent", "17_thorough"],
    "usefulness": ["5_effective", "6_beneficial", "9_suitable"],
    "pleasure": ["7_pleasant", "8_desirable", "20_fortunate"],
    "warmth": ["2_kind", "19_respectable", "18_obedient"],
    "reliability": ["10_valid", "11_genuine", "12_reliable", "14_safe"],
}


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def score_accuracy(better_embs, worse_embs, direction):
    n = len(better_embs)
    correct = sum(
        1 for i in range(n)
        if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)
    )
    return correct / n


def main():
    from sentence_transformers import SentenceTransformer

    battery = load_cases(BATTERY_50) + load_cases(WARMTH_20)
    orig_cases = load_cases(BATTERY_50)
    warmth_cases = load_cases(WARMTH_20)

    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    print(f"Battery: {len(battery)} cases (50 orig + 20 warmth)")
    print(f"Expansion: {len(expansion)} OOS cases")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {short}")
        print(f"{'='*70}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Embed battery responses
        bat_better = model.encode([c["better"] for c in battery], convert_to_numpy=True, batch_size=32)
        bat_worse = model.encode([c["worse"] for c in battery], convert_to_numpy=True, batch_size=32)
        for i in range(len(battery)):
            bat_better[i] /= norm(bat_better[i]) + 1e-12
            bat_worse[i] /= norm(bat_worse[i]) + 1e-12

        # Split indices for orig vs warmth
        orig_idx = list(range(len(orig_cases)))
        warmth_idx = list(range(len(orig_cases), len(battery)))

        # Embed expansion responses
        exp_better = model.encode([c["better"] for c in expansion], convert_to_numpy=True, batch_size=32)
        exp_worse = model.encode([c["worse"] for c in expansion], convert_to_numpy=True, batch_size=32)
        for i in range(len(expansion)):
            exp_better[i] /= norm(exp_better[i]) + 1e-12
            exp_worse[i] /= norm(exp_worse[i]) + 1e-12

        # Embed "good" alone for comparison
        good_emb = model.encode(["good"], convert_to_numpy=True)[0]
        good_emb /= norm(good_emb) + 1e-12

        # Compute the supervised quality direction for comparison
        quality_dir = bat_better.mean(axis=0) - bat_worse.mean(axis=0)
        quality_dir /= norm(quality_dir) + 1e-12

        # Embed each sense cluster
        sense_dirs = {}
        for sense_name, words in SENSES.items():
            embs = model.encode(words, convert_to_numpy=True)
            for i in range(len(embs)):
                embs[i] /= norm(embs[i]) + 1e-12
            sense_dir = embs.mean(axis=0)
            sense_dir /= norm(sense_dir) + 1e-12
            sense_dirs[sense_name] = sense_dir

        # Score each sense on battery and expansion
        print(f"\n  {'Sense':25s} {'Battery':>8s} {'Orig50':>8s} {'Warm20':>8s} {'Expan':>8s} {'cos(qual)':>10s} {'cos(good)':>10s}")
        print(f"  {'-'*85}")

        # First: raw "good" for comparison
        good_bat = score_accuracy(bat_better, bat_worse, good_emb)
        good_orig = score_accuracy(bat_better[orig_idx], bat_worse[orig_idx], good_emb)
        good_warm = score_accuracy(bat_better[warmth_idx], bat_worse[warmth_idx], good_emb)
        good_exp = score_accuracy(exp_better, exp_worse, good_emb)
        cos_gq = float(np.dot(good_emb, quality_dir))
        print(f"  {'[word: good]':25s} {good_bat:7.0%}  {good_orig:7.0%}  {good_warm:7.0%}  {good_exp:7.0%}  {cos_gq:+9.3f}  {'---':>10s}")

        # Supervised direction for comparison
        sup_bat = score_accuracy(bat_better, bat_worse, quality_dir)
        sup_orig = score_accuracy(bat_better[orig_idx], bat_worse[orig_idx], quality_dir)
        sup_warm = score_accuracy(bat_better[warmth_idx], bat_worse[warmth_idx], quality_dir)
        sup_exp = score_accuracy(exp_better, exp_worse, quality_dir)
        print(f"  {'[supervised direction]':25s} {sup_bat:7.0%}  {sup_orig:7.0%}  {sup_warm:7.0%}  {sup_exp:7.0%}  {'---':>10s}  {'---':>10s}")

        print(f"  {'-'*85}")

        for sense_name, sense_dir in sense_dirs.items():
            bat_acc = score_accuracy(bat_better, bat_worse, sense_dir)
            orig_acc = score_accuracy(bat_better[orig_idx], bat_worse[orig_idx], sense_dir)
            warm_acc = score_accuracy(bat_better[warmth_idx], bat_worse[warmth_idx], sense_dir)
            exp_acc = score_accuracy(exp_better, exp_worse, sense_dir)
            cos_q = float(np.dot(sense_dir, quality_dir))
            cos_g = float(np.dot(sense_dir, good_emb))
            print(f"  {sense_name:25s} {bat_acc:7.0%}  {orig_acc:7.0%}  {warm_acc:7.0%}  {exp_acc:7.0%}  {cos_q:+9.3f}  {cos_g:+9.3f}")

        # Root idea composites
        print(f"\n  --- Root idea composites ---")
        print(f"  {'Root idea':25s} {'Battery':>8s} {'Orig50':>8s} {'Warm20':>8s} {'Expan':>8s} {'cos(qual)':>10s} {'cos(good)':>10s}")
        print(f"  {'-'*85}")
        for root_name, sense_names in ROOT_IDEAS.items():
            # Average all synonym embeddings from all senses in this root
            all_words = []
            for sn in sense_names:
                all_words.extend(SENSES[sn])
            all_embs = model.encode(all_words, convert_to_numpy=True)
            for i in range(len(all_embs)):
                all_embs[i] /= norm(all_embs[i]) + 1e-12
            root_dir = all_embs.mean(axis=0)
            root_dir /= norm(root_dir) + 1e-12

            bat_acc = score_accuracy(bat_better, bat_worse, root_dir)
            orig_acc = score_accuracy(bat_better[orig_idx], bat_worse[orig_idx], root_dir)
            warm_acc = score_accuracy(bat_better[warmth_idx], bat_worse[warmth_idx], root_dir)
            exp_acc = score_accuracy(exp_better, exp_worse, root_dir)
            cos_q = float(np.dot(root_dir, quality_dir))
            cos_g = float(np.dot(root_dir, good_emb))
            print(f"  {root_name:25s} {bat_acc:7.0%}  {orig_acc:7.0%}  {warm_acc:7.0%}  {exp_acc:7.0%}  {cos_q:+9.3f}  {cos_g:+9.3f}")

        # Quality-only composite (senses 3, 4, 5, 10, 17) vs warmth composite (2, 7, 19)
        print(f"\n  --- Targeted composites ---")
        quality_senses = ["3_quality", "4_competent", "5_effective", "10_valid", "17_thorough"]
        warmth_senses = ["2_kind", "7_pleasant", "19_respectable"]

        for label, sense_list in [("quality_senses", quality_senses), ("warmth_senses", warmth_senses)]:
            all_words = []
            for sn in sense_list:
                all_words.extend(SENSES[sn])
            all_embs = model.encode(all_words, convert_to_numpy=True)
            for i in range(len(all_embs)):
                all_embs[i] /= norm(all_embs[i]) + 1e-12
            comp_dir = all_embs.mean(axis=0)
            comp_dir /= norm(comp_dir) + 1e-12

            bat_acc = score_accuracy(bat_better, bat_worse, comp_dir)
            orig_acc = score_accuracy(bat_better[orig_idx], bat_worse[orig_idx], comp_dir)
            warm_acc = score_accuracy(bat_better[warmth_idx], bat_worse[warmth_idx], comp_dir)
            exp_acc = score_accuracy(exp_better, exp_worse, comp_dir)
            cos_q = float(np.dot(comp_dir, quality_dir))
            cos_g = float(np.dot(comp_dir, good_emb))
            print(f"  {label:25s} {bat_acc:7.0%}  {orig_acc:7.0%}  {warm_acc:7.0%}  {exp_acc:7.0%}  {cos_q:+9.3f}  {cos_g:+9.3f}")

        # Quality minus warmth: subtract warmth direction from quality direction
        q_words = []
        for sn in quality_senses:
            q_words.extend(SENSES[sn])
        w_words = []
        for sn in warmth_senses:
            w_words.extend(SENSES[sn])

        q_embs = model.encode(q_words, convert_to_numpy=True)
        w_embs = model.encode(w_words, convert_to_numpy=True)
        for i in range(len(q_embs)):
            q_embs[i] /= norm(q_embs[i]) + 1e-12
        for i in range(len(w_embs)):
            w_embs[i] /= norm(w_embs[i]) + 1e-12

        q_center = q_embs.mean(axis=0)
        w_center = w_embs.mean(axis=0)
        diff_dir = q_center - w_center
        diff_dir /= norm(diff_dir) + 1e-12

        bat_acc = score_accuracy(bat_better, bat_worse, diff_dir)
        orig_acc = score_accuracy(bat_better[orig_idx], bat_worse[orig_idx], diff_dir)
        warm_acc = score_accuracy(bat_better[warmth_idx], bat_worse[warmth_idx], diff_dir)
        exp_acc = score_accuracy(exp_better, exp_worse, diff_dir)
        cos_q = float(np.dot(diff_dir, quality_dir))
        cos_g = float(np.dot(diff_dir, good_emb))
        print(f"  {'quality MINUS warmth':25s} {bat_acc:7.0%}  {orig_acc:7.0%}  {warm_acc:7.0%}  {exp_acc:7.0%}  {cos_q:+9.3f}  {cos_g:+9.3f}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
