#!/usr/bin/env python3
"""Large word discrimination sweep.

Goal (per user): test a large pool of candidate words as single-word axes
against the full battery, to find which words discriminate well, which
catch others' failure modes (complementary), and to surface words worth
studying for the score-subtraction tree.

Exploratory. No premature conclusions — just the table.

Scores every word two ways (cosine-to-positive AND bipolar pos-vs-neg),
on all batteries:
  - 50 firmness (original)
  - 20 warmth
  - 20 expansion (nuance/factual/conciseness/creative)
  - 20 anti-sycophancy expansion
Total: 110 cases.

Reports per-word: combined, firmness, warmth, nuance, factual, conciseness,
creative, anti-sycophancy accuracy. Both methods side by side.

For negatives we use the standard antonym where one exists; for words
without a clean antonym we use the generic "Bad" — so the bipolar column
is meaningful (deviation from "Bad" baseline) but the cosine-to-positive
column is the primary signal for most words.

Outputs:
  notes/research_cycles/large_word_sweep/sweep_results.json  (full table)
  + a printed summary of top/bottom words per category and method.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "research_cycles" / "large_word_sweep" / "sweep_results.json"

ORIG = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARM = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXP_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# ---------------------------------------------------------------------------
# Candidate word pool — ~180 words, hand-curated into groups for interpretability.
# Format: (word, antonym_or_None). None -> uses "Bad" as the bipolar negative.
# ---------------------------------------------------------------------------
CANDIDATES = [
    # --- Root evaluative ---
    ("good", "bad"),
    ("great", "terrible"),
    ("excellent", "awful"),
    ("wonderful", "dreadful"),
    ("superb", "atrocious"),
    ("quality", "worthless"),
    ("ideal", "unacceptable"),
    ("exceptional", "mediocre"),
    ("outstanding", "poor"),
    ("satisfactory", "unsatisfactory"),
    ("helpful", "unhelpful"),
    ("useful", "useless"),
    ("relevant", "irrelevant"),
    ("appropriate", "inappropriate"),

    # --- Competence / precision ---
    ("careful", "reckless"),
    ("precise", "imprecise"),
    ("accurate", "inaccurate"),
    ("rigorous", "sloppy"),
    ("methodical", "haphazard"),
    ("meticulous", "careless"),
    ("systematic", "chaotic"),
    ("diligent", "lazy"),
    ("disciplined", "undisciplined"),
    ("attentive", "inattentive"),
    ("deliberate", "impulsive"),
    ("thorough", "superficial"),
    ("complete", "incomplete"),
    ("comprehensive", "partial"),
    ("exhaustive", "sparse"),
    ("detailed", "vague"),
    ("correct", "wrong"),
    ("sound", "flawed"),
    ("logical", "illogical"),
    ("coherent", "incoherent"),
    ("valid", "invalid"),
    ("robust", "fragile"),
    ("reliable", "unreliable"),

    # --- Integrity / honesty ---
    ("honest", "dishonest"),
    ("truthful", "deceitful"),
    ("sincere", "insincere"),
    ("genuine", "fake"),
    ("candid", "evasive"),
    ("frank", "guarded"),
    ("direct", "indirect"),
    ("transparent", "opaque"),
    ("trustworthy", "deceitful"),
    ("authentic", "artificial"),
    ("forthright", "evasive"),
    ("humble", "arrogant"),

    # --- Warmth / empathy ---
    ("kind", "cruel"),
    ("warm", "cold"),
    ("compassionate", "callous"),
    ("empathetic", "indifferent"),
    ("caring", "uncaring"),
    ("gentle", "harsh"),
    ("sympathetic", "unsympathetic"),
    ("considerate", "inconsiderate"),
    ("thoughtful", "thoughtless"),
    ("gracious", "graceless"),
    ("respectful", "disrespectful"),
    ("supportive", "undermining"),
    ("encouraging", "discouraging"),
    ("friendly", "hostile"),
    ("patient", "impatient"),
    ("polite", "rude"),
    ("fair", "unfair"),
    ("responsible", "irresponsible"),

    # --- Restraint / proportion ---
    ("measured", "extreme"),
    ("proportionate", "disproportionate"),
    ("judicious", "imprudent"),
    ("prudent", "imprudent"),
    ("balanced", "unbalanced"),
    ("moderate", "extreme"),
    ("restrained", "unrestrained"),
    ("temperate", "intemperate"),
    ("concise", "verbose"),
    ("clear", "confusing"),
    ("lucid", "murky"),
    ("organized", "disorganized"),
    ("structured", "disorganized"),

    # --- Creative ---
    ("creative", "unoriginal"),
    ("original", "derivative"),
    ("imaginative", " unimaginative"),
    ("insightful", "shallow"),
    ("vivid", "flat"),
    ("engaging", "dull"),
    ("eloquent", "inarticulate"),
    ("elegant", "clunky"),
    ("thought-provoking", "banal"),

    # --- FAILURE-MODE / negative-concept words (for the tree penalty search) ---
    # These are tested the same way: cosine to the word, and bipolar vs antonym.
    # For the score-subtraction sweep, we'll use these as penalty candidates.
    ("sycophantic", "honest"),
    ("flattering", "candid"),
    ("placating", "challenging"),
    ("obsequious", "independent"),
    ("servile", "independent"),
    ("fawning", "dismissive"),
    ("compliant", "defiant"),
    ("agreeable", "disagreeable"),
    ("accommodating", "rigid"),
    ("evasive", "direct"),
    ("vague", "specific"),
    ("verbose", "concise"),
    ("generic", "specific"),
    ("superficial", "thorough"),
    ("misleading", "accurate"),
    ("inflated", "measured"),
    ("pandering", "principled"),
    ("deferential", "defiant"),
    ("gushing", "measured"),
    ("saccharine", "sincere"),

    # --- ODDITIES / length-correlated (controls & red-herring checks) ---
    ("hard", "soft"),
    ("heavy", "light"),
    ("dense", "sparse"),
    ("weighty", "frivolous"),
    ("solid", "hollow"),
    ("strong", "weak"),
    ("firm", "yielding"),
    ("substantial", "trivial"),
    ("deep", "shallow"),
    ("sharp", "dull"),

    # --- RANDOM CONTROLS (chance floor) ---
    ("blue", None),
    ("table", None),
    ("running", None),
    ("however", None),
    ("therefore", None),
    ("purple", None),
    ("external", None),
    ("circular", None),
    ("monthly", None),
    ("northern", None),
]

# Remove a stray space-typo from one antonym
CANDIDATES = [(w, a.strip() if a else None) for w, a in CANDIDATES]
# Dedupe
seen = set()
CANDIDATES = [c for c in CANDIDATES if not (c in seen or seen.add(c))]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def framed(c, k):
    return f"User: {c['prompt']}\nAssistant: {c[k]}"


def cosine_to(embs, anchor):
    an = np.linalg.norm(anchor) + 1e-12
    rn = np.linalg.norm(embs, axis=1) + 1e-12
    return (embs @ anchor) / (rn * an)


def acc(better_scores, worse_scores):
    m = np.asarray(better_scores) - np.asarray(worse_scores)
    return (np.sum(m > 0) + 0.5 * np.sum(m == 0)) / len(m)


def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(ORIG)
    warm = read_jsonl(WARM)
    # Expansion: 4 standard categories + anti-sycophancy
    exp_cases = []
    for f in sorted(EXP_DIR.glob("*.jsonl")):
        exp_cases.extend(read_jsonl(f))
    # category map
    cat_cases = {"firmness": orig, "warmth": warm}
    for c in exp_cases:
        cat_cases.setdefault(c["category"], []).append(c)
    n_per = {k: len(v) for k, v in cat_cases.items()}
    print(f"Batteries: {n_per}  total={sum(n_per.values())}")
    print(f"Candidate words: {len(CANDIDATES)}")
    print(f"Models: {len(MODELS)}")

    results = {
        "metadata": {"n_words": len(CANDIDATES), "categories": n_per,
                     "models": MODELS},
        "per_model": {},
    }

    for model_name in MODELS:
        ms = model_name.split("/")[-1]
        print(f"\n{'='*84}")
        print(f"MODEL: {ms}   ({len(CANDIDATES)} words x {sum(n_per.values())} cases)")
        print(f"{'='*84}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        def emb(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed all responses once per category
        cat_embs = {}
        for cat, cases in cat_cases.items():
            be = emb([framed(c, "better") for c in cases])
            we = emb([framed(c, "worse") for c in cases])
            cat_embs[cat] = (be, we)

        # Embed all candidate words + their antonyms + "Bad" fallback once
        all_words = set("Bad")
        all_words |= {w for w, _ in CANDIDATES}
        all_words |= {a for _, a in CANDIDATES if a}
        word_embs = {w: emb([w])[0] for w in all_words}

        word_rows = {}
        print(f"\n{'word':<18s} {'cos:comb':>8s} {'cos:firm':>8s} {'cos:warm':>8s} "
              f"{'cos:nua':>8s} {'cos:fact':>8s} {'cos:conc':>8s} {'cos:cre':>8s} "
              f"{'cos:syc':>8s} {'bip:comb':>8s}")
        print("-" * 100)

        # Build a list for sorting
        scored = []
        for word, anto in CANDIDATES:
            pos = word_embs[word]
            neg = word_embs[anto] if anto else word_embs["bad"]
            bipolar = pos - neg
            bipolar = bipolar / (np.linalg.norm(bipolar) + 1e-12)

            row = {"word": word, "antonym": anto}
            cos_per, bip_per = {}, {}
            for cat, (be, we) in cat_embs.items():
                cos_per[cat] = round(acc(cosine_to(be, pos), cosine_to(we, pos)), 4)
                bip_per[cat] = round(acc(be @ bipolar, we @ bipolar), 4)
            row["cosine"] = cos_per
            row["bipolar"] = bip_per
            scored.append(row)

        # Sort by cosine combined (sum of category accuracies / n_categories)
        cats = list(cat_cases.keys())
        def combined(r, method):
            return sum(r[method][c] for c in cats) / len(cats)
        scored.sort(key=lambda r: -combined(r, "cosine"))

        print(f"\n--- TOP 25 by cosine combined ---")
        print(f"{'word':<18s} {'comb':>6s} {'firm':>6s} {'warm':>6s} {'nuan':>6s} "
              f"{'fact':>6s} {'conc':>6s} {'cre':>6s} {'syc':>6s} | bip_comb")
        for row in scored[:25]:
            c = row["cosine"]; b = row["bipolar"]
            cb = sum(b[cat] for cat in cats)/len(cats)
            print(f"  {row['word']:<18s} {combined(row,'cosine'):5.0%} "
                  f"{c.get('firmness',0):5.0%} {c.get('warmth',0):5.0%} "
                  f"{c.get('nuance_context',0):5.0%} {c.get('factual_accuracy',0):5.0%} "
                  f"{c.get('conciseness_completeness',0):5.0%} "
                  f"{c.get('creative_quality',0):5.0%} "
                  f"{c.get('anti_sycophancy',0):5.0%} | {cb:5.0%}")

        print(f"\n--- BOTTOM 10 by cosine combined ---")
        for row in scored[-10:]:
            c = row["cosine"]
            print(f"  {row['word']:<18s} {combined(row,'cosine'):5.0%} "
                  f"{c.get('firmness',0):5.0%} {c.get('warmth',0):5.0%} "
                  f"{c.get('anti_sycophancy',0):5.0%}")

        # Random-control baseline (chance floor)
        rand_words = [w for w, _ in CANDIDATES if w in
                      {"blue","table","running","however","therefore","purple",
                       "external","circular","monthly","northern"}]
        if rand_words:
            rand_comb = np.mean([combined(r,'cosine') for r in scored
                                 if r['word'] in rand_words])
            print(f"\n  Random-control chance floor (mean of {len(rand_words)} "
                  f"random words): {rand_comb:.0%}")

        results["per_model"][model_name] = scored
        del model
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
