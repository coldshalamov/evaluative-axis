#!/usr/bin/env python3
"""Does a word's semantic neighborhood predict its warmth bias?

Theory: words whose neighborhoods are warmth-heavy should show warmth bias
in scoring (good pattern). Words whose neighborhoods are competence/restraint-heavy
should be warmth-independent (careful pattern).

We test this by computing neighborhood composition for every axis word we've
tested on the battery, then correlating with their known per-split accuracy.
"""

import json, gc, math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# All axis words we've tested on the battery, with their known warmth d values
# from notes/research_cycles/absolute_score_analysis/absolute_score_results.json
TESTED_AXES = {
    "good":       {"pos": "Good", "neg": "Bad"},
    "careful":    {"pos": "Careful", "neg": "Reckless"},
    "restrained": {"pos": "Restrained", "neg": "Unrestrained"},
    "thorough":   {"pos": "Thorough", "neg": "Superficial"},
    "honest":     {"pos": "Honest", "neg": "Dishonest"},
    "kind":       {"pos": "Kind", "neg": "Cruel"},
    "helpful":    {"pos": "Helpful", "neg": "Unhelpful"},
    "moderate":   {"pos": "Moderate", "neg": "Excessive"},
    "precise":    {"pos": "Precise", "neg": "Vague"},
    "logical":    {"pos": "Logical", "neg": "Illogical"},
    "measured":   {"pos": "Measured", "neg": "Extreme"},
    "disciplined": {"pos": "Disciplined", "neg": "Undisciplined"},
    "prudent":    {"pos": "Prudent", "neg": "Imprudent"},
    "methodical": {"pos": "Methodical", "neg": "Haphazard"},
    "rigorous":   {"pos": "Rigorous", "neg": "Lax"},
    "diligent":   {"pos": "Diligent", "neg": "Negligent"},
    "balanced":   {"pos": "Balanced", "neg": "Extreme"},
    "considerate": {"pos": "Considerate", "neg": "Inconsiderate"},
}

# Same vocabulary as the neighborhood script, for computing neighborhood composition
VOCAB = {
    "kind": "warmth", "friendly": "warmth", "warm": "warmth",
    "caring": "warmth", "supportive": "warmth", "empathetic": "warmth",
    "gentle": "warmth", "compassionate": "warmth", "loving": "warmth",
    "nurturing": "warmth", "tender": "warmth", "affectionate": "warmth",
    "generous": "warmth", "charitable": "warmth", "benevolent": "warmth",
    "considerate": "warmth", "thoughtful": "warmth", "sympathetic": "warmth",
    "welcoming": "warmth", "gracious": "warmth", "cordial": "warmth",
    "amiable": "warmth", "agreeable": "warmth", "pleasant": "warmth",
    "sweet": "warmth", "soft": "warmth", "mild": "warmth",
    "forgiving": "warmth", "tolerant": "warmth", "understanding": "warmth",
    "encouraging": "warmth", "reassuring": "warmth", "comforting": "warmth",
    "careful": "competence", "thorough": "competence", "precise": "competence",
    "accurate": "competence", "rigorous": "competence", "systematic": "competence",
    "methodical": "competence", "diligent": "competence", "meticulous": "competence",
    "attentive": "competence", "detail-oriented": "competence",
    "skilled": "competence", "competent": "competence", "capable": "competence",
    "proficient": "competence", "expert": "competence", "adept": "competence",
    "effective": "competence", "efficient": "competence", "productive": "competence",
    "analytical": "competence", "logical": "competence", "rational": "competence",
    "sharp": "competence", "astute": "competence", "perceptive": "competence",
    "knowledgeable": "competence", "informed": "competence", "educated": "competence",
    "clever": "competence", "resourceful": "competence", "ingenious": "competence",
    "honest": "integrity", "truthful": "integrity", "sincere": "integrity",
    "authentic": "integrity", "genuine": "integrity", "trustworthy": "integrity",
    "reliable": "integrity", "faithful": "integrity", "loyal": "integrity",
    "principled": "integrity", "ethical": "integrity", "moral": "integrity",
    "virtuous": "integrity", "righteous": "integrity", "honorable": "integrity",
    "transparent": "integrity", "candid": "integrity", "forthright": "integrity",
    "fair": "integrity", "just": "integrity", "equitable": "integrity",
    "impartial": "integrity", "objective": "integrity", "unbiased": "integrity",
    "restrained": "restraint", "moderate": "restraint", "measured": "restraint",
    "disciplined": "restraint", "controlled": "restraint", "temperate": "restraint",
    "balanced": "restraint", "cautious": "restraint", "prudent": "restraint",
    "conservative": "restraint", "reserved": "restraint", "composed": "restraint",
    "patient": "restraint", "steady": "restraint", "calm": "restraint",
    "sober": "restraint", "sensible": "restraint", "judicious": "restraint",
    "circumspect": "restraint", "deliberate": "restraint", "calculated": "restraint",
    "strong": "strength", "powerful": "strength", "firm": "strength",
    "resolute": "strength", "decisive": "strength", "bold": "strength",
    "brave": "strength", "courageous": "strength", "fearless": "strength",
    "confident": "strength", "assertive": "strength", "determined": "strength",
    "persistent": "strength", "tenacious": "strength", "resilient": "strength",
    "tough": "strength", "robust": "strength", "vigorous": "strength",
    "forceful": "strength", "commanding": "strength", "authoritative": "strength",
    "polite": "social", "respectful": "social", "courteous": "social",
    "civil": "social", "diplomatic": "social", "tactful": "social",
    "humble": "social", "modest": "social", "deferential": "social",
    "cooperative": "social", "collaborative": "social", "accommodating": "social",
    "obedient": "social", "compliant": "social", "submissive": "social",
    "useful": "utility", "helpful": "utility", "practical": "utility",
    "functional": "utility", "constructive": "utility", "beneficial": "utility",
    "valuable": "utility", "worthwhile": "utility", "meaningful": "utility",
    "relevant": "utility", "applicable": "utility", "instrumental": "utility",
    "handy": "utility", "convenient": "utility", "serviceable": "utility",
    "creative": "creativity", "innovative": "creativity", "original": "creativity",
    "imaginative": "creativity", "inventive": "creativity", "novel": "creativity",
    "unconventional": "creativity", "experimental": "creativity", "visionary": "creativity",
    "inspired": "creativity", "artistic": "creativity", "expressive": "creativity",
    "happy": "emotion", "cheerful": "emotion", "optimistic": "emotion",
    "positive": "emotion", "enthusiastic": "emotion", "passionate": "emotion",
    "joyful": "emotion", "content": "emotion", "satisfied": "emotion",
    "grateful": "emotion", "hopeful": "emotion", "excited": "emotion",
    "reckless": "neg_competence", "careless": "neg_competence",
    "sloppy": "neg_competence", "negligent": "neg_competence",
    "incompetent": "neg_competence", "clumsy": "neg_competence",
    "dishonest": "neg_integrity", "deceitful": "neg_integrity",
    "manipulative": "neg_integrity", "deceptive": "neg_integrity",
    "cruel": "neg_warmth", "cold": "neg_warmth", "harsh": "neg_warmth",
    "hostile": "neg_warmth", "aggressive": "neg_warmth", "mean": "neg_warmth",
    "selfish": "neg_warmth", "indifferent": "neg_warmth",
    "lazy": "neg_discipline", "impulsive": "neg_discipline",
    "rash": "neg_discipline", "hasty": "neg_discipline",
    "chaotic": "neg_discipline", "erratic": "neg_discipline",
    "wise": "wisdom", "sage": "wisdom", "discerning": "wisdom",
    "insightful": "wisdom", "reasonable": "wisdom", "pragmatic": "wisdom",
    "realistic": "wisdom",
    "appropriate": "misc", "proper": "misc", "correct": "misc",
    "right": "misc", "suitable": "misc", "fitting": "misc",
    "legitimate": "misc", "valid": "misc", "sound": "misc",
    "wholesome": "misc", "healthy": "misc", "safe": "misc",
    "clean": "misc", "pure": "misc", "innocent": "misc",
    "natural": "misc", "organic": "misc",
    "complete": "misc", "comprehensive": "misc",
    "consistent": "misc", "coherent": "misc", "clear": "misc",
    "concise": "misc", "succinct": "misc", "articulate": "misc",
    "elegant": "misc", "refined": "misc", "polished": "misc",
    "mature": "misc", "sophisticated": "misc", "nuanced": "misc",
}


def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


def warmth_fraction(word_emb, vocab_embs, vocab_cats, top_n=30):
    """What fraction of a word's top-N neighbors are warmth/emotion?"""
    sims = [(cosine_sim(word_emb, vocab_embs[i]), vocab_cats[i]) for i in range(len(vocab_cats))]
    sims.sort(key=lambda x: -x[0])
    # Skip self if present (sim ~1.0)
    filtered = [(s, c) for s, c in sims if s < 0.999][:top_n]
    warmth_count = sum(1 for s, c in filtered if c in ("warmth", "emotion"))
    restraint_count = sum(1 for s, c in filtered if c in ("competence", "restraint"))
    total = len(filtered)
    return warmth_count / total if total > 0 else 0, restraint_count / total if total > 0 else 0


def main():
    from sentence_transformers import SentenceTransformer

    # Load battery results if available for per-split accuracy
    abs_results_path = ROOT / "notes/research_cycles/absolute_score_analysis/absolute_score_results.json"
    if abs_results_path.exists():
        abs_data = json.loads(abs_results_path.read_text(encoding="utf-8"))
    else:
        abs_data = None
        print("Warning: no absolute_score_results.json found")

    vocab_words = list(VOCAB.keys())
    vocab_cats = [VOCAB[w] for w in vocab_words]

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        vocab_embs = embed(vocab_words)

        # For each tested axis word, compute neighborhood warmth fraction
        axis_words = [info["pos"].lower() for info in TESTED_AXES.values()]
        axis_embs = embed(axis_words)

        model_data = []
        print(f"\n{'Axis':15s} {'WarmthFrac':>10s} {'CompFrac':>10s} {'Warmth d':>10s} {'Firm d':>10s}  Prediction")
        print("-" * 85)

        for i, (axis_name, info) in enumerate(TESTED_AXES.items()):
            wf, cf = warmth_fraction(axis_embs[i], vocab_embs, vocab_cats, top_n=30)

            # Look up warmth d from absolute results if available
            warmth_d = None
            firm_d = None
            if abs_data and short in abs_data:
                model_abs = abs_data[short]
                if axis_name in model_abs:
                    warmth_d = model_abs[axis_name].get("warmth_d")
                    firm_d = model_abs[axis_name].get("firmness_d")

            # Prediction: high warmth fraction -> warmth biased, high comp fraction -> not
            if wf > 0.35:
                pred = "warmth-biased"
            elif cf > 0.25:
                pred = "warmth-independent"
            else:
                pred = "ambiguous"

            wd_str = f"{warmth_d:+.3f}" if warmth_d is not None else "n/a"
            fd_str = f"{firm_d:+.3f}" if firm_d is not None else "n/a"

            print(f"{axis_name:15s} {wf:10.0%} {cf:10.0%} {wd_str:>10s} {fd_str:>10s}  {pred}")

            model_data.append({
                "axis": axis_name,
                "warmth_frac": round(wf, 3),
                "comp_frac": round(cf, 3),
                "warmth_d": warmth_d,
                "firmness_d": firm_d,
                "prediction": pred,
            })

        # Correlation between warmth_frac and warmth_d
        wfs = [d["warmth_frac"] for d in model_data if d["warmth_d"] is not None]
        wds = [d["warmth_d"] for d in model_data if d["warmth_d"] is not None]
        if len(wfs) >= 3:
            corr = float(np.corrcoef(wfs, wds)[0, 1])
            print(f"\n  Correlation(warmth_frac, warmth_d) = {corr:+.3f}  (n={len(wfs)})")

        # Also correlate with firmness_d (should be negative)
        fds = [d["firmness_d"] for d in model_data if d["firmness_d"] is not None]
        if len(fds) >= 3:
            corr_f = float(np.corrcoef(wfs[:len(fds)], fds)[0, 1])
            print(f"  Correlation(warmth_frac, firmness_d) = {corr_f:+.3f}  (n={len(fds)})")

        results[short] = model_data
        del model
        gc.collect()

    # Cross-model summary
    print(f"\n{'='*80}")
    print("CROSS-MODEL: Neighborhood warmth fraction for each axis")
    print(f"{'='*80}")
    print(f"\n{'Axis':15s}", end="")
    for m in results:
        print(f" {m[:12]:>12s}", end="")
    print("  mean")
    print("-" * 70)

    axis_names = [d["axis"] for d in results[list(results.keys())[0]]]
    for axis in axis_names:
        print(f"{axis:15s}", end="")
        vals = []
        for m in results:
            for d in results[m]:
                if d["axis"] == axis:
                    print(f" {d['warmth_frac']:12.0%}", end="")
                    vals.append(d["warmth_frac"])
        print(f"  {np.mean(vals):.0%}")

    # Save
    out = ROOT / "notes/research_cycles/good_neighborhood"
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / "neighborhood_predicts_bias.json"
    outfile.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
