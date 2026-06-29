#!/usr/bin/env python3
"""Map the semantic neighborhood of "good" in embedding space.

Find words closest to "good" that AREN'T synonyms — the conceptual
decomposition. Compare to "careful" neighborhood to see if they differ
in warmth vs competence composition.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

SYNONYMS_OF_GOOD = {
    "great", "excellent", "wonderful", "fantastic", "superb", "terrific",
    "magnificent", "outstanding", "exceptional", "marvelous", "splendid",
    "fine", "nice", "decent", "adequate", "acceptable", "satisfactory",
    "perfect", "ideal", "brilliant", "awesome", "amazing", "fabulous",
    "tremendous", "remarkable", "phenomenal", "glorious", "lovely",
    "delightful", "superior", "first-rate", "top-notch",
}

SYNONYMS_OF_BAD = {
    "terrible", "awful", "horrible", "dreadful", "atrocious", "abysmal",
    "lousy", "poor", "inferior", "subpar", "mediocre", "pathetic",
    "wretched", "ghastly", "horrendous", "appalling", "dire", "dismal",
}

# Tag each word with a semantic category for analysis
VOCAB = {
    # WARMTH / INTERPERSONAL
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

    # COMPETENCE / SKILL
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

    # INTEGRITY / HONESTY
    "honest": "integrity", "truthful": "integrity", "sincere": "integrity",
    "authentic": "integrity", "genuine": "integrity", "trustworthy": "integrity",
    "reliable": "integrity", "faithful": "integrity", "loyal": "integrity",
    "principled": "integrity", "ethical": "integrity", "moral": "integrity",
    "virtuous": "integrity", "righteous": "integrity", "honorable": "integrity",
    "transparent": "integrity", "candid": "integrity", "forthright": "integrity",
    "fair": "integrity", "just": "integrity", "equitable": "integrity",
    "impartial": "integrity", "objective": "integrity", "unbiased": "integrity",

    # RESTRAINT / DISCIPLINE
    "restrained": "restraint", "moderate": "restraint", "measured": "restraint",
    "disciplined": "restraint", "controlled": "restraint", "temperate": "restraint",
    "balanced": "restraint", "cautious": "restraint", "prudent": "restraint",
    "conservative": "restraint", "reserved": "restraint", "composed": "restraint",
    "patient": "restraint", "steady": "restraint", "calm": "restraint",
    "sober": "restraint", "sensible": "restraint", "judicious": "restraint",
    "circumspect": "restraint", "deliberate": "restraint", "calculated": "restraint",

    # STRENGTH / ASSERTIVENESS
    "strong": "strength", "powerful": "strength", "firm": "strength",
    "resolute": "strength", "decisive": "strength", "bold": "strength",
    "brave": "strength", "courageous": "strength", "fearless": "strength",
    "confident": "strength", "assertive": "strength", "determined": "strength",
    "persistent": "strength", "tenacious": "strength", "resilient": "strength",
    "tough": "strength", "robust": "strength", "vigorous": "strength",
    "forceful": "strength", "commanding": "strength", "authoritative": "strength",

    # SOCIAL / POLITENESS
    "polite": "social", "respectful": "social", "courteous": "social",
    "civil": "social", "diplomatic": "social", "tactful": "social",
    "humble": "social", "modest": "social", "deferential": "social",
    "cooperative": "social", "collaborative": "social", "accommodating": "social",
    "obedient": "social", "compliant": "social", "submissive": "social",

    # UTILITY / HELPFULNESS
    "useful": "utility", "helpful": "utility", "practical": "utility",
    "functional": "utility", "constructive": "utility", "beneficial": "utility",
    "valuable": "utility", "worthwhile": "utility", "meaningful": "utility",
    "relevant": "utility", "applicable": "utility", "instrumental": "utility",
    "handy": "utility", "convenient": "utility", "serviceable": "utility",

    # CREATIVITY / NOVELTY
    "creative": "creativity", "innovative": "creativity", "original": "creativity",
    "imaginative": "creativity", "inventive": "creativity", "novel": "creativity",
    "unconventional": "creativity", "experimental": "creativity", "visionary": "creativity",
    "inspired": "creativity", "artistic": "creativity", "expressive": "creativity",

    # EMOTIONAL VALENCE
    "happy": "emotion", "cheerful": "emotion", "optimistic": "emotion",
    "positive": "emotion", "enthusiastic": "emotion", "passionate": "emotion",
    "joyful": "emotion", "content": "emotion", "satisfied": "emotion",
    "grateful": "emotion", "hopeful": "emotion", "excited": "emotion",

    # NEGATIVE POLES (controls)
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

    # WISDOM / JUDGMENT
    "wise": "wisdom", "sage": "wisdom", "discerning": "wisdom",
    "insightful": "wisdom", "thoughtful": "wisdom",
    "reasonable": "wisdom", "pragmatic": "wisdom", "realistic": "wisdom",

    # MISC EVALUATIVE
    "appropriate": "misc", "proper": "misc", "correct": "misc",
    "right": "misc", "suitable": "misc", "fitting": "misc",
    "legitimate": "misc", "valid": "misc", "sound": "misc",
    "wholesome": "misc", "healthy": "misc", "safe": "misc",
    "clean": "misc", "pure": "misc", "innocent": "misc",
    "natural": "misc", "organic": "misc", "authentic": "misc",
    "complete": "misc", "comprehensive": "misc", "thorough": "misc",
    "consistent": "misc", "coherent": "misc", "clear": "misc",
    "concise": "misc", "succinct": "misc", "articulate": "misc",
    "elegant": "misc", "refined": "misc", "polished": "misc",
    "mature": "misc", "sophisticated": "misc", "nuanced": "misc",
}

TARGETS = ["good", "careful", "bad"]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


def main():
    from sentence_transformers import SentenceTransformer

    all_words = list(VOCAB.keys())
    all_categories = [VOCAB[w] for w in all_words]
    print(f"Vocabulary: {len(all_words)} words")
    cats = {}
    for c in all_categories:
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        target_embs = embed(TARGETS)
        word_embs = embed(all_words)

        model_results = {}
        for ti, target in enumerate(TARGETS):
            sims = []
            for wi, word in enumerate(all_words):
                sim = cosine_sim(target_embs[ti], word_embs[wi])
                sims.append((word, VOCAB[word], sim))

            sims.sort(key=lambda x: -x[2])

            # Filter synonyms
            if target == "good":
                filtered = [(w, c, s) for w, c, s in sims if w not in SYNONYMS_OF_GOOD]
            elif target == "bad":
                filtered = [(w, c, s) for w, c, s in sims if w not in SYNONYMS_OF_BAD]
            else:
                filtered = sims

            model_results[target] = {
                "top50_all": [(w, c, round(s, 4)) for w, c, s in sims[:50]],
                "top50_filtered": [(w, c, round(s, 4)) for w, c, s in filtered[:50]],
            }

            print(f"\n--- Top 30 nearest to '{target}' (synonyms excluded) ---")
            for i, (w, c, s) in enumerate(filtered[:30]):
                print(f"  {i+1:2d}. {w:20s} [{c:15s}]  sim={s:.4f}")

            # Category breakdown of top 30
            cat_counts = {}
            for w, c, s in filtered[:30]:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            print(f"\n  Category breakdown (top 30 non-synonym neighbors of '{target}'):")
            for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
                print(f"    {c:15s}: {n:2d} ({n/30:.0%})")

        # Overlap analysis: what's in good's top30 but NOT in careful's top30?
        good_top30 = {w for w, c, s in model_results["good"]["top50_filtered"][:30]}
        care_top30 = {w for w, c, s in model_results["careful"]["top50_filtered"][:30]}
        only_good = good_top30 - care_top30
        only_care = care_top30 - good_top30
        both = good_top30 & care_top30

        print(f"\n--- Overlap: good vs careful top-30 ---")
        print(f"  Only near 'good' ({len(only_good)}): {sorted(only_good)}")
        print(f"  Only near 'careful' ({len(only_care)}): {sorted(only_care)}")
        print(f"  Near both ({len(both)}): {sorted(both)}")

        results[short] = model_results
        del model
        gc.collect()

    # Cross-model consistency: which words are in ALL models' good-top30?
    print(f"\n{'='*80}")
    print("CROSS-MODEL: Words in ALL three models' good-neighborhood (top 30)")
    print(f"{'='*80}")

    good_sets = []
    for m in results:
        s = {w for w, c, sim in results[m]["good"]["top50_filtered"][:30]}
        good_sets.append(s)
    cross_model_good = good_sets[0] & good_sets[1] & good_sets[2]
    print(f"\nWords near 'good' on ALL 3 models ({len(cross_model_good)}):")
    for w in sorted(cross_model_good):
        cat = VOCAB[w]
        sims = []
        for m in results:
            for ww, cc, ss in results[m]["good"]["top50_filtered"]:
                if ww == w:
                    sims.append(ss)
                    break
        print(f"  {w:20s} [{cat:15s}]  sims: {sims}")

    care_sets = []
    for m in results:
        s = {w for w, c, sim in results[m]["careful"]["top50_filtered"][:30]}
        care_sets.append(s)
    cross_model_care = care_sets[0] & care_sets[1] & care_sets[2]
    print(f"\nWords near 'careful' on ALL 3 models ({len(cross_model_care)}):")
    for w in sorted(cross_model_care):
        cat = VOCAB[w]
        sims = []
        for m in results:
            for ww, cc, ss in results[m]["careful"]["top50_filtered"]:
                if ww == w:
                    sims.append(ss)
                    break
        print(f"  {w:20s} [{cat:15s}]  sims: {sims}")

    # Category composition comparison
    print(f"\n{'='*80}")
    print("CATEGORY COMPOSITION: good vs careful neighborhoods (top 30, cross-model avg)")
    print(f"{'='*80}")

    for target in ["good", "careful"]:
        print(f"\n  '{target}' neighborhood category percentages by model:")
        avg_cats = {}
        for m in results:
            top30 = results[m][target]["top50_filtered"][:30]
            cats = {}
            for w, c, s in top30:
                cats[c] = cats.get(c, 0) + 1
            print(f"    {m}:")
            for c, n in sorted(cats.items(), key=lambda x: -x[1]):
                avg_cats[c] = avg_cats.get(c, 0) + n/30
                print(f"      {c:15s}: {n:2d} ({n/30:.0%})")
        print(f"    Cross-model average:")
        for c, v in sorted(avg_cats.items(), key=lambda x: -x[1]):
            print(f"      {c:15s}: {v/3:.0%}")

    # Save
    out = ROOT / "notes/research_cycles/good_neighborhood"
    out.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for m in results:
        save_data[m] = {}
        for target in TARGETS:
            save_data[m][target] = {
                "top50_filtered": results[m][target]["top50_filtered"],
            }

    outfile = out / "good_neighborhood_results.json"
    outfile.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
