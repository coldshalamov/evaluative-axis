#!/usr/bin/env python3
"""Map the semantic tree of "good" empirically.

For a large vocabulary of evaluative terms, compute:
1. Cosine similarity to "good" (what's closest to the root?)
2. Cosine similarity to each other (what clusters together?)
3. For each cluster, find the NEGATIVE children (the failure modes)

This builds the tree from the embedding geometry itself, not from
our assumptions about what should be where.
"""

import json, gc
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# Large vocabulary of evaluative terms (positive only — we'll find negatives separately)
POSITIVE_TERMS = [
    # Broad quality
    "good", "excellent", "outstanding", "quality",
    # Warmth/interpersonal
    "warm", "kind", "empathetic", "compassionate", "supportive",
    "respectful", "considerate", "friendly", "pleasant", "polite",
    # Helpfulness
    "helpful", "useful", "constructive", "valuable",
    # Honesty/integrity
    "honest", "truthful", "sincere", "candid", "transparent",
    "forthright", "genuine", "authentic",
    # Competence/rigor
    "careful", "precise", "accurate", "rigorous", "meticulous",
    "thorough", "diligent", "methodical", "systematic",
    # Restraint/discipline
    "restrained", "measured", "moderate", "disciplined", "prudent",
    "cautious", "balanced",
    # Clarity
    "clear", "concise", "articulate", "coherent",
    # Intelligence/insight
    "insightful", "analytical", "logical", "rational", "wise",
    "thoughtful", "perceptive", "astute",
    # Reliability
    "reliable", "consistent", "dependable", "responsible",
    # Fairness
    "fair", "impartial", "objective", "neutral",
    # Courage/firmness
    "principled", "firm", "resolute", "direct", "frank",
    "skeptical", "critical", "discerning",
    # Safety/care
    "safe", "protective", "responsible",
    "ethical", "moral",
]

# Negative terms — potential failure modes to penalize
NEGATIVE_TERMS = [
    # Sycophancy cluster
    "sycophantic", "placating", "flattering", "obsequious", "fawning",
    "ingratiating", "servile", "submissive",
    # Deception cluster
    "deceptive", "misleading", "dishonest", "manipulative",
    "evasive", "withholding",
    # Recklessness cluster
    "reckless", "careless", "negligent", "hasty", "impulsive",
    "sloppy", "haphazard",
    # Excess cluster
    "verbose", "excessive", "extreme", "overwhelming",
    "overconfident", "presumptuous",
    # Coldness cluster (NOT necessarily bad — but the opposite of warmth)
    "cold", "dismissive", "curt", "harsh", "blunt",
    # Incompetence cluster
    "incompetent", "superficial", "shallow", "simplistic", "vague",
    # Bias cluster
    "biased", "partial", "unfair", "prejudiced",
]

# Remove duplicates
POSITIVE_TERMS = list(dict.fromkeys(POSITIVE_TERMS))
NEGATIVE_TERMS = list(dict.fromkeys(NEGATIVE_TERMS))


def main():
    from sentence_transformers import SentenceTransformer

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Embed all terms
        pos_embs = embed(POSITIVE_TERMS)
        neg_embs = embed(NEGATIVE_TERMS)
        good_emb = pos_embs[POSITIVE_TERMS.index("good")]

        # 1. SIMILARITY TO "GOOD" — what are good's closest children?
        print(f"\n  DISTANCE FROM 'GOOD' (cosine similarity):")
        sims_to_good = []
        for i, term in enumerate(POSITIVE_TERMS):
            if term == "good":
                continue
            sim = float(np.dot(good_emb, pos_embs[i]) / (norm(good_emb) * norm(pos_embs[i]) + 1e-12))
            sims_to_good.append((term, sim))

        sims_to_good.sort(key=lambda x: -x[1])
        print(f"\n  Top 20 closest to 'good':")
        for term, sim in sims_to_good[:20]:
            print(f"    {term:20s}  {sim:.4f}")
        print(f"\n  Bottom 10 farthest from 'good':")
        for term, sim in sims_to_good[-10:]:
            print(f"    {term:20s}  {sim:.4f}")

        # 2. For each top-level child, find ITS closest siblings
        # Take top 5 as candidate children of good
        top_children = [t for t, s in sims_to_good[:5]]
        print(f"\n  CANDIDATE CHILDREN OF GOOD: {', '.join(top_children)}")

        for child in top_children:
            child_emb = pos_embs[POSITIVE_TERMS.index(child)]
            child_sims = []
            for i, term in enumerate(POSITIVE_TERMS):
                if term == child or term == "good":
                    continue
                sim = float(np.dot(child_emb, pos_embs[i]) / (norm(child_emb) * norm(pos_embs[i]) + 1e-12))
                child_sims.append((term, sim))
            child_sims.sort(key=lambda x: -x[1])
            siblings = [t for t, s in child_sims[:5]]
            print(f"\n    '{child}' -> closest: {', '.join(f'{t}({s:.3f})' for t, s in child_sims[:5])}")

        # 3. NEGATIVE CHILDREN — which bad terms are closest to each good child?
        print(f"\n  FAILURE MODES (negative terms closest to each good child):")
        for child in top_children:
            child_emb = pos_embs[POSITIVE_TERMS.index(child)]
            neg_sims = []
            for i, term in enumerate(NEGATIVE_TERMS):
                sim = float(np.dot(child_emb, neg_embs[i]) / (norm(child_emb) * norm(neg_embs[i]) + 1e-12))
                neg_sims.append((term, sim))
            neg_sims.sort(key=lambda x: -x[1])
            print(f"    '{child}' -> failures: {', '.join(f'{t}({s:.3f})' for t, s in neg_sims[:5])}")

        # Also: what negative terms are closest to GOOD itself?
        print(f"\n  FAILURE MODES OF 'GOOD' DIRECTLY:")
        good_neg_sims = []
        for i, term in enumerate(NEGATIVE_TERMS):
            sim = float(np.dot(good_emb, neg_embs[i]) / (norm(good_emb) * norm(neg_embs[i]) + 1e-12))
            good_neg_sims.append((term, sim))
        good_neg_sims.sort(key=lambda x: -x[1])
        for term, sim in good_neg_sims[:10]:
            print(f"    {term:20s}  {sim:.4f}")

        # 4. HIERARCHICAL CLUSTERING — build the tree bottom-up
        print(f"\n  COSINE SIMILARITY BETWEEN TOP CHILDREN:")
        for i, c1 in enumerate(top_children):
            e1 = pos_embs[POSITIVE_TERMS.index(c1)]
            row = []
            for j, c2 in enumerate(top_children):
                e2 = pos_embs[POSITIVE_TERMS.index(c2)]
                sim = float(np.dot(e1, e2) / (norm(e1) * norm(e2) + 1e-12))
                row.append(f"{sim:.3f}")
            print(f"    {c1:15s}  {' '.join(row)}")

        # 5. NEGATIVE TERM CLUSTERING — which negative terms cluster together?
        # (tells us about the structure of failure modes)
        print(f"\n  NEGATIVE TERMS CLOSEST TO 'GOOD' vs FARTHEST:")
        for term, sim in good_neg_sims[:5]:
            print(f"    CLOSE: {term:20s}  {sim:.4f}")
        print(f"    ...")
        for term, sim in good_neg_sims[-5:]:
            print(f"    FAR:   {term:20s}  {sim:.4f}")

        # 6. KEY QUESTION: where does "sycophantic" sit relative to "good" and its children?
        syc_emb = neg_embs[NEGATIVE_TERMS.index("sycophantic")]
        plac_emb = neg_embs[NEGATIVE_TERMS.index("placating")]
        flat_emb = neg_embs[NEGATIVE_TERMS.index("flattering")]

        print(f"\n  SYCOPHANCY CLUSTER RELATIONSHIPS:")
        for neg_name, neg_e in [("sycophantic", syc_emb), ("placating", plac_emb), ("flattering", flat_emb)]:
            good_sim = float(np.dot(good_emb, neg_e) / (norm(good_emb) * norm(neg_e) + 1e-12))
            print(f"    cos(good, {neg_name:15s}) = {good_sim:.4f}")
            for child in top_children:
                child_emb = pos_embs[POSITIVE_TERMS.index(child)]
                child_sim = float(np.dot(child_emb, neg_e) / (norm(child_emb) * norm(neg_e) + 1e-12))
                print(f"      cos({child:12s}, {neg_name:15s}) = {child_sim:.4f}")

        del model
        gc.collect()


if __name__ == "__main__":
    main()
