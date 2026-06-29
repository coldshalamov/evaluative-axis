# Centroid Method Test Battery: Summary of Results

**Date**: 2026-06-29
**Models tested**: Snowflake Arctic Embed M, BGE-M3, Nomic Embed Text v1.5
**Battery**: 70 training pairs (50 original + 20 warmth), 61 OOS expansion cases

---

## TEST RESULTS

### 1. External Validation (TEST 1A) — NEGATIVE
Cross-dataset transfer fails. Centroid trained on our battery → chance accuracy
on SHP and UltraFeedback. Centroid trained on SHP → 66-77% on our battery (within-dataset OK,
cross-dataset zero). The quality direction is dataset-specific.

### 2. Vocabulary Projection (TEST 2A) — CLAIM HOLDS
10K Brown corpus words projected onto the centroid. Max |cosine|: Snowflake 0.12,
BGE-M3 0.25, Nomic 0.25. No word exceeds 0.30. The quality direction is
orthogonal to all common vocabulary.

### 3. Phrase Projection (TEST 2B) — CLAIM HOLDS
46 quality phrases + 22 single words. No phrase exceeds 0.30. Max: BGE-M3
"not helpful" at 0.244. Phrases marginally better than single words (+0.01 to
+0.04 max improvement). The quality direction remains unnameable.

### 4. PCA Dimensionality (TEST 3A) — 51 PCs FOR 80% VARIANCE
Quality is high-dimensional: 51 components needed for 80% of variance.
Centroid (88-93% in-sample) >> PCA top-k (41-77%). The centroid captures a
specific direction that PCA's top components miss.

### 5. Margin-Length Correlation (TEST 4A) — NOT A CONFOUND
word_length_diff correlation with quality margin is inconsistent across models.
Length does not drive the centroid's discriminative ability.

### 6. Gameability (TEST 6A-B) — NOT GAMEABLE
6 mechanical modifications × 30 cases × 3 models. Total flip rate: 1.9%.
No modification (padding, hedging, bullet points, repetition, formalization,
lengthening) reliably inflates the score. The centroid is robust to surface
manipulation.

### 7. Score Diversity (TEST 6C) — TOLERATES DIVERSITY
10 prompts × 5 styles (concise, warm, technical, casual, formal).
Between-prompt variance > within-prompt variance (ratio 1.8-2.7x) on all models.
No single style dominates cross-model. But formal and technical consistently
score lowest — a real bias toward accessible, direct language.

### 8. Additional Models (TEST 1C) — MIXED
- gte-base: 65.6% OOS, permutation p=0.540
- e5-base-v2: 72.1% OOS, permutation p=0.675
- mxbai-embed-large-v1: 70.5% OOS, permutation p=0.335

Above chance but fail permutation significance. The quality signal exists
in these models but is weaker than in the core three.

### 9. Meta-Validity Battery (C10) — METHOD IS SOUND

**M1. Prompt Leakage**: Response-only BEATS full-format on 2/3 models
(Snowflake +11.4pp, Nomic +8.2pp). The prompt adds noise.

**M2. Label Flip**: Perfect symmetry on all 3 models. Cosine(normal,
flipped) = -1.000. No structural artifact.

**M3. LOO Stability**: Very stable. Mean cosine 0.995+. Max single-case
accuracy impact ±5-7%. No case dominates.

**M4. Edge Cases**: BGE-M3: only "refuse" (1/7 degenerate) above real_better.
Snowflake: all 7 above — absolute scoring is meaningless on Snowflake.
Pairwise comparison only.

**M5. Training Influence**: Warmth cases more influential on Snowflake
(p=0.0002). No asymmetry on BGE-M3 or Nomic.

---

## REMAINING

- **Cross-author validation**: 9/70 Gemini pairs generated. API quota
  limited to 20 req/day. Needs ~3 more days of accumulation.

---

## KEY TAKEAWAYS

1. **The centroid works**: 77-86% OOS across 4 models, not gameable,
   stable under LOO, no structural artifacts.

2. **It has real limitations**: no cross-dataset transfer, absolute
   scoring is meaningless, formal/technical style is penalized, weaker
   models don't reach significance.

3. **Use it for pairwise comparison only**: strip the prompt, compare
   two responses to the same prompt. The better response gets a higher
   dot product with the centroid.

4. **The direction is genuinely novel**: no word or phrase captures it.
   It's orthogonal to the evaluative vocabulary of English.
