# Open Research Directions

Last updated: 2026-06-29

This document captures every open question identified so far. New sessions
should read this first instead of re-deriving the research agenda. When a
question is resolved by an experiment, mark it DONE with a date and a
one-line result summary. When a question turns out to be the wrong question,
mark it RETIRED with the reason.

---

## PRIMARY FOCUS: Supervised Centroid Direction

The centroid approach is the central finding. All new experiments should
prioritize validating and stress-testing it. Word-based experiments (section A
below) are diagnostic supporting work explaining WHY words fail as probes.

**Master spec**: `methodology/CENTROID_METHOD_SPEC.md`

### Centroid established results

23. **Centroid achieves 77-86% OOS across 4 models.** Snowflake 77-80%,
    BGE-M3 75-80%, Nomic 66-77%, Gemini 86%. 50% is chance. Permutation
    p-values: 0.003 (Snowflake), 0.053 (BGE-M3), 0.032 (Nomic).

24. **Quality direction is orthogonal to ALL tested English words.**
    Cosine < 0.10 with "good," "careful," "helpful," "honest," etc.
    across all 4 models. The direction captures something words cannot name.

25. **Quality is multi-dimensional.** Firmness and warmth quality directions
    are anti-correlated: cosine -0.35 (Snowflake) to -0.49 (Gemini).
    Training on firmness alone → 15-25% on warmth cases (worse than chance).

26. **Length is NOT a confound.** Length-only classifier: 37-49%. Quality
    direction orthogonal to length direction (cosine < 0.05).

27. **10 labeled pairs suffice for BGE-M3 (78% OOS).** Learning curve
    hasn't fully flattened at 70 pairs.

28. **Margins are self-weighting.** Anti-sycophancy (obvious quality gaps):
    margins 0.10-0.26. Errors: margins -0.003 to -0.035.

29. **The five-term tree's 89-94% is inflated by OR-logic.** Chance baseline
    with 5 binary classifiers under OR: 1 - 0.5^5 = 97%. The centroid's
    77-86% is the honest measure.

30. **Logistic probe (full embedding) beats centroid by 12 points.**
    87-94% on BGE-M3 vs centroid's 75-80%. The centroid's claim is
    "cheapest," not "best."

### Centroid open questions (PRIORITY ORDER)

**C1. External validation [DONE 2026-06-29 — NEGATIVE]** — Our direction
scores chance (43-50%) on SHP and UltraFeedback across all 3 models.
Directions orthogonal (cosine < 0.12). Failure is BIDIRECTIONAL — their
directions also fail on our data (30-50%). But the METHOD works: their own
centroids get 66-77% in-sample. Conclusion: quality is dataset-specific,
not universal. The centroid extracts whatever quality concept is in the
labels, but different label sources define quality in orthogonal directions.
Results: `centroid_deep/external_validation_results.json`.

**C2. Gameability [DONE 2026-06-29 — NOT GAMEABLE]** — Six mechanical
modifications (padded, hedged, bulletized, repeated, formalized, lengthened)
on 30 expansion cases × 3 models. Total flips: 10/540 = 1.9%.
Formalization had ZERO effect across all models. Padding and lengthening
DECREASE scores. Results: `gameability_probe_results.json`.

**C3. Vocabulary projection [DONE 2026-06-29 — CLAIM HOLDS]** — Projected
onto 10K Brown corpus words. No word exceeds cosine 0.30 on any model.
Max: Snowflake 0.12 ("deny"), BGE-M3 0.25 ("doubted"), Nomic 0.25
("efficiently" with NEGATIVE sign — quality points AWAY from performance
vocabulary). The direction is genuinely unnameable by single words.
Results: `vocabulary_projection_results.json`.

**C4. Margin-length correlation [DONE 2026-06-29 — NOT A CONFOUND]** —
word_length_diff inconsistent across models (+0.30, -0.08, 0.00).
|margin| vs |char_diff| consistently NEGATIVE (-0.17 to -0.24). Larger
length differences → smaller margins. Mean length diff near zero for
both correct and incorrect. Results: `margin_length_correlation.json`.

**C5. PCA dimensionality [DONE 2026-06-29]** — ~51 components for 80%
variance on all 3 models. Quality is genuinely high-dimensional. Centroid
(88-93% OOS) massively outperforms PCA-based classification (41-77%).
The centroid works by noise cancellation. Results: `pca_dimensionality_results.json`.

**C6. Additional embedding models [DONE 2026-06-29 — MIXED]** — Three
models tested. All show above-chance OOS accuracy but NONE pass the
200-permutation significance test:
  gte-base: 65.6% OOS, p=0.540.
  e5-base-v2: 72.1% OOS, p=0.675.
  mxbai-embed-large-v1: 70.5% OOS, p=0.335.
In-sample accuracy is high (83-87%) for all three. The signal exists but
doesn't generalize OOS as reliably as on our core four models. Model
quality matters — not all embedding models encode quality structure
strongly enough for the centroid to extract a significant direction.
Results: `additional_models_results.json`.

**C7. Stability at scale** — Does direction stability improve with more
training data? Status: BLOCKED. C1 showed cross-dataset transfer is zero
(our direction and external directions are orthogonal). Testing stability
on external data measures stability of THEIR quality concept, not ours.
Could only test this with more data IN OUR quality framework (cross-author
validation, C9).

**C8. Score diversity** — Does the centroid reward one "flavor" of good
or tolerate diverse quality? Status: NOT STARTED.

**C9. Cross-author validation** — Test whether the centroid works on data
written by a different author (Gemini 2.5 Flash generating pairs with
different system prompts). Addresses the strongest reviewer objection:
all training and test data is self-authored. Status: PARTIALLY STARTED
2026-06-29. Script at `scripts/run_cross_author_validation.py`. 9/70
pairs generated before hitting free-tier daily quota (20 req/day). Needs
accumulation over multiple days. Partial pairs saved at
`notes/research_cycles/centroid_deep/gemini_generated_pairs.jsonl`.

**C10. Meta-validity battery [DONE 2026-06-29]** — Five tests completed:
(M1) Prompt HURTS accuracy on 2/3 models — response-only outperforms full
format by 8-11pp on Snowflake/Nomic. (M2) Label flip is perfectly
symmetric (0% asymmetry, cosine -1.000). (M3) LOO is very stable (mean
cosine 0.995+, max accuracy impact ±5-7%). (M4) Edge case scoring is
mixed — Snowflake bad for absolute scores, BGE-M3 good (1/7 degenerate
above real_better). (M5) Warmth cases more influential on Snowflake only
(p=0.0002); no asymmetry on BGE-M3/Nomic. Results: `meta_validity_results.json`.

---

## Word-Based Experiments (diagnostic, supporting)

The word-based experiments explain WHY words fail as quality probes.
They are NOT the primary finding — the centroid is. These sections are
retained as supporting evidence for the paper.

---

## What we know (established results — word-based)

These are not open questions. They are settled results that constrain the
remaining research. Read them before designing new experiments.

1. **Raw "good/bad" fails as a discriminator on a firmness-biased battery**
   (26% on 50 cases, BGE-M3). But it succeeds on warmth cases (85%).
   The "good" embedding direction is biased toward warmth/agreeableness.

2. **Discrimination != training signal quality.** A conflict battery forces
   the axis to rank warmth-vs-correctness. In actual training, the model
   can generate responses that are BOTH warm and correct. A broad signal
   like "good" might be a better training signal than its discrimination
   score suggests. (See battery methodology limitation, below.)

3. **"Careful/Reckless" is the most consistent single-word axis but does
   NOT reach significance on all local models.** Pooled across 90 cases
   (70 main + 20 expansion), Wilson 95% CI: Nomic 64% [54%, 74%] significant;
   Snowflake 56% [45%, 65%] NOT significant; BGE-M3 53% [43%, 63%] NOT
   significant. On Gemini: 74% [62%, 83%], significant.

4. **No single word tested reaches statistical significance on ALL three
   local models.** The fixed-method pooled analysis (bipolar, standard framing)
   finds no axis with Wilson CI lower bound > 50% on all three models.

5. **Compositing multiple terms into one averaged axis degrades performance.**
   Scoring terms independently and summing (or voting) does NOT help
   out-of-sample. Multi-axis voting overfits: searched combinations gain
   10-15 points in-sample but lose 10-25 points OOS on expansion battery.

6. **Gemini Embedding dramatically outperforms local models** on targeted
   axes (86-98% vs 50-74%). Cause unknown. Gap does not close with scale
   in the 33M-600M range.

7. **Corpus frequency does not predict signal strength.** "Good" is the most
   frequent evaluative word and the worst performer. "Careful" is far less
   frequent and the best single-word performer.

8. **Different axes point in genuinely different geometric directions**
   (cosine 0.01-0.20 between axis vectors). They capture different
   evaluative dimensions independently.

9. **The original 50-case battery is 64% firmness-biased** (32/50 cases
   reward pushback/firmness). This confounds all results measured only
   on that battery.

10. **"Hard/Soft" is a battery artifact, NOT a robust cross-model signal.**
    Appeared strong on firmness-biased original battery (58-68% on 3 local
    models), but drops to 39% on Gemini (inverted, worse than random). It
    captures firmness well but fails on warmth and on balanced test sets.

11. **Cosine-to-positive scoring is model-dependent.** BGE-M3 benefits
    dramatically on many axes (+10-24 points in the 2026-06-28 audit,
    plausibly related to low anchor cosine ~0.50). Snowflake sees no
    robust benefit (high anchor cosine ~0.90). Nomic is mixed. Gemini is
    hurt by cospos. Correlation between anchor geometry and cospos
    advantage exists at model level but NOT within models.

12. **Multi-word anchors generally hurt.** Single words outperform concatenated
    words, grammatical phrases, and sentence-level anchors across all models.

13. **18% of cases are structurally beyond embedding evaluation.** 16/90 cases
    have no single-word axis correct on all 3 local models. These span factual
    accuracy, logical reasoning, persona honesty, writing quality, and emotional
    support. Complementarity is negative: on the best model, adding any other
    axis to "Careful" introduces more failures than it rescues.

14. **Per-model method selection is a researcher degree of freedom.** BGE-M3's
    "Careful" jumps from 51% to 67% with cospos + response_only framing, but
    choosing best-of-several-methods post hoc inflates apparent accuracy. The
    fixed-method pooled numbers are the honest ones.

15. **Two pre-registered predictions returned null.** (a) Within-model
    geometry → cospos advantage: r = -0.11, +0.24, +0.23. (b) Main battery
    accuracy → expansion accuracy: r = +0.20, -0.09, +0.49. No simple
    predictive law governs axis performance.

16. **Content split confirms warmth-bias mechanism on all 4 models.**
    The battery's content split (50 firmness-biased orig cases vs 20 warmth
    cases) is independent of any model's geometry. Good shows 12-69 point
    gaps favoring warmth cases on all 4 models (BGE-M3: 16%→85%, Nomic:
    12%→80%, Gemini: 26%→95%). Careful shows small, direction-inconsistent
    gaps (-23pt to +8pt) — not systematically tracking either pole.
    On Gemini, careful→good score-delta r=+0.24 (weakest of all children),
    and Gemini's accuracy advantage is concentrated in independent terms
    (careful 74%, thorough 61%); warmth-biased children stay near chance.
    A secondary kind_delta split produces more extreme numbers (Nomic good
    14%→57%, careful 64%→64%) but is not independent of §4.18 correlations.

17. **Three prediction tests on 26 novel terms fail to predict warmth-independence.**
    Test 1 (10 terms): 8/10, matching null "all biased" (also 8/10).
    Test 2 (10 terms, 6 predicted independent): 5/10, +1 above null.
    Test 3 (6 terms, restraint hypothesis): 5/6, +2 above null, but only
    2/3 independence predictions correct (at chance with n=3).
    Pooled across all tests: 5/13 independence predictions correct (38%)
    vs ~20% base rate — not a usable predictive criterion.

18. **Comprehensive analysis of 45+ terms reveals a robust asymmetry.**
    ~80% of evaluative terms share good's warmth bias (r > 0.4 on 2+ models).
    All 9 independent terms (careful, thorough, deliberate, cautious,
    methodical, patient, prudent, measured, rigorous) share restraint/
    discipline semantics — a descriptive observation, not a predictive rule.
    Independent terms average 47-51% accuracy (near chance); biased terms
    average 35-41% (below chance, because tracking warmth hurts on this
    battery). Escaping warmth bias removes the anti-signal; only "careful"
    clears chance meaningfully, and only on some models.

19. **Training-signal quality is split-dependent, not universally anti-correlated.**
    An absolute-score analysis shows "good"'s d sign-flips by case type: positive
    on warmth cases (d = +0.74 on BGE-M3, where warmth aligns with correctness),
    strongly negative on firmness cases (d = -0.73, where they conflict), and
    catastrophically negative on sycophancy (d = -4.02). The pooled negative d
    reflects battery composition (64% firmness-biased), not a universal property.
    Whether the net training signal is positive or negative depends on the base
    rate of warmth/correctness conflicts in real training data. No single axis
    provides a universal training signal.

20. **"Restrained" shows model-specific strength on BGE-M3.**
    In-sample: 64% (BGE-M3), 54% (Snowflake), 61% (Nomic).
    OOS expansion battery: 75% (BGE-M3), 55% (Snowflake), 45% (Nomic).
    Strongly anti-correlated with good (r = -0.57 on BGE-M3). But drops to
    near-chance on other models OOS. Not a universal alternative to careful.

21. **The default scoring recipe is fixed.** Confirmatory cross-model,
    unknown-model, and paper-headline results use standard
    `User:/Assistant:` framing and bipolar projection:
    `normalize(mean(pos) - mean(neg))`, then dot product with the response
    embedding. Cosine-to-positive is allowed only as an explicitly declared
    BGE-M3 diagnostic or pre-registered ablation, reported separately. Do
    not choose per-axis or per-model scoring methods post hoc.

22. **Compound anchor strings are not allowed as default anchors.** The
    2026-06-28 compound-anchor audit tested cross-concept strings like
    "careful and kind" against independent component scoring. Most
    compounds diluted signal on >=2 models, and no compound beat its best
    single component on >=2 models under the cosine diagnostic. This is
    the anchor-text version of the no-averaging rule: score concepts as
    separate axes, then combine scores/votes after scoring.

---

## A. Anchor vocabulary optimization

These are questions about WHAT WORDS to use as axis anchors.

### A1. Optimal word length / complexity

**Question**: Are single words, short phrases, or full sentences the best
anchors? We know multi-sentence ML-jargon axes are unstable across models
and single words partially work. What about 2-3 word phrases?

**What we know**: Single words like "Careful" beat multi-sentence ML-jargon
on cross-model stability. But we haven't tested 2-3 word phrases
systematically (e.g., "avoids mistakes" vs "Careful").

**Experiment**: Test anchors at multiple lengths: 1 word, 2-3 word phrase,
1 sentence, 2-3 sentences. Same evaluative concept at each length. Measure
accuracy AND cross-model consistency.

**Status**: DONE (2026-06-28). **Result**: Single words are optimal. Tested
18 anchor pairs across 5 concepts at multiple lengths on all 3 models. All
"good" variants remain warmth-biased regardless of length (BGE-M3: 1-word
18%/85% firm/warm; sentence 13%/85%). Longer anchors degrade clean axes
(Nomic careful: 64%→41%). One model-specific exception: explicit anti-syc
sentence scores 71% on BGE-M3 (84% firm, 100% syc) but 25% warm — a
firmness axis, not balanced. No phrase escapes the warmth/firmness tradeoff.

---

### A2. Tree decomposition of "good"

**Status**: DONE (2026-06-28).

**Design**: Tested 26 terms across 3 tree levels (root, 10 L1 children,
15 L2 grandchildren) on the 70-case balanced battery, all 3 models.
Measured individual accuracy, within-branch vs cross-branch score-delta
correlations, parent-child correlations, and combination strategies.

**Results** (key findings on Nomic, best model):

1. **Best child outperforms root on every model**: On Nomic, every L1
   child beat "good" (31%); on Snowflake and BGE-M3, the best children
   (thorough 60%, kind 53%) beat "good" (51%, 36%) but not all children
   do. The cross-model-true claim: the best child beats root on every model.

2. **Going deeper HURTS**: L2 children worse than L1 parents on every
   branch. Careful (64%) > deliberate (51%), cautious (50%), precise (34%).

3. **Same-branch correlation prediction: FAILED at L1**. Same-branch
   mean r=0.24, cross-branch mean r=0.32 (OPPOSITE of prediction).
   Reason: most L1 children are warmth-biased and correlate with each
   other regardless of nominal branch. Marginally confirmed at L2
   (same=0.32, cross=0.23, difference=0.09).

4. **"Careful" is UNCORRELATED with "good"** (score-delta r=-0.11, p=0.36,
   n.s.). It accesses an independent evaluative dimension. Most other L1
   children correlate strongly WITH "good" (honest r=0.71, wise r=0.73,
   helpful r=0.76, responsible r=0.76 — all 6-8 SE from zero). Only
   "careful" (r=-0.11) and "thorough" (r=-0.12) are independent.

5. **Combination FAILS**: L1 majority vote (34%) is worse than best
   individual child (64%). Reason: 8 of 10 L1 children are warmth-biased
   and outvote the 2 independent ones.

**Interpretation**: The tree decomposition theory is partially right:
decomposition helps at L1 because children are more evaluatively specific.
But "careful" works not because it's a narrower sub-sense of "good" — it
works because it's geometrically INDEPENDENT of "good" (uncorrelated in
score-delta space, near-orthogonal in axis-vector space), capturing the
effort/rigor dimension that "good" does not encode. Most evaluative terms
in natural language share "good's" warmth bias; "careful" is rare in
accessing firmness independently.

**Cross-model verification (2026-06-28)**: Correlation analysis replicated
on all 3 models. "Careful" is independent of "good" on every model:
Snowflake r=+0.09 (n.s.), BGE-M3 r=-0.25 (borderline), Nomic r=-0.11
(n.s.). Meanwhile warmth-biased children show strong, significant
correlations with "good" on all 3 models (e.g., helpful: 0.30/0.92/0.76;
honest: 0.55/0.79/0.71). The geometric independence is model-invariant.

---

### A3. Evaluative specificity vs frequency

**Question**: What predicts which words make good anchors? The theory is
that the optimal word is (a) common enough to have a strong embedding, AND
(b) specific enough that it almost always appears in evaluative contexts.
"Good" fails (b) because "good morning," "good faith," etc. "Careful"
passes both tests.

**Experiment**: Measure evaluative specificity for candidate words (what
fraction of a word's occurrences are in evaluative vs non-evaluative
contexts). Correlate with axis accuracy. This could use a corpus analysis
or a proxy like the variance of the word's contextualized embeddings.

**Status**: DONE (2026-06-28). Three prediction tests (26 novel terms)
and a comprehensive 45+-term analysis. All three hypotheses tested
(distributional-context, personality-praise, restraint semantics) fail as
predictive rules. Pooled: 5/13 independence predictions correct (38%) vs
~20% base rate. Descriptive finding: all 9 independent terms share restraint
semantics, but this doesn't predict novel terms reliably. The robust result
is the asymmetry: warmth bias is predictable (~80% of terms), escaping it
is not. Full corpus-based specificity measurement still not done but the
prediction-test approach is exhausted — a fourth reformulation would not
change the picture.

---

### A4. Unusual / literary / dramatic anchors

**Question**: Could anchors from outside the obvious evaluative vocabulary
work better? Bible quotes, literary metaphors, idioms, unusual combinations?
The search space is enormous.

**Why it matters**: If the optimal anchor is something unexpected (like a
specific literary phrase), our current vocabulary search is way too narrow.

**Experiment**: Test anchors from different registers: formal/academic,
colloquial, literary, religious, technical. Compare with standard evaluative
vocabulary. This is exploratory — look for surprising outliers.

**Status**: NOT STARTED

---

### A5. Bipolar axis vs distance-to-positive

**Question**: Is good-minus-bad (bipolar axis) the best scoring method, or
would distance-to-good (single-pole) or some other formulation work better?

**Status**: DONE (2026-06-27; audited 2026-06-29). **Result**:
Model-dependent. BGE-M3 benefits from cosine-to-positive on many axes,
Snowflake sees no robust benefit, Nomic is mixed, and Gemini is HURT by
cospos (careful drops from 74% to 59%). Default recipe: bipolar with
standard `User:/Assistant:` framing. Cospos is allowed only as an
explicitly declared BGE-M3 diagnostic or pre-registered ablation; it is
not a post-hoc per-axis default.

---

### A6. Optimal term SET considering mutual coverage

**Question**: Given that no single term covers all quality dimensions, what
is the minimum set of terms that together cover "good" completely?

**Status**: DONE (2026-06-27). **Result**: NEGATIVE. Complementarity exists
in-sample (other axes rescue 12-22 of careful's failures per model) but
HURTS net. On Nomic (best model), adding ANY axis to "careful" introduces
more new failures than it rescues (net -3 to -27). Multi-axis voting
combinations overfit: searched 3-axis combos gain 10-15 points in-sample
but lose 10-25 points OOS. "Careful" alone generalizes better than any
combination on every model.

---

## B. Battery and methodology

### B1. Battery balance and composition

**Question**: What is the right mix of case types for a fair test? The
original battery was 64% firmness-biased. The 20 warmth cases partially
rebalanced it. What other quality dimensions are missing?

**Open case types to add**: nuance/context-sensitivity, factual accuracy
without emotional content, creative quality, conciseness-vs-completeness
tradeoffs, safety without sacrificing helpfulness.

**Status**: PARTIALLY DONE (20 warmth cases added, revealed the firmness
confound)

---

### B2. Discrimination vs training signal

**Question**: Does our pairwise accuracy metric (can the axis rank two
pre-written responses?) actually predict how well the axis would work as
a training signal? In training, the model generates responses and gets a
reward. It can avoid the conflicts our battery forces by being good in
all dimensions simultaneously.

**Status**: PARTIALLY RESOLVED (2026-06-28). **Result**: SPLIT-DEPENDENT.
The absolute-score analysis (§4.24) shows "good"'s d sign-flips by case
type: positive on warmth cases (BGE-M3 d = +0.74) but strongly negative
on firmness cases (d = -0.73) and catastrophically negative on sycophancy
(d = -4.02). The pooled negative d (Snowflake -0.14, BGE-M3 -0.40, Nomic
-0.30) reflects the battery's 64% firmness bias — the same composition
artifact policed in Finding #9/#10. Whether pairwise accuracy is pessimistic
or optimistic depends on the base rate of warmth/correctness conflicts in
real training data, which this battery cannot estimate. The question is
unresolved on this evidence.

---

### B3. Input length effects

**Question**: Does the length of the text being scored affect signal
quality? Short inputs might not give enough context for the embedding to
evaluate ("London is the capital of France" — is that false? only if
you have outside knowledge). Long inputs might muddy the signal.

**Experiment**: Take cases with short and long responses. Measure accuracy
as a function of response length. Also test whether providing more context
(e.g., system prompt, conversation history) helps or hurts.

**Status**: NOT STARTED

---

## C. Model understanding

### C1. Why does Gemini Embedding 2 dominate?

**Question**: Is Gemini's advantage from architecture, scale, training
data, training objective (RL for retrieval?), or something else? Is it
deterministic like standard embedding models or does it have LLM-like
variance? Google suggests giving it prompts — does that help?

**Why it matters**: If we understand WHY Gemini works, we can predict
which other models might work and what model development would help.

**What we now know**: Jina v5 (API-based, recent, LoRA-adapted for tasks)
performs like local models, NOT like Gemini. So it's not about being
API-based or recent. Jina's `classification` task mode shifts behavior
(less warmth-biased) but doesn't approach Gemini's accuracy. The
advantage is Gemini-specific.

**Experiments**:
- Test Gemini determinism: embed same text 10x, check variance
- Test Gemini with task prompts vs without
- Compare Gemini to other frontier embedders when available
- Test whether Gemini's advantage is axis-specific or universal

**Status**: PARTIALLY DONE (Jina v5 comparison complete, Gemini experiments
blocked by API quota)

---

### C2. What model properties predict evaluative signal quality?

**Question**: Across the models we've tested, what predicts which models
produce good evaluative signal? Dimensionality? Training objective?
Parameter count? Architecture?

**What we know**: Scale doesn't predict (33M vs 600M performs similarly).
Gemini is dramatically better. We don't know why.

**Experiment**: Systematic comparison across model properties. Need to
test more models in the 1B+ range. Need models with different training
objectives (contrastive vs generative vs retrieval-tuned).

**Status**: PARTIALLY DONE (8 model sweep exists, but analysis of WHY
models differ is missing)

---

### C3. Model development direction

**Question**: If we wanted to build or fine-tune an embedding model
specifically for evaluative signal quality, what would we change? This
could be the most impactful practical recommendation.

**Status**: NOT STARTED (depends on C1 and C2)

---

## D. Gemini-specific experiments (API quota gated)

These are ready to run but blocked by Gemini API daily free-tier limits.

### D1. Gemini vocabulary depth

**Status**: DONE (2026-06-27). **Result**: Single-word "Careful" scores 74%
combined on Gemini (72% orig, 80% warmth) — the best single-word axis.
"Hard" inverts to 39% on Gemini. Multi-sentence targeted axes still
outperform (86-98% on original battery), but the gap between single-word
and multi-sentence is smaller than on local models. Cospos HURTS on Gemini.

### D2. Gemini anchor perturbation

**What it tests**: How robust is Gemini's signal to rewording the anchors?
**Status**: NEEDS SCRIPT, blocked by API quota

### D3. Gemini word-stripping ablation

**What it tests**: Does removing evaluative words from responses affect
Gemini's scoring?
**Status**: NEEDS SCRIPT, blocked by API quota

---

## E. Scaling and applications

### E1. Cross-domain validation

**Question**: Does the evaluative axis work beyond safety/helpfulness?
Code quality, writing quality, summarization, reasoning?

**Status**: PARTIALLY DONE (code/math/tool suites exist for Gemini)

### E2. Training experiment (DPO)

**Question**: Does embedding-axis reward actually improve a model through
training?

**Prerequisites**: Needs GPU. Should only attempt after the signal is well
characterized.

**Status**: NOT STARTED

### E3. Pretraining data curation

**Question**: Can the embedding axis score pretraining data quality at
scale?

**Status**: NOT STARTED

---

## F. Paper revision priorities

### F1. Battery firmness confound

**Status**: DONE (2026-06-28). Revised contribution #8, §4.15, §5.6,
conclusion. Added firmness-bias caveat to §4.14 table. Added battery
composition bias to limitations.

### F2. Methodology limitation section

**Status**: DONE (2026-06-28). Added "structural failure ceiling" to
limitations (18% universal failure rate, complementarity is negative).

### F3. Battery rebalancing results

**Status**: DONE (2026-06-28). Added §4.16 (rebalancing, pooled analysis
with Wilson CIs, voting overfits OOS, per-model method variation, null
prediction results) and §4.17 (Gemini single-word validation).

### F4. Tree decomposition theory

Write up the semantic tree theory as a framework for future work.
This is currently only in conversation, not in the paper.
**Status**: NOT STARTED

---

## Priority ordering

### CENTROID PRIORITIES (publication path)

1. **C1. External validation** [DONE 2026-06-29 — NEGATIVE] — Cross-dataset transfer zero, method validates within-dataset
2. **C2. Gameability probe** [DONE 2026-06-29 — NOT GAMEABLE] — 1.9% flip rate, formalization zero effect
3. **C3. Vocabulary projection** [DONE 2026-06-29 — CLAIM HOLDS] — No word > 0.30, quality unnameable
4. **C4. Margin-length correlation** [DONE 2026-06-29 — NOT A CONFOUND] — Inconsistent across models
5. **C5. PCA dimensionality** [DONE 2026-06-29] — 51 PCs for 80%, centroid >> PCA
6. **C6. Additional models** [DONE 2026-06-29 — MIXED] — 66-72% OOS, all p>0.33, none significant
7. **C7. Stability at scale** [BLOCKED — C1 showed cross-dataset transfer is zero, so external datasets provide no useful stability test for OUR direction]
8. **C8. Score diversity** [NOT STARTED] — Flavor tolerance
9. **C9. Cross-author validation** [PARTIALLY STARTED 2026-06-29] — 9/70 pairs via Gemini Flash. Accumulate over days
10. **C10. Meta-validity battery** [DONE 2026-06-29] — Prompt leakage, label flip, LOO, edge cases, influence

### Word-based priorities (diagnostic/supporting)

**DONE** (resolved in prior sessions):
- F1-F3: Paper revisions, A5: Bipolar vs cospos, A6: Optimal term set
- D1: Gemini vocabulary depth, A2: Tree decomposition, A3: Evaluative specificity
- Scoring recipe audit, compound anchor audit

**Remaining** (low priority, diagnostic):
- B3: Input length effects
- A4: Unusual/literary anchors
- D2-D3: Gemini ablations (API quota gated)

### Long term (needs resources we don't have)
- E2: DPO training experiment (needs GPU)
- E3: Pretraining data curation (needs scale)
- C3: Model development recommendations (needs C1-C2)
