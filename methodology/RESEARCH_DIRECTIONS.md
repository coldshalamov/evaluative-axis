# Open Research Directions

Last updated: 2026-06-28

This document captures every open question identified so far. New sessions
should read this first instead of re-deriving the research agenda. When a
question is resolved by an experiment, mark it DONE with a date and a
one-line result summary. When a question turns out to be the wrong question,
mark it RETIRED with the reason.

---

## What we know (established results)

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

3. **"Careful/Reckless" works on firmness cases (62-80%) but fails on
   warmth cases (35-70%).** It captures one branch of the "good" tree,
   not the whole tree.

4. **No single word tested survives both firmness and warmth splits across
   all three local models.** The combined 70-case battery (50 firmness +
   20 warmth) has no axis scoring >50% on both splits on all models.

5. **Compositing multiple terms into one averaged axis degrades performance.**
   Scoring terms independently and summing (or voting) preserves signal.

6. **Gemini Embedding 2 dramatically outperforms local models** on targeted
   axes (86-98% vs 50-74%). Cause unknown. Gap does not close with scale
   in the 33M-600M range. Jina v5 (API-based, 2025-2026) performs like
   local models, confirming the advantage is Gemini-specific, not a
   property of being API-based or recent.

7. **Corpus frequency does not predict signal strength.** "Good" is the most
   frequent evaluative word and the worst performer. "Careful" is far less
   frequent and the best single-word performer on firmness cases.

8. **Different axes point in genuinely different geometric directions**
   (cosine 0.01-0.20 between axis vectors). They capture different
   evaluative dimensions independently.

9. **The original 50-case battery is 64% firmness-biased** (32/50 cases
   reward pushback/firmness). This confounds all results measured only
   on that battery.

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

**Status**: NOT STARTED

---

### A2. Tree decomposition of "good"

**Question**: Is it better to use child terms of "good" (careful, honest,
kind, wise, etc.) scored independently than to use "good" directly?
At what tree depth is the signal strongest?

**Theory** (from user): "Good" has ~50,000 senses. Any response only
satisfies a fraction. "Careful" has ~12 senses. A careful response satisfies
most of them. So "careful" gives a cleaner projection because the response
matches the full scope of the word.

**Predictions**:
- Child terms from the SAME branch should correlate in their scores
- Child terms from DIFFERENT branches should be uncorrelated
- Going deeper (careful -> deliberate, attentive) gives more specificity
  but narrower coverage
- There's an optimal depth level

**Experiment**: Build a tree: good -> {careful, honest, kind, wise, helpful,
thorough, fair, ...} -> {deliberate, attentive, precise, ...}. Test each
level independently. Check whether child-term combinations outperform
parents. Check whether same-branch children correlate and cross-branch
children don't.

**Status**: NOT STARTED

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

**Status**: NOT STARTED

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

**What we know**: We've used normalize(embed(positive) - embed(negative))
throughout. Haven't tested alternatives.

**Experiment**: Compare: (a) bipolar axis (current), (b) cosine to positive
anchor only, (c) cosine to negative anchor only (inverted), (d) difference
of cosines. Same battery, same anchors, multiple methods.

**Status**: NOT STARTED

---

### A6. Optimal term SET considering mutual coverage

**Question**: Given that no single term covers all quality dimensions, what
is the minimum set of terms that together cover "good" completely? The terms
need to (a) work individually, (b) cover each other's failure modes, and
(c) not introduce unmonitored failure modes of their own.

**What we know**: "Careful" covers firmness, "Good" covers warmth, but
combining them by averaging kills both signals. Independent scoring +
summing partially works. We haven't systematically searched for the optimal
SET considering each term's failure modes.

**Experiment**: For each candidate term, measure not just accuracy but
PER-CASE correctness. Find terms whose correct cases are COMPLEMENTARY
(term A gets right what term B gets wrong). Build the minimum covering set.
Test on balanced battery.

**Status**: PARTIALLY STARTED (optimal basis search ran on firmness-only
battery; needs rerun on balanced battery)

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

**Why it matters**: If discrimination and training performance diverge,
our battery results could systematically undervalue broad terms like "good"
that reward all dimensions but can't resolve conflicts between them.

**Experiment**: This is hard to test without actually training a model.
Possible proxy: generate N responses per prompt, score with the axis,
and check whether the top-scored responses are genuinely good (by human
or LLM judge). If the top-scored responses are good despite the axis
scoring poorly on pairwise discrimination, the axis is better as a
training signal than as a discriminator.

**Status**: NOT STARTED

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

**Script**: `scripts/run_vocab_depth_gemini.py`
**What it tests**: Do single-word anchors work on Gemini like they do on
local models? Or does Gemini's advantage come specifically from multi-
sentence targeted axes?
**Status**: SCRIPT READY, blocked by API quota (HTTP 429)

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

The paper currently presents "Hard/Soft" and several other results as
positive findings. These need revision:
- Section 4.15 (Osgood dimensions): reframe with battery artifact caveat
- Section 5.6 (Osgood interpretation): revise
- Contribution 8: revise
- Conclusion Osgood paragraph: revise

### F2. Methodology limitation section

Add a section acknowledging that pairwise discrimination on a conflict
battery may not predict training signal quality. Reference the "good"
inversion (85% warmth, 16% firmness) as evidence that the battery
methodology has limitations.

### F3. Battery rebalancing results

Write up the rebalancing experiment: what it found, why it matters,
what it means for all prior single-word results.

### F4. Tree decomposition theory

Write up the semantic tree theory as a framework for future work.
This is currently only in conversation, not in the paper.

---

## Priority ordering

**Immediate** (resolve before more theory):
1. F1-F4: Paper revisions to be honest about what we know
2. A2: Tree decomposition experiment (tests the core theory)
3. A6: Optimal term set on balanced battery
4. B1: Expand battery with more case types

**When Gemini quota resets**:
5. D1-D3: Gemini experiments

**Medium term**:
6. A1: Word length optimization
7. A3: Evaluative specificity measurement
8. A5: Bipolar vs single-pole scoring
9. B2: Discrimination vs training signal proxy test
10. B3: Input length effects
11. C1-C2: Model understanding

**Long term (needs resources or prerequisites)**:
12. E2: DPO training experiment
13. E3: Pretraining curation
14. C3: Model development recommendations
15. A4: Unusual/literary anchors (huge search space)
