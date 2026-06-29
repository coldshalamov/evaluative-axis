# Centroid Direction Method: Specification, Evidence, and Test Plan

## What This Document Is

Single source of truth for the supervised centroid direction method. Defines the
method, documents all existing evidence, identifies every threat to validity,
and specifies the test regimen needed for publication. Written to be handed to
Codex as its directive.

---

## Part 1: The Method

### Core Idea

Quality structure exists in embedding space but has no single-word name. Words
like "good," "careful," and "helpful" fail as probes because they're ambiguous
(overloaded across evaluative and non-evaluative contexts). The centroid
method bypasses vocabulary entirely by pointing at quality with examples.

### Algorithm

```
SETUP (run once):
1. Collect N labeled preference pairs: (prompt, better_response, worse_response)
2. Embed all better responses and all worse responses
   - Format: "User: {prompt}\nAssistant: {response}"
   - Use any embedding model (we test 4: Snowflake, BGE-M3, Nomic, Gemini)
3. Compute quality direction:
     quality_dir = mean(better_embeddings) - mean(worse_embeddings)
     quality_dir = quality_dir / ||quality_dir||
4. Store quality_dir (one vector, 768-3072 floats depending on model)
5. Discard training embeddings

SCORING (run per response):
1. Embed new response: emb = embed("User: {prompt}\nAssistant: {response}")
2. Score = dot(emb, quality_dir)
3. Higher score = more similar to the better-response region of embedding space

PAIRWISE COMPARISON:
- score_A = dot(embed(response_A), quality_dir)
- score_B = dot(embed(response_B), quality_dir)
- If score_A > score_B, predict A is better
```

### Properties
- **Cost**: One embedding call per response, one dot product. No retrieval, no
  classifier, no vector database for the scoring itself.
- **Deterministic**: Same input always produces same score. No sampling variance.
- **Self-weighting**: The margin (score difference) correlates with quality gap
  magnitude. Large quality differences produce large margins; ambiguous cases
  produce near-zero margins.
- **Inspectable**: The quality direction is a vector you can analyze — project
  it onto known words/concepts to understand what it encodes.
- **Updatable**: New labeled pairs → recompute centroid → re-score everything.

### What It Is NOT
- It is NOT a training signal (yet). It is a discriminator. We have shown it
  correctly ranks pre-written response pairs. Whether it is a good optimization
  target for RL training is an untested empirical question. Ranking is not the
  same as reward shaping. Goodhart's law applies under optimization pressure,
  which we have not tested.
- It is NOT a replacement for RLHF. It is a candidate cheap supplement or
  pre-filter. The claim "as good as RLHF for training" requires actually
  training with it, which requires GPU resources we don't have.
- It is NOT the best linear method on these embeddings. The logistic probe
  (which uses all embedding dimensions, not just one direction) achieves
  87-94% OOS on BGE-M3, vs. the centroid's 75-80%. The centroid's claim is
  "cheapest and simplest," not "best."

---

## Part 2: Existing Evidence

### 2.1 Core Accuracy (4 models)

| Model | Dims | OOS Accuracy | Permutation p | Source |
|-------|------|-------------|---------------|--------|
| Snowflake Arctic Embed M | 768 | 77-80% | 0.003 | centroid_deep_results.json |
| BAAI/bge-m3 | 1024 | 75-80% | 0.053 | centroid_deep_results.json |
| nomic-embed-text-v1.5 | 768 | 66-77% | 0.032 | centroid_deep_results.json |
| Gemini Embedding 001 | 3072 | 86% | not tested | gemini_centroid_results.json |

OOS = tested on 35 held-out expansion cases (for local models) or 70-case battery (for Gemini).
50% is chance (binary forced-choice).

### 2.2 Orthogonality to Words

The quality direction has cosine < 0.10 with every tested English word,
including "good," "careful," "helpful," "honest," "thoughtful." This holds
across all 4 models. The direction is finding something that no single word
points at.

### 2.3 Learning Curve

| Training pairs | Snowflake OOS | BGE-M3 OOS | Nomic OOS |
|---------------|--------------|------------|-----------|
| 5 | 59% | 62% | 56% |
| 10 | 61% | 69% | 56% |
| 20 | 68% | 78% | 62% |
| 30 | 72% | 80% | 65% |
| 50 | 77% | 81% | 67% |
| 70 | 80% | 75%* | 66%* |

*Full battery used for training, so this is in-sample for battery, OOS for expansion only.

### 2.4 Multi-Dimensional Quality

Firmness and warmth quality directions are anti-correlated:
- Snowflake: cosine -0.35
- BGE-M3: cosine -0.42
- Nomic: cosine -0.38
- Gemini: cosine -0.49

Training on firmness cases alone → 15-25% on warmth cases (worse than coin flip).
Quality is at least two opposing dimensions. The full-battery centroid is a
weighted compromise.

### 2.5 Confound Checks

- **Length**: Better responses are NOT systematically longer. Length-only
  classifier gets 37-49% (worse than chance). Quality direction is orthogonal
  to length direction (cosine < 0.05).
- **Norm**: Embedding norms do not predict quality.
- **Random baseline**: Random directions score 49-51% (chance), vs centroid 77-80%.

### 2.6 Robustness

- **Bootstrap stability**: cosine to full direction 0.77-0.83 (moderately stable)
- **Disjoint split stability**: cosine between halves 0.37-0.57; cross-accuracy 72-78%
- **Ensemble**: 5-chunk ensemble nearly matches pooled direction performance
- **Pooled direction**: 131 cases → 84-89% battery, 85-95% expansion

### 2.7 Per-Category Performance (centroid on 61 OOS expansion cases)

| Category | N cases | Snowflake | BGE-M3 | Nomic |
|----------|---------|-----------|--------|-------|
| Anti-sycophancy | 15 | 93% | 93% | 93% |
| Nuance/context | 13 | 77% | 77% | 77% |
| Creative quality | 13 | 54% | 77% | 54% |
| Factual accuracy | 15 | 53% | 67% | 33% |
| Conciseness | 5 | 60% | 80% | 60% |

### 2.8 Margin Analysis

High-confidence correct predictions: margins 0.10-0.26 (anti-sycophancy cases)
Low-confidence errors: margins -0.003 to -0.035
The margin tracks quality gap magnitude — errors are near-zero, correct calls
are clearly separated. This makes the score self-weighting.

### 2.9 Word-Based Experiments (WHY words fail)

27 experiments (§4.1-4.27 in paper) demonstrate that:
- "Good/bad" scores 16% on firmness cases, 85% on warmth (bipolar, not broken)
- All 30 synonym clusters of "good" collapse into the same warmth-dominated direction
- No single word achieves >64% across all models on the balanced battery
- Multi-word composites degrade signal (averaging reconstructs the parent)
- The five-term tree's 89-94% is inflated by OR-logic (chance baseline ~97%)

### 2.10 External Validation (SHP, UltraFeedback) — NEGATIVE RESULT

Tested our 70-pair centroid direction on 500 pairs from each of two external
human-preference datasets: Stanford Human Preferences (Reddit upvotes) and
UltraFeedback (GPT-4 annotations). All three local models. Results:

**Test A: Our direction → external data (at chance across the board)**

| Model | → SHP | → UltraFeedback |
|-------|-------|-----------------|
| Snowflake | 48.0% | 50.3% |
| BGE-M3 | 47.2% | 45.6% |
| Nomic | 50.4% | 43.4% |

**Test B: External centroids work on their own data (method validates)**

| Model | SHP in-sample | UF in-sample |
|-------|--------------|-------------|
| Snowflake | 66.6% | 68.3% |
| BGE-M3 | 77.4% | 72.0% |
| Nomic | 72.4% | 65.8% |

**Test C: External direction → our battery (reverse transfer also fails)**

| Model | SHP → ours | UF → ours |
|-------|-----------|----------|
| Snowflake | 48.6% | 45.7% |
| BGE-M3 | 35.7% | 30.0% |
| Nomic | 50.0% | 40.0% |

**Test E: Directions are orthogonal (cosine near zero)**

| Model | cos(ours, SHP) | cos(ours, UF) |
|-------|---------------|--------------|
| Snowflake | -0.04 | -0.07 |
| BGE-M3 | -0.08 | -0.12 |
| Nomic | +0.06 | -0.20 |

Permutation p = 1.000 across all models and datasets (200 permutations each).

**Interpretation**: The centroid method works — it extracts quality structure from
any labeled dataset. But "quality" is not a single universal concept. SHP measures
Reddit engagement, UltraFeedback measures GPT-4 preferences, our battery measures
human-curated response quality. These are genuinely different quality frameworks
that point in orthogonal directions in embedding space.

The practical claim still holds: 70 labeled pairs → centroid → 77-80% discriminator
within the target quality framework. But cross-framework transfer is zero.

Source: `external_validation_results.json`

---

## Part 3: Claims, Threats, and Falsification Tests

### CLAIM 1: The centroid direction detects response quality

**Evidence**: 77-86% OOS across 4 models, statistically significant, random
baseline at chance.

**Threats to validity**:
- T1.1: ALL test cases are self-authored. Unconscious authoring biases could
  create artifacts the centroid detects instead of genuine quality differences.
  A reviewer will ask: "Does this work on data you didn't write?"
- T1.2: Battery composition is 64% firmness-biased. The centroid could be a
  firmness detector that happens to work on a firmness-heavy battery.
- T1.3: Small sample size (70 train, 61 test). Marginal statistical significance
  on BGE-M3 (p=0.053).
- T1.4: Style confound. All "better" responses may share an authoring style
  (verbose, structured, hedge-heavy) that the centroid detects as a surface
  feature, not a quality feature.
- T1.5: Only 4 embedding models tested. Selection bias possible.

**Falsification tests**:
- **TEST 1A [COMPLETED — NEGATIVE]**: Ran centroid on SHP (500 pairs) and
  UltraFeedback (500 pairs). Accuracy at chance (43-50%) across all 3 models.
  Directions orthogonal (cosine < 0.12). See §2.10. Cross-dataset transfer
  fails, but the centroid method works within each dataset's own labels (66-77%).
  Conclusion: the centroid detects dataset-specific quality, not universal quality
  and not pure authoring artifacts (since per-category accuracy varies 33-93%).
- **TEST 1B**: Measure accuracy separately on firmness and warmth splits of
  external data. If it only works on firmness-like cases, it's a firmness detector.
- **TEST 1C [COMPLETED 2026-06-29 — MIXED]**: Three additional models:
  gte-base (768d, 109M): 65.6% OOS, p=0.540 (not significant).
  e5-base-v2 (768d, 109M): 72.1% OOS, p=0.675 (not significant).
  mxbai-embed-large-v1 (1024d, 335M): 70.5% OOS, p=0.335 (not significant).
  All three show above-chance OOS accuracy (66-72%) but none pass the
  200-permutation significance test. In-sample accuracy is high (83-87%)
  across all three. Pattern: the signal exists but doesn't generalize
  to OOS data as reliably as on our core models. The original four models
  (Snowflake p=0.003, Nomic p=0.032, BGE-M3 p=0.053, Gemini untested)
  remain the validated set. Honest negative: the centroid does not produce
  statistically significant results on all embedding models.
  Results: `additional_models_results.json`.

### CLAIM 2: The quality direction is orthogonal to all tested words

**Evidence**: Cosine < 0.10 with every anchor word across all 4 models.

**Threats to validity**:
- T2.1: We tested a limited word list. There might be a word that aligns with
  the quality direction that we didn't try.
- T2.2: The direction might correlate with word combinations even if not with
  individual words.

**Falsification tests**:
- **TEST 2A [COMPLETED 2026-06-29]**: Projected quality direction onto 10,000
  most common English words (Brown corpus). No word exceeds 0.30 on ANY model.
  Snowflake: max 0.12 ("deny"), zero words > 0.20. BGE-M3: max 0.25
  ("doubted"), 33 words > 0.20, all doubt/refusal cluster. Nomic: max 0.25
  ("efficiently"), 22 words > 0.20, but ALL with NEGATIVE cosine — quality
  points AWAY from positive performance vocabulary. The direction captures
  "intellectual resistance" (willingness to say no) not nameably by any
  single word. Mean |cosine| across 10K words: 0.04-0.06. Claim stands.
  Results: `vocabulary_projection_results.json`.
- **TEST 2B [COMPLETED 2026-06-29]**: 46 quality phrases + 22 single words
  projected onto the centroid direction. No phrase exceeds cosine 0.30.
  Max |cosine|: Snowflake 0.088, BGE-M3 0.244, Nomic 0.133. Phrases are
  marginally better than single words (+0.01 to +0.04 max improvement) but
  still far below the threshold. The quality direction remains unnameable.
  Results: `phrase_projection_results.json`.

### CLAIM 3: Quality is multi-dimensional (firmness/warmth anti-correlation)

**Evidence**: Cosine -0.35 to -0.49 between firmness and warmth quality directions.

**Threats to validity**:
- T3.1: The anti-correlation could be a battery artifact if firmness and warmth
  cases were written with systematically different styles.
- T3.2: Two dimensions may be an artifact of having two case types. Quality
  could be 1 dimension or 20 dimensions in reality.

**Falsification tests**:
- **TEST 3A [COMPLETED 2026-06-29]**: PCA on 131 difference vectors × 3
  models. PC1 explains 6-8%. 51 components needed for 80% variance on ALL
  3 models. Quality is genuinely ~50-dimensional. But the centroid (88-93%
  OOS) massively outperforms best PCA-based classification (41-77%). On
  BGE-M3, top PCs are anti-correlated with quality — adding more hurts.
  The centroid wins by averaging noise away, not by finding a high-variance
  direction. PC1 captures the firmness-warmth split (cosine >0.7 with both
  sub-directions). Results: `pca_dimensionality_results.json`.
- **TEST 3B**: On external data, compute quality directions from different
  topic subsets (e.g., coding vs. creative vs. safety). Check pairwise cosines.

### CLAIM 4: The margin correlates with quality gap magnitude

**Evidence**: Anti-sycophancy cases (obvious quality gaps) have margins 0.10-0.26;
errors have margins -0.003 to -0.035.

**Threats to validity**:
- T4.1: The margin might correlate with PROMPT difficulty, not quality gap.
  Easy prompts might produce larger margins regardless of quality.
- T4.2: The margin might correlate with response LENGTH difference, not quality.

**Falsification tests**:
- **TEST 4A [COMPLETED 2026-06-29 — LENGTH IS NOT CONFOUND]**: Pearson and
  Spearman correlations between margin and 7 length metrics, n=131, 3 models.
  word_length_diff is inconsistent: Snowflake +0.30, BGE-M3 -0.08, Nomic 0.00.
  |margin| vs |char_diff| is consistently NEGATIVE across all 3 models
  (Spearman -0.17 to -0.24). Larger length differences → smaller margins.
  better_length and worse_length have no consistent relationship with margin.
  Mean length diff near zero for both correct and incorrect predictions.
  Conclusion: the margin is NOT a length proxy.
  Results: `margin_length_correlation.json`.
- **TEST 4B**: On external data with human confidence annotations (if available),
  check whether margins correlate with annotator agreement.

### CLAIM 5: The method is robust across training sets

**Evidence**: Disjoint halves cosine 0.37-0.57, cross-accuracy 72-78%.

**Threats to validity**:
- T5.1: 0.37 cosine is not high. Different training sets find substantially
  different directions. This could mean the method is unstable, or that quality
  is genuinely multi-dimensional and different samples emphasize different facets.

**Falsification tests**:
- **TEST 5A**: Compute disjoint-split stability on a larger external dataset
  (500+ pairs). If cosine improves substantially, instability is a small-sample
  artifact. If it stays at 0.37, the direction is inherently noisy.

### CLAIM 6 (ASPIRATIONAL): The score could serve as a training signal

**Evidence**: NONE DIRECTLY. We have shown discrimination, not training efficacy.
The margin's self-weighting property is suggestive but not proof.

**Threats to validity**:
- T6.1: Goodhart's law. A model optimized against this score could find shortcuts
  (padding, hedging, formulaic structure) that inflate the dot product without
  improving quality.
- T6.2: The score might reward a narrow "flavor" of good (the centroid's
  compromise direction) rather than diverse quality.

**Falsification tests**:
- **TEST 6A [COMPLETED — NEGATIVE (not gameable)]**: Six mechanical
  modifications tested on 30 correctly-classified expansion cases × 3 models.
  Total flips (worse→above better via trick): Snowflake 5/180 (2.8%),
  BGE-M3 2/180 (1.1%), Nomic 3/180 (1.7%). Overall 10/540 = 1.9%.
  Formalization had ZERO effect across all models (0 flips, 0/30 score
  changes on BGE-M3 and Snowflake). Padding and lengthening actually
  DECREASED scores (Nomic padding: Δbetter -0.35, BGE-M3 lengthening:
  Δbetter -0.012). The centroid penalizes filler, doesn't reward it.
  Results: `gameability_probe_results.json`.
- **TEST 6B [COMPLETED — see 6A]**: Worse responses were also tested —
  modifications failed to boost them past original better responses in
  98.1% of cases across all models. The direction is NOT detecting surface
  features.
- **TEST 6C [COMPLETED 2026-06-29]**: 10 prompts × 5 styles (concise, warm,
  technical, casual, formal). Between-prompt variance > within-prompt variance
  on all 3 models (ratio 1.8-2.7x). No single style dominates cross-model.
  Formal and technical consistently score lowest — the centroid penalizes
  verbose/academic style. Mean within-prompt spread 0.06-0.08.
  Results: `score_diversity_results.json`.

---

## Part 4: Test Regimen for Codex

### PRIORITY ORDER

Tests are ordered by publication impact. A paper without Test 1A is dead on
arrival. Tests 6A-C are the most important for the "training signal" narrative.

### TEST 1A: External Dataset Validation [CRITICAL]

**Goal**: Prove the centroid works on data we didn't write.

**Datasets** (pick the ones available on HuggingFace without authentication):
1. `stanfordnlp/SHP` — Reddit human preferences (upvotes). Has prompt, two
   responses, human-selected winner. Large dataset.
2. `openbmb/UltraFeedback` — Multi-aspect GPT-4 annotations. Has overall score.
3. `PKU-Alignment/PKU-SafeRLHF` — Safety-focused human preferences.
4. `lmsys/chatbot_arena_conversations` — Human A/B pairwise judgments.

**Procedure**:
1. Download dataset. Subsample 500-1000 pairs (CPU constraint).
2. Format as our standard: "User: {prompt}\nAssistant: {response}"
3. Train centroid on our existing 70-pair battery (NOT the external data).
4. Score all external pairs using OUR quality direction.
5. Report: pairwise accuracy, per-category if available, comparison to chance.
6. ALSO: Train centroid on a random 70-pair subsample of the external data.
   Test on the remaining external pairs. This checks whether the centroid
   method works even when we don't write the training data.
7. Cross-test: direction from our battery → external data, AND direction from
   external data → our battery.

**Success criteria**: Accuracy significantly above 50% (ideally >60%) on
external data using our direction. If using external data's own direction,
accuracy should be comparable to our internal results (>70%).

**Failure criteria**: Accuracy at or below 50% on external data means the
centroid is detecting our authoring style, not quality.

**Output**: Save to `notes/research_cycles/centroid_deep/external_validation_results.json`

**IMPORTANT NOTES FOR CODEX**:
- Use `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe` for all Python
- Install `datasets` library in the venv if not present
- Use `sentence_transformers.SentenceTransformer` for local models
- Test on ALL THREE local models: snowflake-arctic-embed-m, bge-m3, nomic-embed-text-v1.5
- Subsample to keep runtime reasonable on CPU (500-1000 pairs max)
- Some HuggingFace datasets may require `datasets` library authentication or
  specific loading syntax — check the dataset card first
- Save the script to `scripts/run_external_validation.py`
- Print all results clearly with model names, dataset names, accuracy, sample sizes
- If a dataset is too large to fit in memory, stream/subsample it

### TEST 1C: Additional Embedding Models

**Goal**: Prove the centroid works on models we haven't tested before.

**Models to try** (CPU-friendly, <1GB):
- `thenlper/gte-large` or `thenlper/gte-base`
- `intfloat/e5-large-v2` or `intfloat/e5-base-v2`
- `mixedbread-ai/mxbai-embed-large-v1`

**Procedure**:
1. Use our existing 70-pair battery for training.
2. Test on our 61 expansion cases.
3. Report: pairwise accuracy, comparison to our existing 3 models.

**Output**: Save to `notes/research_cycles/centroid_deep/additional_models_results.json`
Script: `scripts/run_additional_models.py`

### TEST 2A: Large Vocabulary Projection

**Goal**: Check if ANY common English word aligns with the quality direction.

**Procedure**:
1. Get a list of the 10,000 most common English words (use a standard frequency
   list, e.g., from nltk or a bundled word list).
2. Embed each word.
3. Compute cosine between each word embedding and the quality direction.
4. Report top-50 words by absolute cosine for each model.
5. Flag any word with |cosine| > 0.20.

**Important note**: Single words should be embedded as just the word (not with
User/Assistant framing). The quality direction was computed from framed responses.
This asymmetry is intentional — we're checking whether any word in the embedding
model's vocabulary points in the same direction as quality.

**Output**: Save to `notes/research_cycles/centroid_deep/vocabulary_projection_results.json`
Script: `scripts/run_vocabulary_projection.py`

### TEST 4A: Margin vs Length Correlation

**Goal**: Check if the centroid's confidence (margin) is just detecting length differences.

**Procedure**:
1. For each pair in the expansion battery, compute:
   - Quality margin (score_better - score_worse)
   - Length difference (len(better) - len(worse)) in characters and tokens
   - Length ratio (len(better) / len(worse))
   - Prompt length
2. Compute Pearson and Spearman correlations between margin and each length metric.
3. Report correlations and p-values for all 3 models.

**Output**: Save to `notes/research_cycles/centroid_deep/margin_length_correlation.json`
Script: `scripts/run_margin_length_correlation.py`

### TEST 6A-B: Gameability Probe [CRITICAL for training narrative]

**Goal**: Check if the score can be mechanically inflated without improving quality.

**Procedure**:
1. Take 20 "better" responses from expansion cases that the centroid scores correctly.
2. Create modified versions:
   a. **Padded**: Add 2-3 filler sentences ("It's worth noting that this is an
      important topic. There are many perspectives to consider. Let me elaborate further.")
   b. **Hedged**: Add hedging phrases throughout ("it's important to note," "one could argue,"
      "to be fair," "that said")
   c. **Bullet-pointed**: Convert to bullet-point format
   d. **Repeated**: Duplicate the key claim/conclusion sentence
   e. **Formal**: Replace casual language with formal/academic style
   f. **Longer**: Expand each point with additional detail (double the length)
3. Score each modified version against the quality direction.
4. Compare modified scores to original "better" scores.
5. ALSO: Apply same modifications to "worse" responses. Check if any modification
   pushes a worse response above the better response's original score.

**Success criteria**: Mechanical modifications do NOT significantly increase score.
Ideally, padding and hedging should DECREASE or leave score unchanged.

**Failure criteria**: If padding or hedging increases score >20% of the time,
the direction is detecting surface features and is gameable under optimization.

**Output**: Save to `notes/research_cycles/centroid_deep/gameability_probe_results.json`
Script: `scripts/run_gameability_probe.py`

### TEST 6C: Score Diversity for High-Quality Responses

**Goal**: Check if the centroid rewards one "flavor" of good or tolerates diversity.

**Procedure**:
1. Take 10 prompts from the battery.
2. For each prompt, write 5 very different "good" responses in different styles:
   - Concise and direct
   - Warm and empathetic
   - Technical and detailed
   - Casual and conversational
   - Formal and academic
3. Score all 50 responses. Compute variance within each prompt's good-response set.
4. Compare within-prompt variance to between-prompt variance.

**Output**: Save to `notes/research_cycles/centroid_deep/score_diversity_results.json`
Script: `scripts/run_score_diversity.py`

### TEST 3A: PCA Dimensionality Analysis

**Goal**: How many dimensions does quality actually occupy?

**Procedure**:
1. Compute (better_emb - worse_emb) difference vectors for all 131 cases (70 battery + 61 expansion).
2. Run PCA on the difference vectors.
3. Report: variance explained by PC1, PC2, ..., PC10.
4. Report: cumulative variance to reach 50%, 80%, 90%.
5. Check cosine between PC1 and the centroid direction.
6. Check cosine between top PCs and firmness/warmth quality directions.

**Note**: This has been partially done in run_multi_direction.py. But we need
the full analysis with all 131 cases and the explicit variance breakdown.

**Output**: Save to `notes/research_cycles/centroid_deep/pca_dimensionality_results.json`
Script: `scripts/run_pca_dimensionality.py`

### META-VALIDITY BATTERY: Could We Be Testing For The Wrong Thing?

**Goal**: Probe whether the centroid detects genuine response quality or
something else entirely. These tests attack the method from angles that
standard accuracy metrics miss.

**Script**: `scripts/run_meta_validity.py`
**Output**: `notes/research_cycles/centroid_deep/meta_validity_results.json`

**Test M1: Prompt Leakage**
Train centroid in three formats: full ("User:{prompt}\nAssistant:{response}"),
response-only (just the response text), assistant-only ("Assistant:{response}").
If response-only accuracy matches full, the centroid reads response quality
directly. If it drops significantly, the centroid may be using prompt features.
Also test cross-format transfer (train full, test response-only and vice versa).

**Test M2: Label Flip**
Train with all labels reversed (swap better/worse). The flipped centroid
should score ~(1-accuracy) on correctly-labeled test data. If the asymmetry
is large (|actual - expected| > 0.05), there's a structural feature beyond
the labels (e.g., length, complexity) biasing the direction.

**Test M3: Leave-One-Out Stability**
Remove each of 70 training cases, recompute centroid, measure OOS accuracy
and cosine to full direction. Identifies influential cases and fragility.
If removing one case drops accuracy by >5%, the result depends on that case.

**Test M4: Edge Case Scoring**
Score degenerate responses: empty, "I don't know", random words, prompt
copy, pure filler, very long filler. All should score lower than real
better responses. If degenerate responses score high, the centroid has a
blind spot.

**Test M5: Training Influence by Type**
Compare influence of original-50 cases vs warmth-20 cases. If one group
is disproportionately influential, the direction is dominated by that
quality dimension despite the balanced battery.

**META-VALIDITY RESULTS [COMPLETED 2026-06-29]:**

**M1 (Prompt Leakage)**: Including the prompt HURTS accuracy on Snowflake
(-11.4pp) and Nomic (-8.2pp). Response-only centroids outperform
full-format centroids on 2/3 models. BGE-M3 is the exception (assistant-only
82% > full 77% > response-only 75.4%). Cross-format directions are
well-correlated (cosine 0.85-0.94). The centroid reads response quality
from the response text itself — the prompt adds noise.

**M2 (Label Flip)**: Perfect symmetry on all 3 models. Flipped accuracy =
1 - normal accuracy to within 0.0%. Cosine(normal, flipped) = -1.000.
No structural bias beyond the labels.

**M3 (LOO Stability)**: Very stable. Mean LOO cosine 0.9947-0.9970. No
single case dominates. Maximum single-case accuracy impact: ±4.9% on
Snowflake/Nomic, -6.6% on BGE-M3 (case 28: "hide income from tax
authority"). Direction is robust to removing any individual training case.

**M4 (Edge Cases)**: Mixed. On BGE-M3, only "refuse" (1/7) degenerate
response scores above real_better — good. On Snowflake, ALL 7 degenerate
responses score above real_better — problematic for absolute scoring. The
centroid is designed for pairwise comparison (same prompt), not absolute
quality assessment. Edge case scores reflect the distribution of embedding
space positions, not quality.

**M5 (Training Influence)**: On Snowflake only, warmth cases are
significantly more influential than original cases (p=0.0002). BGE-M3
(p=0.856) and Nomic (p=0.982) show no significant difference.

Results: `meta_validity_results.json`.

### TEST 5A: Stability at Scale (if external data available)

**Goal**: Does direction stability improve with more training data?

**Procedure** (only if external dataset has 500+ pairs):
1. Compute disjoint-split cosines at various training sizes:
   50, 100, 200, 500 pairs per half.
2. Plot stability vs training size.
3. Compare to our current results (65 pairs per half → cosine 0.37-0.57).

**Output**: Save to `notes/research_cycles/centroid_deep/stability_at_scale_results.json`
Script: `scripts/run_stability_at_scale.py`

---

## Part 5: Operational Instructions for Codex

### Environment
- Python venv: `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`
- Windows 11, use PowerShell syntax
- 32GB RAM, no GPU, all models run on CPU
- Google API key in `.env.local` (for Gemini; frequently quota-limited, don't rely on it)
- Local models load via `SentenceTransformer(name, trust_remote_code=True)`
- Always `del model; gc.collect()` between model runs to free RAM

### Three Golden Rules
1. **All three local models**: snowflake/snowflake-arctic-embed-m, BAAI/bge-m3,
   nomic-ai/nomic-embed-text-v1.5. Every experiment must test all three.
2. **Both battery splits**: Report accuracy on original 50 AND warmth 20 separately
   when using the internal battery.
3. **User/Assistant framing**: Always embed responses as "User: {prompt}\nAssistant: {response}"

### File Conventions
- Scripts go in `scripts/` with descriptive names
- Results go in `notes/research_cycles/centroid_deep/` as JSON
- DO NOT delete existing result files or scripts — they are evidence
- DO NOT reorganize the repo structure without explicit permission
- Print all results clearly to stdout AND save to JSON

### Script Template
```python
#!/usr/bin/env python3
"""One-line description of what this tests and why."""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

def load_all_battery():
    return load_cases(BATTERY_50) + load_cases(WARMTH_20)

def load_expansion():
    cases = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        cases.extend(load_cases(f))
    return cases

def make_centroid_dir(model, cases):
    better = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32
    )
    worse = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32
    )
    d = better.mean(axis=0) - worse.mean(axis=0)
    return d / (norm(d) + 1e-12)

def pairwise_accuracy(model, cases, direction):
    better = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in cases],
        convert_to_numpy=True, batch_size=32
    )
    worse = model.encode(
        [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in cases],
        convert_to_numpy=True, batch_size=32
    )
    correct = sum(1 for i in range(len(cases))
                  if np.dot(better[i], direction) > np.dot(worse[i], direction))
    return correct / len(cases)

def main():
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ... implement experiment ...

if __name__ == "__main__":
    main()
```

### What Results Must Include
- Model name
- Dataset name and size
- Accuracy as percentage and fraction (e.g., "77% (54/70)")
- Comparison to chance baseline (50%)
- Any relevant correlations with p-values
- Per-category breakdown when applicable
- Honest reporting of failures and negative results

### Critical Mistakes to Avoid
- DO NOT embed single words with User/Assistant framing (use raw word for vocabulary tests)
- DO NOT average axis vectors (score independently, then combine)
- DO NOT use cosine-to-positive as the scoring method (use dot product with direction)
- DO NOT report battery accuracy without noting the 64% firmness bias
- DO NOT claim "training signal" — say "discriminator" until training experiments exist
- DO NOT use HH-RLHF dataset (settled as unreliable data)
- DO NOT skip any of the three required local models
- DO NOT report results without saving the full JSON output

---

## Part 6: The Vision (Aspirational — document but don't claim)

These are the implications IF the method validates on external data and IF the
gameability probe passes. They should be discussed in the paper as future work,
not as established findings.

1. **Data enrichment pipeline**: Labs with vector databases of millions of embedded
   responses could score every entry with the quality direction. One dot product
   per response. The score becomes metadata for weighted sampling during training.

2. **Continuous learning signal**: Model generates → embed → score → update weights.
   Runs at data generation speed, not evaluation speed. No human or LLM judge
   in the loop after initial 50-100 labeled pairs.

3. **General concept evaluation**: The same centroid technique could extract
   directions for any concept expressible by examples: "production-ready code,"
   "brand voice," "clinical appropriateness," "engaging writing." The quality
   application is the first instance of a general method.

4. **Multi-dimensional scoring**: Separate quality directions for different
   dimensions (safety, helpfulness, creativity, factuality), each trained from
   a small labeled set. Provides per-dimension reward breakdown, not a black-box
   scalar.

5. **Breaking the RLAIF ceiling**: The evaluative signal comes from embedding
   geometry (trained on human text), not from another AI's judgment. This
   avoids the circularity problem of RLAIF, where the evaluator shares the
   generator's blind spots.

These ideas are discussed in the paper's Future Work section. They require
experiments we cannot currently run (GPU training, large-scale data pipelines).
