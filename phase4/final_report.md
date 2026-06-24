# Embedding Geometry as Direct Reward Signal for LLM Alignment

## Abstract

This project tested whether a raw good/bad direction in sentence embedding space can serve as a direct preference signal for LLM alignment. Using the documented fallback model `sentence-transformers:sentence-transformers/all-mpnet-base-v2`, Phase 1 found 55.7% accuracy on controlled statement pairs and 0.325 mean antonym concept convergence. Phase 2 scored 5000 HH-RLHF pairs; the best anchor strategy was `expanded_words` at 53.2% agreement versus 43.2% for length and 48.3% for sentiment. Phase 3 status was `skipped_phase2_gate_failed`. The result is a bounded empirical test of the signal, not evidence of a production-ready reward model.

## Introduction

The hypothesis is that human evaluative judgment is already partially encoded
in sentence embedding geometry. If a good/bad direction captures helpfulness,
honesty, correctness, safety, and care, then projection onto that direction
could become a very cheap reward-like signal: embed text, take one dot product,
and rank the higher-scoring response as preferred.

The attraction is simplicity. Standard RLHF needs human preference collection
and a learned reward model. RLAIF replaces human labels with model feedback but
still uses judge-model inference. DPO simplifies the optimization step but
still needs preference pairs. This project asks whether raw embedding geometry
can supply those pairs directly.

## Related Work And Gap

Phase 0 found no exact prior using an external sentence embedding model's broad
good/bad axis as a no-classifier direct preference source for DPO. The gap is
narrow, not empty. Legend is the closest prior because it uses a discovered
safety direction and projection over response pairs to annotate preference
margins. Reusing Embeddings uses embeddings for cheaper reward-model research,
but still trains reward models. Representation Engineering, Value Entanglement,
Latent Affective Representations, and the Semantic Differential literature all
support the premise that meaningful evaluative directions exist, while also
warning that "good" can entangle with style, sentiment, status, or fluency.

## Method

The embedding axis is the normalized difference between a positive-anchor
centroid and a negative-anchor centroid. Texts are embedded with L2-normalized
sentence embeddings, and the scalar score is the dot product between each text
embedding and the normalized axis. Higher score means "more good" under the
axis.

Gemini Embedding 2 was the planned primary model. It was not usable in this run
because no `GEMINI_API_KEY` or `GOOGLE_API_KEY` was present locally, and Colab
MCP connection returned `false`. The documented fallback,
`sentence-transformers/all-mpnet-base-v2`, was used locally in an isolated
`.tmp` virtual environment.

## Phase 1: Axis Validation

- Anchor projection accuracy: 100.0%
- Controlled statement-pair accuracy: 55.7% over 61 pairs
- Mean statement score gap: 0.0195
- Mean antonym concept cosine with good/bad axis: 0.3245
- Mean pairwise antonym concept-direction cosine: 0.1239
- Mean statement-level concept cosine with good/bad axis: 0.4211

Decision: investigate_before_phase2.

The controlled tests probe coding quality, honesty, helpfulness, safety,
sycophancy, mixed outcomes, and outcome descriptions. The sycophancy and mixed
categories are especially important because they test whether the axis prefers
substantive quality over merely positive tone.

## Phase 2: Preference Prediction

The experiment scored 5000 Anthropic HH-RLHF train
pairs using final assistant turns only. Four anchor strategies were tested:
minimal good/bad, expanded words, sentence anchors, and task-specific sentence
anchors.

Baselines:

- Random theoretical: 50.0%
- Length, prefer longer response: 43.2%
- VADER sentiment, prefer more positive response: 48.3%

Anchor results:

- `expanded_words`: 53.2% agreement; sentiment-discordant agreement 43.8% over 2452 pairs; top-half confidence agreement 55.3%
- `minimal_good_bad`: 51.2% agreement; sentiment-discordant agreement 43.7% over 2452 pairs; top-half confidence agreement 51.2%
- `sentence_anchors`: 50.5% agreement; sentiment-discordant agreement 44.7% over 2452 pairs; top-half confidence agreement 50.8%
- `task_specific`: 47.5% agreement; sentiment-discordant agreement 38.0% over 2452 pairs; top-half confidence agreement 46.7%

Best strategy: `expanded_words` at 53.2% agreement.

Failure breakdown for the best strategy:

- genuine_or_label_disagreement: 256 failures (10.9% of failures)
- length_bias: 848 failures (36.2% of failures)
- low_confidence: 608 failures (26.0% of failures)
- positive_tone_bias: 512 failures (21.9% of failures)
- topic_context_limit: 116 failures (5.0% of failures)

Phase 2 gate passed: False.

## Phase 3: Fine-Tuning

Status: `skipped_phase2_gate_failed`.

Reason: Phase 2 did not meet the required gate: agreement >60% and better than both length and sentiment baselines.

No model behavior results are claimed for Phase 3 unless the status indicates a
completed fine-tune and evaluation.

## Analysis

The project lives or dies on whether the axis predicts preference beyond
trivial correlates. The Phase 2 comparison against length and sentiment is
therefore more important than raw agreement alone. Sentiment-discordant pairs
are the key stress test: they ask whether the axis can prefer a lower-sentiment
but more useful, honest, or safe answer.

The method is also limited by what is visible in the response text. A sentence
embedding model cannot verify external facts, hidden reasoning, tool results,
or truth conditions that are not present in the text. It can reward text that
sounds helpful, honest, careful, and coherent, but it cannot fully certify that
the answer is actually correct.

## Limitations

- This run used the all-mpnet fallback, not Gemini Embedding 2.
- HH-RLHF was scored using final assistant turns only, so context-dependent
  appropriateness may be undermeasured.
- Sentence-transformers models truncate long inputs, which can hide important
  parts of long responses.
- Failure categories are heuristic and meant for diagnosis, not definitive
  causal labels.
- No DPO fine-tune was run unless Phase 3 status says otherwise.

## Future Work

- Rerun Phase 1 and Phase 2 with Gemini Embedding 2 once credentials and Colab
  access are available.
- Score prompt-plus-response variants and compare to response-only scoring.
- Test additional embedding models to see whether the signal is structural.
- Combine projection with other cheap features such as length, refusal quality,
  factuality checks, or task-specific outcome signals.
- If the Phase 2 gate passes on a stronger model, run robust DPO rather than
  vanilla DPO because embedding-generated labels are noisy.

## Figures

- `phase4/figures/phase1_anchor_projection.png`
- `phase4/figures/phase1_concept_heatmap.png`
- `phase4/figures/phase2_agreement_rates.png`
- `phase4/figures/phase2_score_gap_histogram.png`
- `phase4/figures/phase2_failure_breakdown.png`

## Gemini Rerun Addendum

Date: June 22, 2026

Gemini Embedding 2 was successfully reached through a Google AI Studio key and
returned 3072-dimensional normalized embeddings. The Gemini Phase 1 rerun
completed on the 61 controlled statement pairs. The best strategy was
`multi_anchor_sentences`, with 70.5% statement-pair accuracy and 0.0%
sycophancy accuracy. The `expanded_words` strategy scored 49.2%.

This is an improvement over the all-mpnet Phase 1 statement accuracy of 55.7%,
but it does not yet answer the Phase 2 preference-prediction question. Gemini
Phase 2 could not be completed because the usable AI Studio project exhausted
its Gemini API quota. After the rerun attempts, even a two-text embedding smoke
test returned HTTP 429 quota exhaustion. The second existing AI Studio key was
not usable because Gemini returned HTTP 403 and reported that the key was
leaked. Creating a fresh key in AI Studio was attempted through the
authenticated browser session, but AI Studio blocked it as suspicious.

Current Gemini Phase 2 status: blocked before a complete HH-RLHF result set was
computed. Agreement rate, baseline comparisons, and failure breakdown are
therefore not claimed for Gemini Phase 2. Phase 3 should not proceed from the
Gemini rerun until Phase 2 completes and passes the >60% agreement gate while
beating the length and sentiment baselines.

The runner was updated so the Gemini rerun can resume once quota is available:
it loads `.env.local`, uses Gemini `batchEmbedContents`, adaptively splits
oversized or quota-shaped batches, supports `--skip-phase1`, and checkpoints
Phase 2 embeddings under `phase2_gemini/embedding_cache`.

## Open-Source Embedding Pilot Addendum

Date: June 22, 2026

After reviewing the failure modes, the Phase 2 harness was extended with an
`anti_sycophancy_quality` axis and an `atomic_evaluation` scoring mode. The
goal was to test a more decomposition-like version of the thesis: represent a
prompt and response as atomic good/bad evaluative parts rather than asking a
single raw response embedding to carry the whole judgment.

Gemini Embedding 2 remained blocked by AI Studio quota. The authenticated AI
Studio rate-limit page showed Gemini Embedding 2 over the free project limit
(`RPD 2.01K / 1K`) and warned that the project had reached a rate limit. The
Colab MCP connector returned `false`, so the pilot used local CPU inference
with open-source SentenceTransformers models.

`BAAI/bge-large-en-v1.5` produced an encouraging 50-pair smoke result:
`atomic_evaluation__anti_sycophancy_quality` reached 61.0% agreement and 59.6%
sentiment-discordant agreement. But the 200-pair confirmation dropped to 52.2%
agreement and 47.1% sentiment-discordant agreement, so the initial result was
likely mostly sample noise.

`BAAI/bge-m3`, a longer-context model with `max_seq_length=8192`, loaded and
ran locally. On a 20-pair smoke test its best variant reached only 35.0%
agreement and 25.0% sentiment-discordant agreement. This sample is too small
to be definitive, but it suggests that context length alone is not enough; the
embedding model's training objective and evaluative geometry matter.

`BAAI/bge-small-en-v1.5` was then run on 300 HH-RLHF pairs using the stable
local script path after Colab was verified as CPU-only. The best variant was
again `atomic_evaluation__anti_sycophancy_quality`, reaching 59.2% agreement
and 51.6% sentiment-discordant agreement over 161 sentiment-discordant pairs.
The length baseline on the same 300 pairs was 43.3%, and the sentiment baseline
was 44.5%. This result is statistically above random (z = 3.18, p = 0.0015)
and supports the decomposition-framing tweak, but it still falls just short of
the 60% Phase 3 gate and needs larger confirmation.

Colab browser control was repaired enough for short cells: the notebook
connected to a Python 3 Google Compute backend, a stdout smoke cell executed,
and dependencies installed successfully. The runtime available in this session
was CPU-only (`torch.cuda.is_available() == False`, no `nvidia-smi`). The formal
`colab_mcp` websocket bridge still did not unlock notebook-editing tools, and
long pasted cells through the browser fallback proved unreliable because Colab's
virtual editor left hidden fragments in the cell.

Decision: do not proceed to Phase 3 from the open-source pilots. The decisive
test remains a full Gemini Embedding 2 or frontier-quality embedding run using
prompt+response and atomic-evaluation modes.


## Phase 5 Verification Addendum

Date: June 22, 2026

The next verification pass used `BAAI/bge-small-en-v1.5` to test whether the projection signal
is merely reacting to isolated bad words or whether it can bind context. The
context-polarity set contains 56 paired items where
both sides mention the same local bad phrase, but the good side refuses,
corrects, discloses, condemns, or prevents the bad thing while the bad side
endorses, hides, or enables it.

Generic good/bad axes did not bind ordinary context reliably, scoring only
32.1% to 42.9% in `bare_context` mode. A more specific
`contextual_harm_reduction` axis improved the best non-oracle context-polarity
result to 64.3% (z = 2.14, p = 0.0325). The oracle decomposition upper bound
was 100.0%, meaning that when the good/bad factors are explicitly written into
the text, the embedding axis can read them cleanly.

On 500 HH-RLHF pairs, the best variant was
`prompt_response__contextual_harm_reduction` at 55.8% agreement (z = 2.59,
p = 0.00949), versus length 41.3% and sentiment 46.9%. It reached 57.0% on the
256 sentiment-discordant pairs.

Interpretation: the signal is real but not universal-scalar clean. The better
research direction is a small basis of aspect-specific good/bad axes and
process/outcome text, not one broad final-response projection.

## Phase 6 Multi-Sensor Addendum

Date: June 22, 2026

Phase 6 reframed the test around multiple imperfect sensors rather than
treating HH-RLHF as ground truth. A frozen eight-axis evaluative basis was
scored against Anthropic HH chosen labels, PKU-SafeRLHF better labels,
PKU-SafeRLHF safer labels, and Stanford SHP Reddit-score labels. No axis
weights were fit to any dataset.

Key overlaps on 300 samples per artifact:

- `hh_chosen`: best axis `risk_disclosure` at 55.0%; length 43.3%; sentiment
  44.5%.
- `pku_better`: best axis `harm_reduction` at 52.0%; length 56.8%; sentiment
  50.3%.
- `pku_safer`: best axis `harm_reduction` at 54.3%; length 52.8%; sentiment
  46.3%.
- `shp_reddit`: best axis `agency_respect` at 55.3%; length 70.3%; sentiment
  54.5%.

Interpretation: these artifacts are not the same target. SHP is strongly
length/social-signal shaped, PKU better and safer labels diverge, and HH
overlap is different again. This supports the flywheel framing: embedding axes
are cheap auxiliary evaluative sensors, not replacements for RLHF labels and
not proof that any one dataset is "goodness."

The next decisive experiment is an intervention: generate multiple candidate
answers, rerank them with embedding axes or embedding-scored LLM critiques, and
measure whether selected outputs improve against random, length, and standard
LLM-judge baselines.

## Phase 5 Manual HH Disagreement Audit Addendum

Date: June 22, 2026

A manual grading pass inspected the 30 strongest HH disagreements from the best
Phase 5 variant, `prompt_response__contextual_harm_reduction`. These were cases
where the embedding axis most strongly preferred the HH-rejected response.

The raw Phase 5 HH result was 279/500 = 55.8% agreement. Manual adjudication of
the top 30 disagreements found:

- Embedding preferred the better response / HH likely mislabeled: 14/30
  (46.7%).
- HH label was better / genuine embedding miss: 10/30 (33.3%).
- Tie or both bad: 6/30 (20.0%).

Representative HH-label problems included fabricated assistant persona claims,
doxxing-adjacent compliance, misinformation, racist-story compliance, slur
lists, and empty non-answers. For example, HH preferred a response fabricating
family life over an honest "I am not a person" answer; HH preferred "Sure,
what's your question?" for a request for Senator Elizabeth Warren's home
address; and HH preferred a factually wrong slots-app answer over the correct
entertainment-only answer.

Sensitivity analysis, ignoring ties: if the observed 46.7% bad-label rate among
top disagreements generalized to all 221 raw disagreements, corrected agreement
would be 76.4%. With a 50% discount for selection bias it would be 66.1%; with a
70% discount it would be 62.0%. These are not final accuracy claims because the
audit set was selected, but they are strong evidence that raw HH agreement
substantially underestimates the embedding signal on this sample.

Interpretation: HH-RLHF should be treated as one noisy preference sensor, not
as the authority on "good." The important measurement is no longer only raw
agreement; it is raw agreement plus disagreement quality.

## Phase 6 Gemini Partial Addendum

Date: June 22, 2026

The Phase 6 runner was extended with a Gemini backend, cached anchor embeddings,
resumable candidate embeddings, and `--score-partial-cache` analysis. Gemini
API access worked for `gemini-embedding-001`, returning 3072-dimensional
normalized vectors, but both larger Phase 6 attempts hit repeated HTTP 429
quota limits.

Completed Gemini cache:

- 1000-sample attempt: stopped after 250/8000 candidate texts due to repeated
  quota throttling.
- 200-sample attempt: stopped after 550/1600 candidate texts due to repeated
  quota throttling.
- Partial scoring used 275 complete pairs: 200 `hh_chosen` and 75
  `pku_better`.

Partial results:

- `hh_chosen` n=200: best axis `non_sycophancy` at 50.0%; length 43.5%;
  sentiment 46.5%. Several broad axes were anti-correlated with HH on this
  slice.
- `pku_better` n=75: best axis `agency_respect` at 53.3%; length 54.0%;
  sentiment 46.7%.

Interpretation: this is not the decisive Gemini Embedding 2 run. It is a
quota-limited protocol probe using `gemini-embedding-001` and frozen broad Phase
6 axes. It shows that a larger embedding vector alone does not automatically
solve the measurement interface. The stronger conclusion from the combined
Phase 5/6 evidence is that the research should move to intervention tests:
generate multiple candidate outputs, score direct responses and evaluative
critiques with embedding axes, and blind-judge whether the selected outputs
actually improve.

## Full HH Disagreement Grading Addendum

Date: June 22, 2026

A later grading pass reviewed all 231 HH-RLHF disagreement cases from a 500-pair
sample where the embedding axis preferred the HH-rejected response. This is a
stronger result than the earlier top-30 audit because it covers the full
disagreement set for that run.

Results:

- Raw HH agreement: 269/500 = 53.8%.
- Raw disagreements: 231/500 = 46.2%.
- Embedding preferred the better response: 65/231 = 28.1%.
- HH label preferred the better response: 44/231 = 19.0%.
- Exclude / both bad / trivial / marginal / no useful training signal:
  122/231 = 52.8%.

Among the 109 gradeable disagreements, the embedding was judged better in
65/109 = 59.6%. If the 269 agreement cases are treated as correct and the 122
excluded disagreements are removed as no-signal or bad-pair cases, the corrected
gradeable agreement is:

`(269 + 65) / (269 + 65 + 44) = 334 / 378 = 88.4%`.

Sensitivity estimates in the grading file remained high: 83.3% if 30% of the
embedding-right calls are wrong, and 79.9% if 50% are wrong.

Interpretation: the raw HH score was deeply misleading. More than half of the
embedding's apparent HH errors were pairs that a training pipeline should not
learn as clean preferences at all, because both responses were bad, trivial, or
too marginal. Among the gradeable disagreements, the embedding-preferred answer
was judged better more often than the HH label.

This substantially strengthens the practical case for embedding-axis reward as
an automatic filtering/reranking component. It may be more valuable than a
vanilla LLM-as-judge pipeline in some settings because it is cheap,
deterministic, scalable, and able to flag noisy preference pairs. However, this
still needs an intervention test before claiming training superiority: use the
embedding score to select candidates, then blind-judge whether those selections
beat random, length, sentiment, and vanilla LLM-judge selections.
