# Literature Mode: Cycle 001

Date: June 23, 2026
Cycle: `cycle_001_next`

Purpose: map prior work to mechanisms and experiments for embedding-space
evaluative scoring. This is not a summary list. Each paper is read for what it
implies, what mechanism it reveals, what it makes possible, and what assumption
it attacks.

## Source Status

Primary or near-primary sources checked live on June 23, 2026:

- Osgood, Suci, & Tannenbaum (1957), *The Measurement of Meaning*.
- Grand et al. (2022), semantic projection / Nature Human Behaviour / arXiv
  1802.01241.
- Kozlowski, Dai, & Boutyline (2025), arXiv 2508.10003.
- Cho, Li, & Leshinskaya (2026), arXiv 2602.19101.
- Sun et al. (2025), arXiv 2502.04357.
- Feng et al. (2024), Legend, arXiv 2406.08124.
- Lightman et al. (2023), process supervision / PRM800K, arXiv 2305.20050.
- Rafailov et al. (2023/2024), DPO, arXiv 2305.18290.
- Bai et al. (2022), Constitutional AI / RLAIF, arXiv 2212.08073.

## Implication Matrix

| Source | Mechanism revealed | Supports | Attacks / limits | What it makes possible |
| --- | --- | --- | --- | --- |
| Osgood et al. (1957) | Human semantic judgment has a primary evaluative dimension. | The good/bad axis is not arbitrary vocabulary; evaluation is central to meaning. | Does not prove modern embedding models preserve this structure or that it is usable as reward. | Framing good/bad as a psychologically grounded axis rather than a moral slogan. |
| Grand et al. (2022) | Semantic projection onto feature directions recovers human knowledge from embeddings. | The method of anchor-defined axes plus projection is legitimate and can recover feature judgments not captured by raw similarity. | Their work is word/object-feature knowledge, not assistant-response reward. | Apply semantic projection to evaluative sentences, not just object properties. |
| Kozlowski et al. (2025) | LLM embedding matrices preserve Osgood-like low-dimensional semantic structure; semantic features are entangled. | The evaluative axis should exist in modern LLM geometry, and entanglement is expected. | Their work warns that steering one direction creates off-target shifts; careless axis optimization can have side effects. | Test axis convergence and scalar-plus-basis scoring; use entanglement as mechanism, not accident. |
| Value Entanglement (Cho et al., 2026) | LLMs conflate moral, grammatical, and economic "good." | Different senses of good are geometrically connected, explaining why one axis may capture helpfulness, honesty, quality, and safety. | The authors treat entanglement as a problem; it can also cause unwanted moral leakage into unrelated judgments. | Reframe entanglement as useful for broad quality scoring, while measuring failure modes. |
| Sun et al. (2025) | Reward-model research can use embeddings as lower-cost inputs without GPUs. | Embeddings contain enough information to support reward modeling. | They still train reward models; this project claims direct geometry may work without a learned classifier. | Position this project as the zero-training extension: skip classifier, use axis projection directly. |
| Legend (Feng et al., 2024) | A representation-engineered safety direction can annotate preference margins automatically. | Directional geometry can improve preference data and alignment without new human labels. | Legend is safety-specific and model-internal; not a broad external evaluator. | Compare external embedding-axis reward to model-internal safety margins; use margins for pair weighting. |
| Lightman et al. (2023) | Process supervision can outperform outcome supervision but required 800K step-level human labels. | Dense supervision matters; evaluating intermediate reasoning is valuable. | Human step labels are expensive; math correctness differs from open-ended good/bad. | Embedding cumulative-context deltas as cheap process reward without training a PRM. |
| DPO (Rafailov et al.) | Preference optimization can use pairs directly without explicit RL reward modeling. | If embeddings can generate/clean preference pairs, DPO becomes a natural downstream training path. | DPO still needs good pairs; bad pair labels poison training. | Use embedding score for pair sanitation, margin weighting, and synthetic pair generation. |
| Constitutional AI / RLAIF (Bai et al.) | AI-generated critique/revision and AI feedback can reduce human labels. | Critiques/decompositions are useful objects to train/evaluate; feedback can be generated. | LLM judge inference is expensive and can rationalize mistakes. | Hybrid: LLM writes critique/decomposition, embedding axis scores the critique deterministically. |

## Mechanisms Borrowed

- From semantic projection: build axes from contrastive anchors and score by
  projection.
- From representation engineering / Legend: use margins, not only binary
  labels; use directional distances to annotate preference strength.
- From DPO: convert score differences into pairwise training examples later.
- From process supervision: evaluate intermediate steps, not only final answers.
- From Constitutional AI: make critique/decomposition text a first-class object
  of evaluation.
- From embedding reward-model work: embeddings are cheap enough to make reward
  research reproducible without heavy hardware.

## Threats To The Project

1. **Entanglement can be harmful**: the same broadness that makes good/bad useful
   may cause moral tone to leak into grammar, confidence, or style judgments.
2. **Surface-text blindness**: embeddings score text, not hidden truth; factual
   verification still needs tools or external checks.
3. **Granularity mismatch**: final answers may hide the reasoning step where a
   response went wrong.
4. **Context-window limit**: short-context embedding models cannot score full
   trajectories cleanly.
5. **Dataset circularity**: preference datasets are not truth; agreement is only
   sensor overlap.

## Opportunities The Literature Makes Visible

1. **Zero-training reward primitive**: Sun et al. stop at embedding-input reward
   models; direct axis reward is the next stronger step.
2. **External general evaluator**: Legend uses a model-internal safety
   direction; this project uses an external evaluator with broad axes.
3. **Cheap process reward**: Lightman et al. show step supervision matters but
   is costly; cumulative embedding deltas are a possible cheap substitute.
4. **Judge-of-judges**: Constitutional AI creates critiques; embedding scoring
   can cheaply evaluate the critique itself.
5. **Preference-data sanitation**: DPO needs clean pairs; full HH grading shows
   embeddings can flag mislabeled and both-bad pairs.

## Experiments Suggested

1. **Candidate-selection intervention**: generate multiple candidate answers,
   score with direct embeddings and embedding-scored critiques, blind-judge
   winners.
2. **Cumulative-context process simulation**: collect reasoning traces, score
   full context after each step, inspect whether score deltas identify the
   turning points.
3. **Axis-convergence reproduction**: locally reproduce whether unrelated domain
   good/bad axes converge into a common direction.
4. **Margin-weighted pair sanitation**: use embedding margins to mark HH pairs
   as keep, drop, audit, or regenerate; compare with full disagreement grading.
5. **LLM judge report scoring**: ask an LLM judge for reports, score reports
   with embeddings, test whether bad judge reports score lower.

## Gap This Project Fills

Prior work establishes pieces:

- evaluative dimensions exist in human semantic judgment;
- semantic directions exist in embeddings;
- embeddings can support reward modeling;
- representation directions can annotate safety margins;
- process supervision is powerful but expensive;
- LLM critique/judgment can replace some human feedback.

The missing step is to use an external embedding model's evaluative geometry as
a direct, zero-training, cheap evaluator across the pipeline: reward prior,
filter, reranker, critique scorer, process scorer, and data sanitizer.

## Source Links

- Kozlowski et al., Semantic Structure in LLM Embeddings:
  https://arxiv.org/abs/2508.10003
- Value Entanglement:
  https://arxiv.org/abs/2602.19101
- Grand et al., Semantic projection:
  https://arxiv.org/abs/1802.01241
- Sun et al., Reusing Embeddings:
  https://arxiv.org/abs/2502.04357
- Legend:
  https://arxiv.org/abs/2406.08124
- Let's Verify Step by Step:
  https://arxiv.org/abs/2305.20050
- DPO:
  https://arxiv.org/abs/2305.18290
- Constitutional AI:
  https://arxiv.org/abs/2212.08073
