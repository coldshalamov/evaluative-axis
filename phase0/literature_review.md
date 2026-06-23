# Phase 0 Literature Review

## Executive Summary

The literature search supports continuing the experiment, but with a narrower
novelty claim than the original plan. I did not find a prior paper that uses an
external sentence embedding model, builds a broad good/bad axis from simple
anchors, and uses raw projection along that axis directly to create DPO
chosen/rejected pairs without a classifier, learned reward model, or LLM judge
in the labeling loop.

The gap is not empty, though. Close prior work exists around representation
directions, valence geometry, and projection-based annotation. The most
important adjacent paper is Legend, which discovers a safety direction in an
LLM representation space and uses projection along that direction to annotate
preference margins. Other adjacent work uses embedding projection as an
analysis metric for moral concepts or uses embeddings as inputs to trained
reward models. The best positioning is therefore:

> Prior work shows that embedding and activation spaces contain alignment-
> relevant directions, and that projection can annotate safety margins. This
> project tests whether a more general external good/bad embedding axis can be
> used as a direct, training-free preference signal for alignment data.

## Required Papers

### Reusing Embeddings: Reproducible Reward Model Research in LLM Alignment without GPUs

Sun et al. (2025) argue that reward-model research can be made cheaper and more
reproducible by reusing embeddings as reward-model inputs. Their case study
still trains reward models or classifiers on top of embeddings, so it does not
remove preference labels or the learned reward layer. It is the closest paper in
the reward-model-efficiency lane, but it is not the same as raw projection used
directly as the reward signal.

### Value Entanglement

Cho, Li, and Leshinskaya (2026) test whether LLMs separate moral, grammatical,
and economic senses of "good." They report entanglement among these values in
behavior, embeddings, and activations. For this project, that result is both
supportive and cautionary: a broad evaluative axis may naturally capture many
quality notions, but it may also confuse moral value with polish, status,
fluency, or other correlated signals.

### Latent Structure of Affective Representations in LLMs

Choi and Weber (2026) study emotion representations in LLM latent spaces and
find geometry aligned with valence-arousal models from psychology. They also
find that nonlinear affective structure can be approximated linearly. This
supports the plausibility of a linear good/bad direction, but the paper does
not test whether that direction predicts human preference labels or supports
fine-tuning.

### The Measurement of Meaning

Osgood, Suci, and Tannenbaum (1957) introduced the semantic differential
framework. Across bipolar adjective scales, they identify Evaluation, Potency,
and Activity as major dimensions of affective meaning, with Evaluation anchored
by good/bad. This is the psychological foundation for the single-axis
hypothesis: good/bad is not an arbitrary word pair, but a central axis in human
meaning judgments.

### InstructGPT / RLHF

Ouyang et al. (2022) fine-tune GPT-3 using supervised demonstrations, human
rankings, a learned reward model, and RLHF. The resulting InstructGPT models
are preferred over much larger base GPT-3 models and improve truthfulness and
toxicity metrics. This is the canonical expensive alignment pipeline that the
embedding-axis approach tries to simplify by removing both human preference
collection and the trained reward model.

### Direct Preference Optimization

Rafailov et al. (2023) show that language models can be aligned directly from
preference pairs using a classification-style objective that implicitly
optimizes the same kind of preference objective as RLHF. DPO removes the
explicit reward-model-and-RL training loop, but it still requires preference
pairs. In this project, DPO is a possible downstream training method; the
research question is whether embedding-axis scoring can supply the pairs.

### Constitutional AI / RLAIF

Bai et al. (2022) reduce human labeling by using constitutional principles,
model-generated critiques and revisions, and AI preference feedback. This is an
important non-human-supervision baseline. It still depends on LLM inference and
a preference model for the reward signal, whereas the proposed embedding axis
is deterministic and much cheaper once embeddings are available.

### Representation Engineering

Zou et al. (2023) frame representation engineering as a top-down approach for
reading and controlling high-level concepts such as honesty, harmlessness, and
power-seeking in neural representations. This strongly supports the idea that
high-level alignment concepts can correspond to directions. The project differs
by using an external sentence embedding space for preference scoring rather
than model-internal activations for interpretability or inference-time control.

## Closest Adjacent Work Found In Gap Search

### Legend

Legend is the closest prior art. It constructs a safety direction in an LLM
embedding/representation space and uses response-pair projection along that
direction to annotate preference margins. It then evaluates those annotations
in reward modeling and harmless alignment. This overlaps with the projection
mechanism, but differs in four important ways: it is safety-specific rather
than broad good/bad, uses internal model representations rather than an
external sentence embedding model, annotates margins for preference datasets
rather than generating new DPO pairs from scratch, and does not test the simple
anchor-defined good/bad axis proposed here.

### Virtue Semantics

The Virtue Semantics workshop paper uses projection onto a moral-goodness
vector to produce a scalar metric for virtue concepts. This is directly
adjacent to the idea of a moral/evaluative projection score, but it is a probe
of moral concept consistency rather than a reward signal for preference
optimization or fine-tuning.

### Intrinsic Guardrails / Semantic Valence Vector

Intrinsic Guardrails introduces a Semantic Valence Vector and uses causal
interventions in model representations to study and regulate emergent
misalignment. It is relevant because it treats social valence as a meaningful
direction in representation space. It does not use an external text embedding
good/bad axis as a direct reward signal for DPO data generation.

### Semantic Structure in LLM Embeddings

Kozlowski, Dai, and Boutyline report that projections onto antonym-defined
semantic directions correlate with human semantic ratings and reduce to a
low-dimensional structure resembling the semantic differential. This is strong
supporting evidence for Phase 1, but the paper studies semantic ratings and
embedding structure rather than preference prediction or alignment training.

## Gap Search

Searches included:

- `embedding projection reward signal LLM`
- `good bad axis embedding reward model`
- `semantic valence reward model language model`
- `embedding direction preference optimization reward`
- `semantic valence vector LLM reward`
- `global embedding projection score LLM moral values`

The search found projection-based annotation, representation-direction
intervention, embedding-projection moral probes, and embedding-input reward
models. It did not find an exact prior where raw projection along an external
sentence embedding model's broad good/bad axis is used directly as the
preference source for DPO without a classifier or learned reward model.

## Interpretation

The research gap exists, but it should be described carefully. The broad idea
that meaningful representation directions can support alignment is established.
The specific minimal version being tested here remains open:

1. External embedding model rather than target-model activations.
2. Broad evaluative good/bad axis rather than safety-only or virtue-only axes.
3. Raw projection only, with no trained classifier or reward head.
4. Preference-pair generation for DPO, not just probing, steering, or margin
   annotation.

The main risk is that the axis may reward positive-sounding text rather than
high-quality text. Phase 1 therefore needs sycophancy and mixed-outcome pairs,
and Phase 2 needs sentiment-discordant analysis.

## Decision

Continue to Phase 1. The exact method was not found, but close prior art
requires positioning the work as a minimal external-axis baseline and
generalization of projection-based alignment signals, not as a completely
untouched research area.
