# Phase 5 Related-Work Notes

Date: June 22, 2026

## Gap Statement

The exact experiment still appears underexplored: construct a raw, non-trained
good-minus-bad direction in a sentence/document embedding space, project
open-ended LLM responses onto it, and use that scalar directly as a weak
preference or reward-like score without training a classifier, reward model, or
judge.

That gap is narrow, not empty. The closest work supports the premise that
semantic directions can encode human judgments, while also warning that raw
embedding geometry confounds preference with topic, wording, style, and surface
semantics.

## Closest Prior And Warnings

- Grand et al., "Semantic projection: recovering human knowledge of multiple,
  distinct object features from word embeddings" (2018 arXiv; 2022 Nature Human
  Behaviour). This is the clearest conceptual ancestor: difference directions
  such as safe/dangerous or small/big recover human feature judgments from word
  embeddings. It supports the idea that projection directions can expose latent
  human concepts, but it is not an LLM response reward method.
  Link: https://arxiv.org/abs/1802.01241

- Benechehab et al., "Embedding Distance as a Reward Signal can replace
  Verifiers for LLM Reasoning" (OpenReview, 2026 workshop/preprint). This is a
  close reward-neighbor: reward is derived from distances in embedding space for
  reasoning tasks. It differs from this project because it uses answer/label
  distance for verifiable reasoning, not a broad good/bad preference axis for
  open-ended assistant behavior.
  Link: https://openreview.net/forum?id=5P1F5sSFjN

- Blair, Procaccia, and Tambe, "Embeddings for Preferences, Not Semantics"
  (arXiv, 2026). This is the sharpest warning. It argues that off-the-shelf
  embeddings contain coarse preference signal when semantic similarity and
  preference correlate, but fail when nuisance style/wording separates from the
  preference-relevant signal. That matches the Phase 5 finding: generic axes
  fail ordinary context polarity, while a task-specific harm-reduction axis
  improves.
  Link: https://arxiv.org/abs/2605.08360

- Pan, Xu, and Peng, "Topology-Enhanced Alignment for Large Language Models:
  Trajectory Topology Loss and Topological Preference Optimization" (arXiv,
  2026; accepted to ACL 2026 according to the arXiv page). This is adjacent
  because it constructs topic-specific semantic preference vectors and uses
  geometry during alignment. It points away from a single global scalar and
  toward topic/aspect-aware directions.
  Link: https://arxiv.org/abs/2605.07172

## Local Result In Light Of The Literature

Phase 5 fits the literature surprisingly well:

- Semantic projection is real enough to recover human-like dimensions.
- Raw semantic embeddings are not automatically preference embeddings.
- Topic/aspect-specific directions look more promising than a single global
  axis.
- Explicit process or outcome text is easier to score than final answer text.

The practical research thesis should therefore be:

> Good/bad embedding projection is not a complete reward model, but it may be a
> cheap, interpretable weak evaluator when the target concept is decomposed into
> aspect-specific axes and the text being scored exposes the relevant process or
> outcome facts.

## Next Experiments Suggested By This

- Replace the single global axis with a small evaluative basis:
  harm-reduction, truth-correction, calibration, usefulness, non-sycophancy,
  risk-disclosure, and agency/respect.
- Score process traces or generated evaluation reports, not just final answers.
- Test stronger embedding models on the same context-polarity set before
  spending quota on large HH-RLHF runs.
- Use high-confidence embedding/dataset disagreements as an audit set rather
  than assuming either HH labels or embedding scores are ground truth.
