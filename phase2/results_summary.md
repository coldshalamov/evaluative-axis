# Phase 2 Preference Prediction Results

Run timestamp: 2026-06-21T04:55:41.872782
Embedding model: `sentence-transformers:sentence-transformers/all-mpnet-base-v2`
Dataset: Anthropic HH-RLHF, first 5000 streamed train pairs

## Baselines

- Random theoretical: 50.0%
- Length baseline, prefer longer final assistant turn: 43.2%
- VADER sentiment baseline, prefer more positive final assistant turn: 48.3%

## Anchor Strategy Comparison

- expanded_words: agreement 53.2%; mean gap 0.00731; sentiment-discordant agreement 43.8% over 2452 pairs; top-half confidence agreement 55.3%
- minimal_good_bad: agreement 51.2%; mean gap 0.00177; sentiment-discordant agreement 43.7% over 2452 pairs; top-half confidence agreement 51.2%
- sentence_anchors: agreement 50.5%; mean gap 0.00170; sentiment-discordant agreement 44.7% over 2452 pairs; top-half confidence agreement 50.8%
- task_specific: agreement 47.5%; mean gap -0.00353; sentiment-discordant agreement 38.0% over 2452 pairs; top-half confidence agreement 46.7%

## Best Strategy

Best strategy: `expanded_words` with 53.2% agreement.
Top-quartile confidence agreement: 55.9%.
Sentiment-discordant subset: 43.8% agreement over 2452 pairs.

## Failure Breakdown

- genuine_or_label_disagreement: 256 (10.9% of failures)
- length_bias: 848 (36.2% of failures)
- low_confidence: 608 (26.0% of failures)
- positive_tone_bias: 512 (21.9% of failures)
- topic_context_limit: 116 (5.0% of failures)

## Decision

Decision: **do_not_proceed_to_phase3**.

Criterion: Phase 3 requires agreement above 60% and better than both length and sentiment baselines.
