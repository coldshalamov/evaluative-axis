# Phase 5 Verification Probe

Run timestamp: 2026-06-22T18:04:18.546825
Embedding model: `BAAI/bge-small-en-v1.5`

## Key Findings

- Generic good/bad axes did not bind ordinary context reliably. On the same 56 paired items, broad axes scored only 32.1% to 42.9% in `bare_context` mode.
- A more specific `contextual_harm_reduction` axis rescued the non-oracle context test to 64.3% (z=2.14, p=0.0325), with strongest categories on bad-act refusal (100.0%), sycophancy resistance (87.5%), falsehood correction (75.0%), and bad-act condemnation (75.0%).
- The same context-specific axis reached 55.8% on 500 HH-RLHF pairs, beating random, length, and sentiment. It also reached 57.0% on sentiment-discordant pairs, which is the most important anti-surface-tone check in this run.
- Explicit `oracle_decomposition` text scored 100.0% across all axes. This is not an independent evaluator; it is an upper-bound/process-scratchpad probe showing the embedding axis can read the good/bad decomposition when the decomposition is made visible.

## Context Polarity Test

Items: 56 paired contexts. Each pair contains the same local bad phrase, but one context refuses/corrects/discloses/prevents the bad thing while the other endorses/hides/enables it.

### bare_context

- `contextual_harm_reduction`: context accuracy 64.3%; phrase rescue 57.1%; mean gap 0.0915; z=2.14, p=0.0325
- `anti_sycophancy_quality`: context accuracy 42.9%; phrase rescue 51.8%; mean gap -0.0233; z=-1.07, p=0.285
- `conceptual_good_bad`: context accuracy 33.9%; phrase rescue 55.4%; mean gap -0.0674; z=-2.41, p=0.0162
- `multi_anchor_sentences`: context accuracy 32.1%; phrase rescue 62.5%; mean gap -0.0672; z=-2.67, p=0.00753

### evaluation_wrapper

- `contextual_harm_reduction`: context accuracy 58.9%; phrase rescue 42.9%; mean gap 0.0213; z=1.34, p=0.181
- `anti_sycophancy_quality`: context accuracy 39.3%; phrase rescue 83.9%; mean gap -0.0075; z=-1.60, p=0.109
- `multi_anchor_sentences`: context accuracy 35.7%; phrase rescue 71.4%; mean gap -0.0172; z=-2.14, p=0.0325
- `conceptual_good_bad`: context accuracy 32.1%; phrase rescue 66.1%; mean gap -0.0147; z=-2.67, p=0.00753

### oracle_decomposition

- `multi_anchor_sentences`: context accuracy 100.0%; phrase rescue 69.6%; mean gap 0.0718; z=7.48, p=7.25e-14
- `anti_sycophancy_quality`: context accuracy 100.0%; phrase rescue 85.7%; mean gap 0.0738; z=7.48, p=7.25e-14
- `conceptual_good_bad`: context accuracy 100.0%; phrase rescue 75.0%; mean gap 0.0834; z=7.48, p=7.25e-14
- `contextual_harm_reduction`: context accuracy 100.0%; phrase rescue 69.6%; mean gap 0.0418; z=7.48, p=7.25e-14

Best non-oracle context variant: `bare_context__contextual_harm_reduction` at 64.3%.
Oracle decomposition upper bound: `oracle_decomposition__multi_anchor_sentences` at 100.0%.

## HH-RLHF Reranking Probe

Sample size: 500 HH-RLHF train pairs.

Baselines:
- Random: 50.0%
- Length, prefer longer response: 41.3%
- Sentiment, prefer more positive response: 46.9%

Variants:
- `prompt_response__contextual_harm_reduction`: agreement 55.8%; sentiment-discordant 57.0% (n=256); length-discordant 53.1% (n=290); z=2.59, p=0.00949
- `response_only__contextual_harm_reduction`: agreement 55.6%; sentiment-discordant 60.2% (n=256); length-discordant 55.2% (n=290); z=2.50, p=0.0123
- `process_wrapper__anti_sycophancy_quality`: agreement 55.3%; sentiment-discordant 50.8% (n=256); length-discordant 65.0% (n=290); z=2.37, p=0.0178
- `process_wrapper__conceptual_good_bad`: agreement 54.9%; sentiment-discordant 50.4% (n=256); length-discordant 63.6% (n=290); z=2.19, p=0.0284
- `process_wrapper__multi_anchor_sentences`: agreement 53.9%; sentiment-discordant 46.9% (n=256); length-discordant 60.9% (n=290); z=1.74, p=0.0811
- `prompt_response__anti_sycophancy_quality`: agreement 51.0%; sentiment-discordant 41.0% (n=256); length-discordant 47.9% (n=290); z=0.45, p=0.655
- `prompt_response__multi_anchor_sentences`: agreement 50.6%; sentiment-discordant 38.3% (n=256); length-discordant 49.7% (n=290); z=0.27, p=0.788
- `response_only__anti_sycophancy_quality`: agreement 49.8%; sentiment-discordant 35.5% (n=256); length-discordant 50.0% (n=290); z=-0.09, p=0.929
- `process_wrapper__contextual_harm_reduction`: agreement 49.5%; sentiment-discordant 48.4% (n=256); length-discordant 36.0% (n=290); z=-0.22, p=0.823
- `response_only__conceptual_good_bad`: agreement 49.0%; sentiment-discordant 35.9% (n=256); length-discordant 49.3% (n=290); z=-0.45, p=0.655
- `prompt_response__conceptual_good_bad`: agreement 49.0%; sentiment-discordant 37.5% (n=256); length-discordant 45.2% (n=290); z=-0.45, p=0.655
- `response_only__multi_anchor_sentences`: agreement 47.6%; sentiment-discordant 32.8% (n=256); length-discordant 47.9% (n=290); z=-1.07, p=0.283

Best HH variant: `prompt_response__contextual_harm_reduction` at 55.8%.

## Interpretation

This phase separates two hypotheses that were previously tangled together:

1. Does an embedding good/bad axis contain a context-sensitive evaluative direction at all?
2. Does one embedding of the final assistant response recover HH-RLHF labels well enough to act as a reward source?

A high context-polarity score with only moderate HH agreement would support the narrower thesis that the signal exists, while showing that raw final-text scoring is the wrong measurement interface. A low context-polarity score would weaken the core idea much more directly.

The `oracle_decomposition` mode is intentionally labeled as an upper-bound/process-scratchpad probe: it asks whether the axis can read explicit good/bad decomposition when the relevant concepts are made visible in text.

## Research Implication

The result supports a narrower, stronger version of the thesis: evaluative directions are present in embedding space, but a single broad "good - bad" direction is too blunt. The cleaner formulation is likely a small basis of good/bad axes such as harm-reduction, truth-correction, calibration, usefulness, agency, non-sycophancy, and risk disclosure. A reward system could then score decomposed process traces or outcome descriptions across those axes, rather than scoring only the final answer text with one scalar projection.
