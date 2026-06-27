# Cycle 012 Experiment

Date: June 27, 2026

## Frozen Protocol

Prompt spec:

- `notes/research_cycles/cycle_012_length_controlled_openended_pool/prompt_spec_v1.json`

Builder:

- `scripts/build_length_controlled_openended_pool.py`

Selection runner:

- `scripts/run_cycle001_intervention.py`

Blind-review infrastructure:

- `scripts/build_pairwise_review.py`
- `scripts/judge_pairwise_with_gemini.py`
- `scripts/analyze_pairwise_review.py`

## Planned Pool Design

- mixed-category open-ended prompts
- routed categories only:
  - `persona_honesty`
  - `harmful_request`
  - `anti_sycophancy`
  - `false_premise`
  - `general_helpfulness`
- 4 generated candidates per prompt
- fixed generation style bank frozen before running
- exact target word count per prompt when feasible
- pilot and holdout splits reserved in the prompt spec before reviewing results

## Baselines And Focus Methods

Baselines:

- `random`
- `length`
- `sentiment`
- `refusal_heuristic`

Initial focus method:

- `direct_category_axis`

Why this focus first:

- it is the most natural routed-selector baseline for the fresh pool
- it avoids decomposition leakage while the candidate pool itself is being
  validated
- it gives the cleanest first answer to whether the fresh pool behaves better
  than the inherited one

## Failure Criteria

- the builder cannot keep candidate word counts effectively tied
- candidates collapse into duplicates or near-duplicates too often
- the blind-review pilot produces too few real method disagreements to be
  informative
- the fresh pool still lets trivial baselines dominate for obvious non-semantic
  reasons

## Status

This cycle is exploratory.

The goal is not yet a partner-grade claim. The goal is to build the cleanest
affordable open-ended intervention surface the repo currently lacks.
