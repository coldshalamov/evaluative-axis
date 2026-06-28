# Cycle 007 Idea

Date: June 27, 2026

Cycle: `cycle_007_pairwise_blind_review_pilot`

## Question

If we stop scoring against a proxy answer key and instead blind-judge method
selected winners head-to-head, do the cheap open-source embedding selectors
still look good?

## Why This Matters

The repo already knew the old 50-prompt intervention pilot was contaminated by
length. But that does not tell us how the embedding-selected outputs actually
look under blind review.

This cycle tests a better question:

- when a cheap embedding selector and a cheap baseline choose different answers
  from the same candidate pool,
- which one looks better under blinded pairwise adjudication?

## Falsification

Treat the cheap open-ended selection story as weakened if:

- the embedding selector loses to length or refusal heuristics under blind
  review;
- wins are concentrated only against sentiment or random;
- or order-flip instability is so high that the LLM judge is not usable even as
  a sensor.
