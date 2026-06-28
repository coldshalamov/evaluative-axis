# Cycle 013 Idea

Date: June 27, 2026

## Question

How do we make the objective reranking lane serious enough that a positive
cross-domain result would actually mean something under laptop-only
constraints?

## Motivation

The current v1 objective math and tool-interpretation suites are useful, but
they are still only 8 tasks each. That is enough for a pilot or a scaffolding
check, not enough for a partner-grade claim.

The right response is not to abandon the objective lane. It is to scale it in
the direction that preserves its best property:

- exact final grading;
- no blind human reviewer needed;
- no LLM judge needed;
- cheap reruns across multiple embedding families.

## Proposal

Build frozen v2 suites that are:

1. materially larger than the v1 objective suites;
2. low in length confounding;
3. randomized in candidate order so `C1` does not always mean `correct`;
4. deterministic and reproducible from a saved builder script.

Then use those suites as the new capacity-ladder surface for cheap OSS
embedders first, followed by stronger OSS or Gemini runs when quota allows.
