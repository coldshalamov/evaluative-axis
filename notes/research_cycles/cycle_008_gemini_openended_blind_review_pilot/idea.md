# Cycle 008 Idea

Date: June 27, 2026

Cycle: `cycle_008_gemini_openended_blind_review_pilot`

## Question

If the same open-ended blind-review pilot that looked weak with cheap BGE
selections is rerun with stronger Gemini embedding selections, does the
open-ended lane materially improve?

## Why This Matters

Cycle 007 showed that cheap OSS open-ended selectors do not beat the stronger
cheap baselines on the inherited Cycle 001 candidate pool.

That still leaves two live possibilities:

- the open-ended task itself is too messy or length-biased to support a clean
  claim right now;
- or the cheap embedding family is the main bottleneck.

This cycle probes that distinction by holding the old candidate pool and blind
review machinery fixed while swapping in the stronger embedding backend.

## Falsification

Treat the stronger-backend story as weakened if:

- Gemini-selected winners do not improve over the cheap BGE pilot;
- or Gemini still cannot beat random on the same focus methods.

Treat it as strengthened if:

- Gemini materially improves the blind-review win rates over cheap BGE on the
  same inherited pool;
- especially on the more targeted direct methods such as harm reduction;
- while still preserving the honest limitation that the old pool remains
  length-biased.
