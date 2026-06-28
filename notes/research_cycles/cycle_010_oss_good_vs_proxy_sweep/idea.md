# Cycle 010 Idea

Date: June 27, 2026

Cycle: `cycle_010_oss_good_vs_proxy_sweep`

## Question

Does the raw `good/bad` versus nearby proxy-word story look different on the
strongest free/local embedding models than it does on Gemini and BGE, or does
the broad-word failure mostly persist across the local model family too?

## Why This Matters

The repo already had one strong-model word-level result and one cheap-BGE
word-level result:

- Gemini raw `good/bad` failed badly on the 50-case conflict battery
- BGE raw `good/bad` also failed badly

That still left too much ambiguity. The failure could have been:

- a BGE-specific weakness;
- a Gemini-specific quirk;
- or a broader property of current direct word-level axes.

This cycle maps the local family so the claim can become more specific and more
credible.

## Falsification

Treat the broad-word failure as weakened if:

- several strong local models recover near-baseline or better raw `good/bad`
  performance;
- or raw `good/bad` consistently beats nearby proxies on the better local
  models.

Treat it as strengthened if:

- most local models still fail on raw `good/bad`;
- while only richer targeted axes remain useful on the same battery.
