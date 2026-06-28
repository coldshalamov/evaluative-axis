# Cycle 004 Idea

Date: June 27, 2026

Cycle: `cycle_004_good_vs_proxy_conflicts`

## Question

Does raw `good/bad` carry a broader evaluative signal than nearby proxy words
 such as `true`, `honest`, `useful`, `helpful`, `accurate`, `correct`, and
 `safe`?

## Why This Matters

The user's core thesis is stronger than "embeddings can score some tasks."

It is:

- human judgment may compress many tradeoffs into broad `good/bad` evaluation;
- narrower words like `correct` or `useful` may be partial projections of that
  broader signal;
- therefore `good/bad` might be a better cheap reward basis than optimizing one
  narrower proxy.

The fastest zero-budget way to probe that is not training. It is a frozen
conflict battery scored by many word-level axes on the same cached embeddings.

## Falsification

Treat the cheap version of the thesis as weakened if:

- raw `good/bad` is near chance or worse on curated conflict cases;
- nearby proxy words are materially stronger than raw `good/bad`;
- stronger models do not rescue the raw-word result.

Treat it as strengthened if:

- raw `good/bad` beats most nearby proxy words on conflict cases;
- the advantage is strongest in mixed tradeoff categories;
- cheap OSS models flatten while stronger models improve.
