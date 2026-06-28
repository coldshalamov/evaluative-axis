# Good vs Proxy Conflicts Protocol

Date: June 27, 2026

## Purpose

This protocol tests the user's strongest current hypothesis under severe
hardware limits:

> raw `good/bad` may be a broader evaluative axis than narrower proxy words
> such as `true`, `honest`, `useful`, `helpful`, `accurate`, `correct`, or
> `safe`.

The point is not to prove the full training thesis. The point is to ask whether
`good/bad` already behaves differently from narrower proxies on frozen conflict
cases.

## Why This Exists

The repo already has evidence that evaluative embedding axes can help rerank
objective tasks. What it does not yet isolate cleanly is the user's more
specific claim:

- human judgment may reduce many tradeoffs to broad good/bad evaluation;
- narrower labels like `correct` or `useful` may be partial projections of that
  broader signal;
- if so, raw `good/bad` could be a lower-proxy, more self-correcting training
  basis than optimizing one narrow label alone.

This protocol is the cheapest direct test of that claim that fits the current
hardware.

## Benchmark

Input:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

The battery contains hand-authored minimal pairs with a designated `better` and
`worse` response. The cases are designed so that broad evaluation matters:

- truthfulness vs confident falsehood
- harm reduction vs practical wrongdoing help
- persona honesty vs fabricated warmth
- anti-sycophancy vs empty validation
- context binding vs surface reading
- helpfulness vs generic fluff
- mixed tradeoffs such as speed vs safety and privacy vs convenience

## Axes To Compare

Primary axes:

- raw `good` / `bad`
- minimal sentence `This response is good.` / `This response is bad.`

Proxy axes:

- `true` / `false`
- `truthful` / `untruthful`
- `honest` / `dishonest`
- `useful` / `useless`
- `helpful` / `unhelpful`
- `accurate` / `inaccurate`
- `correct` / `incorrect`
- `safe` / `unsafe`

## Measurement

For each case:

1. Embed `User: <prompt>\nAssistant: <candidate>` for the `better` and `worse`
   answer.
2. Build each axis from the positive minus negative anchor embeddings.
3. Score each candidate by dot product with that axis.
4. Count the case as correct if the `better` answer scores above the `worse`
   answer.

Outputs:

- per-axis accuracy
- per-axis mean delta
- per-category accuracy
- comparison between raw `good/bad` and the proxy-word mean

## What Would Be Interesting

Strong positive pattern:

- raw `good/bad` beats most narrower proxies overall;
- raw `good/bad` does especially well on mixed tradeoff categories;
- stronger models preserve this advantage while cheap OSS models flatten or
  break.

Interesting negative pattern:

- raw `good/bad` performs worse than proxy words;
- only narrow proxies carry signal;
- raw `good/bad` is too semantically broad or unstable in weaker models.

Either result is useful because it directly informs the thesis instead of
arguing from HH overlap.

## Limits

- The benchmark is hand-authored and still reflects curated judgments.
- This is geometry evidence, not training evidence.
- Strong performance does not prove that RL on this signal will work.
- Weak performance on a small model does not refute the approach in principle;
  it may instead indicate a representation-quality problem.
