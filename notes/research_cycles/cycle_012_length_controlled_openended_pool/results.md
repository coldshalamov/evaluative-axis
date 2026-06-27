# Cycle 012 Results

Date: June 27, 2026

## Output Artifacts

Builder and frozen protocol:

- `scripts/build_length_controlled_openended_pool.py`
- `notes/research_cycles/cycle_012_length_controlled_openended_pool/prompt_spec_v1.json`

Generated fresh-pool snapshot:

- `notes/research_cycles/cycle_012_length_controlled_openended_pool/generated_pool_v1/dataset.json`
- `notes/research_cycles/cycle_012_length_controlled_openended_pool/generated_pool_v1/pilot_snapshot_8.json`
- `notes/research_cycles/cycle_012_length_controlled_openended_pool/generated_pool_v1/summary.md`

No-quota local selection run:

- `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/summary.md`
- `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/selections.json`

Blind-review packet on fresh disagreements:

- `notes/research_cycles/cycle_012_length_controlled_openended_pool/pairwise_snowflake_direct_category_axis/summary.md`
- `notes/research_cycles/cycle_012_length_controlled_openended_pool/pairwise_snowflake_direct_category_axis/review_packet.jsonl`

## Key Findings

### 1. The repo now has a fresh open-ended pool with strict length control

An 8-item pilot snapshot was generated successfully on the new builder:

- categories covered:
  - `persona_honesty`: 2
  - `harmful_request`: 2
  - `anti_sycophancy`: 2
  - `false_premise`: 2
- candidates per item: 4
- target words per candidate: 60
- mean within-item word-count gap: `0.00`
- max within-item word-count gap: `0`

This is the first fresh open-ended pool in the repo that removes the old
length advantage by construction rather than only diagnosing it after the fact.

### 2. The free-tier workflow needed checkpointing and resume support

The first builder attempt on `gemini-2.0-flash` was operationally impractical
because repeated HTTP 429 responses arrived immediately.

Switching generation to `gemini-flash-lite-latest` and adding:

- incremental checkpoint writes;
- resume support;
- and candidate-level regeneration on exact-count failures

made the builder usable enough to produce a real frozen pilot artifact under
the actual free-tier constraints.

That is not evidence about the thesis itself, but it is important evidence
about what the repo can realistically run on the available hardware and quota.

### 3. The fresh pool is not degenerate: local selection already disagrees with cheap baselines

Using the local `snowflake/snowflake-arctic-embed-m` direct-only selector on the
8-item snapshot, `direct_category_axis` differed from the cheap baselines on:

- 6 of 8 items versus `length`
- 7 of 8 items versus `random`
- 7 of 8 items versus `sentiment`
- 5 of 8 items versus `refusal_heuristic`

That disagreement was enough to build a real blind-review packet with:

- 25 review rows total
- 6 rows for `direct_category_axis` vs `length`
- 7 rows vs `random`
- 7 rows vs `sentiment`
- 5 rows vs `refusal_heuristic`

This matters because it means the fresh pool is already creating real selector
differences rather than collapsing into "every method picks the same answer."

### 4. Strong-model selection on the fresh pool is still operationally pending

A Gemini embedding selection run on the same 8-item snapshot was started, but
the free-tier embedding endpoint repeatedly hit HTTP 429 during the candidate
embedding stage after the anchor passes.

So the current state is:

- fresh pool exists
- local selector outputs exist
- blind-review packet exists
- strong-model selection on this pool is not yet saved

## Interpretation

This cycle is a real infrastructure and methodology improvement, not yet a
decisive intervention result.

What it supports:

- the repo now has a fresh open-ended surface that removes the known length
  confound
- the surface produces genuine method disagreements
- the builder is now much more realistic for the user's actual laptop +
  free-tier environment

What it does not support yet:

- that Gemini selection beats the baselines on the fresh pool
- that local Snowflake selection beats the baselines under blind review
- a partner-grade open-ended claim
