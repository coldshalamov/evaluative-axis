# Cycle 009 Results

Date: June 27, 2026

## Output Artifacts

Aggregate sweep summary:

- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/sweep_results.json`

Per-model outputs:

- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/BAAI_bge-small-en-v1.5/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/BAAI_bge-base-en-v1.5/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/thenlper_gte-base/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/snowflake_snowflake-arctic-embed-m/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/jinaai_jina-embeddings-v2-small-en/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/jinaai_jina-embeddings-v2-base-en/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/nomic-ai_nomic-embed-text-v1.5-Q/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/mixedbread-ai_mxbai-embed-large-v1/summary.md`

Gemini comparison reference:

- `notes/research_system_v1/battery_v3_gemini_direct_v1/summary.md`

## Key Findings

### 1. No local model approached Gemini on the cleaner direct-only aggregate metrics

Best local direct-only results:

- best `direct_combined`: `snowflake/snowflake-arctic-embed-m` at 34.0%
- best `direct_category_axis`: `snowflake/snowflake-arctic-embed-m` at 50.0%

Gemini on the same battery:

- `direct_combined`: 86.0%
- `direct_category_axis`: 86.0%

So the local gap is large:

- combined gap: 52 points
- category-axis gap: 36 points

### 2. The local landscape is not flat

Different local models show different narrow strengths:

- best `direct_harm_reduction`: `jinaai/jina-embeddings-v2-small-en` and
  `jinaai/jina-embeddings-v2-base-en` at 64.0%
- best `direct_persona_honesty`: `snowflake/snowflake-arctic-embed-m` at 74.0%
- best `direct_anti_sycophancy`: `BAAI/bge-small-en-v1.5` at 62.0%

That matters because the result is not "every OSS model is equally useless."
There is a real gradient. But the strengths are narrow and fragmented.

### 3. Even the best locals still do not look like general evaluative selectors

Cheap baselines on this battery are:

- `length`: 50.0%
- `refusal`: 57.0%

Against that reference:

- no local model beat `refusal` on `direct_combined`
- no local model beat `refusal` on `direct_category_axis`
- the best local `direct_general_evaluative` score was only 32.0%

So the local models may carry usable pieces of the signal, but not a strong
general answer selector under this direct-only protocol.

### 4. Gemini still dominates every targeted axis that currently matters most

Best local vs Gemini:

- `direct_truthfulness`: 48.0% local best vs 90.0% Gemini
- `direct_harm_reduction`: 64.0% local best vs 94.0% Gemini
- `direct_persona_honesty`: 74.0% local best vs 96.0% Gemini
- `direct_anti_sycophancy`: 62.0% local best vs 98.0% Gemini

That is a broad capability gap, not just a one-axis gap.

## Interpretation

This cycle sharpens the model-quality story.

What it supports:

- there is a meaningful performance gradient across free/local embedders
- some local models are clearly better than cheap BGE on selected axes
- but none of them approach Gemini on the clean direct-only battery

What it does not support:

- the crude claim that "all OSS embeddings are equally bad"
- the stronger causal claim that parameter count alone explains the gap

The honest current conclusion is:

> Better embedding families help, but the local/free models we can run on this
> laptop still behave like partial narrow evaluators rather than strong general
> evaluative selectors on the clean battery.
