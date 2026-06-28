# Cycle 010 Results

Date: June 27, 2026

## Output Artifacts

Aggregate sweep summary:

- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/sweep_results.json`

Per-model outputs:

- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/BAAI_bge-small-en-v1.5/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/BAAI_bge-base-en-v1.5/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/thenlper_gte-base/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/snowflake_snowflake-arctic-embed-m/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/jinaai_jina-embeddings-v2-small-en/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/jinaai_jina-embeddings-v2-base-en/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/nomic-ai_nomic-embed-text-v1.5-Q/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/mixedbread-ai_mxbai-embed-large-v1/summary.md`

Gemini comparison reference:

- `notes/research_system_v1/good_vs_proxy_conflicts_gemini_v1/summary.md`

## Key Findings

### 1. The broad-word failure mostly persists across the local model family

Raw `good/bad` accuracy across the eight local models:

- 14.0%
- 16.0%
- 20.0%
- 20.0%
- 20.0%
- 22.0%
- 28.0%
- 48.0%

Sentence `This response is good/bad.` ranged from 12.0% to 36.0%.

So most of the local family still fails badly on the minimalist broad-word
setup.

### 2. Snowflake is a real partial exception, but not a broad-word win

`snowflake/snowflake-arctic-embed-m` reached:

- raw `good/bad`: 48.0%
- sentence `good/bad`: 36.0%
- best proxy `helpful/unhelpful`: 58.0%

That matters because it shows the raw-word failure is not perfectly universal.
But it still does not produce a strong broad evaluative signal:

- raw `good/bad` remains below its own best nearby proxy
- and it is still far below Gemini targeted-axis performance on the same
  battery

### 3. Nearby proxy words also remain weak or fragmented

Best proxy accuracy by model ranged from 24.0% to 58.0%.

The best proxy pair varied by model:

- `useful/useless`
- `honest/dishonest`
- `helpful/unhelpful`

There is no stable broad proxy winner across the local family.

### 4. The result now looks more nuanced and more credible

Taken together with Cycle 009:

- targeted direct axes on the same local models can hit 50-74% on selected
  dimensions
- but raw `good/bad` and its nearby proxy neighborhood usually stay far below
  that

So the emerging pattern is not:

- "everything outside Gemini is equally bad"

It is:

- broad raw-word evaluative axes are generally weak on current local models
- while richer targeted evaluative directions recover much more of the usable
  signal

## Interpretation

This cycle strengthens the narrower version of the thesis.

What it supports:

- the surrounding-word question is now grounded across a real local model
  family rather than only Gemini plus one BGE baseline
- the minimalist raw-word story is weak across most local models
- some local models can partially recover the signal, but not enough to make
  raw `good/bad` look robust

What it does not support:

- the claim that raw `good/bad` is already a generally reliable zero-shot
  answer selector
- the claim that a single nearby proxy word consistently fixes the problem
