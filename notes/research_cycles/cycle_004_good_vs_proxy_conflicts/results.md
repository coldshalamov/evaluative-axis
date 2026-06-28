# Cycle 004 Results

Date: June 27, 2026

## Output Artifacts

Word-level runs:

- `notes/research_system_v1/good_vs_proxy_conflicts_bge_v1/summary.md`
- `notes/research_system_v1/good_vs_proxy_conflicts_gemini_v1/summary.md`

Richer-axis control on the same 50-case battery:

- `notes/research_system_v1/battery_v3_gemini_direct_v1/summary.md`

## Key Findings

### 1. Raw `good/bad` did not work on this battery

On `gemini-embedding-2`:

- `raw_good_bad`: 26.0%
- `sentence_good_bad`: 30.0%

On `BAAI/bge-base-en-v1.5`:

- `raw_good_bad`: 28.0%
- `sentence_good_bad`: 24.0%

Both are below chance on a 50-case better/worse battery.

### 2. Nearby single-word proxies also mostly failed

Gemini proxy mean:

- 34.8%

Best Gemini proxy:

- `raw_useful_useless`: 42.0%

BGE proxy mean:

- 21.8%

Best BGE proxy:

- `raw_honest_dishonest`: 30.0%

So the failure is not unique to raw `good/bad`. The entire single-word
neighborhood is weak in this direct zero-shot format.

### 3. The same battery is not impossible

Using the existing richer cycle-001 evaluative axes on the exact same 50 cases
with Gemini direct scoring:

- `direct_combined`: 86.0%
- `direct_category_axis`: 86.0%
- `direct_general_evaluative`: 46.0%
- `direct_truthfulness`: 90.0%
- `direct_harm_reduction`: 94.0%
- `direct_persona_honesty`: 96.0%
- `direct_anti_sycophancy`: 98.0%

This matters a lot. It shows the benchmark itself is not the main reason the
word-level runs failed.

## Interpretation

This is a meaningful negative result for the strongest minimalist version of the
thesis.

What failed:

- raw single-word `good/bad` as a direct zero-shot evaluative axis
- nearby single-word proxy axes in the same style
- a broad richer `general_evaluative` axis on the same battery

What did not fail:

- targeted richer evaluative axes such as harm reduction, truthfulness, persona
  honesty, and anti-sycophancy

So the clean current conclusion is:

> Under the current no-training embedding-only setup, the strong model does not
> appear to resolve these conflict cases through raw `good/bad` alone. The
> useful signal is present in richer, targeted evaluative directions, not in the
> minimalist raw-word axis.

That does not kill the broader training thesis. It does narrow it.

The repo now has evidence that:

- evaluative geometry exists and can be useful;
- the raw one-word version is much weaker than hoped;
- decomposition into targeted evaluative criteria currently works much better
  than expecting a single broad word to carry the full burden.
