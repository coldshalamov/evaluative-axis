# Cycle 011 Results

Date: June 27, 2026

## Output Artifacts

Direct routed battery:

- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/battery_direct_only/summary.md`
- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/battery_direct_only/summary.json`
- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/battery_direct_only/results.json`

Raw `good/bad` companion:

- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/good_vs_proxy/summary.md`
- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/good_vs_proxy/summary.json`
- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/good_vs_proxy/results.json`

Older comparison reference:

- `notes/research_system_v1/battery_v3_gemini_direct_v1/summary.md`
- `notes/research_system_v1/good_vs_proxy_conflicts_gemini_v1/summary.md`

## Key Findings

### 1. The direct routed Gemini battery reproduced exactly

The rerun matched the older saved direct-only battery result:

- `length`: 50.0%
- `sentiment`: 44.0%
- `refusal`: 57.0%
- `direct_combined`: 86.0%
- `direct_category_axis`: 86.0%
- `direct_general_evaluative`: 46.0%
- `direct_truthfulness`: 90.0%
- `direct_harm_reduction`: 94.0%
- `direct_persona_honesty`: 96.0%
- `direct_anti_sycophancy`: 98.0%

So the strongest clean measurement result on this battery is not a one-off.
It reproduces on the current frozen file under the current harness.

### 2. The misses are structured, not random

`direct_combined` missed 7 cases, concentrated in:

- `helpfulness`: 3
- `reasoning_rigor`: 2
- `harm_reduction`: 1
- `persona_honesty`: 1

`direct_category_axis` also missed 7 cases, concentrated in:

- `helpfulness`: 4
- `mixed`: 2
- `truthfulness`: 1

`direct_general_evaluative` failed badly at 46.0% and missed 27 cases total.
Its failures span most categories and include total collapse on:

- `anti_sycophancy`: 0.0%
- `persona_honesty`: 0.0%

That is an important boundary: the strong result is not "broad good/bad works
raw." The strong result is that routed targeted evaluative axes work well on
this exact-length battery.

### 3. The raw broad-word companion stayed weak on the same frozen cases

The companion run also reproduced the older broad-word limitation:

- `raw_good_bad`: 26.0%
- `sentence_good_bad`: 30.0%
- proxy mean: 34.8%
- best proxy: `raw_useful_useless` at 42.0%
- `raw_good_bad` minus proxy mean: -8.7%

So on this battery:

- raw `good/bad` is not the winning zero-shot axis
- nearby single-word proxies do not rescue it cleanly either

### 4. The raw broad-word failures are especially severe on some categories

For `raw_good_bad`:

- `anti_sycophancy`: 0.0%
- `persona_honesty`: 0.0%
- `mixed`: 0.0%
- `harm_reduction`: 14.3%

The only category where broad words looked even partially usable was
`context_binding` at 60.0%.

This sharpens the current interpretation:

- richer targeted evaluative geometry is doing the useful work
- the bare broad-word interface is still too weak and underspecified in
  zero-shot form

## Interpretation

This cycle strengthens the repo's current measurement story by reproduction,
not by novelty.

What it supports:

- the strong Gemini routed-axis result on the v3 battery is stable and
  reproducible
- the battery really is length-controlled in the saved artifact, not just in
  description
- the split between targeted evaluative geometry and raw `good/bad` remains
  sharp on the exact same cases

What it does not support:

- a decisive intervention claim
- a training-readiness claim
- the claim that raw `good/bad` already works as a generally reliable zero-shot
  selector
