# Cycle 009 Idea

Date: June 27, 2026

Cycle: `cycle_009_oss_direct_battery_v3_sweep`

## Question

On the current 50-case direct-only controlled battery, do any free/local
embedding models approach Gemini, or is there a clear capability gradient with
Gemini still standing apart?

## Why This Matters

The repo already knows that cheap BGE is weak on the direct-only conflict
battery. That alone is not enough, because the failure could still be:

- a BGE-specific weakness;
- a local ONNX implementation issue;
- or evidence that every non-Gemini model is useless.

This cycle maps the local model landscape on the same battery so the project
can make a narrower, stronger claim:

- there is a gradient among cheap/free embedders;
- but none of them currently match Gemini on the clean direct-only battery.

## Falsification

Treat the capability-gap story as weakened if:

- one or more local models approach Gemini on `direct_combined` or
  `direct_category_axis`;
- or multiple local models beat the cheap baselines consistently across the
  broader direct metrics.

Treat it as strengthened if:

- the local models show only fragmented narrow strengths;
- while Gemini remains much stronger on the same battery.
