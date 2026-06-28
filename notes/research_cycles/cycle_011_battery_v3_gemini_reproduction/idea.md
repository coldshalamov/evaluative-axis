# Cycle 011 Idea

Date: June 27, 2026

Cycle: `cycle_011_battery_v3_gemini_reproduction`

## Question

If we rerun the exact current 50-case length-controlled battery file with
Gemini on this workspace, do we reproduce the older `research_system_v1`
results, and does the matching raw `good/bad` companion still show the same
boundary on the very same frozen cases?

## Why This Matters

The repo already had strong Gemini battery outputs, but they were easy to treat
as inherited artifacts unless they were reproduced directly from the current
battery file.

This cycle is a cheap but important seriousness check:

- exact frozen file
- exact zero length-gap condition
- current runnable harnesses
- current saved outputs
- both the strong targeted-axis story and the weak broad-word story on the same
  cases

That makes the evidence easier to inspect and harder to dismiss as stale or
accidental.

## Falsification

Treat the current measurement story as weakened if:

- the reproduced direct-only Gemini battery drops materially below the older
  saved result;
- or the raw `good/bad` companion suddenly looks strong on the same battery.

Treat the current story as strengthened if:

- the direct routed axes reproduce strongly on the exact file;
- while the raw broad-word probes remain weak on that same exact test surface.
