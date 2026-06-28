# Cycle 008 Experiment

Date: June 27, 2026

## Protocol

Candidate-generation and selection source:

- `notes/research_cycles/cycle_001_next/pilot_50_candidates.json`

Gemini intervention runner:

- `scripts/run_cycle001_intervention.py`

Blind-review runner:

- `scripts/run_pairwise_blind_review_pilot.py`

Gemini selection run:

- backend: `gemini`
- model: `gemini-embedding-2`
- interfaces: `direct`
- output: `.tmp/cycle001_gemini_direct_openended_pilot/`

Blind judge:

- model: `gemini-flash-lite-latest`
- order-flip stability required, else tie
- max sampled rows per comparison: `10`

Focus methods:

- `direct_category_axis`
- `direct_harm_reduction`

Baselines:

- `length`
- `random`
- `sentiment`
- `refusal_heuristic`

Matched cheap comparison added:

- `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.json`
- focus method: `direct_harm_reduction`
- same blind-judge protocol

## Why This Is Cleaner

- it upgrades the old open-ended lane from proxy-key overlap to blinded review
- it keeps the candidate pool fixed so backend differences are interpretable
- it adds a matched BGE-vs-Gemini comparison for the same focus method

## Limits

- the inherited Cycle 001 pool is still not the clean length-controlled
  open-ended benchmark the repo ultimately wants
- blinded Gemini adjudication is a sensor, not human gold truth
