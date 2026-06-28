# Cycle 007 Experiment

Date: June 27, 2026

## Protocol

Starting point:

- `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.json`

Blind-review tooling:

- `scripts/build_pairwise_review.py`
- `scripts/judge_pairwise_with_gemini.py`
- `scripts/analyze_pairwise_review.py`
- `scripts/run_pairwise_blind_review_pilot.py`

Judge model used:

- `gemini-flash-lite-latest`

Focus methods piloted:

- `direct_category_axis`
- `direct_anti_sycophancy`

Baselines:

- `length`
- `random`
- `sentiment`
- `refusal_heuristic`

Sampling rule:

- build blinded head-to-head rows only where the focus method and baseline
  chose different candidates
- sample at most 10 rows per comparison
- require stable order-flip agreement from the Gemini judge, otherwise treat as
  tie

## Why This Is Cleaner

- no proxy key is used as the headline metric
- the packet is actually blind and order-randomized
- ties and judge instability are preserved instead of hidden
- the result directly tests selection quality, even if only with an LLM judge
  sensor
