# Cycle 007 Results

Date: June 27, 2026

## Output Artifacts

- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/runs/combined_summary.md`
- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/runs/direct_category_axis/`
- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/runs/direct_anti_sycophancy/`

## Key Findings

### 1. The blind-review tooling now runs end-to-end on a quota-available Gemini judge path

The original judge path was blocked because `gemini-2.0-flash` was quota
exhausted for this key. Switching the blind judge to
`gemini-flash-lite-latest` made the pilot runnable.

### 2. The cheap BGE-small selectors do not beat strong cheap baselines on this open-ended pilot

For `direct_category_axis`:

- vs `length`: 1 win, 8 losses, 1 tie -> 11.1% decided win rate
- vs `random`: 5 wins, 3 losses, 2 ties -> 62.5%
- vs `refusal_heuristic`: 2 wins, 5 losses, 3 ties -> 28.6%
- vs `sentiment`: 8 wins, 1 loss, 1 tie -> 88.9%

For `direct_anti_sycophancy`:

- vs `length`: 3 wins, 6 losses, 1 tie -> 33.3%
- vs `random`: 3 wins, 5 losses, 2 ties -> 37.5%
- vs `refusal_heuristic`: 1 win, 7 losses, 2 ties -> 12.5%
- vs `sentiment`: 5 wins, 1 loss, 4 ties -> 83.3%

### 3. The result is informative even though it is not yet a decisive positive claim

This pilot uses:

- a length-biased candidate pool inherited from Cycle 001
- a blinded LLM judge as a sensor rather than human gold truth
- only cheap open-source embedding selection, not Gemini embeddings

Even with those limitations, the result is useful because it says the current
cheap BGE-small open-ended selection story is not strong enough to promote.

### 4. The open-ended negative result fits the rest of the repo's capacity story

This new pilot is consistent with the broader picture:

- cheap OSS embedders are weak in the repo's harder settings
- stronger embedding models do much better on objective reranking and targeted
  evaluative batteries
- the remaining open-ended practical claim likely depends on cleaner candidate
  pools and stronger embedders

## Interpretation

This is a valuable negative result, not a failure of the project.

What it shows:

- the blind-review tooling works
- LLM-judged pairwise review can now be run cheaply on sampled packets
- the current cheap open-source open-ended selectors do not beat length or
  refusal heuristics on this inherited pilot

What it does not show:

- that embedding selection fails in general
- that stronger embedding models would behave the same way
- that a clean length-controlled open-ended reranking test would be negative
