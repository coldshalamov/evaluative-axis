# Cycle 008 Results

Date: June 27, 2026

## Output Artifacts

Gemini-selector pilot:

- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/combined_summary.md`
- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_category_axis/analysis/summary.md`
- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/analysis/summary.md`

Matched cheap BGE harm-reduction comparison:

- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/analysis/summary.md`

## Key Findings

### 1. Gemini improved the open-ended blind-review lane over cheap BGE

For Gemini `direct_category_axis`:

- vs `length`: 20.0%
- vs `random`: 55.6%
- vs `refusal_heuristic`: 0.0%
- vs `sentiment`: 100.0%

For Gemini `direct_harm_reduction`:

- vs `length`: 30.0%
- vs `random`: 88.9%
- vs `refusal_heuristic`: 37.5%
- vs `sentiment`: 100.0%

Compared with the earlier cheap-BGE lane, the stronger backend is clearly more
competitive, especially on the targeted harm-reduction selector.

### 2. The old candidate pool still blocks a strong open-ended claim

Even with the stronger embedding backend:

- both focus methods still lose to `length`
- both still lose to `refusal_heuristic`

So the fair reading is not "Gemini solved open-ended selection." It is that the
stronger selector helps, but the inherited pool remains a bad surface for a
decisive claim.

### 3. The matched harm-reduction comparison shows a real backend gap

Gemini `direct_harm_reduction` vs baselines:

- `length`: 30.0%
- `random`: 88.9%
- `refusal_heuristic`: 37.5%
- `sentiment`: 100.0%

Cheap BGE `direct_harm_reduction` vs the same baselines:

- `length`: 12.5%
- `random`: 25.0%
- `refusal_heuristic`: 20.0%
- `sentiment`: 71.4%

That is not a tiny wobble. On the same inherited pool and same blind-judge
protocol, the stronger embedding family is materially better.

## Interpretation

This is a useful capability-gap result, not yet a partner-grade intervention
result.

What the cycle supports:

- open-ended blind-review performance is sensitive to the embedding backend
- targeted Gemini selectors are materially better than the cheap BGE version on
  the same inherited pool
- the harm-reduction axis is a stronger open-ended selector than the broader
  category-routed axis on this pilot

What it does not support:

- that the repo already has a decisive open-ended reranking benchmark
- that the current open-ended lane beats the strongest cheap baselines on a
  clean pool
