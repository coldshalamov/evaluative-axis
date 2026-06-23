# Gemini Phase 2 Preference Prediction

Status: blocked by Gemini API quota before preference-prediction results could
be computed.

## Target

- Dataset: Anthropic HH-RLHF train split
- Target sample: 5000 preference pairs
- Planned scoring modes: response-only, prompt+response, and
  instruction-prefixed prompt+response
- Planned anchor strategies: `expanded_words` and `multi_anchor_sentences`
- Embedding model: `gemini-embedding-2`

## Completed Prerequisites

- Gemini Embedding 2 probe succeeded before quota exhaustion.
- Phase 1 Gemini axis validation completed.
- Best Phase 1 strategy: `multi_anchor_sentences`
- Phase 1 statement-pair accuracy: 70.5%
- Phase 1 sycophancy accuracy: 0.0%

## Blocker

The usable AI Studio key reached Gemini API quota exhaustion during the Phase 2
rerun attempts. Afterward, even a two-text embedding smoke test returned HTTP
429 quota exhaustion. The second existing AI Studio key is unusable because the
Gemini API returns HTTP 403 and reports that the key was leaked. Attempting to
create a fresh key in AI Studio from the browser was blocked by AI Studio with:
`Failed to generate API key, The request is suspicious. Please try again.`

## Current Phase 2 Result

- Gemini HH-RLHF pairs scored: 0 complete result sets
- Agreement rate: not computed
- Baseline comparison: not computed for Gemini
- Failure breakdown: not computed for Gemini
- Phase 3 gate: not evaluated; do not proceed to Phase 3 from Gemini results

## Recovery Path

The runner now supports local `.env.local` loading, Gemini batch embeddings,
adaptive batch splitting, Phase 1 skipping, and Phase 2 embedding checkpoints.
After adding a non-leaked Gemini API key from a project with enough quota, run:

```powershell
& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50
```

Multiple keys can be supplied as comma-separated values in `GOOGLE_API_KEYS`.
