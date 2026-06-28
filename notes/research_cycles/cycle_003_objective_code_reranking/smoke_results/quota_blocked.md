# Generated-Mode Smoke Attempt Blocked

Date: June 25, 2026

## What Was Attempted

A generated-candidate smoke run of `scripts/run_objective_code_reranking.py`
was attempted so the full direct-vs-critique protocol could run without manual
candidate curation.

## What Happened

- Gemini embeddings succeeded.
- Candidate generation hit repeated Gemini HTTP 429 responses with long
  retry delays.
- `mcp__colab_mcp.open_colab_browser_connection` also returned `false`, so the
  run could not be moved into a connected Colab session.

## Fallback Chosen

Use curated local candidate sets for the first objective pilot and keep the
critique lane queued for a later rerun when generation quota or Colab access is
available.

## Why This Note Exists

Without this note, the `smoke_results` directory could look like an abandoned
partial run instead of a quota-blocked attempt with a documented fallback.
