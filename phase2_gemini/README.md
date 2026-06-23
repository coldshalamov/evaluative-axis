# Gemini Phase 2 Rerun

This directory is reserved for the v2 Gemini rerun artifacts.

Run from the workspace root after adding a Gemini API key as either:

- local environment variable `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Colab Secret `GOOGLE_API_KEY` or `GEMINI_API_KEY`

Command:

```bash
python scripts/run_gemini_rerun.py --sample-size 5000
```

The runner intentionally refuses to fall back to all-mpnet. It uses the current
Gemini embedding model candidate `gemini-embedding-2` first, then falls back
only to other Gemini API embedding models if needed.
