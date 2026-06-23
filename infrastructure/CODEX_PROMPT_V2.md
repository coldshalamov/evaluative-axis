# Codex Goal Prompt v2 — Gemini Rerun

## Context

This is the SECOND run of the embedding reward signal experiment. The first run used sentence-transformers/all-mpnet-base-v2 as a fallback and got 53.2% agreement on HH-RLHF — statistically significant (p < 0.000003) but below the 60% Phase 3 gate.

Read these files first:
- `RESEARCH_PLAN.md` — original plan
- `PHASE2_DIAGNOSIS.md` — detailed diagnosis of why v1 was weak and what to fix
- `CODEX_GUIDANCE.md` — axis construction strategies and Phase 2 interpretation guide
- `RESEARCH_NOTES.md` — parallel research findings

The v1 results are in phase1/ and phase2/. Do NOT overwrite them — save Gemini results to phase1_gemini/ and phase2_gemini/.

## What Changed

1. **Model**: Use Gemini Embedding 2 instead of all-mpnet-base-v2
2. **Task instructions**: Gemini Embedding 2 does NOT use `task_type`. Prepend instructions to text instead.
3. **Prompt+response scoring**: Test embedding the full "User: {prompt}\nAssistant: {response}" string, not just the response
4. **Multi-anchor sentences**: Test sentence-level anchor pairs alongside word-level anchors
5. **rDPO**: If Phase 3 runs, use loss_type="robust" with label_smoothing calibrated to agreement rate

## Gemini API Usage

```python
import google.generativeai as genai
# Try environment variable first, then Colab secret
import os
api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get('GOOGLE_API_KEY')
    except:
        pass
if not api_key:
    raise RuntimeError("No Gemini API key found. Do not fall back to all-mpnet — we already have those results.")

genai.configure(api_key=api_key)

# Embed with task instruction
result = genai.embed_content(
    model="models/gemini-embedding-exp-03-07",
    content="Evaluate whether the following text is helpful, honest, and safe: " + text_to_embed
)
embedding = result['embedding']  # 3072-dim vector
```

Check model availability — the model name may have been updated. If gemini-embedding-exp-03-07 is unavailable, try text-embedding-004 or list available models with `genai.list_models()`.

## Key Diagnostics

The single most important number from this rerun: **sycophancy category accuracy in Phase 1**. v1 got 0%. If Gemini gets >50%, the model upgrade works. If it still gets <50%, the problem is deeper than model capacity.

The second most important number: **sentiment-discordant agreement in Phase 2**. v1 got 43.8% (below random). If Gemini gets >50%, the axis IS separating quality from sentiment. If not, the hybrid approach (embedding-scored LLM judge) may be needed.

## Comparison Reporting

In every results file, include explicit comparison to v1:
- "Phase 1 sycophancy: X% (v1: 0%)"
- "Phase 2 agreement: X% (v1: 53.2%)"
- "Sentiment-discordant: X% (v1: 43.8%)"

## Execution Environment

- **Gemini API**: Free tier, 10M tokens/min. More than enough for all experiments.
- **Colab**: If Colab MCP works, run there. If not, run locally with the API key.
- **GPU**: Only needed for Phase 3 (DPO training). All embedding work is API-based.
- **Storage**: Save only results summaries locally. Embeddings can be computed on the fly.

## Start

1. Verify Gemini API key is available. If not, stop immediately.
2. Create phase1_gemini/ and phase2_gemini/ directories.
3. Run Phase 1 with both axis strategies.
4. Run Phase 2 with response-only AND prompt+response scoring.
5. Evaluate Phase 3 gate.
6. Update research_log.md after every phase.
