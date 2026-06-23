# Open-Source Embedding Pilot Analysis

Date: June 22, 2026

## Why This Was Run

Gemini Embedding 2 remains blocked by AI Studio rate limits for the available
project. The AI Studio rate-limit page showed Gemini Embedding 2 over quota for
the selected project (`RPD 2.01K / 1K`) and displayed a rate-limit warning.
The Colab MCP connector also returned `false`, so the fallback test used local
CPU inference with open-source SentenceTransformers models.

The test was changed after reviewing the earlier failures:

- Added an `anti_sycophancy_quality` axis where calibrated truthfulness and
  good/bad decomposition are positive, while flattery, placating agreement, and
  overconfidence are negative.
- Added an `atomic_evaluation` text mode that asks the embedding model to
  represent prompt+response text through decomposed good/bad evaluative parts.

## Results

### BAAI/bge-large-en-v1.5

This model has a short context window (`max_seq_length=512`) but is stronger
than all-mpnet in common retrieval benchmarks.

- 50-pair smoke test:
  - Best variant: `atomic_evaluation__anti_sycophancy_quality`
  - Agreement: 61.0%
  - Sentiment-discordant agreement: 59.6% over 26 pairs
  - Length baseline: 42.0%
  - Sentiment baseline: 46.0%

- 200-pair confirmation:
  - Best variant: `atomic_evaluation__anti_sycophancy_quality`
  - Agreement: 52.2%
  - Sentiment-discordant agreement: 47.1% over 104 pairs
  - Length baseline: 43.5%
  - Sentiment baseline: 46.5%

Interpretation: the 50-pair result was encouraging but did not hold at 200
pairs. The decomposition-framed mode still beat length and sentiment, but the
effect was weak and not statistically reliable.

### BAAI/bge-m3

This model loaded successfully with `max_seq_length=8192`, making it the best
local test of the context-window hypothesis so far.

- 20-pair smoke test:
  - Best variant: `atomic_evaluation__anti_sycophancy_quality`
  - Agreement: 35.0%
  - Sentiment-discordant agreement: 25.0% over 8 pairs
  - Length baseline: 35.0%
  - Sentiment baseline: 55.0%

Interpretation: this tiny sample is not definitive, but it is not promising
enough to justify a long CPU-only BGE-M3 run. It does show that a larger
context window alone is not sufficient; the embedding model's training objective
and evaluative geometry matter.

### BAAI/bge-small-en-v1.5

This model was run after verifying that the currently connected Colab runtime
is CPU-only and that long pasted notebook cells are unreliable through the
browser-control fallback. The run used the stable local script path on 300
HH-RLHF pairs with the same decomposition-oriented scoring modes.

- 300-pair pilot:
  - Best variant: `atomic_evaluation__anti_sycophancy_quality`
  - Agreement: 59.2%
  - Sentiment-discordant agreement: 51.6% over 161 pairs
  - Length baseline: 43.3%
  - Sentiment baseline: 44.5%
  - z vs random: 3.18, p = 0.0015

Interpretation: the small BGE model surprisingly produced the cleanest
open-source pilot since the 50-pair BGE-large smoke test. It still misses the
60% Phase 3 gate by a hair and needs a larger confirmation run, but the fact
that `atomic_evaluation` beats response-only by 11.8 percentage points on the
same model supports the idea that the test framing matters at least as much as
the anchor words.

## Colab Status

The Colab browser path is now usable for short cells. A connected Python 3
runtime was verified, stdout was captured from a smoke cell, and dependencies
installed successfully with `pip`. The formal `colab_mcp` websocket bridge still
does not unlock notebook tools: it starts ephemeral localhost websocket ports,
but the browser tab and active MCP server can get out of sync, causing
`open_colab_browser_connection` to time out.

The current Colab runtime is CPU-only:

- `torch`: 2.11.0+cpu
- `torch.cuda.is_available()`: False
- `nvidia-smi`: unavailable

Long code-cell pastes through the browser-control fallback are unreliable
because Colab's virtual editor can leave hidden fragments in the cell. Short
cells work; for larger experiments, either create/upload an `.ipynb` file or
run the local script path until a true Colab notebook-control tool is available.

## Current Inference

The strongest current empirical pattern is:

- all-mpnet full 5000-pair Phase 2: weak but real signal at 53.2%
- Gemini Phase 1 controlled test: improved to 70.5%
- BGE-large 200-pair decomposition pilot: weak signal at 52.2%
- BGE-M3 20-pair long-context smoke: poor, not worth expanding locally
- BGE-small 300-pair decomposition pilot: 59.2%, statistically above random
  and just below the Phase 3 gate

The thesis that better embeddings should clean up the good/bad signal remains
plausible, but the tests now suggest "better" does not mean context length
alone. The next decisive test is still Gemini Embedding 2 or another
frontier-quality embedding model on prompt+response and atomic-evaluation modes
at 5000 pairs.

## Decision

Do not proceed to Phase 3 from these open-source pilots. They are useful
diagnostics, not training-quality preference sources.

## Next Step

Use a Gemini project with billing/prepay or enough daily quota, then rerun:

```powershell
& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50 --scoring-modes prompt_response_instruction,atomic_evaluation
```
