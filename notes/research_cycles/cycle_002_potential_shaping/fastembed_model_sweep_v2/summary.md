# FastEmbed Model Sweep: Length-Balanced Battery V2

Date: June 23, 2026

Battery: `controlled_evaluative_axis_battery_v2_length_balanced.jsonl`

All candidate pairs in this battery have exact word-count ties, so the
length baseline should be 50%. The sample is small (12 pairs), so this
is a diagnostic sweep, not a publishable model ranking.

## Key Metrics

| Model | Status | Best Axis | Best Acc | Direct Combined | Direct Broad | Direct Category | Decomp Category | Anti-Syc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-small-en-v1.5` | ok | `decomposition_anti_sycophancy` | 66.7% | 8.3% | 0.0% | 25.0% | 58.3% | 66.7% |
| `BAAI/bge-base-en-v1.5` | ok | `decomposition_category_axis` | 58.3% | 8.3% | 8.3% | 41.7% | 58.3% | 33.3% |
| `thenlper/gte-base` | ok | `direct_harm_reduction` | 75.0% | 8.3% | 8.3% | 41.7% | 50.0% | 16.7% |
| `snowflake/snowflake-arctic-embed-m` | ok | `direct_persona_honesty` | 75.0% | 33.3% | 25.0% | 50.0% | 58.3% | 50.0% |
| `jinaai/jina-embeddings-v2-small-en` | ok | `decomposition_category_axis` | 91.7% | 25.0% | 16.7% | 50.0% | 91.7% | 33.3% |
| `jinaai/jina-embeddings-v2-base-en` | ok | `decomposition_harm_reduction` | 50.0% | 25.0% | 0.0% | 41.7% | 41.7% | 33.3% |
| `nomic-ai/nomic-embed-text-v1.5-Q` | ok | `decomposition_harm_reduction` | 75.0% | 16.7% | 0.0% | 25.0% | 33.3% | 33.3% |
| `mixedbread-ai/mxbai-embed-large-v1` | ok | `decomposition_anti_sycophancy` | 66.7% | 8.3% | 8.3% | 25.0% | 66.7% | 50.0% |

## Interpretation Rule

This sweep answers a narrow question: whether a stronger local ONNX
embedding model changes the result on the same exact-length battery.
If no model beats cheap baselines except on narrow axes, the next move
is trajectory-potential testing and/or Gemini, not more final-answer
BGE-style scoring.
