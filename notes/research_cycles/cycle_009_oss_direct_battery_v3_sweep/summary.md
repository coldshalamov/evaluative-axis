# FastEmbed Model Sweep

Date: June 27, 2026

Battery: `controlled_evaluative_axis_battery_v3_50_cases.jsonl`
Interfaces: `direct`

This sweep is diagnostic rather than decisive. Its purpose is to map how
free/local embedding models behave on the same frozen battery so the repo
can separate model-capacity limits from benchmark limits.

## Key Metrics

| Model | Status | Best Direct | Best Acc | Length | Refusal | Direct Combined | Direct Category | Direct Harm | Direct Persona | Direct Anti-Syc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-small-en-v1.5` | ok | `direct_anti_sycophancy` | 62.0% | 50.0% | 57.0% | 20.0% | 30.0% | 58.0% | 42.0% | 62.0% |
| `BAAI/bge-base-en-v1.5` | ok | `direct_harm_reduction` | 50.0% | 50.0% | 57.0% | 16.0% | 32.0% | 50.0% | 34.0% | 32.0% |
| `thenlper/gte-base` | ok | `direct_harm_reduction` | 52.0% | 50.0% | 57.0% | 20.0% | 32.0% | 52.0% | 46.0% | 28.0% |
| `snowflake/snowflake-arctic-embed-m` | ok | `direct_persona_honesty` | 74.0% | 50.0% | 57.0% | 34.0% | 50.0% | 60.0% | 74.0% | 52.0% |
| `jinaai/jina-embeddings-v2-small-en` | ok | `direct_persona_honesty` | 68.0% | 50.0% | 57.0% | 28.0% | 46.0% | 64.0% | 68.0% | 36.0% |
| `jinaai/jina-embeddings-v2-base-en` | ok | `direct_harm_reduction` | 64.0% | 50.0% | 57.0% | 32.0% | 42.0% | 64.0% | 62.0% | 40.0% |
| `nomic-ai/nomic-embed-text-v1.5-Q` | ok | `direct_harm_reduction` | 52.0% | 50.0% | 57.0% | 24.0% | 28.0% | 52.0% | 42.0% | 50.0% |
| `mixedbread-ai/mxbai-embed-large-v1` | ok | `direct_persona_honesty` | 58.0% | 50.0% | 57.0% | 26.0% | 36.0% | 52.0% | 58.0% | 56.0% |

## Interpretation Rule

Treat this as a model-landscape map, not as training proof.
If direct-only local models stay near or below cheap baselines while
Gemini remains strong on the same battery, that supports a real
capability gap story rather than a benchmark-only story.

These direct-only runs avoid decomposition leakage from hand-authored
notes, so they are the cleaner evidence lane for this battery.
