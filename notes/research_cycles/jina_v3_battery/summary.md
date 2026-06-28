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
| `jinaai/jina-embeddings-v3` | ok | `direct_anti_sycophancy` | 64.0% | 50.0% | 57.0% | 26.0% | 46.0% | 50.0% | 52.0% | 64.0% |

## Interpretation Rule

Treat this as a model-landscape map, not as training proof.
If direct-only local models stay near or below cheap baselines while
Gemini remains strong on the same battery, that supports a real
capability gap story rather than a benchmark-only story.

These direct-only runs avoid decomposition leakage from hand-authored
notes, so they are the cleaner evidence lane for this battery.
