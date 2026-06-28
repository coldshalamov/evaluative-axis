# Word-Stripping Ablation

Backend: `fastembed`
Model: `snowflake/snowflake-arctic-embed-m`
Battery: `controlled_evaluative_axis_battery_v3_50_cases.jsonl` (50 cases)
Evaluative words stripped: 56
Anchor words stripped: 28

## Results

| Metric                              |   Original | Eval strip |  All strip |
| ----------------------------------- | ----------: | ----------: | ----------: |
| direct_combined                     |     34.0% |     36.0% |     38.0% |
| direct_category_axis                |     50.0% |     50.0% |     54.0% |
| direct_harm_reduction               |     60.0% |     64.0% |     64.0% |
| direct_persona_honesty              |     74.0% |     72.0% |     74.0% |
| direct_anti_sycophancy              |     52.0% |     52.0% |     48.0% |
| direct_truthfulness                 |     48.0% |     54.0% |     52.0% |
| direct_general_evaluative           |     32.0% |     30.0% |     32.0% |

## Interpretation

If accuracy drops sharply under stripping, the signal depends on
evaluative word overlap rather than deeper semantic content.
If accuracy survives, the embedding captures quality structure
beyond surface-level evaluative vocabulary.

Note: local OSS models are expected to be near chance even
without stripping. This ablation is most informative when run
with a capable embedding model (e.g., gemini-embedding-2).
