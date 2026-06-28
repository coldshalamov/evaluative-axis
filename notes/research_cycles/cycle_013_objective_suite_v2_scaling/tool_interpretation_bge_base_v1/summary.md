# Objective Text Reranking

Backend: `sentence_transformers`
Model: `BAAI/bge-base-en-v1.5`
Tasks: 32

## Method Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 37.5% | 12 / 32 | 37.5% | 37.5% | 0.625 |
| `length` | 50.0% | 16 / 32 | 50.0% | 50.0% | 0.500 |
| `direct_combined` | 43.8% | 14 / 32 | 43.8% | 43.8% | 0.562 |
| `direct_general_evaluative` | 43.8% | 14 / 32 | 43.8% | 43.8% | 0.562 |
| `direct_target_axis` | 40.6% | 13 / 32 | 40.6% | 40.6% | 0.594 |

Best direct method: `direct_combined` at 43.8% solve rate.

## Interpretation Rule

This suite is objective at the final metric layer because each task has an exact grading rule. Treat it as selection evidence rather than as proof of full training value.
