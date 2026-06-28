# Objective Text Reranking

Backend: `sentence_transformers`
Model: `BAAI/bge-base-en-v1.5`
Tasks: 8

## Method Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 37.5% | 3 / 8 | 37.5% | 37.5% | 0.625 |
| `length` | 37.5% | 3 / 8 | 37.5% | 37.5% | 0.625 |
| `direct_combined` | 37.5% | 3 / 8 | 37.5% | 37.5% | 0.625 |
| `direct_general_evaluative` | 37.5% | 3 / 8 | 37.5% | 37.5% | 0.625 |
| `direct_target_axis` | 50.0% | 4 / 8 | 50.0% | 50.0% | 0.500 |

Best direct method: `direct_target_axis` at 50.0% solve rate.

## Interpretation Rule

This suite is objective at the final metric layer because each task has an exact grading rule. Treat it as selection evidence rather than as proof of full training value.
