# Objective Text Reranking

Backend: `sentence_transformers`
Model: `sentence-transformers/all-mpnet-base-v2`
Tasks: 32

## Method Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 37.5% | 12 / 32 | 37.5% | 37.5% | 0.625 |
| `length` | 50.0% | 16 / 32 | 50.0% | 50.0% | 0.500 |
| `direct_combined` | 18.8% | 6 / 32 | 18.8% | 18.8% | 0.812 |
| `direct_general_evaluative` | 28.1% | 9 / 32 | 28.1% | 28.1% | 0.719 |
| `direct_target_axis` | 18.8% | 6 / 32 | 18.8% | 18.8% | 0.812 |

Best direct method: `direct_general_evaluative` at 28.1% solve rate.

## Interpretation Rule

This suite is objective at the final metric layer because each task has an exact grading rule. Treat it as selection evidence rather than as proof of full training value.
