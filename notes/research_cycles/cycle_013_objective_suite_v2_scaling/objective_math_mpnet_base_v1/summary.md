# Objective Text Reranking

Backend: `sentence_transformers`
Model: `sentence-transformers/all-mpnet-base-v2`
Tasks: 48

## Method Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 47.9% | 23 / 48 | 47.9% | 50.0% | 0.500 |
| `length` | 35.4% | 17 / 48 | 35.4% | 37.5% | 0.625 |
| `direct_combined` | 35.4% | 17 / 48 | 35.4% | 37.5% | 0.625 |
| `direct_general_evaluative` | 22.9% | 11 / 48 | 22.9% | 25.0% | 0.750 |
| `direct_target_axis` | 29.2% | 14 / 48 | 29.2% | 31.2% | 0.688 |

Best direct method: `direct_combined` at 35.4% solve rate.

## Interpretation Rule

This suite is objective at the final metric layer because each task has an exact grading rule. Treat it as selection evidence rather than as proof of full training value.
