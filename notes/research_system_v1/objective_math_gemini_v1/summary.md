# Objective Text Reranking

Backend: `gemini`
Model: `gemini-embedding-2`
Tasks: 8

## Method Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 37.5% | 3 / 8 | 37.5% | 37.5% | 0.625 |
| `length` | 50.0% | 4 / 8 | 50.0% | 50.0% | 0.500 |
| `direct_combined` | 100.0% | 8 / 8 | 100.0% | 100.0% | 0.000 |
| `direct_general_evaluative` | 100.0% | 8 / 8 | 100.0% | 100.0% | 0.000 |
| `direct_target_axis` | 75.0% | 6 / 8 | 75.0% | 75.0% | 0.250 |

Best direct method: `direct_combined` at 100.0% solve rate.

## Interpretation Rule

This suite is objective at the final metric layer because each task has an exact grading rule. Treat it as selection evidence rather than as proof of full training value.
