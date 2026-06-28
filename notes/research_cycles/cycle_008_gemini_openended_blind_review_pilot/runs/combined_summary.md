# Pairwise Blind Review Pilot

Judge model: `gemini-flash-lite-latest`
Max sampled rows per comparison: 10

This pilot uses blinded LLM adjudication as a sensor, not as human gold truth.

## Aggregate Results

| Focus method | Baseline | Focus wins | Baseline wins | Ties | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_category_axis` | `length` | 2 | 8 | 0 | 20.0% |
| `direct_category_axis` | `random` | 5 | 4 | 1 | 55.6% |
| `direct_category_axis` | `refusal_heuristic` | 0 | 8 | 2 | 0.0% |
| `direct_category_axis` | `sentiment` | 10 | 0 | 0 | 100.0% |
| `direct_harm_reduction` | `length` | 3 | 7 | 0 | 30.0% |
| `direct_harm_reduction` | `random` | 8 | 1 | 1 | 88.9% |
| `direct_harm_reduction` | `refusal_heuristic` | 3 | 5 | 2 | 37.5% |
| `direct_harm_reduction` | `sentiment` | 9 | 0 | 1 | 100.0% |

## Interpretation Rule

Treat wins over sentiment as weak evidence. The more important comparisons are versus random, length, and refusal heuristics.
