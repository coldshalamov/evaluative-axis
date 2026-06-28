# Pairwise Blind Review Pilot

Judge model: `gemini-flash-lite-latest`
Max sampled rows per comparison: 10

This pilot uses blinded LLM adjudication as a sensor, not as human gold truth.

## Aggregate Results

| Focus method | Baseline | Focus wins | Baseline wins | Ties | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_harm_reduction` | `length` | 1 | 7 | 2 | 12.5% |
| `direct_harm_reduction` | `random` | 2 | 6 | 2 | 25.0% |
| `direct_harm_reduction` | `refusal_heuristic` | 2 | 8 | 0 | 20.0% |
| `direct_harm_reduction` | `sentiment` | 5 | 2 | 2 | 71.4% |

## Interpretation Rule

Treat wins over sentiment as weak evidence. The more important comparisons are versus random, length, and refusal heuristics.
