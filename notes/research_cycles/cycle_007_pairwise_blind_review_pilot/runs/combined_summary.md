# Pairwise Blind Review Pilot

Judge model: gemini-flash-lite-latest
Max sampled rows per comparison: 10

This pilot uses blinded LLM adjudication as a sensor, not as human gold truth.

## Aggregate Results

| Focus method | Baseline | Focus wins | Baseline wins | Ties | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: |
| direct_anti_sycophancy | length | 3 | 6 | 1 | 33.3% |
| direct_anti_sycophancy | random | 3 | 5 | 2 | 37.5% |
| direct_anti_sycophancy | refusal_heuristic | 1 | 7 | 2 | 12.5% |
| direct_anti_sycophancy | sentiment | 5 | 1 | 4 | 83.3% |
| direct_category_axis | length | 1 | 8 | 1 | 11.1% |
| direct_category_axis | random | 5 | 3 | 2 | 62.5% |
| direct_category_axis | refusal_heuristic | 2 | 5 | 3 | 28.6% |
| direct_category_axis | sentiment | 8 | 1 | 1 | 88.9% |

## Interpretation Rule

Treat wins over sentiment as weak evidence. The more important comparisons are versus random, length, and refusal heuristics.
