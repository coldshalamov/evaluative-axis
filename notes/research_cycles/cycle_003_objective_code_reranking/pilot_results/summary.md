# Objective Code Reranking Pilot

Generation model: `gemini-2.0-flash`
Embedding model: `gemini-embedding-2`
Tasks: 6
Candidates per task: 3

## Method Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 50.0% | 3 / 6 | 83.3% | 50.0% | 0.167 |
| `length` | 50.0% | 3 / 6 | 83.3% | 50.0% | 0.167 |
| `direct_broad` | 83.3% | 5 / 6 | 96.7% | 83.3% | 0.033 |
| `direct_code` | 83.3% | 5 / 6 | 96.7% | 83.3% | 0.033 |

## Interpretation Rule

This pilot is objective at the final metric layer because hidden unit tests decide whether the selected candidate is actually good. It is still a small pilot and should be read as directional evidence, not a publishable final benchmark.
