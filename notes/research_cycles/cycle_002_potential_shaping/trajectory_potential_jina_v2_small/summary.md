# Trajectory Potential Test

Backend: `fastembed`
Model: `jinaai/jina-embeddings-v2-small-en`
Cases: 12
Trajectory states: 36

## Stage Accuracy

| Stage | Category Axis | Combined |
| ---: | ---: | ---: |
| 1 | 0.0% | 0.0% |
| 2 | 33.3% | 8.3% |
| 3 | 41.7% | 16.7% |

## Cumulative Potential

| Metric | Accuracy | Mean Delta |
| --- | ---: | ---: |
| Category-axis integral | 33.3% | -0.0366 |
| Combined integral | 0.0% | -0.0596 |

## Interpretation Rule

This is a mechanism probe, not training evidence. It asks whether a cumulative context state carries a better/worse potential signal before the final answer, which is the core requirement for dense process supervision.
