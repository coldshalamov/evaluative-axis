# Trajectory Potential Test

Backend: `fastembed`
Model: `BAAI/bge-small-en-v1.5`
Cases: 12
Trajectory states: 36

## Stage Accuracy

| Stage | Category Axis | Combined |
| ---: | ---: | ---: |
| 1 | 0.0% | 0.0% |
| 2 | 66.7% | 50.0% |
| 3 | 41.7% | 33.3% |

## Cumulative Potential

| Metric | Accuracy | Mean Delta |
| --- | ---: | ---: |
| Category-axis integral | 25.0% | -0.0349 |
| Combined integral | 0.0% | -0.0559 |

## Interpretation Rule

This is a mechanism probe, not training evidence. It asks whether a cumulative context state carries a better/worse potential signal before the final answer, which is the core requirement for dense process supervision.
