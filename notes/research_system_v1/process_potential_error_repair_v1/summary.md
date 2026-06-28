# Process Potential Error-Repair V1

Backend: `gemini`
Model: `gemini-embedding-2`
Traces: 12

## Main Metric

| Scorer | Error drop | Repair rise | Error loc top1 | Repair loc top1 | Dense score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `category_axis` | 91.7% | 83.3% | 33.3% | 66.7% | 50.0% |
| `combined` | 75.0% | 75.0% | 58.3% | 66.7% | 62.5% |
| `length` | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| `sentiment` | 41.7% | 16.7% | 16.7% | 0.0% | 8.3% |
| `final_answer_only_category_axis` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `final_answer_only_combined` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Interpretation Rule

This suite is the bridge from selection to training. A strong result means the evaluative score changes at the injected error and the later repair step, not just at the final answer. A weak result means the signal may still be useful for reranking while remaining too blunt for dense process supervision.
