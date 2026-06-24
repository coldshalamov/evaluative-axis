# Controlled Evaluative-Axis Battery

Backend: `fastembed`
Model: `mixedbread-ai/mxbai-embed-large-v1`
Cases: 12
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Pairwise Accuracy

| Method | Accuracy | n |
| --- | ---: | ---: |
| `length` | 50.0% | 12 |
| `sentiment` | 50.0% | 12 |
| `refusal` | 58.3% | 12 |
| `direct_combined` | 8.3% | 12 |
| `direct_category_axis` | 25.0% | 12 |
| `decomposition_combined` | 50.0% | 12 |
| `decomposition_category_axis` | 66.7% | 12 |
| `direct_general_evaluative` | 8.3% | 12 |
| `direct_truthfulness` | 8.3% | 12 |
| `direct_harm_reduction` | 50.0% | 12 |
| `direct_persona_honesty` | 50.0% | 12 |
| `direct_anti_sycophancy` | 50.0% | 12 |
| `decomposition_anti_sycophancy` | 66.7% | 12 |
| `decomposition_general_evaluative` | 33.3% | 12 |
| `decomposition_harm_reduction` | 58.3% | 12 |
| `decomposition_persona_honesty` | 66.7% | 12 |
| `decomposition_truthfulness` | 50.0% | 12 |

## Interpretation Rule

This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.
