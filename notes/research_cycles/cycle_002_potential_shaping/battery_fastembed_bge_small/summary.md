# Controlled Evaluative-Axis Battery

Backend: `fastembed`
Model: `BAAI/bge-small-en-v1.5`
Cases: 23
Mean absolute length gap: 3.57 words
Max absolute length gap: 12 words

## Pairwise Accuracy

| Method | Accuracy | n |
| --- | ---: | ---: |
| `length` | 89.1% | 23 |
| `sentiment` | 45.7% | 23 |
| `refusal` | 56.5% | 23 |
| `direct_combined` | 26.1% | 23 |
| `direct_category_axis` | 39.1% | 23 |
| `decomposition_combined` | 52.2% | 23 |
| `decomposition_category_axis` | 52.2% | 23 |
| `direct_general_evaluative` | 26.1% | 23 |
| `direct_truthfulness` | 34.8% | 23 |
| `direct_harm_reduction` | 43.5% | 23 |
| `direct_persona_honesty` | 39.1% | 23 |
| `direct_anti_sycophancy` | 52.2% | 23 |
| `decomposition_anti_sycophancy` | 43.5% | 23 |
| `decomposition_general_evaluative` | 39.1% | 23 |
| `decomposition_harm_reduction` | 47.8% | 23 |
| `decomposition_persona_honesty` | 43.5% | 23 |
| `decomposition_truthfulness` | 56.5% | 23 |

## Interpretation Rule

This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.
