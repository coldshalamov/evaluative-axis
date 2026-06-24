# Controlled Evaluative-Axis Battery

Backend: `lexical`
Model: `default`
Cases: 23
Mean absolute length gap: 3.57 words
Max absolute length gap: 12 words

## Pairwise Accuracy

| Method | Accuracy | n |
| --- | ---: | ---: |
| `length` | 89.1% | 23 |
| `sentiment` | 45.7% | 23 |
| `refusal` | 56.5% | 23 |
| `direct_combined` | 56.5% | 23 |
| `direct_category_axis` | 47.8% | 23 |
| `decomposition_combined` | 65.2% | 23 |
| `decomposition_category_axis` | 45.7% | 23 |
| `direct_general_evaluative` | 41.3% | 23 |
| `direct_truthfulness` | 60.9% | 23 |
| `direct_harm_reduction` | 50.0% | 23 |
| `direct_persona_honesty` | 69.6% | 23 |
| `direct_anti_sycophancy` | 56.5% | 23 |
| `decomposition_anti_sycophancy` | 54.3% | 23 |
| `decomposition_general_evaluative` | 43.5% | 23 |
| `decomposition_harm_reduction` | 65.2% | 23 |
| `decomposition_persona_honesty` | 71.7% | 23 |
| `decomposition_truthfulness` | 65.2% | 23 |

## Interpretation Rule

This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.
