# Controlled Evaluative-Axis Battery

Backend: `lexical`
Model: `default`
Cases: 12
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Pairwise Accuracy

| Method | Accuracy | n |
| --- | ---: | ---: |
| `length` | 50.0% | 12 |
| `sentiment` | 50.0% | 12 |
| `refusal` | 58.3% | 12 |
| `direct_combined` | 87.5% | 12 |
| `direct_category_axis` | 62.5% | 12 |
| `decomposition_combined` | 75.0% | 12 |
| `decomposition_category_axis` | 62.5% | 12 |
| `direct_general_evaluative` | 54.2% | 12 |
| `direct_truthfulness` | 54.2% | 12 |
| `direct_harm_reduction` | 54.2% | 12 |
| `direct_persona_honesty` | 70.8% | 12 |
| `direct_anti_sycophancy` | 75.0% | 12 |
| `decomposition_anti_sycophancy` | 70.8% | 12 |
| `decomposition_general_evaluative` | 58.3% | 12 |
| `decomposition_harm_reduction` | 66.7% | 12 |
| `decomposition_persona_honesty` | 62.5% | 12 |
| `decomposition_truthfulness` | 54.2% | 12 |

## Interpretation Rule

This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.
