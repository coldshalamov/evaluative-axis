# Controlled Evaluative-Axis Battery

Backend: `fastembed`
Model: `BAAI/bge-small-en-v1.5`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Pairwise Accuracy

| Method | Accuracy | n |
| --- | ---: | ---: |
| `length` | 50.0% | 50 |
| `sentiment` | 44.0% | 50 |
| `refusal` | 57.0% | 50 |
| `direct_combined` | 20.0% | 50 |
| `direct_category_axis` | 30.0% | 50 |
| `direct_general_evaluative` | 12.0% | 50 |
| `direct_truthfulness` | 24.0% | 50 |
| `direct_harm_reduction` | 58.0% | 50 |
| `direct_persona_honesty` | 42.0% | 50 |
| `direct_anti_sycophancy` | 62.0% | 50 |

## Interpretation Rule

This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.
