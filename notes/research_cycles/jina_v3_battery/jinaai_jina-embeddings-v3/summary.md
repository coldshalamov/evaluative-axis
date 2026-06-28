# Controlled Evaluative-Axis Battery

Backend: `fastembed`
Model: `jinaai/jina-embeddings-v3`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Pairwise Accuracy

| Method | Accuracy | n |
| --- | ---: | ---: |
| `length` | 50.0% | 50 |
| `sentiment` | 44.0% | 50 |
| `refusal` | 57.0% | 50 |
| `direct_combined` | 26.0% | 50 |
| `direct_category_axis` | 46.0% | 50 |
| `direct_general_evaluative` | 20.0% | 50 |
| `direct_truthfulness` | 38.0% | 50 |
| `direct_harm_reduction` | 50.0% | 50 |
| `direct_persona_honesty` | 52.0% | 50 |
| `direct_anti_sycophancy` | 64.0% | 50 |

## Interpretation Rule

This controlled battery is still hand-authored, but it is designed to test conceptual discrimination under tighter confound control than the 50-prompt pilot. Inspect failures before treating aggregate accuracy as evidence.
