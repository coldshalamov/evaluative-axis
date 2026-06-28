# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `mixedbread-ai/mxbai-embed-large-v1`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 20.0% | -0.0289 | 50 |
| `sentence_good_bad` | 20.0% | -0.0336 | 50 |
| `raw_true_false` | 24.0% | -0.0254 | 50 |
| `raw_truthful_untruthful` | 20.0% | -0.0266 | 50 |
| `raw_honest_dishonest` | 32.0% | -0.0218 | 50 |
| `raw_useful_useless` | 20.0% | -0.0255 | 50 |
| `raw_helpful_unhelpful` | 22.0% | -0.0232 | 50 |
| `raw_accurate_inaccurate` | 18.0% | -0.0327 | 50 |
| `raw_correct_incorrect` | 16.0% | -0.0298 | 50 |
| `raw_safe_unsafe` | 20.0% | -0.0292 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 20.0%
- `sentence_good_bad`: 20.0%
- proxy mean: 21.5%
- best proxy: `raw_honest_dishonest` at 32.0%
- raw good/bad minus proxy mean: -1.5%
- sentence good/bad minus proxy mean: -1.5%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| `harm_reduction` | 57.1% | 42.9% | 42.9% | 28.6% | 42.9% |
| `helpfulness` | 20.0% | 60.0% | 40.0% | 20.0% | 20.0% |
| `mixed` | 0.0% | 0.0% | 25.0% | 25.0% | 0.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% |
| `reasoning_rigor` | 9.1% | 9.1% | 18.2% | 18.2% | 18.2% |
| `truthfulness` | 33.3% | 22.2% | 33.3% | 33.3% | 22.2% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
