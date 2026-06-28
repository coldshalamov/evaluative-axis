# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `BAAI/bge-small-en-v1.5`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 14.0% | -0.0317 | 50 |
| `sentence_good_bad` | 12.0% | -0.0386 | 50 |
| `raw_true_false` | 20.0% | -0.0298 | 50 |
| `raw_truthful_untruthful` | 18.0% | -0.0394 | 50 |
| `raw_honest_dishonest` | 18.0% | -0.0317 | 50 |
| `raw_useful_useless` | 26.0% | -0.0246 | 50 |
| `raw_helpful_unhelpful` | 18.0% | -0.0269 | 50 |
| `raw_accurate_inaccurate` | 6.0% | -0.0449 | 50 |
| `raw_correct_incorrect` | 14.0% | -0.0319 | 50 |
| `raw_safe_unsafe` | 6.0% | -0.0403 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 14.0%
- `sentence_good_bad`: 12.0%
- proxy mean: 15.8%
- best proxy: `raw_useful_useless` at 26.0%
- raw good/bad minus proxy mean: -1.7%
- sentence good/bad minus proxy mean: -3.8%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| `harm_reduction` | 42.9% | 28.6% | 28.6% | 0.0% | 14.3% |
| `helpfulness` | 20.0% | 20.0% | 20.0% | 40.0% | 0.0% |
| `mixed` | 0.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `reasoning_rigor` | 9.1% | 9.1% | 27.3% | 27.3% | 9.1% |
| `truthfulness` | 11.1% | 11.1% | 22.2% | 33.3% | 0.0% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
