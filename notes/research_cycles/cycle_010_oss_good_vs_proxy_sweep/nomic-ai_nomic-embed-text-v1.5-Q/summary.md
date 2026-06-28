# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `nomic-ai/nomic-embed-text-v1.5-Q`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 20.0% | -0.0229 | 50 |
| `sentence_good_bad` | 18.0% | -0.0367 | 50 |
| `raw_true_false` | 28.0% | -0.0213 | 50 |
| `raw_truthful_untruthful` | 24.0% | -0.0237 | 50 |
| `raw_honest_dishonest` | 34.0% | -0.0070 | 50 |
| `raw_useful_useless` | 24.0% | -0.0233 | 50 |
| `raw_helpful_unhelpful` | 28.0% | -0.0170 | 50 |
| `raw_accurate_inaccurate` | 16.0% | -0.0259 | 50 |
| `raw_correct_incorrect` | 20.0% | -0.0272 | 50 |
| `raw_safe_unsafe` | 28.0% | -0.0168 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 20.0%
- `sentence_good_bad`: 18.0%
- proxy mean: 25.2%
- best proxy: `raw_honest_dishonest` at 34.0%
- raw good/bad minus proxy mean: -5.2%
- sentence good/bad minus proxy mean: -7.3%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 20.0% | 0.0% | 0.0% | 40.0% | 20.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 40.0% |
| `harm_reduction` | 14.3% | 28.6% | 42.9% | 42.9% | 14.3% |
| `helpfulness` | 60.0% | 40.0% | 60.0% | 40.0% | 60.0% |
| `mixed` | 0.0% | 0.0% | 50.0% | 0.0% | 50.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 25.0% | 25.0% |
| `reasoning_rigor` | 27.3% | 36.4% | 27.3% | 27.3% | 27.3% |
| `truthfulness` | 11.1% | 0.0% | 22.2% | 22.2% | 11.1% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
