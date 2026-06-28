# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `jinaai/jina-embeddings-v2-base-en`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 16.0% | -0.0296 | 50 |
| `sentence_good_bad` | 18.0% | -0.0325 | 50 |
| `raw_true_false` | 26.0% | -0.0171 | 50 |
| `raw_truthful_untruthful` | 8.0% | -0.0409 | 50 |
| `raw_honest_dishonest` | 14.0% | -0.0276 | 50 |
| `raw_useful_useless` | 24.0% | -0.0287 | 50 |
| `raw_helpful_unhelpful` | 32.0% | -0.0284 | 50 |
| `raw_accurate_inaccurate` | 20.0% | -0.0389 | 50 |
| `raw_correct_incorrect` | 20.0% | -0.0278 | 50 |
| `raw_safe_unsafe` | 24.0% | -0.0250 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 16.0%
- `sentence_good_bad`: 18.0%
- proxy mean: 21.0%
- best proxy: `raw_helpful_unhelpful` at 32.0%
- raw good/bad minus proxy mean: -5.0%
- sentence good/bad minus proxy mean: -3.0%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 0.0% | 40.0% | 20.0% | 0.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| `harm_reduction` | 14.3% | 28.6% | 57.1% | 57.1% | 14.3% |
| `helpfulness` | 20.0% | 20.0% | 0.0% | 40.0% | 20.0% |
| `mixed` | 0.0% | 25.0% | 25.0% | 0.0% | 0.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% |
| `reasoning_rigor` | 27.3% | 27.3% | 27.3% | 36.4% | 36.4% |
| `truthfulness` | 22.2% | 11.1% | 22.2% | 44.4% | 44.4% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
