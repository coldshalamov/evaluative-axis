# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `BAAI/bge-base-en-v1.5`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 28.0% | -0.0276 | 50 |
| `sentence_good_bad` | 24.0% | -0.0345 | 50 |
| `raw_true_false` | 22.0% | -0.0333 | 50 |
| `raw_truthful_untruthful` | 20.0% | -0.0342 | 50 |
| `raw_honest_dishonest` | 30.0% | -0.0203 | 50 |
| `raw_useful_useless` | 26.0% | -0.0229 | 50 |
| `raw_helpful_unhelpful` | 18.0% | -0.0266 | 50 |
| `raw_accurate_inaccurate` | 14.0% | -0.0449 | 50 |
| `raw_correct_incorrect` | 20.0% | -0.0369 | 50 |
| `raw_safe_unsafe` | 24.0% | -0.0229 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 28.0%
- `sentence_good_bad`: 24.0%
- proxy mean: 21.8%
- best proxy: `raw_honest_dishonest` at 30.0%
- raw good/bad minus proxy mean: +6.3%
- sentence good/bad minus proxy mean: +2.2%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| `harm_reduction` | 57.1% | 28.6% | 57.1% | 28.6% | 42.9% |
| `helpfulness` | 20.0% | 40.0% | 60.0% | 20.0% | 20.0% |
| `mixed` | 0.0% | 25.0% | 0.0% | 0.0% | 0.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% |
| `reasoning_rigor` | 36.4% | 27.3% | 27.3% | 18.2% | 36.4% |
| `truthfulness` | 44.4% | 33.3% | 0.0% | 33.3% | 22.2% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
