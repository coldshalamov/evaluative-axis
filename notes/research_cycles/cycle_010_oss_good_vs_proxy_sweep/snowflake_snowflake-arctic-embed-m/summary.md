# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `snowflake/snowflake-arctic-embed-m`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 48.0% | -0.0095 | 50 |
| `sentence_good_bad` | 36.0% | -0.0105 | 50 |
| `raw_true_false` | 30.0% | -0.0159 | 50 |
| `raw_truthful_untruthful` | 34.0% | -0.0118 | 50 |
| `raw_honest_dishonest` | 26.0% | -0.0112 | 50 |
| `raw_useful_useless` | 38.0% | -0.0067 | 50 |
| `raw_helpful_unhelpful` | 58.0% | 0.0022 | 50 |
| `raw_accurate_inaccurate` | 22.0% | -0.0189 | 50 |
| `raw_correct_incorrect` | 26.0% | -0.0208 | 50 |
| `raw_safe_unsafe` | 36.0% | -0.0136 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 48.0%
- `sentence_good_bad`: 36.0%
- proxy mean: 33.8%
- best proxy: `raw_helpful_unhelpful` at 58.0%
- raw good/bad minus proxy mean: +14.2%
- sentence good/bad minus proxy mean: +2.2%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 20.0% | 0.0% | 0.0% | 100.0% | 20.0% |
| `context_binding` | 60.0% | 20.0% | 20.0% | 20.0% | 40.0% |
| `harm_reduction` | 57.1% | 42.9% | 14.3% | 42.9% | 42.9% |
| `helpfulness` | 100.0% | 60.0% | 60.0% | 60.0% | 60.0% |
| `mixed` | 0.0% | 50.0% | 50.0% | 75.0% | 25.0% |
| `persona_honesty` | 75.0% | 50.0% | 25.0% | 25.0% | 25.0% |
| `reasoning_rigor` | 45.5% | 36.4% | 54.5% | 63.6% | 27.3% |
| `truthfulness` | 33.3% | 33.3% | 11.1% | 66.7% | 44.4% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
