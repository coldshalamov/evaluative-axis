# Good vs Proxy Conflicts

Backend: `gemini`
Model: `gemini-embedding-2`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 26.0% | -0.0216 | 50 |
| `sentence_good_bad` | 30.0% | -0.0215 | 50 |
| `raw_true_false` | 40.0% | -0.0117 | 50 |
| `raw_truthful_untruthful` | 36.0% | -0.0183 | 50 |
| `raw_honest_dishonest` | 38.0% | -0.0074 | 50 |
| `raw_useful_useless` | 42.0% | -0.0067 | 50 |
| `raw_helpful_unhelpful` | 26.0% | -0.0178 | 50 |
| `raw_accurate_inaccurate` | 36.0% | -0.0177 | 50 |
| `raw_correct_incorrect` | 24.0% | -0.0316 | 50 |
| `raw_safe_unsafe` | 36.0% | -0.0164 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 26.0%
- `sentence_good_bad`: 30.0%
- proxy mean: 34.8%
- best proxy: `raw_useful_useless` at 42.0%
- raw good/bad minus proxy mean: -8.7%
- sentence good/bad minus proxy mean: -4.7%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `context_binding` | 60.0% | 60.0% | 80.0% | 40.0% | 60.0% |
| `harm_reduction` | 14.3% | 14.3% | 42.9% | 28.6% | 71.4% |
| `helpfulness` | 20.0% | 40.0% | 40.0% | 40.0% | 40.0% |
| `mixed` | 0.0% | 25.0% | 25.0% | 25.0% | 25.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `reasoning_rigor` | 36.4% | 45.5% | 45.5% | 27.3% | 45.5% |
| `truthfulness` | 44.4% | 33.3% | 55.6% | 33.3% | 22.2% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
