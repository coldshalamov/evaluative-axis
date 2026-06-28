# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `thenlper/gte-base`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 22.0% | -0.0200 | 50 |
| `sentence_good_bad` | 20.0% | -0.0169 | 50 |
| `raw_true_false` | 20.0% | -0.0160 | 50 |
| `raw_truthful_untruthful` | 12.0% | -0.0187 | 50 |
| `raw_honest_dishonest` | 24.0% | -0.0131 | 50 |
| `raw_useful_useless` | 24.0% | -0.0178 | 50 |
| `raw_helpful_unhelpful` | 14.0% | -0.0164 | 50 |
| `raw_accurate_inaccurate` | 10.0% | -0.0232 | 50 |
| `raw_correct_incorrect` | 18.0% | -0.0186 | 50 |
| `raw_safe_unsafe` | 16.0% | -0.0184 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 22.0%
- `sentence_good_bad`: 20.0%
- proxy mean: 17.2%
- best proxy: `raw_honest_dishonest` at 24.0%
- raw good/bad minus proxy mean: +4.8%
- sentence good/bad minus proxy mean: +2.8%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| `harm_reduction` | 28.6% | 14.3% | 28.6% | 14.3% | 14.3% |
| `helpfulness` | 40.0% | 40.0% | 60.0% | 20.0% | 40.0% |
| `mixed` | 25.0% | 50.0% | 25.0% | 0.0% | 0.0% |
| `persona_honesty` | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `reasoning_rigor` | 27.3% | 18.2% | 9.1% | 9.1% | 27.3% |
| `truthfulness` | 11.1% | 22.2% | 22.2% | 33.3% | 11.1% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
