# Good vs Proxy Conflicts

Backend: `fastembed`
Model: `jinaai/jina-embeddings-v2-small-en`
Cases: 50
Mean absolute length gap: 0.00 words
Max absolute length gap: 0 words

## Overall Accuracy

| Axis | Accuracy | Mean delta | n |
| --- | ---: | ---: | ---: |
| `raw_good_bad` | 20.0% | -0.0324 | 50 |
| `sentence_good_bad` | 26.0% | -0.0336 | 50 |
| `raw_true_false` | 18.0% | -0.0347 | 50 |
| `raw_truthful_untruthful` | 20.0% | -0.0311 | 50 |
| `raw_honest_dishonest` | 18.0% | -0.0274 | 50 |
| `raw_useful_useless` | 28.0% | -0.0321 | 50 |
| `raw_helpful_unhelpful` | 26.0% | -0.0236 | 50 |
| `raw_accurate_inaccurate` | 18.0% | -0.0400 | 50 |
| `raw_correct_incorrect` | 22.0% | -0.0248 | 50 |
| `raw_safe_unsafe` | 14.0% | -0.0326 | 50 |

## Good/Bad vs Proxy Summary

- `raw_good_bad`: 20.0%
- `sentence_good_bad`: 26.0%
- proxy mean: 20.5%
- best proxy: `raw_useful_useless` at 28.0%
- raw good/bad minus proxy mean: -0.5%
- sentence good/bad minus proxy mean: +5.5%

## Category Snapshot

| Category | raw_good_bad | sentence_good_bad | raw_true_false | raw_helpful_unhelpful | raw_safe_unsafe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anti_sycophancy` | 0.0% | 20.0% | 0.0% | 20.0% | 0.0% |
| `context_binding` | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| `harm_reduction` | 28.6% | 42.9% | 28.6% | 57.1% | 14.3% |
| `helpfulness` | 40.0% | 40.0% | 40.0% | 40.0% | 20.0% |
| `mixed` | 0.0% | 25.0% | 25.0% | 0.0% | 0.0% |
| `persona_honesty` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `reasoning_rigor` | 27.3% | 27.3% | 18.2% | 27.3% | 18.2% |
| `truthfulness` | 22.2% | 22.2% | 11.1% | 22.2% | 22.2% |

## Interpretation Rule

This battery is still hand-authored and does not prove training efficacy by itself. Its purpose is narrower: test whether raw `good/bad` behaves like a broader evaluative axis than nearby proxy words on conflict cases where narrow proxies can disagree.
