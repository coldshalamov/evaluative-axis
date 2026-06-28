# Pairwise Review Analysis

Review packet: `.tmp/pairwise_bge_small_direct_anti_sycophancy_pilot/gemini_review.jsonl`
Key: `.tmp/pairwise_bge_small_direct_anti_sycophancy_pilot/review_key.json`

## Comparison Results

| Comparison | Focus wins | Baseline wins | Ties | Reject both | Incomplete | Focus win rate (decided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_anti_sycophancy vs length` | 3 | 6 | 1 | 0 | 0 | 33.3% |
| `direct_anti_sycophancy vs random` | 3 | 5 | 2 | 0 | 0 | 37.5% |
| `direct_anti_sycophancy vs refusal_heuristic` | 1 | 7 | 2 | 0 | 0 | 12.5% |
| `direct_anti_sycophancy vs sentiment` | 5 | 1 | 4 | 0 | 0 | 83.3% |

## Category Breakdown

| Comparison | Category | Focus wins | Baseline wins | Ties | Reject both | Decided | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_anti_sycophancy vs length` | `anti_sycophancy` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_anti_sycophancy vs length` | `false_premise` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_anti_sycophancy vs length` | `general_helpfulness` | 0 | 3 | 1 | 0 | 3 | 0.0% |
| `direct_anti_sycophancy vs length` | `harmful_request` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_anti_sycophancy vs length` | `persona_honesty` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_anti_sycophancy vs random` | `anti_sycophancy` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_anti_sycophancy vs random` | `false_premise` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_anti_sycophancy vs random` | `general_helpfulness` | 0 | 2 | 2 | 0 | 2 | 0.0% |
| `direct_anti_sycophancy vs random` | `harmful_request` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_anti_sycophancy vs random` | `persona_honesty` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_anti_sycophancy vs refusal_heuristic` | `anti_sycophancy` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_anti_sycophancy vs refusal_heuristic` | `false_premise` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_anti_sycophancy vs refusal_heuristic` | `general_helpfulness` | 0 | 2 | 2 | 0 | 2 | 0.0% |
| `direct_anti_sycophancy vs refusal_heuristic` | `harmful_request` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_anti_sycophancy vs sentiment` | `anti_sycophancy` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_anti_sycophancy vs sentiment` | `factuality` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_anti_sycophancy vs sentiment` | `false_premise` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_anti_sycophancy vs sentiment` | `general_helpfulness` | 0 | 1 | 1 | 0 | 1 | 0.0% |
| `direct_anti_sycophancy vs sentiment` | `harmful_request` | 1 | 0 | 1 | 0 | 1 | 100.0% |
| `direct_anti_sycophancy vs sentiment` | `persona_honesty` | 1 | 0 | 1 | 0 | 1 | 100.0% |

## Interpretation Rule

Treat incomplete reviews, ties, and reject-both cases as signal about benchmark quality, not noise to hide.
