# Pairwise Review Analysis

Review packet: `.tmp/pairwise_bge_small_direct_category_axis_pilot/gemini_review.jsonl`
Key: `.tmp/pairwise_bge_small_direct_category_axis_pilot/review_key.json`

## Comparison Results

| Comparison | Focus wins | Baseline wins | Ties | Reject both | Incomplete | Focus win rate (decided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_category_axis vs length` | 1 | 8 | 1 | 0 | 0 | 11.1% |
| `direct_category_axis vs random` | 5 | 3 | 2 | 0 | 0 | 62.5% |
| `direct_category_axis vs refusal_heuristic` | 2 | 5 | 3 | 0 | 0 | 28.6% |
| `direct_category_axis vs sentiment` | 8 | 1 | 1 | 0 | 0 | 88.9% |

## Category Breakdown

| Comparison | Category | Focus wins | Baseline wins | Ties | Reject both | Decided | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_category_axis vs length` | `anti_sycophancy` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs length` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs length` | `factuality` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs length` | `false_premise` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_category_axis vs length` | `general_helpfulness` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_category_axis vs length` | `harmful_request` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_category_axis vs length` | `persona_honesty` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs random` | `anti_sycophancy` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_category_axis vs random` | `factuality` | 1 | 0 | 1 | 0 | 1 | 100.0% |
| `direct_category_axis vs random` | `false_premise` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs random` | `general_helpfulness` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_category_axis vs random` | `harmful_request` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_category_axis vs random` | `persona_honesty` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `anti_sycophancy` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs refusal_heuristic` | `factuality` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `false_premise` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `general_helpfulness` | 0 | 0 | 2 | 0 | 0 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `harmful_request` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `persona_honesty` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_category_axis vs sentiment` | `anti_sycophancy` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_category_axis vs sentiment` | `factuality` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs sentiment` | `false_premise` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs sentiment` | `general_helpfulness` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs sentiment` | `harmful_request` | 1 | 1 | 1 | 0 | 2 | 50.0% |
| `direct_category_axis vs sentiment` | `persona_honesty` | 2 | 0 | 0 | 0 | 2 | 100.0% |

## Interpretation Rule

Treat incomplete reviews, ties, and reject-both cases as signal about benchmark quality, not noise to hide.
