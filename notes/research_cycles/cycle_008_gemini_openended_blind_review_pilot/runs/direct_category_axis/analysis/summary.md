# Pairwise Review Analysis

Review packet: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_category_axis/gemini_review.jsonl`
Key: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_category_axis/review_key.json`

## Comparison Results

| Comparison | Focus wins | Baseline wins | Ties | Reject both | Incomplete | Focus win rate (decided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_category_axis vs length` | 2 | 8 | 0 | 0 | 0 | 20.0% |
| `direct_category_axis vs random` | 5 | 4 | 1 | 0 | 0 | 55.6% |
| `direct_category_axis vs refusal_heuristic` | 0 | 8 | 2 | 0 | 0 | 0.0% |
| `direct_category_axis vs sentiment` | 10 | 0 | 0 | 0 | 0 | 100.0% |

## Category Breakdown

| Comparison | Category | Focus wins | Baseline wins | Ties | Reject both | Decided | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_category_axis vs length` | `anti_sycophancy` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_category_axis vs length` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs length` | `false_premise` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs length` | `general_helpfulness` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_category_axis vs length` | `harmful_request` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs length` | `persona_honesty` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_category_axis vs random` | `anti_sycophancy` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_category_axis vs random` | `factuality` | 1 | 0 | 1 | 0 | 1 | 100.0% |
| `direct_category_axis vs random` | `false_premise` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_category_axis vs random` | `general_helpfulness` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs random` | `harmful_request` | 2 | 2 | 0 | 0 | 4 | 50.0% |
| `direct_category_axis vs refusal_heuristic` | `anti_sycophancy` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `factuality` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `general_helpfulness` | 0 | 2 | 1 | 0 | 2 | 0.0% |
| `direct_category_axis vs refusal_heuristic` | `harmful_request` | 0 | 3 | 1 | 0 | 3 | 0.0% |
| `direct_category_axis vs sentiment` | `anti_sycophancy` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_category_axis vs sentiment` | `factuality` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_category_axis vs sentiment` | `harmful_request` | 4 | 0 | 0 | 0 | 4 | 100.0% |
| `direct_category_axis vs sentiment` | `persona_honesty` | 2 | 0 | 0 | 0 | 2 | 100.0% |

## Interpretation Rule

Treat incomplete reviews, ties, and reject-both cases as signal about benchmark quality, not noise to hide.
