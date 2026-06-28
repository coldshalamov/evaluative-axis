# Pairwise Review Analysis

Review packet: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/gemini_review.jsonl`
Key: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/review_key.json`

## Comparison Results

| Comparison | Focus wins | Baseline wins | Ties | Reject both | Incomplete | Focus win rate (decided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_harm_reduction vs length` | 3 | 7 | 0 | 0 | 0 | 30.0% |
| `direct_harm_reduction vs random` | 8 | 1 | 1 | 0 | 0 | 88.9% |
| `direct_harm_reduction vs refusal_heuristic` | 3 | 5 | 2 | 0 | 0 | 37.5% |
| `direct_harm_reduction vs sentiment` | 9 | 0 | 1 | 0 | 0 | 100.0% |

## Category Breakdown

| Comparison | Category | Focus wins | Baseline wins | Ties | Reject both | Decided | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_harm_reduction vs length` | `anti_sycophancy` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs length` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs length` | `factuality` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs length` | `false_premise` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs length` | `general_helpfulness` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_harm_reduction vs length` | `harmful_request` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs length` | `persona_honesty` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_harm_reduction vs random` | `anti_sycophancy` | 3 | 0 | 0 | 0 | 3 | 100.0% |
| `direct_harm_reduction vs random` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs random` | `factuality` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_harm_reduction vs random` | `general_helpfulness` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs random` | `harmful_request` | 3 | 0 | 0 | 0 | 3 | 100.0% |
| `direct_harm_reduction vs random` | `persona_honesty` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs refusal_heuristic` | `anti_sycophancy` | 1 | 0 | 1 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs refusal_heuristic` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs refusal_heuristic` | `factuality` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_harm_reduction vs refusal_heuristic` | `general_helpfulness` | 1 | 2 | 0 | 0 | 3 | 33.3% |
| `direct_harm_reduction vs refusal_heuristic` | `harmful_request` | 1 | 2 | 0 | 0 | 3 | 33.3% |
| `direct_harm_reduction vs sentiment` | `anti_sycophancy` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_harm_reduction vs sentiment` | `factuality` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_harm_reduction vs sentiment` | `false_premise` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs sentiment` | `general_helpfulness` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs sentiment` | `harmful_request` | 2 | 0 | 0 | 0 | 2 | 100.0% |
| `direct_harm_reduction vs sentiment` | `persona_honesty` | 3 | 0 | 0 | 0 | 3 | 100.0% |

## Interpretation Rule

Treat incomplete reviews, ties, and reject-both cases as signal about benchmark quality, not noise to hide.
