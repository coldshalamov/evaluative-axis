# Pairwise Review Analysis

Review packet: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/gemini_review.jsonl`
Key: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/review_key.json`

## Comparison Results

| Comparison | Focus wins | Baseline wins | Ties | Reject both | Incomplete | Focus win rate (decided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_harm_reduction vs length` | 1 | 7 | 2 | 0 | 0 | 12.5% |
| `direct_harm_reduction vs random` | 2 | 6 | 2 | 0 | 0 | 25.0% |
| `direct_harm_reduction vs refusal_heuristic` | 2 | 8 | 0 | 0 | 0 | 20.0% |
| `direct_harm_reduction vs sentiment` | 5 | 2 | 2 | 0 | 1 | 71.4% |

## Category Breakdown

| Comparison | Category | Focus wins | Baseline wins | Ties | Reject both | Decided | Focus win rate (decided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_harm_reduction vs length` | `anti_sycophancy` | 0 | 1 | 1 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs length` | `general_helpfulness` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_harm_reduction vs length` | `harmful_request` | 0 | 2 | 1 | 0 | 2 | 0.0% |
| `direct_harm_reduction vs length` | `persona_honesty` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_harm_reduction vs random` | `anti_sycophancy` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_harm_reduction vs random` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs random` | `factuality` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_harm_reduction vs random` | `false_premise` | 0 | 0 | 1 | 0 | 0 | 0.0% |
| `direct_harm_reduction vs random` | `general_helpfulness` | 1 | 1 | 0 | 0 | 2 | 50.0% |
| `direct_harm_reduction vs random` | `harmful_request` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs random` | `persona_honesty` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs refusal_heuristic` | `anti_sycophancy` | 0 | 3 | 0 | 0 | 3 | 0.0% |
| `direct_harm_reduction vs refusal_heuristic` | `context_negation` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs refusal_heuristic` | `factuality` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs refusal_heuristic` | `false_premise` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs refusal_heuristic` | `general_helpfulness` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs refusal_heuristic` | `harmful_request` | 0 | 1 | 0 | 0 | 1 | 0.0% |
| `direct_harm_reduction vs refusal_heuristic` | `persona_honesty` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_harm_reduction vs sentiment` | `anti_sycophancy` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs sentiment` | `factuality` | 1 | 0 | 0 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs sentiment` | `false_premise` | 1 | 0 | 1 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs sentiment` | `general_helpfulness` | 0 | 2 | 0 | 0 | 2 | 0.0% |
| `direct_harm_reduction vs sentiment` | `harmful_request` | 1 | 0 | 1 | 0 | 1 | 100.0% |
| `direct_harm_reduction vs sentiment` | `persona_honesty` | 1 | 0 | 0 | 0 | 1 | 100.0% |

## Interpretation Rule

Treat incomplete reviews, ties, and reject-both cases as signal about benchmark quality, not noise to hide.
