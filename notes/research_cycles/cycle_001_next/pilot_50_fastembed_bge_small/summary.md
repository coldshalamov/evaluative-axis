# Cycle 001 Intervention Summary

Backend: `fastembed`
Items: 50
Candidates: 200
Interfaces: direct, decomposition

## Proxy Expected-Hit Rates

These numbers compare against the current proxy answer key. They are not a substitute for blind review.

| Method | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| `decomposition_anti_sycophancy` | 10 | 50 | 20.0% |
| `decomposition_category_axis` | 12 | 50 | 24.0% |
| `decomposition_combined` | 12 | 50 | 24.0% |
| `decomposition_general_evaluative` | 10 | 50 | 20.0% |
| `decomposition_harm_reduction` | 17 | 50 | 34.0% |
| `decomposition_persona_honesty` | 17 | 50 | 34.0% |
| `decomposition_truthfulness` | 11 | 50 | 22.0% |
| `direct_anti_sycophancy` | 23 | 50 | 46.0% |
| `direct_category_axis` | 19 | 50 | 38.0% |
| `direct_combined` | 12 | 50 | 24.0% |
| `direct_general_evaluative` | 5 | 50 | 10.0% |
| `direct_harm_reduction` | 19 | 50 | 38.0% |
| `direct_persona_honesty` | 15 | 50 | 30.0% |
| `direct_truthfulness` | 12 | 50 | 24.0% |
| `length` | 33 | 50 | 66.0% |
| `random` | 16 | 50 | 32.0% |
| `refusal_heuristic` | 25 | 50 | 50.0% |
| `sentiment` | 15 | 50 | 30.0% |

## Category Proxy-Hit Rates

Focused methods only. Full per-method selections are in `selections.csv`.

| Category | Method | Hits | Total | Rate |
| --- | --- | ---: | ---: | ---: |
| `anti_sycophancy` | `random` | 4 | 10 | 40.0% |
| `anti_sycophancy` | `length` | 6 | 10 | 60.0% |
| `anti_sycophancy` | `sentiment` | 3 | 10 | 30.0% |
| `anti_sycophancy` | `refusal_heuristic` | 8 | 10 | 80.0% |
| `anti_sycophancy` | `direct_combined` | 2 | 10 | 20.0% |
| `anti_sycophancy` | `decomposition_combined` | 2 | 10 | 20.0% |
| `anti_sycophancy` | `direct_category_axis` | 6 | 10 | 60.0% |
| `anti_sycophancy` | `decomposition_category_axis` | 0 | 10 | 0.0% |
| `context_negation` | `random` | 1 | 1 | 100.0% |
| `context_negation` | `length` | 1 | 1 | 100.0% |
| `context_negation` | `sentiment` | 0 | 1 | 0.0% |
| `context_negation` | `refusal_heuristic` | 1 | 1 | 100.0% |
| `context_negation` | `direct_combined` | 0 | 1 | 0.0% |
| `context_negation` | `decomposition_combined` | 0 | 1 | 0.0% |
| `context_negation` | `direct_category_axis` | 0 | 1 | 0.0% |
| `context_negation` | `decomposition_category_axis` | 1 | 1 | 100.0% |
| `factuality` | `random` | 2 | 5 | 40.0% |
| `factuality` | `length` | 2 | 5 | 40.0% |
| `factuality` | `sentiment` | 0 | 5 | 0.0% |
| `factuality` | `refusal_heuristic` | 1 | 5 | 20.0% |
| `factuality` | `direct_combined` | 1 | 5 | 20.0% |
| `factuality` | `decomposition_combined` | 1 | 5 | 20.0% |
| `factuality` | `direct_category_axis` | 1 | 5 | 20.0% |
| `factuality` | `decomposition_category_axis` | 1 | 5 | 20.0% |
| `false_premise` | `random` | 1 | 5 | 20.0% |
| `false_premise` | `length` | 4 | 5 | 80.0% |
| `false_premise` | `sentiment` | 2 | 5 | 40.0% |
| `false_premise` | `refusal_heuristic` | 3 | 5 | 60.0% |
| `false_premise` | `direct_combined` | 1 | 5 | 20.0% |
| `false_premise` | `decomposition_combined` | 1 | 5 | 20.0% |
| `false_premise` | `direct_category_axis` | 1 | 5 | 20.0% |
| `false_premise` | `decomposition_category_axis` | 1 | 5 | 20.0% |
| `general_helpfulness` | `random` | 2 | 10 | 20.0% |
| `general_helpfulness` | `length` | 7 | 10 | 70.0% |
| `general_helpfulness` | `sentiment` | 2 | 10 | 20.0% |
| `general_helpfulness` | `refusal_heuristic` | 5 | 10 | 50.0% |
| `general_helpfulness` | `direct_combined` | 2 | 10 | 20.0% |
| `general_helpfulness` | `decomposition_combined` | 2 | 10 | 20.0% |
| `general_helpfulness` | `direct_category_axis` | 2 | 10 | 20.0% |
| `general_helpfulness` | `decomposition_category_axis` | 2 | 10 | 20.0% |
| `harmful_request` | `random` | 4 | 8 | 50.0% |
| `harmful_request` | `length` | 5 | 8 | 62.5% |
| `harmful_request` | `sentiment` | 2 | 8 | 25.0% |
| `harmful_request` | `refusal_heuristic` | 2 | 8 | 25.0% |
| `harmful_request` | `direct_combined` | 1 | 8 | 12.5% |
| `harmful_request` | `decomposition_combined` | 2 | 8 | 25.0% |
| `harmful_request` | `direct_category_axis` | 4 | 8 | 50.0% |
| `harmful_request` | `decomposition_category_axis` | 3 | 8 | 37.5% |
| `persona_honesty` | `random` | 1 | 10 | 10.0% |
| `persona_honesty` | `length` | 7 | 10 | 70.0% |
| `persona_honesty` | `sentiment` | 5 | 10 | 50.0% |
| `persona_honesty` | `refusal_heuristic` | 4 | 10 | 40.0% |
| `persona_honesty` | `direct_combined` | 4 | 10 | 40.0% |
| `persona_honesty` | `decomposition_combined` | 3 | 10 | 30.0% |
| `persona_honesty` | `direct_category_axis` | 4 | 10 | 40.0% |
| `persona_honesty` | `decomposition_category_axis` | 3 | 10 | 30.0% |
| `privacy_safety` | `random` | 1 | 1 | 100.0% |
| `privacy_safety` | `length` | 1 | 1 | 100.0% |
| `privacy_safety` | `sentiment` | 1 | 1 | 100.0% |
| `privacy_safety` | `refusal_heuristic` | 1 | 1 | 100.0% |
| `privacy_safety` | `direct_combined` | 1 | 1 | 100.0% |
| `privacy_safety` | `decomposition_combined` | 1 | 1 | 100.0% |
| `privacy_safety` | `direct_category_axis` | 1 | 1 | 100.0% |
| `privacy_safety` | `decomposition_category_axis` | 1 | 1 | 100.0% |

## Artifacts

- Scores CSV: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/scores.csv`
- Scores JSON: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/scores.json`
- Selections CSV: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.csv`
- Blind packet: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/blind_review_packet.jsonl`
- Blind packet key: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/blind_review_packet_key.json`

## Interpretation Rule

Do not treat proxy-hit rates as final evidence. The decisive result is the blind review of method-selected winners plus example autopsy.
