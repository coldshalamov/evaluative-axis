# Cycle 001 Intervention Summary

Backend: `fastembed`
Items: 25
Candidates: 100
Interfaces: direct, decomposition

## Proxy Expected-Hit Rates

These numbers compare against the current proxy answer key. They are not a substitute for blind review.

| Method | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| `decomposition_anti_sycophancy` | 6 | 25 | 24.0% |
| `decomposition_category_axis` | 10 | 25 | 40.0% |
| `decomposition_combined` | 11 | 25 | 44.0% |
| `decomposition_general_evaluative` | 10 | 25 | 40.0% |
| `decomposition_harm_reduction` | 11 | 25 | 44.0% |
| `decomposition_persona_honesty` | 14 | 25 | 56.0% |
| `decomposition_truthfulness` | 10 | 25 | 40.0% |
| `direct_anti_sycophancy` | 18 | 25 | 72.0% |
| `direct_category_axis` | 13 | 25 | 52.0% |
| `direct_combined` | 10 | 25 | 40.0% |
| `direct_general_evaluative` | 4 | 25 | 16.0% |
| `direct_harm_reduction` | 9 | 25 | 36.0% |
| `direct_persona_honesty` | 10 | 25 | 40.0% |
| `direct_truthfulness` | 9 | 25 | 36.0% |
| `length` | 24 | 25 | 96.0% |
| `random` | 7 | 25 | 28.0% |
| `refusal_heuristic` | 21 | 25 | 84.0% |
| `sentiment` | 12 | 25 | 48.0% |

## Category Proxy-Hit Rates

Focused methods only. Full per-method selections are in `selections.csv`.

| Category | Method | Hits | Total | Rate |
| --- | --- | ---: | ---: | ---: |
| `anti_sycophancy` | `random` | 0 | 5 | 0.0% |
| `anti_sycophancy` | `length` | 5 | 5 | 100.0% |
| `anti_sycophancy` | `sentiment` | 3 | 5 | 60.0% |
| `anti_sycophancy` | `refusal_heuristic` | 5 | 5 | 100.0% |
| `anti_sycophancy` | `direct_combined` | 2 | 5 | 40.0% |
| `anti_sycophancy` | `decomposition_combined` | 2 | 5 | 40.0% |
| `anti_sycophancy` | `direct_category_axis` | 5 | 5 | 100.0% |
| `anti_sycophancy` | `decomposition_category_axis` | 0 | 5 | 0.0% |
| `context_negation` | `random` | 0 | 1 | 0.0% |
| `context_negation` | `length` | 1 | 1 | 100.0% |
| `context_negation` | `sentiment` | 0 | 1 | 0.0% |
| `context_negation` | `refusal_heuristic` | 1 | 1 | 100.0% |
| `context_negation` | `direct_combined` | 0 | 1 | 0.0% |
| `context_negation` | `decomposition_combined` | 0 | 1 | 0.0% |
| `context_negation` | `direct_category_axis` | 0 | 1 | 0.0% |
| `context_negation` | `decomposition_category_axis` | 1 | 1 | 100.0% |
| `false_premise` | `random` | 1 | 5 | 20.0% |
| `false_premise` | `length` | 4 | 5 | 80.0% |
| `false_premise` | `sentiment` | 2 | 5 | 40.0% |
| `false_premise` | `refusal_heuristic` | 3 | 5 | 60.0% |
| `false_premise` | `direct_combined` | 1 | 5 | 20.0% |
| `false_premise` | `decomposition_combined` | 1 | 5 | 20.0% |
| `false_premise` | `direct_category_axis` | 1 | 5 | 20.0% |
| `false_premise` | `decomposition_category_axis` | 1 | 5 | 20.0% |
| `general_helpfulness` | `random` | 3 | 5 | 60.0% |
| `general_helpfulness` | `length` | 5 | 5 | 100.0% |
| `general_helpfulness` | `sentiment` | 2 | 5 | 40.0% |
| `general_helpfulness` | `refusal_heuristic` | 5 | 5 | 100.0% |
| `general_helpfulness` | `direct_combined` | 2 | 5 | 40.0% |
| `general_helpfulness` | `decomposition_combined` | 2 | 5 | 40.0% |
| `general_helpfulness` | `direct_category_axis` | 2 | 5 | 40.0% |
| `general_helpfulness` | `decomposition_category_axis` | 2 | 5 | 40.0% |
| `harmful_request` | `random` | 1 | 3 | 33.3% |
| `harmful_request` | `length` | 3 | 3 | 100.0% |
| `harmful_request` | `sentiment` | 1 | 3 | 33.3% |
| `harmful_request` | `refusal_heuristic` | 2 | 3 | 66.7% |
| `harmful_request` | `direct_combined` | 1 | 3 | 33.3% |
| `harmful_request` | `decomposition_combined` | 2 | 3 | 66.7% |
| `harmful_request` | `direct_category_axis` | 1 | 3 | 33.3% |
| `harmful_request` | `decomposition_category_axis` | 2 | 3 | 66.7% |
| `persona_honesty` | `random` | 2 | 5 | 40.0% |
| `persona_honesty` | `length` | 5 | 5 | 100.0% |
| `persona_honesty` | `sentiment` | 3 | 5 | 60.0% |
| `persona_honesty` | `refusal_heuristic` | 4 | 5 | 80.0% |
| `persona_honesty` | `direct_combined` | 3 | 5 | 60.0% |
| `persona_honesty` | `decomposition_combined` | 3 | 5 | 60.0% |
| `persona_honesty` | `direct_category_axis` | 3 | 5 | 60.0% |
| `persona_honesty` | `decomposition_category_axis` | 3 | 5 | 60.0% |
| `privacy_safety` | `random` | 0 | 1 | 0.0% |
| `privacy_safety` | `length` | 1 | 1 | 100.0% |
| `privacy_safety` | `sentiment` | 1 | 1 | 100.0% |
| `privacy_safety` | `refusal_heuristic` | 1 | 1 | 100.0% |
| `privacy_safety` | `direct_combined` | 1 | 1 | 100.0% |
| `privacy_safety` | `decomposition_combined` | 1 | 1 | 100.0% |
| `privacy_safety` | `direct_category_axis` | 1 | 1 | 100.0% |
| `privacy_safety` | `decomposition_category_axis` | 1 | 1 | 100.0% |

## Artifacts

- Scores CSV: `notes/research_cycles/cycle_001_next/adversarial_fastembed_bge_small/scores.csv`
- Scores JSON: `notes/research_cycles/cycle_001_next/adversarial_fastembed_bge_small/scores.json`
- Selections CSV: `notes/research_cycles/cycle_001_next/adversarial_fastembed_bge_small/selections.csv`
- Blind packet: `notes/research_cycles/cycle_001_next/adversarial_fastembed_bge_small/blind_review_packet.jsonl`
- Blind packet key: `notes/research_cycles/cycle_001_next/adversarial_fastembed_bge_small/blind_review_packet_key.json`

## Interpretation Rule

Do not treat proxy-hit rates as final evidence. The decisive result is the blind review of method-selected winners plus example autopsy.
