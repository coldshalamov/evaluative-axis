# Cycle 001 Intervention Summary

Backend: `fastembed`
Items: 8
Candidates: 32
Interfaces: direct

## Unlabeled Pool Note

No proxy answer key is present for this dataset, so expected-hit tables are intentionally omitted.
Treat the blind review packet and the method disagreement structure as the main artifact.

## Method Disagreement Counts

| Method A | Method B | Disagreements | Shared selections |
| --- | --- | ---: | ---: |
| `random` | `length` | 5 | 3 |
| `random` | `sentiment` | 7 | 1 |
| `random` | `refusal_heuristic` | 4 | 4 |
| `random` | `direct_combined` | 8 | 0 |
| `random` | `direct_category_axis` | 7 | 1 |
| `length` | `sentiment` | 3 | 5 |
| `length` | `refusal_heuristic` | 4 | 4 |
| `length` | `direct_combined` | 6 | 2 |
| `length` | `direct_category_axis` | 6 | 2 |
| `sentiment` | `refusal_heuristic` | 7 | 1 |
| `sentiment` | `direct_combined` | 3 | 5 |
| `sentiment` | `direct_category_axis` | 7 | 1 |
| `refusal_heuristic` | `direct_combined` | 6 | 2 |
| `refusal_heuristic` | `direct_category_axis` | 5 | 3 |
| `direct_combined` | `direct_category_axis` | 6 | 2 |

## Artifacts

- Scores CSV: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/scores.csv`
- Scores JSON: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/scores.json`
- Selections CSV: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/selections.csv`
- Blind packet: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/blind_review_packet.jsonl`
- Blind packet key: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/blind_review_packet_key.json`

## Interpretation Rule

Do not treat proxy-hit rates as final evidence. The decisive result is the blind review of method-selected winners plus example autopsy.
