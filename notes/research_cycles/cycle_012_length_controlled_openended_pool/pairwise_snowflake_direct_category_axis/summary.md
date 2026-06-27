# Pairwise Blind Review Packet

Dataset: `notes/research_cycles/cycle_012_length_controlled_openended_pool/generated_pool_v1/pilot_snapshot_8.json`
Selections: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pilot_snapshot_8_snowflake_direct/selections.json`
Review rows: 25
Focus methods: `direct_category_axis`
Baseline methods: `length`, `random`, `sentiment`, `refusal_heuristic`

## Comparison Counts

| Focus method | Baseline method | Available review pairs | Sampled review pairs | Same-candidate skips | Missing-method skips |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_category_axis` | `length` | 6 | 6 | 2 | 0 |
| `direct_category_axis` | `random` | 7 | 7 | 1 | 0 |
| `direct_category_axis` | `sentiment` | 7 | 7 | 1 | 0 |
| `direct_category_axis` | `refusal_heuristic` | 5 | 5 | 3 | 0 |

## Review Rule

Judge only the prompt and the two blinded responses. Do not open the key until review is finished.

## Files

- Review packet: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pairwise_snowflake_direct_category_axis/review_packet.jsonl`
- Hidden key: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pairwise_snowflake_direct_category_axis/review_key.json`
- Manifest: `notes/research_cycles/cycle_012_length_controlled_openended_pool/pairwise_snowflake_direct_category_axis/manifest.json`
