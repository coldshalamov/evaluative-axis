# Pairwise Blind Review Packet

Dataset: `C:/Users/93rob/Documents/GitHub/Colab_exp/notes/research_cycles/cycle_001_next/pilot_50_candidates.json`
Selections: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.json`
Review rows: 40
Focus methods: `direct_category_axis`
Baseline methods: `length`, `random`, `sentiment`, `refusal_heuristic`

## Comparison Counts

| Focus method | Baseline method | Available review pairs | Sampled review pairs | Same-candidate skips | Missing-method skips |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_category_axis` | `length` | 28 | 10 | 22 | 0 |
| `direct_category_axis` | `random` | 39 | 10 | 11 | 0 |
| `direct_category_axis` | `sentiment` | 31 | 10 | 19 | 0 |
| `direct_category_axis` | `refusal_heuristic` | 31 | 10 | 19 | 0 |

## Review Rule

Judge only the prompt and the two blinded responses. Do not open the key until review is finished.

## Files

- Review packet: `.tmp/pairwise_bge_small_direct_category_axis_pilot/review_packet.jsonl`
- Hidden key: `.tmp/pairwise_bge_small_direct_category_axis_pilot/review_key.json`
- Manifest: `.tmp/pairwise_bge_small_direct_category_axis_pilot/manifest.json`
