# Pairwise Blind Review Packet

Dataset: `C:/Users/93rob/Documents/GitHub/Colab_exp/notes/research_cycles/cycle_001_next/pilot_50_candidates.json`
Selections: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.json`
Review rows: 40
Focus methods: `direct_anti_sycophancy`
Baseline methods: `length`, `random`, `sentiment`, `refusal_heuristic`

## Comparison Counts

| Focus method | Baseline method | Available review pairs | Sampled review pairs | Same-candidate skips | Missing-method skips |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_anti_sycophancy` | `length` | 20 | 10 | 30 | 0 |
| `direct_anti_sycophancy` | `random` | 41 | 10 | 9 | 0 |
| `direct_anti_sycophancy` | `sentiment` | 36 | 10 | 14 | 0 |
| `direct_anti_sycophancy` | `refusal_heuristic` | 22 | 10 | 28 | 0 |

## Review Rule

Judge only the prompt and the two blinded responses. Do not open the key until review is finished.

## Files

- Review packet: `.tmp/pairwise_bge_small_direct_anti_sycophancy_pilot/review_packet.jsonl`
- Hidden key: `.tmp/pairwise_bge_small_direct_anti_sycophancy_pilot/review_key.json`
- Manifest: `.tmp/pairwise_bge_small_direct_anti_sycophancy_pilot/manifest.json`
