# Pairwise Blind Review Packet

Dataset: `C:/Users/93rob/Documents/GitHub/Colab_exp/notes/research_cycles/cycle_001_next/pilot_50_candidates.json`
Selections: `.tmp/cycle001_gemini_direct_openended_pilot/selections.json`
Review rows: 40
Focus methods: `direct_harm_reduction`
Baseline methods: `length`, `random`, `sentiment`, `refusal_heuristic`

## Comparison Counts

| Focus method | Baseline method | Available review pairs | Sampled review pairs | Same-candidate skips | Missing-method skips |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_harm_reduction` | `length` | 28 | 10 | 22 | 0 |
| `direct_harm_reduction` | `random` | 36 | 10 | 14 | 0 |
| `direct_harm_reduction` | `sentiment` | 39 | 10 | 11 | 0 |
| `direct_harm_reduction` | `refusal_heuristic` | 29 | 10 | 21 | 0 |

## Review Rule

Judge only the prompt and the two blinded responses. Do not open the key until review is finished.

## Files

- Review packet: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/review_packet.jsonl`
- Hidden key: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/review_key.json`
- Manifest: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/manifest.json`
