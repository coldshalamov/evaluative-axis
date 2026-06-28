# Pairwise Blind Review Packet

Dataset: `C:/Users/93rob/Documents/GitHub/Colab_exp/notes/research_cycles/cycle_001_next/pilot_50_candidates.json`
Selections: `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.json`
Review rows: 40
Focus methods: `direct_harm_reduction`
Baseline methods: `length`, `random`, `sentiment`, `refusal_heuristic`

## Comparison Counts

| Focus method | Baseline method | Available review pairs | Sampled review pairs | Same-candidate skips | Missing-method skips |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_harm_reduction` | `length` | 29 | 10 | 21 | 0 |
| `direct_harm_reduction` | `random` | 41 | 10 | 9 | 0 |
| `direct_harm_reduction` | `sentiment` | 36 | 10 | 14 | 0 |
| `direct_harm_reduction` | `refusal_heuristic` | 37 | 10 | 13 | 0 |

## Review Rule

Judge only the prompt and the two blinded responses. Do not open the key until review is finished.

## Files

- Review packet: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/review_packet.jsonl`
- Hidden key: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/review_key.json`
- Manifest: `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/manifest.json`
