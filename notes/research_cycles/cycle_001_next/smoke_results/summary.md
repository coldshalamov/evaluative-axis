# Cycle 001 Intervention Smoke Summary

Backend: `lexical`
Items: 5
Candidates: 15
Interfaces: direct, decomposition

## Fixture Expected-Hit Rates

These numbers are plumbing checks against hand-written expected winners, not publishable evidence.

| Method | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| `decomposition_anti_sycophancy` | 1 | 5 | 20.0% |
| `decomposition_combined` | 4 | 5 | 80.0% |
| `decomposition_general_evaluative` | 4 | 5 | 80.0% |
| `decomposition_harm_reduction` | 2 | 5 | 40.0% |
| `decomposition_persona_honesty` | 2 | 5 | 40.0% |
| `decomposition_truthfulness` | 2 | 5 | 40.0% |
| `direct_anti_sycophancy` | 1 | 5 | 20.0% |
| `direct_combined` | 3 | 5 | 60.0% |
| `direct_general_evaluative` | 1 | 5 | 20.0% |
| `direct_harm_reduction` | 1 | 5 | 20.0% |
| `direct_persona_honesty` | 1 | 5 | 20.0% |
| `direct_truthfulness` | 0 | 5 | 0.0% |
| `length` | 3 | 5 | 60.0% |
| `random` | 2 | 5 | 40.0% |
| `sentiment` | 1 | 5 | 20.0% |

## Artifacts

- Scores CSV: `notes/research_cycles/cycle_001_next/smoke_results/scores.csv`
- Scores JSON: `notes/research_cycles/cycle_001_next/smoke_results/scores.json`
- Selections CSV: `notes/research_cycles/cycle_001_next/smoke_results/selections.csv`
- Blind packet: `notes/research_cycles/cycle_001_next/smoke_results/blind_review_packet.jsonl`

## Interpretation Rule

Do not treat this smoke result as evidence for or against the thesis. The fixture is too small, known in advance, and hand-authored. Its purpose is to verify that the benchmark interface, axes, outputs, and review packet work before a blinded pilot.
