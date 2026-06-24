# Cycle 001 Pilot Build Summary

Date: June 23, 2026

## Pilot Candidate Set

- Prompts: 50
- Candidates per prompt: 4
- Total candidates: 200

## Category Counts

- `anti_sycophancy`: 10
- `context_negation`: 1
- `factuality`: 5
- `false_premise`: 5
- `general_helpfulness`: 10
- `harmful_request`: 8
- `persona_honesty`: 10
- `privacy_safety`: 1

## Source Counts

- `constructed_adversarial_context_test`: 25
- `hh_disagreement_audit`: 25

## HH Blind Adjudication Prep

- Gradeable disagreement cases: 108
- Prior audit EMBEDDING_RIGHT: 63
- Prior audit HH_RIGHT: 45
- Source table EXCLUDE: 123
- Note: prior prose summary says 109 gradeable cases, but the table currently parses as 108.

## Files

- `pilot_50_candidates.json`
- `pilot_answer_key.json`
- `adversarial_negation_tests.json`
- `hh_blind_adjudication/gradeable_108_review.jsonl`
- `hh_blind_adjudication/gradeable_108_key.json`

## Interpretation

This build creates review and scoring artifacts, not final evidence. The expected labels are proxy labels from the prior audit or curated adversarial design. The publishable result comes from blind review plus autopsy.
