# HH Gradeable Disagreement Blind Adjudication Packet

Date: June 23, 2026

This packet strips method labels from the 109 gradeable HH disagreement cases.
Reviewers see only the prompt and two shuffled responses.

## Counts

- Total gradeable cases: 108
- Prior audit EMBEDDING_RIGHT: 63
- Prior audit HH_RIGHT: 45
- Source table EXCLUDE: 123

## Count Discrepancy

The prose summary in `disagreement_audit/full_grading.md` says 65
`EMBEDDING_RIGHT`, 44 `HH_RIGHT`, and 122 `EXCLUDE` cases, implying
109 gradeable cases. The actual grading table currently parses as 63
`EMBEDDING_RIGHT`, 45 `HH_RIGHT`, and 123 `EXCLUDE`, implying 108
gradeable cases. This packet follows the table rows because those are
the auditable examples reviewers can inspect.

## Files

- `gradeable_108_review.jsonl`: blinded review packet
- `gradeable_108_key.json`: hidden key for analysis after review

## Review Rule

Do not open the key while adjudicating. The key exists only for analysis after blind labels are collected.
