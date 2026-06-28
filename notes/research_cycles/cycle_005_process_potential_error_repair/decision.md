# Cycle 005 Decision

Date: June 27, 2026

## Decision

Promote this cycle as the first direct evidence that evaluative embedding
geometry can respond to trajectory quality rather than only final answer
quality.

Do not promote it as proof that the signal is already ready for dense training.

The report should now say:

- process sensitivity exists;
- stronger models show much more of it than cheap OSS embedders;
- trivial controls do not explain it;
- but the frozen training-readiness gate still fails.

## What To Do Next

1. Keep `process_potential_error_repair_v1` as the standing bridge test from
   selection to training.
2. Expand the suite from 12 traces toward 30-50 traces, especially in the weak
   categories:
   - reasoning rigor
   - persona honesty
   - harm reduction edge cases
3. Add more than one repair style per phenomenon so the benchmark tests
   general process sensitivity rather than a single wording pattern.
4. Keep the gate frozen for the current version. Do not relabel `combined =
   62.5%` as a pass after the fact.
5. If future budget allows, test whether longer-context or post-trained
   embedders improve the weak localization categories rather than only average
   selection accuracy.
