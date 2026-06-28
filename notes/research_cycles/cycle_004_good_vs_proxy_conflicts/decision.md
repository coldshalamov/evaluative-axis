# Cycle 004 Decision

Date: June 27, 2026

## Decision

Do not promote the raw one-word `good/bad` thesis as if it is already supported
by the current zero-shot embedding evidence.

Promote a narrower and more defensible claim instead:

- stronger embedding models carry useful evaluative geometry;
- that geometry is much easier to recover with richer and more targeted axes
  than with single raw evaluative words;
- if raw `good/bad` is ever going to work as the primary training signal, it
  likely requires either a stronger representation, more context, training on
  the signal, or some combination of those.

## What To Do Next

1. Keep the word-level benchmark as a standing falsification test.
2. Do not spend more time trying many tiny single-word variants unless there is
   a principled reason.
3. Shift near-term effort toward:
   - targeted-axis conflict batteries,
   - process-potential tests,
   - and objective reranking suites that already show lift.
4. Preserve this result prominently because it is scientifically useful:
   it rules out the easiest and most seductive overclaim.
