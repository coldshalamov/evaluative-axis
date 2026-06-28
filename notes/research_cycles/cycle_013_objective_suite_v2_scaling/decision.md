# Cycle 013 Decision

Date: June 27, 2026

## Decision

Keep the v2 objective suites and treat them as the new default surface for
future capacity-ladder work.

Do not treat the old 8-task objective suites as decisive evidence anymore.
They remain useful pilot artifacts, but they are too small to carry the claim.

## What This Cycle Shows

1. The repo now has a materially stronger exact-grading benchmark surface that
   does not require blind reviewers or LLM judges.
2. Cheap OSS embedders remain weak or anti-correlated on that larger surface.
3. The capacity-ladder story is still alive, but the benchmark is now strict
   enough that a future strong-model win would mean more than the old pilot
   wins did.

## What It Does Not Show Yet

1. A strong-model positive result on the new v2 suites.
2. A final cross-domain confirmatory claim.
3. Training-readiness evidence.

## Next Steps

1. Run `gemini-embedding-2` on the same v2 suites when quota allows.
2. Use the working Chrome-approved Colab notebook surface to test at least one
   stronger OSS embedder on the same frozen v2 files.
3. Update the serious research report so current objective v1 wins are framed
   as pilot evidence and v2 becomes the new confirmatory target.
