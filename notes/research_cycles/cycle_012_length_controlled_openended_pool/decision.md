# Cycle 012 Decision

Date: June 27, 2026

## Decision

Promote this cycle as the creation of the missing clean open-ended intervention
surface, with a frozen partial pilot already saved.

The repo can now say something more concrete than before:

- the old open-ended lane was blocked by a known length-biased pool
- the new builder removes that confound directly
- the fresh pool already yields real selector disagreements
- and the remaining blocker is now mostly operational Gemini quota, not the
  absence of a serious candidate surface

## What To Do Next

1. Finish the remaining pilot prompts in the fresh pool with the resume-enabled
   builder when the free-tier generation endpoint is cooperative again.
2. Rerun Gemini embedding selection on the frozen fresh pool and save that run
   as the main strong-model open-ended pilot.
3. Complete the blind-review pass on the fresh pool with the hardened
   `--resume` judge path rather than on the old inherited Cycle 001 pool.
4. Only after that, decide whether to scale prompt count or add a neutral
   critique/decomposition interface.
