# Cycle 012 Idea

Date: June 27, 2026

Cycle: `cycle_012_length_controlled_openended_pool`

## Question

Can we replace the inherited length-biased open-ended candidate pool with a
fresh, mixed-category, strict-length pool that is clean enough to support the
next real blind reranking test?

## Why This Matters

The repo's current bottleneck is no longer hidden:

- the objective suites are positive;
- the direct controlled battery is reproduced;
- the old open-ended lane is informative but still inherits a bad candidate
  pool.

So the highest-value infrastructure step is not another proxy analysis. It is a
fresh open-ended pool where:

- all candidates are generated from the same prompt only;
- candidate lengths are controlled tightly enough that length cannot dominate;
- categories map cleanly onto the routed evaluative axes;
- a holdout split is reserved before tuning on results.

## Falsification

Treat this cycle as weakened if:

- exact or near-exact length control proves too brittle to run cheaply;
- the generation styles collapse to near-duplicates so the pool creates little
  selection pressure;
- or the fresh pool still cannot produce a useful blind-review pilot because
  every method mostly ties.

Treat it as strengthened if:

- the builder reliably creates exact-length mixed-category candidates;
- the pool supports real method disagreements without leaking labels;
- and the first blind-review run on the new pool is at least interpretable,
  even if still exploratory.
