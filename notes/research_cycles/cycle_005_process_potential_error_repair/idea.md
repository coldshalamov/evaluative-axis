# Cycle 005 Idea

Date: June 27, 2026

Cycle: `cycle_005_process_potential_error_repair`

## Question

Do evaluative embedding scores respond at the point where an error enters a
reasoning trace and then recover when the trace repairs itself, or do they only
look good or bad at the final answer level?

## Why This Matters

Selection wins are useful, but they do not by themselves justify training on a
dense reward.

The bridge to the stronger thesis is process sensitivity:

- a useful training signal should drop when the trace becomes worse;
- it should rise when the trace repairs itself;
- and that should beat trivial controls like length, sentiment, or looking only
  at the final answer.

If that does not happen, the method may still be a reranker while remaining too
blunt for potential shaping or dense supervision.

## Falsification

Treat the training-readiness story as weakened if:

- the score barely changes at the injected error or repair step;
- final-answer-only scoring performs similarly to the process-aware score;
- or trivial controls localize the same transitions just as well.

Treat it as strengthened if:

- the error step reliably causes a score drop;
- the repair step reliably causes a score rise;
- process-aware scoring beats final-answer-only, length, and sentiment on the
  same traces;
- and a stronger backend materially outperforms a cheap OSS embedder.
