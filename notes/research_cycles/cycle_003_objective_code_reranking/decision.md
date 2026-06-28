# Decision Mode: Cycle 003

Date: June 25, 2026
Cycle: `cycle_003_objective_code_reranking`

## Decision

Promote objective reranking with hidden end metrics to the primary no-GPU proof
path.

Do not treat HH agreement as the main gate for the thesis. Use HH, PKU, and
similar preference datasets as sensors for disagreement analysis and coverage,
not as the final arbiter of whether the evaluative-geometry signal is useful.

## Evidence Used

- The cycle-wide landscape scan ranked 20 candidate tests by thesis fit,
  falsification power, budget fit, contamination resistance, and decision
  value.
- Objective code reranking ranked first because it gives a real end metric
  without human labels or oracle decomposition.
- A local fallback pilot completed on 6 tasks with 3 candidates per task.
- Hidden-test results:
  - random: 3/6 solved
  - length: 3/6 solved
  - direct broad evaluative: 5/6 solved
  - direct code-quality evaluative: 5/6 solved
- The only miss was a near-correct bracket parser that failed one
  order-sensitive hidden case, with a very small score margin over the truly
  correct candidate.

## Remaining Uncertainty

- Whether the result survives scale to 30-50 tasks.
- Whether critique-based scoring beats direct answer scoring.
- Whether this transfers to math, tool interpretation, and open-ended language
  tasks.
- Whether the signal remains useful once candidates are fully model-generated
  instead of partly curated.
- Whether small score-margin cases can be identified reliably enough to support
  abstention or resampling.

## Next Action

Run a scaled version of Target 1 and then Target 3.

Operationally:

1. keep the current code reranking harness;
2. expand the task set with harder hidden checks and more close-call distractor
   candidates;
3. add critique scoring when Gemini generation quota or Colab access becomes
   available;
4. carry the same protocol over to objective math.

Owner: Codex.

Completion evidence:

- `scripts/run_objective_code_reranking.py` exists.
- Cycle 003 `idea.md`, `experiment.md`, and `results.md` exist.
- `pilot_results/summary.md` records the objective outcomes.
- `notes/research_log.md` records the cycle and fallback conditions.
