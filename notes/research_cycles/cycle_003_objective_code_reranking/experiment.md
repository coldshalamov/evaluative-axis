# Experiment Mode: Cycle 003

Date: June 25, 2026
Cycle: `cycle_003_objective_code_reranking`
Experiment name: Objective code reranking pilot

## Research Question

Can evaluative embedding scoring choose better code answers from several
candidates for the same task when the final metric is hidden unit-test success
rather than HH overlap or LLM-judge preference?

## Why This Test

This is the cleanest practical test available under current constraints:

- no GPU training;
- no hired annotators;
- no reliance on HH labels as truth;
- no oracle decomposition;
- objective final metric.

## Core Falsification

The test should be considered negative if:

- direct embedding scoring loses to random or code length;
- critique-plus-embedding scoring also loses to random or length;
- the selected candidates look nicer or more verbose but do not pass more
  hidden tests;
- Gemini-produced critique text only helps when it smuggles in answer labels
  rather than surfacing real failure modes.

## Protocol

1. Use a small set of pure-Python coding tasks with hidden tests.
2. Generate multiple code candidates per task with Gemini using different style
   prompts to create realistic quality variation.
3. Score each candidate with:
   - random;
   - longer-code baseline;
   - direct broad evaluative embedding score;
   - direct code-quality embedding score;
   - blind critique broad evaluative embedding score;
   - blind critique code-quality embedding score.
4. Select one candidate per method.
5. Execute hidden tests safely.
6. Compare:
   - solved-task count;
   - average hidden tests passed;
   - hit rate on picking the best candidate available for each task.

## What This Can Tell Us

- whether the signal is already useful as a reranker in an objective domain;
- whether critique/decomposition interfaces outperform raw answer scoring;
- whether broad good/bad alone is too blunt for code;
- whether this project should next lean toward:
  - objective-domain scaling,
  - critique scoring,
  - process scoring,
  - or abandonment of the current interface.

## What This Cannot Tell Us

- whether the method solves open-ended alignment;
- whether training on the signal improves model weights;
- whether code success generalizes to safety/helpfulness;
- whether the best embedding model has already been identified.

## Chosen Pilot Scale

- 6 tasks;
- 3 candidates per task;
- Gemini generation;
- Gemini embeddings;
- hidden unit tests.

This is intentionally small. The point is to get a clean directional answer
fast, not to oversell a pilot as a definitive benchmark.
