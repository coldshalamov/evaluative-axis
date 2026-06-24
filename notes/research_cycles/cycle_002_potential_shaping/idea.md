# Idea Mode: Cycle 002

Date: June 23, 2026
Cycle: `cycle_002_potential_shaping`

## Seed Idea

The strongest version of the project is not "pick the best final answer with a
good/bad dot product." It is:

> Use evaluative embedding geometry as a semantic potential over full context,
> then use changes in that potential to supply dense supervision.

This avoids double-counting local goodness and directly tests whether each new
step improves or degrades the whole trajectory.

## Core Formulation

Let `f(x)` be a normalized embedding and `v` an evaluative direction built from
positive and negative anchors.

For context `c` and generated prefix `r_1:t`:

```text
Phi_t = v dot f(c, r_1:t)
```

The dense signal is the change:

```text
F_t = gamma * Phi_t - Phi_(t-1)
```

With `gamma = 1`, the deltas telescope:

```text
sum_t F_t = Phi_T - Phi_0
```

That is the clean credit-assignment hypothesis: score whether the new step made
the trajectory better, not whether the step contains good-sounding words in
isolation.

## Why This Is Stronger Than Final-Answer Scoring

Final-answer scoring has local-good accumulation problems:

- long wrong answers can contain true facts and polished structure;
- harmful helpfulness can be fluent and detailed;
- sycophancy can be warm and coherent;
- a bad global conclusion can be surrounded by locally good statements.

Potential deltas ask whether the whole state improved after each addition. A
locally good sentence can still produce a negative delta if it pushes the answer
toward a bad conclusion.

## What This Cycle Tests First

Before trajectory scoring, test whether the evaluator can detect controlled
conceptual differences when length and style are constrained:

- right vs wrong factual conclusion;
- valid vs invalid reasoning step;
- sunk-cost reasoning vs sunk-cost correction;
- quoted harmful phrase vs endorsed harmful phrase;
- helpful refusal vs blanket refusal;
- persona honesty vs fabricated human persona;
- truth with discomfort vs comforting falsehood;
- safety boundary vs harmful compliance.

## Best Current Hypothesis

The broad evaluative axis alone will be unstable. A scalar-plus-basis design is
more plausible:

- broad evaluative potential;
- truth/factual support;
- harm reduction;
- persona honesty;
- anti-sycophancy;
- reasoning rigor.

The scientific question is whether these axes provide useful deltas after
controlling for length, tone, and verbosity.
