# Experiment Mode: Cycle 002

Date: June 23, 2026
Cycle: `cycle_002_potential_shaping`
Experiment name: Controlled evaluative-axis battery and potential-shaping prep

## Research Question

Can embedding evaluative axes distinguish controlled conceptual differences
after obvious confounds such as length, tone, and verbosity are constrained?

## What This Can Answer

- Whether BGE-small or another local embedding can detect conceptual wrongness
  in length-matched pairs.
- Which axes work for which failure modes.
- Whether direct scoring or decomposition scoring is less confounded.
- Whether the next step should be trajectory deltas or better axis design.

## What This Cannot Answer

- Whether the method improves trained model weights.
- Whether Gemini or a frontier embedding model would pass the same battery.
- Whether the signal is calibrated as probability of correctness.
- Whether a broad axis alone is enough.

## Protocol Frozen Before Run

- Dataset / prompt source: hand-built controlled minimal pairs in
  `controlled_evaluative_axis_battery.jsonl`.
- Sample size: initial v0 battery of 24+ pairs.
- Embedding model: local FastEmbed `BAAI/bge-small-en-v1.5`, with lexical mode
  as a cheap debugging baseline.
- Axes: broad good/bad plus truthfulness, harm reduction, persona honesty,
  anti-sycophancy, and category-routed axis.
- Text interface / granularity:
  - direct prompt+answer;
  - decomposition text when available;
  - later cumulative-prefix traces.
- Scoring formula: pairwise score gap, better answer minus worse answer.
- Baselines:
  - length;
  - sentiment;
  - refusal heuristic;
  - random theoretical 50%.
- Primary metric: pairwise accuracy by method.
- Subgroup metrics: accuracy by phenomenon/category and length gap.
- Failure criteria:
  - length dominates;
  - broad axis fails but specific axes pass;
  - all axes fail on factual or reasoning minimal pairs.

## Autopsy Sampling Plan

- Every false negative where length favors the correct answer but embedding
  still prefers the wrong answer.
- Every case where broad good/bad fails but category axis succeeds.
- Every case where direct succeeds and decomposition fails, or vice versa.
- Every case with high length imbalance.

## Expected Interpretation

If positive:

Proceed to cumulative-context trajectory deltas. The evaluator can distinguish
conceptual quality under controlled pair conditions.

If negative:

Either BGE-small is too weak, axes are wrong, or external embeddings cannot
reliably detect this conceptual structure without explicit decomposition.

If mixed:

Use scalar-plus-basis. The broad axis supplies general valence; diagnostic axes
handle truth, harm, persona, sycophancy, and rigor.
