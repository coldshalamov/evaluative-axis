# Process Potential Error-Repair V1

## Purpose

This suite is the bridge between reranking evidence and training evidence.

It asks whether evaluative embedding scores behave like a usable dense reward on
process traces instead of only on final answers.

## Core Construction

For each problem, build a short trace family with four aligned variants:

1. `clean`
   A fully correct solution trace.
2. `error_injected`
   The same trace with one deliberately wrong local step.
3. `error_persisted`
   The wrong step remains and contaminates later reasoning.
4. `repaired`
   The wrong step is explicitly corrected and the later trace is fixed.

The content before the injection point should stay as similar as possible across
variants so that the main signal comes from the error and repair itself.

## Domains

Start with small, objective domains:

- arithmetic derivations
- code reasoning / bug-fix traces
- tool-log interpretation
- factual reasoning with false-premise correction

## Scoring Procedure

For each prefix of each trace:

1. score the prefix with the same evaluative embedding methods used elsewhere;
2. compute score deltas between adjacent prefixes;
3. compare the delta pattern to the known injected error and known repair step.

## Required Metrics

The suite summary should expose at least these metrics:

- `error_drop_accuracy`
  Fraction of traces where the score drops after the injected wrong step.
- `repair_rise_accuracy`
  Fraction of traces where the score rises after the explicit repair step.
- `error_localization_top1_accuracy`
  Fraction of traces where the largest negative local delta points to the
  injected error step.
- `repair_localization_top1_accuracy`
  Fraction of traces where the largest positive local delta points to the
  repair step.
- `dense_reward_localization_score`
  Main gate metric.
  Recommended definition: mean of
  `error_localization_top1_accuracy` and
  `repair_localization_top1_accuracy`.

## Baselines

Always compare against:

- length delta
- sentiment delta
- final-answer-only scoring

If evaluative embeddings cannot beat those baselines on localization, the
training-readiness claim should stay blocked.

## Initial Pass Condition

Treat the first version as promising only if:

- `dense_reward_localization_score >= 0.65`
- `error_drop_accuracy > 0.75`
- `repair_rise_accuracy > 0.75`

These are not universal truth thresholds. They are the first internal bar for
claiming that the signal looks dense enough to justify cheap training work.

## Output Contract

The eventual `summary.json` for this lane should include:

```json
{
  "error_drop_accuracy": 0.0,
  "repair_rise_accuracy": 0.0,
  "error_localization_top1_accuracy": 0.0,
  "repair_localization_top1_accuracy": 0.0,
  "dense_reward_localization_score": 0.0,
  "n_traces": 0
}
```

That contract is what the manifest gate consumes.
