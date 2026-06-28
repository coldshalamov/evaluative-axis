# Cycle 005 Results

Date: June 27, 2026

## Output Artifacts

Gemini:

- `notes/research_system_v1/process_potential_error_repair_v1/summary.md`

Cheap OSS baseline:

- `notes/research_system_v1/process_potential_error_repair_bge_v1/summary.md`

System report refresh:

- `notes/research_system_v1/report/report.md`

## Key Findings

### 1. Gemini clearly tracks process changes better than cheap OSS

On `gemini-embedding-2`:

- `category_axis` dense score: 50.0%
- `combined` dense score: 62.5%
- `error_drop_accuracy`: 91.7%
- `repair_rise_accuracy`: 83.3%

On `BAAI/bge-base-en-v1.5`:

- `category_axis` dense score: 20.8%
- `combined` dense score: 16.7%
- `error_drop_accuracy`: 33.3%
- `repair_rise_accuracy`: 75.0%

So the stronger model is doing something real here that the cheap OSS embedder
mostly is not.

### 2. Process-aware scoring beats final-answer-only scoring decisively

For Gemini:

- `final_answer_only_category_axis`: 0.0% dense score
- `final_answer_only_combined`: 0.0% dense score

This is one of the most important outcomes of the cycle. The signal is not just
coming from whether the last answer looks nice in isolation. It is responding
to changes inside the trajectory.

### 3. Trivial controls do not explain the result

For Gemini:

- `length`: 0.0% dense score
- `sentiment`: 8.3% dense score

So the benchmark is not being won by verbosity or positive wording.

### 4. The frozen training-readiness gate still failed

The current manifest gate is:

- `process_potential_gate`: pass if `dense_reward_localization_score >= 0.65`

Observed Gemini result:

- `dense_reward_localization_score = 0.50`

That means the lane is promising but still below the precommitted bar needed to
say dense reward localization is strong enough for training-readiness.

### 5. The remaining weakness is concentrated, not random

Trace-level misses were concentrated in:

- `reasoning_rigor`
- `persona_honesty`
- parts of `harm_reduction`

The `combined` scorer was much stronger than the single per-category axis on:

- truthfulness-heavy traces
- the tool interpretation traces
- some arithmetic traces

That pattern suggests the geometry is useful but still incomplete for more
subtle reasoning-error localization.

## Interpretation

This is a meaningful positive result with an honest ceiling.

What the cycle supports:

- stronger embedding models can detect process deterioration and later repair
  inside cumulative traces
- that process signal is not reducible to final-answer-only, length, or
  sentiment controls
- there is now direct evidence for a bridge from reranking toward denser
  supervision

What it does not support yet:

- that the current embedding signal is already strong enough to serve as a
  robust dense reward without more work
- that training-readiness has been demonstrated

So the strongest fair statement is:

> The process-aware signal is real and capability-sensitive, but it has not yet
> crossed the repo's frozen threshold for training-readiness.
