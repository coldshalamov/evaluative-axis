# Cycle 005 Experiment

Date: June 27, 2026

## Protocol

Frozen spec:

- `experiments/research_system_v1/benchmarks/process_potential_error_repair_v1_spec.md`

Runner:

- `scripts/run_process_potential_error_repair.py`

Benchmark:

- `experiments/research_system_v1/benchmarks/process_potential_error_repair_v1.json`

Trace design:

- 12 traces
- injected `error_step`
- later `repair_step`
- natural cumulative traces across arithmetic, code reasoning, tool
  interpretation, factual reasoning, harm reduction, persona honesty, and
  anti-sycophancy

Backends run:

- `gemini-embedding-2`
- `BAAI/bge-base-en-v1.5`

Controls:

- `length`
- `sentiment`
- `final_answer_only_category_axis`
- `final_answer_only_combined`

## Why This Is Cleaner

- no HH overlap or label agreement as the headline metric
- no post-hoc changing of the success threshold after seeing results
- same traces reused for strong-model, cheap-model, and trivial-control
  comparisons
- final-answer-only controls make it explicit whether the signal is actually
  process-aware
