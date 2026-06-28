# Cycle 013 Experiment

Date: June 27, 2026

## Status

Benchmark construction: confirmatory / frozen once generated

Local OSS verification: exploratory

## Frozen Builder

Script:

- `scripts/build_objective_reranking_suite_v2.py`

Outputs:

- `experiments/research_system_v1/benchmarks/objective_math_reranking_v2.json`
- `experiments/research_system_v1/benchmarks/tool_interpretation_reranking_v2.json`
- `experiments/research_system_v1/benchmarks/objective_suite_v2_build_summary.json`

## Design Rules

1. No model-specific filtering after generation.
2. Candidate order randomized per task before assigning `C1` / `C2` / `C3`.
3. Candidate texts kept close in structure and length.
4. Final grading remains exact:
   - numeric;
   - boolean;
   - canonical string with aliases.

## v2 Suite Sizes

- math: 48 tasks / 144 candidates
- tool interpretation: 32 tasks / 96 candidates

## Verification Runs

Math:

- `python scripts/run_objective_text_reranking.py --tasks experiments/research_system_v1/benchmarks/objective_math_reranking_v2.json --output notes/research_cycles/cycle_013_objective_suite_v2_scaling/objective_math_bge_base_v1 --backend sentence_transformers --model BAAI/bge-base-en-v1.5`
- `python scripts/run_objective_text_reranking.py --tasks experiments/research_system_v1/benchmarks/objective_math_reranking_v2.json --output notes/research_cycles/cycle_013_objective_suite_v2_scaling/objective_math_mpnet_base_v1 --backend sentence_transformers --model sentence-transformers/all-mpnet-base-v2`

Tool interpretation:

- `python scripts/run_objective_text_reranking.py --tasks experiments/research_system_v1/benchmarks/tool_interpretation_reranking_v2.json --output notes/research_cycles/cycle_013_objective_suite_v2_scaling/tool_interpretation_bge_base_v1 --backend sentence_transformers --model BAAI/bge-base-en-v1.5`
- `python scripts/run_objective_text_reranking.py --tasks experiments/research_system_v1/benchmarks/tool_interpretation_reranking_v2.json --output notes/research_cycles/cycle_013_objective_suite_v2_scaling/tool_interpretation_mpnet_base_v1 --backend sentence_transformers --model sentence-transformers/all-mpnet-base-v2`
