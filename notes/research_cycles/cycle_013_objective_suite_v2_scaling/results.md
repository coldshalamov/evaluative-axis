# Cycle 013 Results

Date: June 27, 2026

## Builder Summary

From `experiments/research_system_v1/benchmarks/objective_suite_v2_build_summary.json`:

- math task count: 48
- tool task count: 32
- math mean within-item word gap: 1.08
- tool mean within-item word gap: 1.56
- math max within-item word gap: 5
- tool max within-item word gap: 3
- correct-position distribution:
  - math: `C1=15`, `C2=15`, `C3=18`
  - tool: `C1=8`, `C2=13`, `C3=11`

The important structural gain is that the objective lane is no longer built on
tiny fixed-order sets where the correct answer is always in the same slot.

## Local OSS Verification

### Objective Math v2

From:

- `notes/research_cycles/cycle_013_objective_suite_v2_scaling/objective_math_bge_base_v1/summary.md`
- `notes/research_cycles/cycle_013_objective_suite_v2_scaling/objective_math_mpnet_base_v1/summary.md`

Results:

- random: 47.9%
- length: 35.4%
- BGE-base best direct: 29.2%
- MPNet-base best direct: 35.4%

Interpretation:

- BGE-base is materially below random on the larger frozen math suite.
- MPNet-base improves over BGE-base on one interface, but still fails to beat
  random and only ties the weak length baseline.

### Tool Interpretation v2

From:

- `notes/research_cycles/cycle_013_objective_suite_v2_scaling/tool_interpretation_bge_base_v1/summary.md`
- `notes/research_cycles/cycle_013_objective_suite_v2_scaling/tool_interpretation_mpnet_base_v1/summary.md`

Results:

- random: 37.5%
- length: 50.0%
- BGE-base best direct: 43.8%
- MPNet-base best direct: 28.1%

Interpretation:

- BGE-base does not beat the cheap length baseline.
- MPNet-base collapses badly on this suite and goes below random.

## Colab Operational Note

The direct Colab MCP transport still timed out after browser approval, but the
Chrome-approved live Colab notebook itself is usable as a fallback surface.
This matters operationally because it means stronger OSS model runs remain
possible even when the direct MCP transport is flaky.
