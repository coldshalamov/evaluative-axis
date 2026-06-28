# Serious Research System V1

Date: June 25, 2026

## Purpose

This file turns the project from a sequence of interesting experiments into a
frozen evidence program.

The goal is not to produce one more positive result. The goal is to define the
minimum system that would make a skeptical third party say:

> this was tested across multiple domains, against cheap baselines, with
> explicit failure gates, and the value of the method is real rather than
> anecdotal.

The executable manifest for this system is:

- `experiments/research_system_v1/program_manifest.json`

The orchestrator is:

- `scripts/orchestrate_research_system.py`

The process-potential bridge specification is:

- `experiments/research_system_v1/benchmarks/process_potential_error_repair_v1_spec.md`

## The Four Claims

1. **Geometry claim**
   Evaluative structure exists in embedding space.
2. **Selection claim**
   Evaluative scoring can choose better answers than cheap baselines.
3. **Capacity claim**
   The effect is much stronger in more capable embedding families than in cheap
   OSS embedders.
4. **Training-readiness claim**
   The signal is strong enough across domains and process tests to justify
   actual training work.

These claims must stay separate.

## Benchmark Families

### 1. Objective Reranking

This is the main practical lane.

Use exact-answer or hidden-test tasks where the final metric is objective:

- code
- math
- tool-output interpretation

Each suite must compare:

- random
- length
- direct general evaluative score
- direct task-targeted score
- combined score when applicable

Pass condition:

- best direct method beats the best cheap baseline by a meaningful margin on
  multiple domains, not just one.

### 2. Behavior Basis Diagnostics

This lane asks whether the signal carries specific behaviors the project cares
about, not just broad `good/bad`.

Use controlled minimal-pair batteries that test:

- truthfulness
- harm reduction
- persona honesty
- anti-sycophancy
- reasoning rigor / false-premise correction

Pass condition:

- targeted axes or category-aware scoring beat broad cheap heuristics on the
  controlled battery.

### 3. Capacity Ladder

This lane tests the user's strongest practical hypothesis:

> the method may require a much more capable embedding model than typical small
> open-source embedders.

Run the same exact objective suite across:

- cheap OSS encoders
- stronger OSS encoders
- Gemini-family embeddings

Pass condition:

- a clear capability gradient on the same frozen benchmark.

### 4. Process Potential

This lane is the bridge from selection to training.

Use injected error-repair traces and cumulative-context deltas.

Pass condition:

- potential deltas localize degradation and repair better than final-only or
  isolated-step scoring.

## Unequivocal Value Standard

The project becomes serious enough for a strong external packet when all of the
following are true:

1. the code reranking lane is positive;
2. at least two more objective domains are also positive;
3. cheap OSS models fail or flatten while stronger models improve;
4. the behavior-basis battery shows meaningful discrimination on targeted
   qualities;
5. cost remains dramatically lower than LLM-as-judge;
6. failure modes are documented honestly.

Without Items 2 and 3, the project is promising but still too easy to dismiss
as benchmark luck.

## Highest-Value Targets

The program should spend most of its near-term budget on these five targets, in
this order:

1. `objective_math_gemini_v1`
   This is the cleanest first cross-domain check because the final metric is
   exact and cheap.
2. `tool_interpretation_gemini_v1`
   This tests whether the signal survives in a more operational, log-reading
   domain instead of only code.
3. `objective_math_oss_bge_v1`
   This is the first frozen capacity-ladder comparison on the same math suite.
4. `tool_interpretation_oss_bge_v1`
   This tells us whether the capacity gap persists in a non-code objective
   domain.
5. `behavior_basis_v2_gemini`
   This is not decisive by itself, but it tells us whether the geometry carries
   targeted behaviors the project actually cares about.

`process_potential_error_repair_v1` should start after the first four targets
because it is the bridge from "selection works" to "training is worth doing."

Until that lane exists and passes, the program must not mark the
training-readiness claim as passed, even if the selection lanes look strong.

## How To Use The System

Build the current report without running new experiments:

```bash
python scripts/orchestrate_research_system.py ^
  --manifest experiments/research_system_v1/program_manifest.json ^
  --output notes/research_system_v1\report
```

Execute all currently `ready` lanes:

```bash
python scripts/orchestrate_research_system.py ^
  --manifest experiments/research_system_v1/program_manifest.json ^
  --output notes/research_system_v1\report ^
  --execute-ready
```

Execute a focused slice:

```bash
python scripts/orchestrate_research_system.py ^
  --manifest experiments/research_system_v1/program_manifest.json ^
  --output notes/research_system_v1\report ^
  --execute-ready ^
  --only objective_math_gemini_v1 tool_interpretation_gemini_v1
```

## Hard Rules

- Do not add new lanes without specifying what claim they support.
- Do not promote a lane result unless its benchmark file is frozen in the repo.
- Do not use dataset overlap alone as the headline claim.
- Do not treat a positive single-domain result as training evidence.
- Always keep cheap baselines visible.

## Why This Makes The Project More Serious

This system creates three things the repo did not fully have before:

1. a **frozen manifest** of what the project is actually trying to prove;
2. a **cross-domain benchmark ladder** instead of one-off experiments;
3. a **claim gate report** that tells us what is proven, what is promising, and
   what still blocks training claims.

That is the difference between an interesting idea and a real research program.
