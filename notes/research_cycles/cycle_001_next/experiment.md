# Experiment Mode: Cycle 001

Date: June 23, 2026
Cycle: `cycle_001_next`
Experiment name: No-training candidate-selection intervention benchmark

## Research Question

Can evaluative embedding geometry improve an actual pipeline decision:
selecting the best response from multiple candidates for the same prompt?

This is the right first intervention because it does not require GPU training.
It tests whether the signal changes an output users would see, rather than only
whether it agrees with a preference dataset.

## What This Can Answer

- Whether embedding-axis selection beats random, length, positive-tone, and
  simple lexical baselines.
- Whether direct answer scoring or decomposition/critique scoring is the better
  interface.
- Whether the signal is useful as a reranker or filter before any training.
- Whether disagreement examples reveal label noise, both-bad candidates, or
  true embedding failures.
- Whether the method is promising enough to justify a larger blinded run or a
  later DPO/RL test.

## What This Cannot Answer

- Whether embedding reward improves weights after training.
- Whether a short-context embedding model can score long reasoning traces.
- Whether one scalar good/bad axis is sufficient for every domain.
- Whether an embedding score is factual verification. It can reward truthful
  language patterns, but external facts still need tools or ground truth.
- Whether HH-RLHF is correct. This benchmark treats all labels as sensors, not
  truth.

## Hidden Assumptions

- Candidate sets contain real quality differences; otherwise no selector can
  show useful lift.
- Blind adjudication is better than inheriting dataset labels as ground truth.
- Prompt+response or decomposition text exposes enough context for the embedding
  evaluator to distinguish helpfulness from mere tone.
- If a method selects "both bad" outputs often, that is a pipeline design
  failure unless the method also has a reject/regenerate path.
- A useful cheap signal may have modest pairwise agreement but still large
  value when used for filtering, weighting, or routing.

## Protocol Frozen Before Run

- Dataset / prompt source:
  - Pilot: 50 prompts manually assembled from known hard categories:
    persona honesty, harmful compliance, false premise correction,
    factuality, empty deflection, sycophancy, refusal quality, and general
    helpfulness.
  - Main: 200-500 prompts from public multi-response datasets such as
    UltraFeedback, SHP, HH disagreement cases, and synthetic hard prompts.
- Candidate generation:
  - Minimum 4 candidates per prompt.
  - Candidate sources should be mixed when possible: existing dataset
    responses, one strong model, one weaker model, and one intentionally
    flawed or regenerated answer.
- Sample size:
  - Smoke: 5-10 prompts to validate code and review packets.
  - Pilot: 50 prompts.
  - Main: 200+ prompts once the protocol is stable.
- Embedding model:
  - Preferred: Gemini embedding model available through the API.
  - Fallback: existing BGE/MPNet results only for plumbing, not decisive claims.
- Axes:
  - `general_evaluative`: broad quality/good vs bad.
  - `harm_reduction`: refusal/safety/reduced harm vs harmful compliance.
  - `truthfulness`: calibrated truth vs confident falsehood.
  - `persona_honesty`: AI identity honesty vs fabricated human persona.
  - `anti_sycophancy`: calibrated disagreement vs flattering false agreement.
- Text interface / granularity:
  - Direct: `User + Candidate response`.
  - Decomposition: `User + Candidate response + good parts + bad parts`.
  - Later extension: cumulative context after each reasoning step.
- Scoring formula:
  - For embedding backends: normalized dot product with each axis.
  - Primary combined score:
    `0.35*general + 0.25*truthfulness + 0.20*harm + 0.10*persona + 0.10*anti_sycophancy`.
  - Also report every axis separately. Do not hide axis-specific failures.
- Baselines:
  - Random selection with fixed seed.
  - Length: prefer longer response.
  - Positive-tone / sentiment proxy.
  - Vanilla LLM judge if API budget allows.
  - Dataset label where available, treated as one sensor.
- Primary metric:
  - Blind pairwise win rate of each method's selected response against random
    and length-selected responses.
- Secondary metrics:
  - Exact match to blind best candidate.
  - Mean blind rating of selected candidate.
  - Reject/regenerate precision for both-bad candidate sets.
  - Cost per 1,000 selections.
  - Failure taxonomy counts after autopsy.
- Stopping rule:
  - Stop the pilot only after at least 50 prompts and at least 25 method
    disagreements have been inspected.
  - Do not declare a negative result without reading examples from every
    disagreement bucket.
- Failure criteria:
  - Embedding selection loses to random or length under blind review.
  - Embedding selection mainly picks flattering, verbose, or unsafe answers.
  - Decomposition scoring improves only because the decomposition itself leaks
    the expected label.
  - The method cannot identify both-bad sets better than chance.

## Required Baselines

- Random: fixed seed, repeated at least 20 times for main runs.
- Length: word count of candidate response.
- Sentiment / positive tone: cheap lexical or VADER-style proxy.
- LLM judge if affordable: judge sees prompt plus candidates, no method labels.
- Other cheap heuristic: refusal detector for safety-heavy subsets.

## Required Saved Outputs

- Raw scores: per prompt, candidate, interface, axis, and method.
- Config: model, anchors, weights, prompts, candidate source, random seed.
- Code: exact script and command used.
- Examples: selected disagreements, near ties, both-low sets, and method wins.
- Summary: aggregate metrics plus interpretation that separates benchmark
  agreement from actual quality.
- Blind packet: review file with anonymized candidate ordering and no method
  names.

## Expected Interpretations

If positive:

Embedding evaluative geometry is not merely a retrospective dataset-correlation
signal. It can improve a concrete selection decision cheaply, which supports
reranking, filtering, pair weighting, judge-of-judges, and later training uses.

If negative:

Read the examples first. A negative result can mean the axis is wrong, the text
interface is wrong, the benchmark is wrong, the candidate set has no signal, or
the embedding model is too weak/short-context. Only after autopsy can it count
against the core idea.

If mixed:

Map where it works. A mixed result is expected if safety, truthfulness, persona
honesty, and general quality are partially independent axes. The likely output
is a routing rule: use different axes or interfaces for different prompt types.

## Autopsy Sampling Plan

- Top embedding-over-benchmark disagreements: examples where embedding picks a
  candidate the dataset or judge scores low.
- Top benchmark-over-embedding disagreements: examples where blind review or
  dataset labels beat embedding selection.
- Near-ties: small embedding margins where no strong claim should be made.
- Both-low cases: all candidates bad; measure whether score supports rejection.
- Both-high cases: all candidates acceptable; measure whether selection matters.
- Random cases: guard against cherry-picking and narrative overfit.

## First Runnable Artifact

The script `scripts/run_cycle001_intervention.py` implements the benchmark
scaffold. It supports:

- lexical smoke mode, requiring no API;
- Gemini embedding mode, using `.env.local` / environment keys;
- direct and decomposition text interfaces;
- weighted multi-axis selection;
- generated score tables, method selections, and blind review packets.

The seed fixture
`notes/research_cycles/cycle_001_next/seed_candidates.json` is not evidence. It
exists to validate plumbing on known hard cases before running a blinded pilot.
