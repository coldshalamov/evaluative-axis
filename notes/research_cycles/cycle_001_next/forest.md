# Forest Mode: Cycle 001

Date: June 23, 2026
Cycle: `cycle_001_next`
Trigger: phase close and research-process correction

## Did This Test The Real Idea Or A Proxy?

The early HH-RLHF agreement tests mostly tested a proxy: overlap between an
embedding score and a historically noisy pairwise preference dataset.

The real idea is broader. It asks whether evaluative geometry can act as a
cheap, deterministic signal for making better training and inference decisions:
select, reject, weight, route, critique, or score process trajectories.

The next cycle must therefore test an intervention, not only a correlation.

## What Assumption Might Be False?

- HH-RLHF labels may not represent current or ideal assistant behavior.
- "Agreement" may punish the embedding for correcting bad labels.
- Final-response scoring may hide the reasoning or context that determines
  whether a response is good.
- One axis may be too blunt; the useful structure may be scalar plus basis:
  broad good/bad plus safety, truthfulness, persona honesty, and
  anti-sycophancy axes.
- A weak embedding model may reveal the mechanism but underestimate the ceiling.

## Did The Benchmark Lie?

The HH benchmark did not exactly lie, but it answered a narrower question than
the project needs. It measured overlap with Anthropic preference labels from a
particular collection process and policy era.

The full disagreement audit shows that the benchmark often treated bad labels,
both-bad pairs, and low-signal pairs as if they were clean ground truth. That
made the raw 53.8% number structurally misleading.

## What Did The Examples Show?

The examples showed that embedding disagreements were often meaningful:

- It preferred privacy-preserving refusal over address-seeking compliance.
- It preferred factual correction over false or unsupported claims.
- It preferred ontological honesty over assistant persona fabrication.
- It sometimes preferred more informative answers over empty deflection.
- It also failed in expected ways: tone, length, missing context, and
  sycophancy can still confuse short-context embedding models.

## What New Mechanism Became Visible?

- Dataset sanitation is central. The score can flag pairs that should not train
  a model at all.
- Candidate reranking is the cleanest no-training intervention.
- Critique/decomposition scoring may expose the conceptual structure the
  direct answer hides.
- Cumulative-context deltas are the plausible dense reward mechanism for later
  process supervision.
- Axis disagreement is diagnostic: different axes may reveal which objective a
  dataset actually measures.

## What New Application Became Plausible?

- Cheap best-of-N reranking.
- Preference-pair keep/drop/audit routing.
- LLM judge report scoring.
- Synthetic data filtering.
- Process reward from cumulative context.
- Tool-trace reward.
- Human review triage.
- Dataset quality audits.

## What Would Change With Better Embeddings?

A stronger embedding model should improve three bottlenecks:

- semantic resolution: fewer shallow jumps from "pleasant tone" to "good";
- context binding: better distinction between "lying" and "refused to lie";
- axis separation: better isolation of truthfulness, safety, quality, and
  sycophancy.

Longer context matters as much as dimension. Many statements reverse value
under context, so scoring only the final answer is a weak interface.

## What Would Change With Different Granularity?

- Response-only:
  - Cheapest, but most vulnerable to missing why the response was good or bad.
- Prompt+response:
  - Minimum viable interface for safety and factuality.
- Critique/decomposition:
  - Lets an LLM expose the evaluative parts, then lets embeddings score the
    report cheaply.
- Cumulative context:
  - Scores the trajectory after each step and can provide dense reward deltas.
- Tool trace:
  - Evaluates whether the model used external evidence well.

## Training Pipeline Implications

- Before training:
  - Filter low-quality corpora, detect bad pairs, weight examples.
- During SFT:
  - Prefer examples with high evaluative score and low contradiction/sycophancy
    markers.
- During preference training:
  - Drop both-bad pairs; margin-weight clean pairs; generate synthetic pairs
    from score gaps.
- During RL:
  - Use embedding reward as an auxiliary dense signal, especially on full
    context or critique text.
- During inference:
  - Rerank multiple candidates or trigger regeneration.
- During data generation:
  - Score and filter synthetic examples cheaply at scale.
- During human-review routing:
  - Prioritize high-disagreement and low-confidence cases.

## Broadest Useful Interpretation

The broadest useful interpretation is:

> Evaluative geometry is a cheap sensor of value-laden structure in language.
> Its first serious use is not replacing all human judgment. Its first serious
> use is making the pipeline less blind: choosing better candidates, rejecting
> bad data, scoring critiques, weighting pairs, and routing uncertainty.

That is already a publishable research direction if the intervention benchmark
shows practical lift.

## Next Experiment That Would Most Change Belief

One experiment:

Run the no-training candidate-selection intervention benchmark with direct and
decomposition scoring, then blind-judge method winners.

Why this one:

It tests the pipeline value directly without needing GPUs, training runs, or
faith in old preference labels.

What result would matter:

- Strong positive: embedding/decomposition selection beats random, length, and
  sentiment under blind review and approaches or beats vanilla LLM judge per
  dollar.
- Mechanistic positive: embedding does not always pick the blind-best answer,
  but reliably flags both-bad or mislabeled cases.
- Negative after autopsy: embedding loses because it tracks surface tone or
  length even under decomposition and prompt+response scoring.
