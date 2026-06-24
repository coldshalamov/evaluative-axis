# Rigor Guardrails For This Research

Date: June 22, 2026

## Purpose

This file exists because the project is vulnerable to the exact failure mode the
user identified: running messy tests, over-interpreting convenient results,
changing the framing after seeing numbers, and treating weak evidence as if it
settles the idea.

The goal is to make future work harder to fool ourselves with.

## Rules For Future Experiments

### 1. Separate Hypothesis From Exploration

Every run must be labeled as one of:

- exploratory: useful for finding ideas, not for making strong claims;
- confirmatory: frozen protocol, held-out data, report all results;
- intervention: tests whether the signal improves an actual selection or
  training-adjacent decision.

Never present an exploratory result as confirmatory.

### 2. Freeze Before Running

For confirmatory or intervention runs, write down before running:

- dataset or prompt source;
- sample size;
- embedding model;
- axes;
- scoring formula;
- baselines;
- primary metric;
- subgroup metrics;
- stopping rule;
- failure criteria.

After seeing the numbers, do not change these and call it the same experiment.

### 3. Do Not Treat Any Dataset As Ground Truth

HH-RLHF, PKU-SafeRLHF, SHP, LLM judges, human graders, and embedding axes are
all sensors. They are not the target itself.

Reports must say "overlap with [sensor]" rather than "accuracy" unless the
label source is specifically being treated as ground truth for a narrow reason.

### 4. Always Include Cheap Baselines

Every preference or reranking experiment must include:

- random;
- length;
- sentiment or positive-tone score;
- direct lexical/surface heuristic if relevant;
- standard LLM judge when affordable.

An embedding method is not interesting if it only beats random while losing to
cheap obvious baselines.

### 4A. No Label Leakage In Scored Inputs

For any confirmatory or intervention claim, the text passed to the embedding
model or judge must not contain:

- the answer key;
- labels such as "better answer", "worse answer", "preferred", or "rejected";
- experimenter-written "Good parts" / "Bad parts" notes;
- decompositions that already identify the intended winner;
- method names or category labels that reveal which candidate should score
  higher.

If explicit evaluative decomposition is supplied by the experimenter, the result
must be labeled `oracle`, `upper_bound`, or `leakage_check`. It may verify that
the scoring code can read evaluative language, but it is not evidence that the
method inferred quality.

Valid decomposition tests must use one of:

- blind LLM-generated decompositions produced without labels or answer keys;
- human-written decompositions produced blind to the expected label;
- neutral feature extraction with evaluative words removed;
- raw answer scoring with no decomposition.

Every decomposition experiment must include a leakage audit before any result
is promoted.

### 5. Always Audit Disagreements

For every dataset-overlap result, inspect disagreements.

At minimum:

- top high-confidence embedding-over-dataset disagreements;
- top high-confidence dataset-over-embedding disagreements;
- near-tie cases;
- random disagreement sample.

Manual or LLM grading must be blind when used for confirmatory claims.

### 6. Do Not Optimize On The Test Set

If an axis, prompt wrapper, decomposition format, or model is chosen after
looking at a dataset, it must be evaluated on a different held-out dataset or
split before making claims.

### 7. Report Negative And Anti-Correlated Results

If an axis goes below random, report it. Anti-correlation may mean:

- the axis is bad;
- the dataset measures a different objective;
- the prompt wrapper is bad;
- the embedding model is not suitable;
- the sample is small or skewed.

Do not silently drop failed axes.

### 8. Reproduce Reported Findings Before Promoting Them

If a result is reported from an attached conversation, another assistant, an
external notebook, or an unsaved command run, label it as reported until it is
reproduced in this workspace with saved inputs, code, and outputs.

This especially applies to:

- axis-convergence numbers;
- claims about which axis is "best";
- literature-map claims;
- API/model capability claims;
- any result used as a paper headline.

Reported findings can guide exploration. They should not become central
evidence until reproduced.

### 9. Distinguish Three Claims

Keep these separate:

1. Geometry claim: evaluative directions exist in embedding space.
2. Measurement claim: a scoring protocol can recover useful judgments.
3. Intervention claim: using that score improves outputs, labels, or training.

The project is currently strongest on the geometry/measurement hints and
weakest on the intervention claim.

### 10. Prefer Intervention Tests Over More Correlation

Dataset overlap is limited because datasets are noisy and historically
contingent. The more decisive question is:

> If we use the embedding score to select one output from several candidates,
> is the selected output better under blind review?

Future effort should prioritize this.

### 11. Preserve The Human Framing

Do not sand the idea down into generic ML boilerplate. The human framing is part
of the contribution:

- good/bad as the primary evaluative axis;
- good as a self-regularizing signal across many meanings;
- reasoning as decomposition into good-making and bad-making parts;
- cumulative context scoring as dense supervision;
- persona honesty as an example of latent evaluative structure anticipating
  later assistant norms.

Translate these ideas into ML terms, but keep the original conceptual force.

### 12. Use Skeptical Language

Avoid:

- "proves";
- "solves";
- "alignment";
- "objective goodness";
- "the effect is weak" when only crude tests were run;
- "the model failed" when the benchmark may be noisy.

Prefer:

- "overlap";
- "sensor";
- "evidence";
- "exploratory";
- "label-noise candidate";
- "measurement interface";
- "under this protocol."

## Minimum Report Template

Each future experiment should include:

1. Research question.
2. Frozen protocol or exploratory status.
3. Dataset/sensor description.
4. Model and axes.
5. Baselines.
6. Main numbers.
7. Confidence intervals or statistical test.
8. Disagreement audit.
9. Failure modes.
10. Interpretation with limitations.
11. Decision.
12. Next experiment.

## Current Best Next Experiment

Run a no-training candidate-selection benchmark.

Protocol:

1. Choose 100 prompts from mixed sources.
2. Generate 4 candidate answers per prompt.
3. Ask an LLM to produce a short blind feature report for each answer:
   strengths, weaknesses, risks, uncertainties, factual claims, missing context,
   and practical usefulness. Do not provide answer labels or ask for the winner
   in this pass.
4. Score both raw answers and decompositions with frozen embedding axes.
5. Select winners by random, length, sentiment, LLM judge, direct embedding,
   and embedding-scored decomposition.
6. Blind-judge the selected winners pairwise.
7. Report win rates and cost.

Primary question:

> Does embedding-scored blind decomposition select better answers than random,
> length, sentiment, refusal heuristics, direct answer scoring, and direct LLM
> judging?

This tests the user's core decomposition thesis without requiring training
hardware.
