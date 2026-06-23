# Research Operating Mode

Date: June 23, 2026

This project should not be run like a benchmark-chasing script. The central
failure mode so far has been premature narrowing: treating a noisy dataset as
truth, optimizing around the metric it supplied, and missing the more important
phenomenon visible in the examples.

## Prime Directive

Before declaring what a result means, inspect what the model actually did.

If an embedding score disagrees with a dataset, do not call it an error until
the prompt and both answers have been read. The disagreement may be:

- an embedding failure;
- a dataset label error;
- a both-bad pair;
- a low-signal pair;
- evidence that the embedding tracks a newer or deeper norm than the dataset;
- evidence that the dataset measures a different objective.

## Required Interpretation Pass

Every experiment must include a written interpretation pass with these
questions:

1. What assumption did this metric make?
2. What would it mean if that assumption were false?
3. What did the top successes actually look like?
4. What did the top failures actually look like?
5. Are failures really failures, or did the benchmark mislabel them?
6. Are there pairs where both options are bad and should be filtered?
7. Does the result suggest a different use than the one tested?
8. What would a stronger embedding model, longer context, or different
   granularity change?
9. What training mechanism does this suggest beyond binary preference ranking?
10. What is the broadest useful interpretation that still fits the evidence?

## Never Use One Dataset As The Finish Line

HH-RLHF is not "goodness." PKU is not "goodness." SHP is not "goodness." An LLM
judge is not "goodness." Human graders are not "goodness."

They are sensors. They can be wrong, stale, noisy, socially shaped, or forced
into bad pairwise choices.

Raw agreement with any one of them is not the result. The result is the pattern
of agreement, disagreement, and what the disagreements reveal.

## Read The Examples

Any result over a preference dataset must include:

- at least 20 high-confidence embedding-over-dataset disagreements;
- at least 20 high-confidence dataset-over-embedding disagreements;
- at least 20 near-tie cases;
- examples where both answers score low;
- examples where both answers score high.

The written report must include what these examples mean.

## Treat Disagreement As Data

The full HH disagreement audit showed that a majority of apparent embedding
"errors" were not simple failures. Many were bad HH labels or pairs that should
not be used for training.

That means disagreement is not cleanup after the main result. Disagreement is
the result.

## Generate Mechanisms, Not Just Scores

After every result, ask what mechanism it suggests.

Possible mechanisms:

- data sanitation: remove bad pairs;
- pair weighting: train more on high-margin clean pairs;
- candidate reranking: choose better outputs from multiple samples;
- critique scoring: score an evaluator's decomposition instead of the answer;
- process scoring: score cumulative context after each reasoning step;
- tool-call scoring: score tool outputs and the model's interpretation of them;
- regeneration: if all candidates score low, generate new candidates;
- routing: send ambiguous or high-disagreement cases to humans or stronger
  judges.

## Granularity Is A Research Variable

Do not assume final-answer scoring is the right interface.

Test:

- response only;
- prompt plus response;
- prompt plus scratchpad plus response;
- generated critique;
- generated decomposition;
- cumulative context after each step;
- cumulative context after each turn;
- tool call plus interpretation;
- summary of reasoning;
- chunks versus whole context.

The user's core hypothesis is about evaluative decomposition, so the interface
may matter more than the raw embedding model.

## Think In Training Uses

The goal is not only to predict a dataset label. The broad question is:

> How could embedding-space evaluation improve model training, inference, data,
> or supervision?

Candidate uses:

- automatic preference-pair cleaning;
- cheap reward prior;
- auxiliary reward in RL;
- DPO pair generation;
- quality-weighted SFT data;
- process reward over reasoning traces;
- evaluation of tool-use trajectories;
- filtering generated synthetic data;
- pretraining-data curation;
- active-learning queue for human review.

## Hardware-Aware Strategy

Lack of GPU does not mean only weak experiments are possible.

No-hardware experiments that matter:

- reproduce axis convergence;
- rerank generated candidates and blind-judge winners;
- audit all disagreements, not only top examples;
- simulate cumulative process scoring on saved traces;
- compare scoring granularities;
- test adversarial examples;
- estimate cost versus LLM judge;
- package clean evidence for collaborators.

Hardware-needed experiments:

- long-context embedding model training;
- DPO/RL fine-tuning;
- process-reward training;
- large-scale pretraining data curation.

The local goal is to produce evidence strong enough to attract help for the
hardware-needed work.

## Conclusion Discipline

Never end a phase with only:

> It got X% agreement.

Always end with:

- what the examples show;
- what the metric hid;
- what the disagreement reveals;
- what mechanism this suggests;
- what broader hypothesis becomes more plausible;
- what experiment would actually change the decision.

The project is valuable precisely because the obvious metric can be misleading.
