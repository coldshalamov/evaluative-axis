# Full HH-RLHF Disagreement Grading Summary

Date: June 22, 2026

Source: full manual/LLM-assisted grading of all 231 disagreement cases from a
500-pair HH-RLHF sample.

Run described in grading file:

- Embedding model: `BAAI/bge-small-en-v1.5`
- Axis/scoring: contextual harm-reduction axis
- Raw HH agreement: 269/500 = 53.8%
- Raw disagreements: 231/500 = 46.2%

## Main Result

All 231 disagreement cases were graded into three buckets:

| Bucket | Count | Percent of Disagreements |
| --- | ---: | ---: |
| Embedding preferred the better response | 65 | 28.1% |
| HH label preferred the better response | 44 | 19.0% |
| Exclude: both bad, both trivial, marginal, or no useful signal | 122 | 52.8% |

Among the 109 gradeable disagreements, the embedding-preferred response was
judged better in 65/109 cases = 59.6%.

## Corrected-Agreement Interpretation

If the 269 agreement cases are treated as embedding-correct and the 122
excluded disagreements are removed as non-informative training pairs, then:

- total gradeable pairs: 269 + 65 + 44 = 378
- embedding-correct gradeable pairs: 269 + 65 = 334
- corrected gradeable agreement: 334/378 = 88.4%

Sensitivity estimates from the grading file:

| Assumption | Corrected Gradeable Agreement |
| --- | ---: |
| Use full grading as written | 88.4% |
| Assume 30% of `EMBEDDING_RIGHT` calls are wrong | 83.3% |
| Assume 50% of `EMBEDDING_RIGHT` calls are wrong | 79.9% |

## Why This Changes The Interpretation

The raw 53.8% HH agreement is not a good estimate of the embedding signal's
quality. The full disagreement grading suggests that most apparent errors fall
into one of three categories:

1. HH appears to prefer the worse response.
2. Both responses are bad and a training system should reject both.
3. The pair is trivial, marginal, or lacks enough signal to train on.

This is directly relevant to automatic training. A scalar embedding reward would
not have to force a preference between two bad responses. It could score both
low and filter the pair, use an absolute threshold, or request a regenerated
candidate. That means many "excluded" HH disagreements are not failures for an
embedding-reward pipeline; they are examples where pairwise preference labels
are the wrong interface.

## Common HH Label Problems Found

The full grading identified recurring HH-RLHF label issues:

- rewarding compliance or near-compliance with harmful requests;
- rewarding misinformation over correction;
- rewarding empty or evasive non-answers over substantive answers;
- rewarding fabricated assistant persona claims;
- rewarding discriminatory agreement or harmful stereotypes;
- rewarding old deflection-heavy behavior over direct ethical engagement.

## What This Supports

This result strongly supports the idea that embedding-axis scoring can be useful
for:

- label-noise discovery;
- filtering bad preference pairs;
- reranking generated candidates;
- weighting LLM-judge or critique outputs;
- creating cheap auxiliary reward signals;
- selecting high-value cases for human review.

It also supports the user's central objection to the original analysis: raw
overlap with HH-RLHF is not the same thing as quality, because HH-RLHF is a
noisy, historically situated sensor.

## What It Does Not Yet Prove

This still does not by itself prove that embedding reward improves model
training. Remaining limitations:

- the grading was not blind human adjudication;
- the grader was another LLM/research assistant, not an independent panel;
- the 269 agreement cases were assumed correct rather than fully audited;
- the result is from one 500-pair HH sample and one scoring setup;
- it is still a retrospective audit, not an intervention.

The next decisive test should therefore be a no-training intervention:
generate multiple candidate responses, score and select candidates using the
embedding-axis method, then blind-judge whether the selected outputs beat
random, length, sentiment, and vanilla LLM-as-judge selection.

## Current Bottom Line

The evidence now points to a stronger claim than "there is a weak correlation
with HH." A better summary is:

> A tiny frozen embedding model plus a simple evaluative projection disagrees
> with HH-RLHF in ways that often look better than HH labels, and more than half
> of its apparent HH errors are low-signal or both-bad pairs that an automatic
> training pipeline should filter rather than imitate.

That makes the approach much more plausible as an automatic reward/filtering
component than the raw 53.8% agreement number suggested.
